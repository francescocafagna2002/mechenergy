"""
model.py — Energy Fingerprints Hackathon (AEW)

Módulo reutilizable para entrenar, evaluar y explicar un Random Forest
por cada activo (EV, PV, Heat Pump, Batería).

Uso esperado una vez tengáis la tabla de features real:

    import pandas as pd
    from model import entrenar_modelo, evaluar_modelo, explicar_prediccion, aplicar_a_todos

    X = pd.read_csv("features_labeled.csv", index_col="MP_ID")   # ~337 filas fiables
    y_ev = ...  # Series con 1/0 para EV (excluyendo los "?" desconocidos)

    modelo_ev, X_test, y_test, X_train, y_train = entrenar_modelo(X, y_ev, nombre="EV")
    evaluar_modelo(modelo_ev, X_test, y_test, nombre="EV")

    X_completo = pd.read_csv("features_90k.csv", index_col="MP_ID")
    predicciones_ev = aplicar_a_todos(modelo_ev, X_completo, nombre_columna="prob_EV")

    explicar_prediccion(modelo_ev, X, mp_id=100234)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False


# ---------------------------------------------------------------------------
# 0. PREPARAR LABELS — convierte los valores crudos de GIGI.csv a 0/1
# ---------------------------------------------------------------------------

def preparar_labels(columna: pd.Series) -> pd.Series:
    """
    Convierte una columna cruda de GIGI.csv (valores 'x', 'X', '-', NaN, u otros)
    en 0/1 para poder entrenar el modelo.

    Regla actual (naive, a revisar en el futuro cuando se implemente PU-learning):
    - 'x' / 'X'    -> 1  (confirmado que SÍ tiene el activo)
    - '-' / vacío  -> 0  (tratado como NO, aunque en realidad significa "no se sabe con certeza")
    - cualquier otro valor raro (ej. 'Boiler', '.') -> se descarta, no se adivina
    """
    # Los nulos reales (None/NaN) se tratan aparte: .astype(str) los convertiría
    # a la cadena "none", que NO coincide con "nan" y se escaparía del mapeo.
    es_nulo = columna.isna()

    columna_limpia = columna.astype(str).str.strip().str.lower()

    mapa = {
        "x": 1,
        "-": 0,
    }

    y = columna_limpia.map(mapa)
    y[es_nulo] = 0  # vacío -> 0, misma regla que '-'

    no_reconocidos = columna[y.isna()]
    if len(no_reconocidos) > 0:
        print(f"Aviso: {len(no_reconocidos)} valores no reconocidos, se descartan: "
              f"{no_reconocidos.unique().tolist()}")

    return y


# ---------------------------------------------------------------------------
# 1. ENTRENAMIENTO
# ---------------------------------------------------------------------------

def entrenar_modelo(X: pd.DataFrame, y: pd.Series, nombre: str = "activo",
                     test_size: float = 0.2, random_state: int = 42,
                     n_estimators: int = 200, max_depth: int | None = None):
    """
    Entrena un Random Forest para un único activo (EV, PV, HP o batería).

    X: DataFrame de features, índice = MP_ID, solo columnas numéricas.
    y: Series con 1 (confirmado sí) / 0 (tratado como no), MISMO índice que X.
       Antes de llamar a esta función, filtrad las filas con label "?" (desconocido)
       si estáis usando la estrategia PU-learning "reliable negative".

    Devuelve: (modelo_entrenado, X_test, y_test, X_train, y_train) — X_train/y_train
    sirven para comprobar overfitting (comparar AUC en train vs en test).
    """
    # Alineamos X e y por índice, por si acaso no vienen ya en el mismo orden
    X_alineado = X.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_alineado, y,
        test_size=test_size,
        stratify=y,  # importante con pocos datos: mantiene la proporción de clases
        random_state=random_state,
    )

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,     # None = sin límite (por defecto); pon un número para limitar y reducir overfitting
        min_samples_leaf=3,      # evita hojas basadas en 1 solo caso (peligroso con pocos datos)
        class_weight="balanced", # compensa el desbalance típico (pocos positivos)
        random_state=random_state,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    print(f"[{nombre}] Entrenado con {len(X_train)} casas (train) / {len(X_test)} (test)")
    print(f"[{nombre}] Positivos en train: {y_train.sum()} de {len(y_train)}")

    return modelo, X_test, y_test, X_train, y_train


# ---------------------------------------------------------------------------
# 2. EVALUACIÓN
# ---------------------------------------------------------------------------

def evaluar_modelo(modelo, X_test: pd.DataFrame, y_test: pd.Series, nombre: str = "activo"):
    """Imprime precision/recall/F1 y AUC, y devuelve la importancia de features ordenada."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(f"\n=== Evaluación: {nombre} ===")
    print(classification_report(y_test, y_pred, target_names=["No", "Sí"]))
    if len(set(y_test)) > 1:  # AUC necesita ambas clases presentes en el test
        print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

    importancias = pd.Series(
        modelo.feature_importances_, index=X_test.columns
    ).sort_values(ascending=False)

    print("\nTop 10 features más importantes:")
    print(importancias.head(10))

    return importancias


# ---------------------------------------------------------------------------
# 3. EXPLICABILIDAD (SHAP) — para el bloque "EVIDENCE" tipo Household A-10482
# ---------------------------------------------------------------------------

def explicar_prediccion(modelo, X: pd.DataFrame, mp_id, top_n: int = 5):
    """
    Descompone la predicción de UNA casa concreta en qué features empujaron
    hacia "sí" o hacia "no". Requiere `pip install shap`.
    """
    if not SHAP_DISPONIBLE:
        print("SHAP no está instalado. Ejecuta: pip install shap")
        return None

    fila = X.loc[[mp_id]]
    proba = modelo.predict_proba(fila)[0, 1]

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(fila)

    # El formato de salida varía según la versión de shap:
    # - lista [array_clase0, array_clase1]              -> nos quedamos con clase 1
    # - array de forma (1, n_features)                  -> una sola fila, una sola clase
    # - array de forma (1, n_features, n_clases)         -> nos quedamos con la última clase
    if isinstance(shap_values, list):
        valores = shap_values[1][0]
    else:
        valores = np.asarray(shap_values)
        if valores.ndim == 3:       # (n_muestras, n_features, n_clases)
            valores = valores[0, :, -1]
        else:                        # (n_muestras, n_features)
            valores = valores[0]

    contribuciones = pd.Series(valores, index=X.columns).sort_values(key=abs, ascending=False)

    print(f"\n=== Household {mp_id} — probabilidad: {proba:.0%} ===")
    print(f"Top {top_n} features que más influyeron en la predicción:")
    for feat, val in contribuciones.head(top_n).items():
        direccion = "↑ hacia SÍ" if val > 0 else "↓ hacia NO"
        print(f"  {feat}: valor={fila[feat].values[0]:.3f}  |  impacto SHAP={val:+.3f} ({direccion})")

    return contribuciones


# ---------------------------------------------------------------------------
# 4. APLICAR A LOS 90K (inferencia a escala)
# ---------------------------------------------------------------------------

def aplicar_a_todos(modelo, X_completo: pd.DataFrame, nombre_columna: str = "probabilidad") -> pd.Series:
    """
    Aplica el modelo ya entrenado a TODAS las casas (incluidas las ~89.663 sin label).
    Devuelve una Series con la probabilidad de "sí" para cada MP_ID.
    """
    probas = modelo.predict_proba(X_completo)[:, 1]
    resultado = pd.Series(probas, index=X_completo.index, name=nombre_columna)
    print(f"Predicciones generadas para {len(resultado)} casas.")
    print(f"Distribución: media={resultado.mean():.2%}, "
          f"% con prob>50%={((resultado > 0.5).mean()):.2%}")
    return resultado


# ---------------------------------------------------------------------------
# 4b. GUARDAR / CARGAR EL MODELO — para no tener que reentrenar cada vez
# ---------------------------------------------------------------------------

def guardar_modelo(modelo, ruta: str):
    """
    Guarda el modelo entrenado en disco (formato .pkl vía joblib).
    Guarda TAMBIÉN, automáticamente, el orden y nombre de las columnas que
    vio en el entrenamiento (sklearn lo almacena en modelo.feature_names_in_
    si se entrenó con un DataFrame) — al cargar y predecir, hay que darle
    las columnas EXACTAMENTE en ese mismo orden y con esos mismos nombres.
    """
    import joblib
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str):
    """Carga un modelo previamente guardado con guardar_modelo()."""
    import joblib
    modelo = joblib.load(ruta)
    print(f"Modelo cargado desde: {ruta}")
    return modelo


# ---------------------------------------------------------------------------
# 5. DEMO CON DATOS SINTÉTICOS — para probar el pipeline YA, sin esperar
#    a que lleguen las features reales del equipo.
# ---------------------------------------------------------------------------

def _generar_datos_sinteticos(n_casas: int = 337, n_features: int = 20, random_state: int = 42):
    """Genera una X/y de juguete con la misma forma que tendréis en el proyecto real."""
    rng = np.random.default_rng(random_state)

    columnas = [
        "media_noche", "media_manana", "media_mediodia", "media_tarde_noche",
        "std_noche", "consumo_negativo_pct", "corr_temperatura",
        "n_escalones", "magnitud_pico_kW", "duracion_pico_h", "hora_inicio_pico",
        "n_platos_totales", "n_platos_por_semana", "magnitud_media_platos",
        "energia_freq_diaria", "energia_freq_semanal", "autocorr_24h",
        "plz_agrupado", "mes_sin", "mes_cos",
    ][:n_features]

    X = pd.DataFrame(
        rng.normal(size=(n_casas, len(columnas))),
        columns=columnas,
        index=pd.RangeIndex(100000, 100000 + n_casas, name="MP_ID"),
    )

    # y sintético: relacionado con un par de features, para que el modelo tenga algo que aprender
    logit = 1.5 * X["magnitud_pico_kW"] + 1.2 * X["n_platos_totales"] - 0.5
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.random(n_casas) < prob).astype(int), index=X.index, name="EV")

    return X, y


if __name__ == "__main__":
    print("=== DEMO con datos sintéticos (mientras llegan las features reales) ===\n")

    X, y = _generar_datos_sinteticos()
    print(f"Datos sintéticos: {X.shape[0]} casas, {X.shape[1]} features, "
          f"{y.sum()} positivos de EV\n")

    modelo, X_test, y_test, X_train, y_train = entrenar_modelo(X, y, nombre="EV (sintético)")
    importancias = evaluar_modelo(modelo, X_test, y_test, nombre="EV (sintético)")

    # Explicar una casa concreta del set de test
    mp_id_ejemplo = X_test.index[0]
    explicar_prediccion(modelo, X, mp_id=mp_id_ejemplo)

    # Simular "aplicar a todos" usando el mismo X sintético como si fueran los 90k
    predicciones = aplicar_a_todos(modelo, X, nombre_columna="prob_EV")
    print("\nPrimeras 5 predicciones:")
    print(predicciones.head())