"""
pu_montecarlo.py — Energy Fingerprints Hackathon (AEW)

Monte Carlo PU-learning, genérico para cualquier activo (PV, EV, HeatPump,
Batería). No toca model.py ni load.py: son funciones nuevas que se apoyan en
las ya existentes (construir_vector_labels con tratar_ausencia_como_negativo
en sus dos modos).

Problema: "X" = confirmado, "-"/vacío = NO REPORTADO (no "no tiene"). Por eso
NO hay negativos fiables, solo positivos (P) y un pool sin etiqueta (U).

Uso esperado (ver entrenar_pv_pu.py):

    feature, signo = seleccionar_feature(X_P, X_U)
    es_monotona, tabla = verificar_monotonia(y_naive, X_labelled, feature, signo)
    p = construir_pesos(X_U, feature, signo, pi, s)
    scores_U, scores_held = entrenar_mc_pu(X_P_train, X_U, p, n_iter=100, X_eval=X_P_held)
    resumen_evaluacion(scores_held, scores_U, pi)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Mismos hiperparámetros que entrenar_modelo() en model.py, para que el RF de
# cada ronda del Monte Carlo sea comparable al baseline. Se duplican aquí en
# vez de importar entrenar_modelo porque esa función hace train_test_split y
# evalúa internamente, y en el bucle MC no queremos ninguna de las dos cosas.
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# 1. Selección de la feature guía (AUC univariante, congelada antes del bucle)
# ---------------------------------------------------------------------------

def seleccionar_feature(X_P: pd.DataFrame, X_U: pd.DataFrame):
    """
    Elige la feature con mayor poder de separación UNIVARIANTE entre P (1) y
    U (0), usando esa sola columna como score de un AUC. No se usa la
    importancia del random forest porque esa mide utilidad en combinación con
    el resto de features, no separación monótona en solitario.

    AUC < 0.5 no se descarta: se invierte el signo (la feature separa, pero
    "al revés").

    Devuelve (nombre_feature, signo) — guardar y reutilizar esto en vez de
    recalcularlo dentro del bucle Monte Carlo (si no, sería circular).
    """
    y_aux = np.concatenate([np.ones(len(X_P)), np.zeros(len(X_U))])

    mejor_feature, mejor_signo, mejor_distancia, mejor_auc = None, None, -1.0, None
    for col in X_P.columns:
        score = np.concatenate([X_P[col].to_numpy(), X_U[col].to_numpy()])
        auc = roc_auc_score(y_aux, score)
        distancia = abs(auc - 0.5)
        if distancia > mejor_distancia:
            mejor_feature = col
            mejor_signo = 1 if auc >= 0.5 else -1
            mejor_distancia = distancia
            mejor_auc = auc

    print(f"[PU-MC] Feature seleccionada: '{mejor_feature}' (AUC={mejor_auc:.3f}, signo={mejor_signo:+d})")
    return mejor_feature, mejor_signo


# ---------------------------------------------------------------------------
# 2. Chequeo de monotonía sobre los datos etiquetados (diagnóstico, no entra
#    en el entrenamiento)
# ---------------------------------------------------------------------------

def verificar_monotonia(y_naive: pd.Series, X_labelled: pd.DataFrame, feature: str,
                         signo: int, n_bandas: int = 3, alpha: float = 0.05):
    """
    Corta las filas etiquetadas (y_naive: "X"->1, "-"/vacío->0, del modo naive
    de construir_vector_labels) en n_bandas por la feature orientada, para
    inspección visual (tabla de proporción de "X" por banda).

    El criterio de monotonía en sí NO se decide sobre esas pocas bandas: con
    solo 3, la media cruda es sensible a un único household de diferencia,
    sobre todo cuando la proporción ya está saturada cerca del techo (p.ej.
    128/137 vs 127/138 en PV — un solo caso cambia el veredicto sin que eso
    signifique que la feature no separe). En su lugar se usa la correlación
    de Spearman (rango) entre la feature orientada y el label naive sobre
    TODOS los datos etiquetados, que mide la tendencia monótona global y es
    robusta a ese ruido puntual.

    Devuelve (es_monotona: bool, tabla: Series banda->proporción de X).
    Si es_monotona es False, el heurístico no aporta señal y hay que usar
    s=0 (PU bagging uniforme) en vez del Monte Carlo ponderado.
    """
    idx_comun = y_naive.index.intersection(X_labelled.index)
    valores = signo * X_labelled.loc[idx_comun, feature]
    etiquetas = y_naive.loc[idx_comun]

    bandas = pd.qcut(valores, n_bandas, labels=False, duplicates="drop")
    tabla = etiquetas.groupby(bandas).mean()
    print(f"[PU-MC] Proporción de 'X' por banda de '{feature}' (orientada):")
    print(tabla)

    rho, p_valor = spearmanr(valores, etiquetas)
    es_monotona = bool(rho > 0 and p_valor < alpha)
    print(f"[PU-MC] Spearman(feature, label) = {rho:.3f} (p={p_valor:.2e}) "
          f"-> ¿monotonía confirmada? {es_monotona}")

    return es_monotona, tabla


# ---------------------------------------------------------------------------
# 3. Pesos de muestreo p (ancla en pi, no en 0.5)
# ---------------------------------------------------------------------------

def construir_pesos(X_U: pd.DataFrame, feature: str, signo: int, pi: float, s: float) -> pd.Series:
    """
    p = clip(pi + s*(r - 0.5), 0, 1), con r = rank percentil de la feature
    orientada dentro de U. s=0 reduce esto exactamente a PU bagging uniforme
    (p constante = pi para todas las filas de U).
    """
    r = (signo * X_U[feature]).rank(pct=True)
    p = (pi + s * (r - 0.5)).clip(lower=0.0, upper=1.0)
    p.name = "p_muestreo"
    return p


# ---------------------------------------------------------------------------
# 4. Bucle Monte Carlo: muestrea labels de U, RE-ENTRENA, acumula predicciones
# ---------------------------------------------------------------------------

def entrenar_mc_pu(X_P_train: pd.DataFrame, X_U: pd.DataFrame, p: pd.Series,
                    n_iter: int = 100, random_state: int = 42,
                    X_eval: pd.DataFrame = None):
    """
    Por cada una de las n_iter rondas:
      - etiqueta de U ~ Bernoulli(p) (una muestra nueva cada ronda)
      - entrena un RF NUEVO sobre P_train (todo 1) + esa U muestreada
      - predice sobre X_U (y opcionalmente sobre X_eval) y acumula
    Al final promedia. Promediar las ETIQUETAS sin reentrenar convergería a p
    y no aportaría nada (E[Bernoulli(p)] = p) — lo que se promedia aquí son
    las PREDICCIONES del modelo, no los labels muestreados.

    X_eval: puntos extra a puntuar cada ronda con el mismo modelo (p.ej. un
    held-out de P, para poder medir recall). Opcional.

    Devuelve (scores_U, scores_eval) — scores_eval es None si no se pasó X_eval.
    """
    rng = np.random.default_rng(random_state)
    p_vals = p.loc[X_U.index].to_numpy()

    acumulado_U = np.zeros(len(X_U))
    acumulado_eval = np.zeros(len(X_eval)) if X_eval is not None else None

    for _ in range(n_iter):
        etiquetas_U = rng.binomial(1, p_vals)

        X_train = pd.concat([X_P_train, X_U])
        y_train = np.concatenate([np.ones(len(X_P_train)), etiquetas_U])

        modelo = RandomForestClassifier(
            **RF_PARAMS, random_state=int(rng.integers(0, 2**31 - 1))
        )
        modelo.fit(X_train, y_train)

        acumulado_U += modelo.predict_proba(X_U)[:, 1]
        if X_eval is not None:
            acumulado_eval += modelo.predict_proba(X_eval)[:, 1]

    scores_U = pd.Series(acumulado_U / n_iter, index=X_U.index, name="prob_mc_pu")
    scores_eval = (
        pd.Series(acumulado_eval / n_iter, index=X_eval.index, name="prob_mc_pu")
        if X_eval is not None else None
    )
    return scores_U, scores_eval


# ---------------------------------------------------------------------------
# 4b. PU bagging (protocolo sin Monte Carlo, sin p ponderado: subconjunto
#     uniforme de U del tamaño de P en cada ronda, promedio solo sobre las
#     rondas out-of-bag)
# ---------------------------------------------------------------------------

def entrenar_pu_bagging(X_P: pd.DataFrame, X_U: pd.DataFrame, n_iter: int = 100,
                         random_state: int = 42, X_eval: pd.DataFrame = None):
    """
    Por cada una de las n_iter rondas:
      - se muestrea de U un subconjunto SIN reemplazo, del mismo tamaño que P,
        tratado como negativo provisional
      - se entrena un RF NUEVO sobre P (todo 1) vs ese subconjunto (todo 0)
      - se predice sobre las filas de U que NO se muestrearon esa ronda
        (out-of-bag) y se acumula
    Al final, cada fila de U se promedia solo sobre las rondas en las que
    estuvo out-of-bag (no todas las filas lo están el mismo número de veces).

    X_eval: puntos extra a puntuar cada ronda con el mismo modelo (p.ej. un
    held-out de P), promediados sobre las n_iter rondas completas. Opcional.

    Devuelve (scores_U, scores_eval) — scores_eval es None si no se pasó X_eval.
    """
    rng = np.random.default_rng(random_state)
    n_P = len(X_P)
    n_U = len(X_U)
    indices_U = np.arange(n_U)

    suma_oob = np.zeros(n_U)
    conteo_oob = np.zeros(n_U)
    acumulado_eval = np.zeros(len(X_eval)) if X_eval is not None else None

    for _ in range(n_iter):
        muestreados = rng.choice(indices_U, size=n_P, replace=False)
        es_oob = np.ones(n_U, dtype=bool)
        es_oob[muestreados] = False

        X_train = pd.concat([X_P, X_U.iloc[muestreados]])
        y_train = np.concatenate([np.ones(n_P), np.zeros(n_P)])

        modelo = RandomForestClassifier(
            **RF_PARAMS, random_state=int(rng.integers(0, 2**31 - 1))
        )
        modelo.fit(X_train, y_train)

        suma_oob[es_oob] += modelo.predict_proba(X_U.iloc[es_oob])[:, 1]
        conteo_oob[es_oob] += 1

        if X_eval is not None:
            acumulado_eval += modelo.predict_proba(X_eval)[:, 1]

    promedio_oob = np.divide(
        suma_oob, conteo_oob, out=np.full(n_U, np.nan), where=conteo_oob > 0
    )
    scores_U = pd.Series(promedio_oob, index=X_U.index, name="prob_pu_bagging")
    scores_eval = (
        pd.Series(acumulado_eval / n_iter, index=X_eval.index, name="prob_pu_bagging")
        if X_eval is not None else None
    )
    return scores_U, scores_eval


# ---------------------------------------------------------------------------
# 5. Evaluación: recall en held-out + tasa de positivos en U vs pi
# ---------------------------------------------------------------------------

def resumen_evaluacion(scores_held_out: pd.Series, scores_U: pd.Series, pi: float,
                        umbral: float = 0.5, nombre: str = "activo"):
    """
    Accuracy no sirve aquí (los negativos dominan). Se usa:
      - recall sobre positivos held-out (métrica principal)
      - fracción de U predicha positiva, comparada con pi
    Un recall alto con una fracción de U muy por encima de pi significa que
    el modelo está diciendo "sí" a casi todo, no que haya aprendido algo.
    """
    recall = float((scores_held_out > umbral).mean())
    frac_U_positivo = float((scores_U > umbral).mean())

    print(f"\n=== Evaluación Monte Carlo PU: {nombre} ===")
    print(f"Recall en held-out positivos: {recall:.2%} (n={len(scores_held_out)})")
    print(f"Fracción de U predicha positiva: {frac_U_positivo:.2%}  (pi={pi:.2%})")

    return recall, frac_U_positivo
