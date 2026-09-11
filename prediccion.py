"""
predecir_nuevas_casas.py — Energy Fingerprints Hackathon (AEW)

Aplica los 4 modelos ya entrenados y guardados (.pkl) a un conjunto de casas
nuevas (ej. las 5 de evaluación final), donde CADA ACTIVO tiene su propio
archivo de features (calculado por la función de features de cada persona),
no un único archivo con todas las columnas juntas.

Uso: ajusta las rutas en RUTAS_MODELOS y RUTAS_FEATURES más abajo, y ejecuta:
    python predecir_nuevas_casas.py
"""

import pandas as pd
from guardado_modelo import cargar_modelo


# Nombre del activo -> ruta del modelo .pkl ya entrenado
RUTAS_MODELOS = {
    "PV": "modelo_pv.pkl",
    "EV": "modelo_ev.pkl",
    "Bateria": "modelo_bat.pkl",
    "HeatPump": "modelo_hp.pkl",  # ajustar si el nombre real es distinto
}

# Nombre del activo -> ruta del CSV de features de las casas nuevas PARA ESE ACTIVO
RUTAS_FEATURES = {
    "PV": "features_pv_5_casas.csv",
    "EV": "features_ev_5_casas.csv",
    "Bateria": "features_bat_5_casas.csv",
    "HeatPump": "features_hp_5_casas.csv",
}


def predecir_activo(nombre: str, ruta_modelo: str, ruta_features: str) -> pd.Series:
    """
    Carga un modelo y su archivo de features de casas nuevas, verifica que
    las columnas necesarias existen, reordena exactamente como el modelo
    las espera, y devuelve la probabilidad (0-1) por MP_ID.
    """
    modelo = cargar_modelo(ruta_modelo)
    X_nuevas = pd.read_csv(ruta_features, index_col="mp_id")

    columnas_esperadas = list(modelo.feature_names_in_)
    faltan = [c for c in columnas_esperadas if c not in X_nuevas.columns]
    if faltan:
        raise ValueError(
            f"[{nombre}] Faltan {len(faltan)} columnas que el modelo necesita "
            f"y no están en '{ruta_features}': {faltan}"
        )

    X_ok = X_nuevas[columnas_esperadas]  # mismo orden que en el entrenamiento

    probas = modelo.predict_proba(X_ok)[:, 1]
    resultado = pd.Series(probas, index=X_ok.index, name=f"prob_{nombre}")
    print(f"[{nombre}] Predicho para {len(resultado)} casas nuevas "
          f"(usando {len(columnas_esperadas)} features)")
    return resultado


def predecir_todos(rutas_modelos: dict, rutas_features: dict) -> pd.DataFrame:
    """Ejecuta predecir_activo para cada activo y junta todo en una sola tabla."""
    columnas = {}
    for nombre in rutas_modelos:
        columnas[f"prob_{nombre}"] = predecir_activo(
            nombre, rutas_modelos[nombre], rutas_features[nombre]
        )
    return pd.DataFrame(columnas)


if __name__ == "__main__":
    resultados = predecir_todos(RUTAS_MODELOS, RUTAS_FEATURES)
    print("\n=== Resultado final ===")
    print(resultados)
    resultados.to_csv("predicciones_5_casas_evaluacion.csv")
    print("\nGuardado en: predicciones_5_casas_evaluacion.csv")