"""
entrenar_pv_pu.py — Energy Fingerprints Hackathon (AEW)

Monte Carlo PU-learning para PV, en paralelo a entrenar_pv.py (que no se
toca: ese sigue con el baseline naive "-" -> 0). Este script es nuevo y
autocontenido, solo se apoya en funciones que ya existían en load.py y en
las nuevas de pu_montecarlo.py.

P = filas "X" de labelled_features.csv para PV.
U = house_10000, quitando cualquier mp_id que ya esté en P (para no
    reentrenar/evaluar con filas cuya etiqueta real ya conocemos).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from load import cargar_tablas, construir_vector_labels
from pu_montecarlo import (
    seleccionar_feature,
    verificar_monotonia,
    construir_pesos,
    entrenar_mc_pu,
    entrenar_pu_bagging,
    resumen_evaluacion,
)

N_ITER_MC = 100        # parametrizable, rondas del Monte Carlo (reentrena sobre P+U cada vez)
N_ITER_BAGGING = 100   # parametrizable, rondas del PU bagging (cada ronda entrena sobre solo 2*|P| filas,
                       # mucho más barato; ~100 ya deja a casi toda U out-of-bag en la inmensa mayoría
                       # de rondas, que es lo que pide la estabilidad del promedio — ver pu_montecarlo.py)
HELD_OUT_FRAC = 0.2    # fracción de P apartada para medir recall
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Features — mismas 28 columnas, mismo orden, comprobado explícitamente
# ---------------------------------------------------------------------------
X_labelled = pd.read_csv("labelled_features.csv", index_col="mp_id")
X_U_all = pd.read_csv("houses_10000.csv", index_col="mp_id")

assert list(X_labelled.columns) == list(X_U_all.columns), (
    "labelled_features.csv y houses_10000.csv no tienen las mismas columnas/orden"
)

# ---------------------------------------------------------------------------
# 2. Labels de PV: naive (para el chequeo de monotonía) y solo-positivos (P)
# ---------------------------------------------------------------------------
gigi, zgp, mapping = cargar_tablas(
    "../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
    "../aew-data/test-blob/input_data/Zähler-GP.csv",
    "../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
)
y_naive = construir_vector_labels(
    gigi, zgp, mapping, columna_activo=" PV", tratar_ausencia_como_negativo=True
)
y_pos = construir_vector_labels(
    gigi, zgp, mapping, columna_activo=" PV", tratar_ausencia_como_negativo=False
)

mp_ids_P = y_pos.index.intersection(X_labelled.index)
X_P = X_labelled.loc[mp_ids_P]
print(f"[PV] P (positivos confirmados con features): {len(X_P)}")

# U: quitamos cualquier solapamiento con P para no contaminar entrenamiento/evaluación
X_U = X_U_all.drop(index=X_P.index, errors="ignore")
print(f"[PV] U tras quitar solapamiento con P: {len(X_U)} de {len(X_U_all)}")

# ---------------------------------------------------------------------------
# 3. Prior pi: placeholder con la tasa real de PV en Suiza (no 0.5 — probado
#    antes y descartado: con pi=0.5 cada ronda del Monte Carlo etiqueta
#    ~50% de las 9965 filas de U como "positivo" al azar, un ruido de
#    etiqueta tan grande que ahoga la señal real y los scores colapsan hacia
#    ~0.5 sin separar nada; el PU bagging no sufre esto porque su tamaño de
#    negativos por ronda es |P|, no depende de pi).
#    338.270 instalaciones FV a cierre de 2025 (pv magazine, datos Swissolar/BFE)
#    / 1,8M de edificios residenciales a cierre de 2024 (BFS) ~= 18.8%.
#    s = pi ("s ~ pi" según la descripción).
# ---------------------------------------------------------------------------
pi = 338_270 / 1_800_000  # ~0.188 — placeholder nacional, a sustituir si hay un número mejor
s = pi
print(f"[PV] pi (placeholder, % PV en Suiza) = {pi:.3f}, s = {s:.3f}")

# ---------------------------------------------------------------------------
# 4. Feature guía + chequeo de monotonía (congelados antes del bucle)
# ---------------------------------------------------------------------------
feature, signo = seleccionar_feature(X_P, X_U)
es_monotona, _tabla_bandas = verificar_monotonia(y_naive, X_labelled, feature, signo)

if not es_monotona:
    print("[PV] Monotonía NO confirmada -> fallback a s=0 (PU bagging uniforme)")
    s = 0.0

# ---------------------------------------------------------------------------
# 5. Pesos de muestreo para U
# ---------------------------------------------------------------------------
p = construir_pesos(X_U, feature, signo, pi, s)

# ---------------------------------------------------------------------------
# 6. Split de P: una parte entrena en cada ronda, otra se aparta para recall
# ---------------------------------------------------------------------------
P_train_idx, P_held_idx = train_test_split(
    X_P.index, test_size=HELD_OUT_FRAC, random_state=RANDOM_STATE
)
X_P_train = X_P.loc[P_train_idx]
X_P_held = X_P.loc[P_held_idx]
print(f"[PV] P_train={len(X_P_train)} / P_held_out={len(X_P_held)}")

# ---------------------------------------------------------------------------
# 7. Bucle Monte Carlo PU
# ---------------------------------------------------------------------------
scores_U_mc, scores_held_mc = entrenar_mc_pu(
    X_P_train, X_U, p, n_iter=N_ITER_MC, random_state=RANDOM_STATE, X_eval=X_P_held
)

# ---------------------------------------------------------------------------
# 7b. PU bagging (protocolo sin Monte Carlo, para comparar)
# ---------------------------------------------------------------------------
scores_U_bagging, scores_held_bagging = entrenar_pu_bagging(
    X_P_train, X_U, n_iter=N_ITER_BAGGING, random_state=RANDOM_STATE, X_eval=X_P_held
)

# ---------------------------------------------------------------------------
# 8. Evaluación
# ---------------------------------------------------------------------------
resumen_evaluacion(scores_held_mc, scores_U_mc, pi, nombre="PV (Monte Carlo PU)")
resumen_evaluacion(scores_held_bagging, scores_U_bagging, pi, nombre="PV (PU bagging)")

# ---------------------------------------------------------------------------
# 9. Guardar resultados fuera del repo, según CLAUDE.md
# ---------------------------------------------------------------------------
scores_U_mc.to_csv("../store/pv_mc_pu_scores.csv")
scores_U_bagging.to_csv("../store/pv_pu_bagging_scores.csv")
print("[PV] Guardado: ../store/pv_mc_pu_scores.csv y ../store/pv_pu_bagging_scores.csv")
