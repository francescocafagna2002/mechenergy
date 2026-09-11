import pandas as pd
from sklearn.metrics import roc_auc_score
from load import cargar_tablas, construir_vector_labels
from guardado_modelo import entrenar_modelo, evaluar_modelo, explicar_prediccion
from guardado_modelo import guardar_modelo

# 1. Cargar la matriz real
X = pd.read_csv("labelled_features.csv", index_col="mp_id")

# 2. Cargar labels
gigi, zgp, mapping = cargar_tablas(
    "../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
    "../aew-data/test-blob/input_data/Zähler-GP.csv",
    "../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
)
y_pv = construir_vector_labels(gigi, zgp, mapping, columna_activo=" PV")

# 3. Entrenar y evaluar
modelo_pv, X_test, y_test, X_train, y_train = entrenar_modelo(X, y_pv, nombre="pv", max_depth=6)
evaluar_modelo(modelo_pv, X_test, y_test, nombre="pv")

# 4. Diagnóstico: ¿hay overfitting? Comparamos AUC en train vs en test
proba_train = modelo_pv.predict_proba(X_train)[:, 1]
proba_test = modelo_pv.predict_proba(X_test)[:, 1]
print("AUC en TRAIN:", roc_auc_score(y_train, proba_train))
print("AUC en TEST:", roc_auc_score(y_test, proba_test))

# 5. Explicar una casa concreta
explicar_prediccion(modelo_pv, X, mp_id=X.index[0])

guardar_modelo(modelo_pv, "modelo_pv.pkl")