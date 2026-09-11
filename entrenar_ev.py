import pandas as pd
from load import cargar_tablas, construir_vector_labels
from model import entrenar_modelo, evaluar_modelo, explicar_prediccion

# 1. Cargar la matriz real que te dieron (ojo: la columna se llama "mp_id" en minúsculas)
X = pd.read_csv("labelled_features_ev.csv", index_col="mp_id")

# 2. Cargar labels
gigi, zgp, mapping = cargar_tablas(
    "../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
    "../aew-data/test-blob/input_data/Zähler-GP.csv",
    "../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
)
y_pv = construir_vector_labels(gigi, zgp, mapping, columna_activo="Ladestation für Elektrofahrzeuge")

# 3. Entrenar y evaluar
modelo_pv, X_test, y_test = entrenar_modelo(X, y_pv, nombre="EV")
evaluar_modelo(modelo_pv, X_test, y_test, nombre="EV")

# 4. Explicar una casa concreta
explicar_prediccion(modelo_pv, X, mp_id=X.index[0])