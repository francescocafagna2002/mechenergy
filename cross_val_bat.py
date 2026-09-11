import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from load import cargar_tablas, construir_vector_labels

X = pd.read_csv("labelled_features_bat.csv", index_col="mp_id")

gigi, zgp, mapping = cargar_tablas(
    "../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
    "../aew-data/test-blob/input_data/Zähler-GP.csv",
    "../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
)
y_bat = construir_vector_labels(gigi, zgp, mapping, columna_activo="Batterie/Speicher")

X_alineado = X.loc[y_bat.index]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo_cv = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1,
)

scores = cross_val_score(modelo_cv, X_alineado, y_bat, cv=cv, scoring="roc_auc")

print("AUC por fold:", scores)
print(f"Media: {scores.mean():.3f} ± {scores.std():.3f}")