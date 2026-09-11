"""
exportar_mp_ids.py — Energy Fingerprints Hackathon (AEW)

Genera el archivo que hay que compartir con el equipo de feature engineering:
la lista de los MP_ID (413) para los que hace falta calcular features, porque
son los únicos con label conocido tras cruzar GIGI + Zähler-GP + mapping.

Uso: python exportar_mp_ids.py
Genera: mp_ids_para_features.csv
"""

from load import cargar_tablas, construir_vector_labels, COLUMNAS_ACTIVOS

gigi, zgp, mapping = cargar_tablas(
    ruta_gigi="../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
    ruta_zgp="../aew-data/test-blob/input_data/Zähler-GP.csv",
    ruta_mapping="../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
)

# Los 413 MP_ID son los mismos para los 4 activos (viene del mismo cruce de
# tablas), así que basta con calcularlo para uno solo (ej. PV) y usar su índice.
y_pv = construir_vector_labels(gigi, zgp, mapping, columna_activo=" PV", verbose=False)

mp_ids = y_pv.index.to_series(name="MP_ID")
mp_ids.to_csv("mp_ids_para_features.csv", index=False)

print(f"Guardado: mp_ids_para_features.csv con {len(mp_ids)} MP_ID")
print(mp_ids.head(10).to_list())