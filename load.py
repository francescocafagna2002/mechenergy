"""
load.py — Energy Fingerprints Hackathon (AEW)

Construye el vector de labels (y) para cada activo, cruzando:
    GIGI.csv (GP-Nr + labels) -> Zähler-GP.csv (GPartner -> Zählpunktbezeichnung)
    -> mpid_zähler_mapping.csv (Zählpunktbezeichnung -> MP ID)

Uso:
    from load import cargar_tablas, construir_vector_labels

    gigi, zgp, mapping = cargar_tablas(
        ruta_gigi="HackDays2026 - GIGI.csv",
        ruta_zgp="Zähler-GP.csv",
        ruta_mapping="mpid_zähler_mapping.csv",
    )

    y_ev = construir_vector_labels(gigi, zgp, mapping, columna_activo="Ladestation für Elektrofahrzeuge")
    y_pv = construir_vector_labels(gigi, zgp, mapping, columna_activo=" PV")
"""

import pandas as pd
from model import preparar_labels


COLUMNAS_ACTIVOS = {
    "Bateria": "Batterie/Speicher",
    "HeatPump": "WärmePumpe",
    "PV": " PV",
    "EV": "Ladestation für Elektrofahrzeuge",
}


def cargar_tablas(ruta_gigi: str, ruta_zgp: str, ruta_mapping: str):
    """Carga las 3 tablas de metadatos con el encoding/separador correctos."""
    gigi = pd.read_csv(ruta_gigi, sep=";", encoding="utf-8-sig")
    zgp = pd.read_csv(ruta_zgp, sep=";", encoding="utf-8-sig")
    mapping = pd.read_csv(ruta_mapping, sep=";", encoding="utf-8-sig")
    return gigi, zgp, mapping


def construir_vector_labels(gigi: pd.DataFrame, zgp: pd.DataFrame, mapping: pd.DataFrame,
                             columna_activo: str, tratar_ausencia_como_negativo: bool = True,
                             verbose: bool = True) -> pd.Series:
    """
    Construye un Series de labels indexado por MP_ID para UN activo.

    tratar_ausencia_como_negativo:
        True  (por defecto, modo "naive baseline"): '-' y vacío -> 0. Devuelve
               1/0 para todos los MP_ID con label válido.
        False (modo PU-learning real): '-' y vacío se excluyen del resultado.
               La función devuelve SOLO los MP_ID con el activo CONFIRMADO (1)
               — es vuestro conjunto de "positivos fiables". El resto de
               MP_ID (los que no aparecen en este Series) son el "pool de
               desconocidos" que necesitaréis para la estrategia PU-learning
               (ej. muestrear de ahí para seleccionar negativos fiables).

    columna_activo: nombre exacto de la columna en GIGI.csv, ej. " PV" (con espacio),
                     "Ladestation für Elektrofahrzeuge", "WärmePumpe", "Batterie/Speicher".
                     Ver COLUMNAS_ACTIVOS para los nombres exactos.

    IMPORTANTE — dos tipos de "duplicado" distintos, tratados de forma distinta:

    1. Un mismo GP-Nr puede aparecer en VARIAS FILAS de GIGI.csv (solicitudes de
       subvención en fechas distintas, ej. PV en 2008 y batería en 2024 para el
       mismo cliente). Aquí AGREGAMOS por GP-Nr tomando el máximo (equivalente a
       un OR): si en cualquiera de sus filas el activo aparece confirmado ('x'),
       el cliente se considera positivo. Quedarnos solo con la "primera" fila
       perdería información real (ver ejemplo GP-Nr 698970 en el chat).

    2. Un mismo GP-Nr puede tener VARIOS MEDIDORES distintos tras cruzar con
       Zähler-GP (ej. un edificio con varios pisos, cada uno con su propio
       MP_ID). Aquí NO tenemos forma de saber qué medidor concreto "ve" el
       activo, así que asignamos el mismo label a TODOS sus medidores en vez
       de descartar filas.
    """
    gigi = gigi.copy()
    zgp = zgp.copy()

    # 1. Convertir labels crudos (x / - / vacío / raros) a 1/0-o-NaN/NaN,
    #    según el modo elegido
    etiquetas = preparar_labels(gigi[columna_activo])
    if not tratar_ausencia_como_negativo:
        # Deshacemos el 0 que preparar_labels asigna a '-'/vacío, dejándolo NaN
        crudo = gigi[columna_activo].astype(str).str.strip().str.lower()
        es_confirmado_no = crudo.isin(["-"]) | gigi[columna_activo].isna()
        etiquetas = etiquetas.where(~es_confirmado_no, other=pd.NA)
    gigi["_label"] = etiquetas

    # 2. Quitar filas sin GP-Nr (no se pueden agregar ni cruzar nunca)
    gigi_util = gigi.dropna(subset=["GP-Nr"]).copy()
    gigi_util["GP-Nr"] = gigi_util["GP-Nr"].astype(str).str.strip()

    # 3. Agregar por GP-Nr: si CUALQUIER fila de ese cliente dice "x", es positivo.
    #    groupby().max() ignora NaN automáticamente; si todas las filas de un
    #    cliente son NaN (valores no reconocidos), el resultado queda NaN.
    labels_por_cliente = gigi_util.groupby("GP-Nr")["_label"].max()
    n_clientes_antes_agregar = gigi_util["GP-Nr"].nunique()
    if verbose:
        n_afectados = gigi_util["GP-Nr"].duplicated().sum()
        print(f"[{columna_activo}] Clientes únicos: {n_clientes_antes_agregar} "
              f"({n_afectados} filas agregadas por tener el mismo GP-Nr en distintas fechas)")

    labels_por_cliente = labels_por_cliente.dropna()
    if verbose:
        print(f"[{columna_activo}] Clientes con label válido tras agregar: {len(labels_por_cliente)}")

    labels_por_cliente = labels_por_cliente.reset_index()  # columnas: GP-Nr, _label

    # 4. Igualar tipos antes de cruzar (GP-Nr es texto, GPartner es número)
    zgp["GPartner"] = zgp["GPartner"].astype(str).str.strip()

    # 5. Cruce 1: labels por cliente -> Zähler-GP (medidor)
    paso1 = labels_por_cliente.merge(zgp, left_on="GP-Nr", right_on="GPartner", how="inner")
    if verbose:
        print(f"[{columna_activo}] Tras cruzar con Zähler-GP: {len(paso1)} filas "
              f"({paso1['GP-Nr'].nunique()} clientes, algunos con varios medidores)")

    # 6. Cruce 2: resultado -> mapping (MP_ID)
    paso2 = paso1.merge(mapping, on="Zählpunktbezeichnung", how="inner")
    if verbose:
        print(f"[{columna_activo}] Tras cruzar con mapping (MP_ID): {len(paso2)} filas")

    # 7. Indexar por MP_ID. NO deduplicamos: cada MP_ID (medidor) ya es único
    #    aquí de forma natural (mapping.csv asigna un MP_ID por medidor), así
    #    que un cliente con varios medidores simplemente aporta varias filas,
    #    cada una con el mismo label (limitación conocida, ver docstring).
    y = paso2.set_index("MP ID")["_label"]

    duplicados_mp_id = y.index.duplicated().sum()
    if duplicados_mp_id > 0 and verbose:
        print(f"[{columna_activo}] Aviso: {duplicados_mp_id} MP_ID repetidos "
              f"de forma inesperada, se resuelven con 'primero'")
        y = y[~y.index.duplicated(keep="first")]

    if verbose:
        print(f"[{columna_activo}] MP_ID finales: {len(y)} | "
              f"Positivos: {int(y.sum())} / Negativos: {int((y == 0).sum())}\n")

    y.name = columna_activo
    return y


def construir_fechas_instalacion(gigi: pd.DataFrame, zgp: pd.DataFrame, mapping: pd.DataFrame,
                                   columna_activo: str, verbose: bool = True) -> pd.Series:
    """
    Para cada MP_ID con el activo CONFIRMADO (x), devuelve la fecha de puesta
    en marcha (InBetrieb-Datum) de ESE activo concreto — usando la fila donde
    esa columna vale 'x' (si hay varias filas con 'x' para el mismo cliente,
    poco habitual, nos quedamos con la más antigua).

    Necesario porque un mismo MP_ID puede tener varias filas en GIGI.csv
    (solicitudes en fechas distintas para activos distintos), y la fecha de
    instalación de UN activo concreto solo aparece en SU propia fila.

    Uso previsto en features.py: para un MP_ID con EV=1 y fecha=2024-03-15,
    calcular las features de EV solo sobre el tramo de la serie posterior a
    esa fecha, para no diluir la señal con meses en los que el EV no existía.

    Devuelve una Series indexada por MP_ID (NaT donde no se conoce la fecha
    exacta, ~30% de los casos — en features.py, tratar como "usar la serie
    completa" al no tener mejor información).
    """
    gigi = gigi.copy()
    zgp = zgp.copy()

    gigi["_label"] = preparar_labels(gigi[columna_activo])
    gigi["_fecha"] = pd.to_datetime(gigi["InBetrieb-Datum"], format="%d.%m.%Y", errors="coerce")

    positivos = gigi[gigi["_label"] == 1].dropna(subset=["GP-Nr"]).copy()
    positivos["GP-Nr"] = positivos["GP-Nr"].astype(str).str.strip()

    fecha_por_cliente = positivos.groupby("GP-Nr")["_fecha"].min().reset_index()

    zgp["GPartner"] = zgp["GPartner"].astype(str).str.strip()
    paso1 = fecha_por_cliente.merge(zgp, left_on="GP-Nr", right_on="GPartner", how="inner")
    paso2 = paso1.merge(mapping, on="Zählpunktbezeichnung", how="inner")

    fechas = paso2.set_index("MP ID")["_fecha"]
    fechas = fechas[~fechas.index.duplicated(keep="first")]

    if verbose:
        conocidas = fechas.notna().sum()
        print(f"[{columna_activo}] Fechas de instalación conocidas: {conocidas} de {len(fechas)} positivos")

    fechas.name = f"fecha_instalacion_{columna_activo}"
    return fechas


if __name__ == "__main__":
    # DEMO: ajusta las rutas a donde tengáis los 3 CSV en vuestro entorno
    gigi, zgp, mapping = cargar_tablas(
        ruta_gigi="../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv",
        ruta_zgp="../aew-data/test-blob/input_data/Zähler-GP.csv",
        ruta_mapping="../aew-data/test-blob/input_data/mpid_zähler_mapping.csv",
    )

    for nombre, columna in COLUMNAS_ACTIVOS.items():
        y = construir_vector_labels(gigi, zgp, mapping, columna_activo=columna)

    # Guardamos cada vector como CSV para poder abrirlo y revisarlo visualmente
    # en el explorador de VSCodium (columnas: MP ID, <nombre_columna_activo>)
    ruta_salida = f"y_{nombre}_preview.csv"
    y.to_csv(ruta_salida)
    print(f"[{nombre}] Guardado en: {ruta_salida}\n")