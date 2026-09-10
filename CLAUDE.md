# MechEnergy

## Data — CRITICAL RULE
The dataset lives in `../aew-data` (read-only, ~78 GB). NEVER read the data
files directly — they would blow up the context window.

To understand structure and columns, read these repo-root files instead:
- `DATA_MANIFEST.txt` — full folder structure and per-folder sizes
- `DATA_HEADERS.txt` — headers and one sample row per file type

Only open an actual data file if I explicitly ask you to, and even then use
sampling (`head`, `shuf -n`, or pandas `nrows=`/`chunksize=`), never the whole file.
All data files are semicolon-separated (`;`) and some have a UTF-8 BOM
(read with `sep=';', encoding='utf-8-sig'`).

## Data schema
Path: `../aew-data/test-blob/input_data/`

Monthly load-curve files — `{year}/{Month year}/LG_AIM*_kWh_*.csv` (the bulk, ~78 GB):
  One row = one meter (`MP ID`) on one day (`Datum`, dd.mm.yyyy).
  Columns: MP ID; OBIS-Code; Datum; PLZ; then 96 quarter-hour kWh readings
  (`00:15`…`00:00`). Value column = electricity consumption per 15-min interval.
  A monthly file can be 1–3 GB — never load a whole one at once; use chunks.

Reference tables (small, in the root of input_data/):
- `HackDays2026 - GIGI.csv` — connection points: GP-Nr; PLZ; Ort; Kanton;
  WärmePumpe; PV; PV-Leistung (kWp); Batterie; Ladestation (EV); Wärmepumpenboiler;
  plus installation dates (Datum Unterschrift, geplanter Baustart, Übergabe, InBetrieb-Datum).
- `Zähler-GP.csv` — maps Zählpunktbezeichnung ↔ GPartner ↔ Anlage.
- `mpid_zähler_mapping.csv` — maps MP ID ↔ Zählpunktbezeichnung.

Join path: monthly `MP ID` → `mpid_zähler_mapping` → `Zähler-GP` (GPartner) → GIGI attributes.

## Folders
- `../aew-data` — input data, READ-ONLY. Do not write here.
- `../store` — write results and outputs here (persistent storage).
- `mechenergy/` (this repo) — code, versioned with Git.

## Workflow
- Results and generated files go in `../store`, not in the repo.
- Commit and push code changes to the shared repo, but let me review before pushing.
