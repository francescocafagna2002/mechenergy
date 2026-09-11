"""
features_hp21.py — TWENTY physically-motivated HEAT-PUMP main features from the AEW smart-meter
CSVs (import counter only) + feature 21, the heating energy signature, which also uses the
canton-average air TEMPERATURE (MeteoSwiss; built by meteo_aargau.py, which must sit in the same
folder — the same weather file features_pv21.py and features_ev21.py use).

Standalone: it does not import pv20_features.py / features_pv21.py / features_ev21.py
(only meteo_aargau.py, and only for feature 21).

Run (inside the Renku session, from ~/work/mechenergy):
    NO ARGUMENTS:  python features_hp21.py
        -> uses the DEFAULTS block below: store aew_store, MP IDs from DEFAULT_IDS_FILE (first
           column, header skipped), output DEFAULT_OUT (+ .xlsx).  Edit those four lines once.
    a list of houses (e.g. the labelled ones):
        python features_hp21.py --store aew_store --mp-ids-file mp_ids_para_features.csv -o hp_labelled.csv
    named houses / the first N / all 89k (8 buckets at a time so memory stays bounded):
        python features_hp21.py --store aew_store --mp-ids 53628,45390 -o hp_two.csv
        python features_hp21.py --store aew_store --max-houses 10000 -o hp_10000.csv
        python features_hp21.py --store aew_store -o hp_all.csv
    without the weather file (feature 21 = 0, no internet needed):
        python features_hp21.py --no-weather
    CSV path, no store yet:
        python features_hp21.py ../aew-data/test-blob/input_data --max-houses 10 -o hp_ten.csv

HOW THE FILES ARE READ — identical to features_pv21.py / features_ev21.py: with a house list each
monthly file is first filtered as TEXT (grep in byte mode, cached in --cache-dir) and only those
lines are parsed; without a list the file is streamed in chunks.  The binary store (build_store.py)
is used automatically when ./aew_store exists.

PHYSICS OF A HEAT PUMP (what the features look for)
    An air- or ground-source heat pump for space heating draws 1-4 kW electrical in a single-family
    house and runs WHENEVER IT IS COLD, so its signature is seasonal and continuous rather than
    event-like:
      * the whole WINTER BASELINE rises — the house never idles at 0.2 kW in January the way it
        does in July; this is the most robust meter-only evidence (features 2, 3, 5);
      * it runs MANY HOURS a day, often through the night and the early morning (7, 9, 17);
      * an ON/OFF (older) unit CYCLES: many short runs a day at a fixed level (9, 10);
        an INVERTER unit MODULATES: long runs whose power varies continuously (14) and whose level
        follows the weather from day to day (13);
      * the daily profile flattens: less peaky than a plain household (15, 16);
      * energy is weather-driven, so it is strongly correlated from one day to the next (20) and
        varies a lot across days (6), unlike an appliance used on a fixed routine.
    Confounders and how they differ:
      direct electric heating  same temperature dependence but ~3x the slope (COP 1 vs 3) and a
                               much bigger winter peak (feature 19, and 21 >> a heat pump's)
      EV charger               flat blocks at a fixed level, a few days a week, evening, NOT
                               temperature-driven: 14 low, 20 low, 21 ~ 0
      electric boiler (DHW)    one fixed nightly block all year: 9 ~ 1 cycle/day, 4 and 8 ~ 0
      plain house              1, 3 ~ 1.1-1.3 (lighting), 2 ~ 0, 21 ~ 0.05
    A heat pump that also makes hot water keeps a small daily cycle in summer (feature 11).

PER-DAY REDUCTION (import, calendar day of 96 quarter-hours)
    base      = 10th percentile of the day (what the house draws when nothing is running)
    running   = quarter-hours with import >= base + RUN_KW (0.5 kW)
    per day: energy, base, daily max, mean, hours running, number of runs >= 30 min, mean level
    above base while running, mean |power change| between consecutive quarter-hours while running,
    energy 02-06 h, number of valid values.

==============================================================================
THE TWENTY-ONE FEATURES  (summer = May-Aug, winter = Nov-Feb; every one is a computed
number for every house — a season with no days falls back to all the house's days)
==============================================================================
A. SEASONAL AMPLITUDE AND BASELINE — the core of heat-pump detection

 1. hp_winter_summer_energy_ratio    [>= 0]  mean daily import Nov-Feb / May-Aug.
    Plain house 1.1-1.3 (lighting); heat pump 2-5; direct electric heating 4-10.
 2. hp_base_winter_minus_summer_kw   [kW]  daily BASE load (10th percentile) in winter minus summer.
    A heat pump lifts the floor of the whole day by 0.3-2 kW; an EV or a boiler does not (it only
    adds peaks, and the 10th percentile ignores peaks).  ~0 for everything that is not heating.
 3. hp_base_winter_over_summer       [>= 0]  the same as a ratio (size-free).  Plain ~1.0-1.2.
 4. hp_monthly_energy_amplitude      [>= 0]  (highest month - lowest month) / mean month of the mean
    daily import.  Heating -> 1-2; a flat consumer -> 0.2-0.4.
 5. hp_daily_energy_winter_kwh       [kWh]  mean daily import Nov-Feb: the size of the winter load.
 6. hp_daily_energy_spread_winter    [0..10]  (p90 - p10) / median of the daily winter import.
    Weather-driven -> wide (0.6-1.2); a fixed routine -> narrow.  Capped at 10.

B. RUNNING PATTERN — hours, cycling, modulation

 7. hp_run_hours_winter              [h/day, 0-24]  hours per winter day at >= base + 0.5 kW.
    Heat pump 6-16; EV 1-3; boiler 2-4.
 8. hp_run_hours_winter_minus_summer [h/day]  the same, winter minus summer: space heating only
    exists in winter -> large; a boiler or an EV -> ~0.
 9. hp_cycles_per_day_winter         [1/day]  number of separate runs >= 30 min per winter day.
    On/off heat pump 4-15; modulating heat pump 2-6; EV ~1; boiler ~1.
10. hp_cycle_length_h_winter         [h]  mean length of those runs (heat pump 0.5-3 h).
11. hp_cycles_per_day_summer         [1/day]  the same in summer: a heat pump that also makes hot
    water keeps ~1 short cycle a day; a heating-only unit goes quiet.
12. hp_level_winter_kw               [kW]  mean power above base while running in winter
    ~ the compressor's electrical input (heat pump 1-4; direct electric heating 4-9).
13. hp_level_spread_winter           [0..10]  (p90 - p10) / median of that level across winter days
    (the median is floored at 0.5 kW, the value capped at 10, so a house that almost never runs
    cannot produce an absurd number).  A heat pump follows the weather -> wide; a fixed appliance
    -> narrow; an EV (11 kW on some days, nothing on others) hits the cap.
14. hp_modulation_index_winter       [>= 0]  mean |power change between consecutive quarter-hours
    while running| / the running level, winter.  An inverter heat pump varies continuously -> high
    (0.15-0.6); a flat EV or boiler block -> low (< 0.1).

C. SHAPE OF THE DAY

15. hp_load_factor_winter            [0..1]  mean daily power / daily maximum power, winter.
    Heating raises the whole day -> 0.4-0.7; a peaky household -> 0.2-0.3.
16. hp_load_factor_winter_minus_summer [-1..1]  the same, winter minus summer (positive for heating).
17. hp_early_morning_share_winter    [0..1]  share of the winter day's import between 02 h and 06 h.
    Nobody is awake, so a plain house is at ~0.10; a heat pump (often boosted at night) 0.15-0.25.
18. hp_early_morning_share_winter_minus_summer [-1..1]  the same, winter minus summer.

D. SIZE OF THE WINTER LOAD

19. hp_peak_winter_minus_summer_kw   [kW]  99th percentile of the power in winter minus in summer:
    the extra capacity installed for heating.  Heat pump 1-4; direct electric heating 5-10; EV ~0
    (it charges in summer too).
20. hp_daily_energy_persistence_winter [-1..1]  correlation of the day's winter import ANOMALY
    (the day minus that month's mean, so the seasonal trend is removed) with the PREVIOUS day's
    anomaly.  Cold spells last several days, so a weather-driven load keeps 0.3-0.7, while a load
    that follows people's routines is uncorrelated from one day to the next (~0).  This is the
    meter-only stand-in for feature 21, and it still works when no weather file is available.

E. TEMPERATURE — needs data/weather/aargau_weather_15min.csv (built automatically on the first run
   if missing: meteo_aargau.py downloads MeteoSwiss data, needs internet, ~1 min; --no-weather skips it)

21. hp_energy_cold_slope_kwh_per_degc  [kWh/°C; heat pump >> 0, no heating ~ 0]  the classic
    HEATING ENERGY SIGNATURE: over the days whose canton mean temperature is below HEAT_LIMIT_C
    (15 °C, where Swiss houses start heating), x = that temperature, y = the day's import [kWh];
    least-squares line y = a + b·x; feature = -b, the extra kWh per degree colder.
    Physically -b = (heat loss of the building [kW/K]) / COP: a well-insulated house with a heat
    pump gives 0.5-1.5, a badly insulated one 2-3, DIRECT ELECTRIC heating 2-6 (no COP), and a
    house without electric heat ~0.05 (a little more light and cooking on cold dark days).
    This is the same regression as feature 1 but continuous, so it cannot be fooled by a mild
    winter or a cold August, and it is the feature that separates a heat pump from an EV charger
    (~0) and grades it against direct electric heating.
    < 5 usable cold days or no temperature variation -> 0.  Weather file missing / --no-weather -> 0.

Data-quality columns: n_days, n_summer_days, n_winter_days, mean_import_kw, base_kw (median daily
base), n_cold_days (days used by feature 21).

Source of the weather data: MeteoSwiss Open Data (SwissMetNet), stations BUS Buchs/Aarau and
BEZ Beznau.  Cite as "Source: MeteoSwiss".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# DEFAULTS — what  python features_hp21.py  does when run with no arguments.
# Edit these lines to your files; any command-line option still overrides them.
# =============================================================================
DEFAULT_STORE = "aew_store"                 # the binary store folder (built by build_store.py)
DEFAULT_IDS_FILE = "labelled_houses.csv"    # CSV/TXT with the MP IDs in the FIRST column, header on line 1
DEFAULT_OUT = "labelled_features_hp21.csv"  # output table (an .xlsx copy is written next to it)
DEFAULT_WEATHER = os.path.join("data", "weather", "aargau_weather_15min.csv")   # built by meteo_aargau.py
DEFAULT_WORKERS = 2                         # parallel processes (use the number of cores)

N_SLOTS = 96
SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
ZERO_KW = 0.02               # import <= 20 W == "the house imports nothing"
EXPORT_NOISE_KW = 0.05       # export noise floor (kept so the reader matches features_pv21.py)

RUN_KW = 0.5             # "running" = at least this much above the day's base load
SPREAD_MAX = 10.0    # relative spreads are capped here so one odd house cannot dominate a split
MIN_RUN_Q = 2            # a run must last >= 2 quarter-hours (30 min) to count as a cycle
EARLY = (2, 6)           # hours: the early-morning window
HEAT_LIMIT_C = 15.0      # feature 21: days below this temperature are heating days
MIN_COLD_DAYS = 5        # feature 21: fewer usable cold days -> 0
MIN_TEMP_STD = 0.5       # feature 21: °C; no temperature variation -> 0

SLOT_HOUR = (np.arange(N_SLOTS) + 0.5) / 4.0
EARLY_MASK = (SLOT_HOUR >= EARLY[0]) & (SLOT_HOUR < EARLY[1])

FEATURE_NAMES = [
    "hp_winter_summer_energy_ratio", "hp_base_winter_minus_summer_kw", "hp_base_winter_over_summer",
    "hp_monthly_energy_amplitude", "hp_daily_energy_winter_kwh", "hp_daily_energy_spread_winter",
    "hp_run_hours_winter", "hp_run_hours_winter_minus_summer", "hp_cycles_per_day_winter",
    "hp_cycle_length_h_winter", "hp_cycles_per_day_summer", "hp_level_winter_kw",
    "hp_level_spread_winter", "hp_modulation_index_winter", "hp_load_factor_winter",
    "hp_load_factor_winter_minus_summer", "hp_early_morning_share_winter",
    "hp_early_morning_share_winter_minus_summer", "hp_peak_winter_minus_summer_kw",
    "hp_daily_energy_persistence_winter",
    "hp_energy_cold_slope_kwh_per_degc",
]

FEATURE_DESC = {
    "hp_winter_summer_energy_ratio": "mean daily import Nov-Feb / May-Aug (plain 1.1-1.3; heat pump 2-5; direct electric heating 4-10)",
    "hp_base_winter_minus_summer_kw": "daily base load (10th percentile) winter minus summer [kW]: heating lifts the floor of the whole day (EV/boiler ~0)",
    "hp_base_winter_over_summer": "the same as a ratio (size-free); plain house ~1.0-1.2",
    "hp_monthly_energy_amplitude": "(highest month - lowest month) / mean month of the mean daily import (heating 1-2; flat consumer 0.2-0.4)",
    "hp_daily_energy_winter_kwh": "mean daily import Nov-Feb [kWh]",
    "hp_daily_energy_spread_winter": "(p90 - p10) / median of the daily winter import (weather-driven wide; fixed routine narrow)",
    "hp_run_hours_winter": "hours per winter day at >= base + 0.5 kW (heat pump 6-16; EV 1-3; boiler 2-4)",
    "hp_run_hours_winter_minus_summer": "the same, winter minus summer (space heating only exists in winter)",
    "hp_cycles_per_day_winter": "separate runs >= 30 min per winter day (on/off heat pump 4-15; EV ~1; boiler ~1)",
    "hp_cycle_length_h_winter": "mean length of those runs [h] (heat pump 0.5-3)",
    "hp_cycles_per_day_summer": "the same in summer: a heat pump making hot water keeps ~1 cycle a day",
    "hp_level_winter_kw": "mean power above base while running, winter [kW] ~ compressor input (heat pump 1-4; direct electric 4-9)",
    "hp_level_spread_winter": "(p90 - p10) / median of that level across winter days (heat pump follows the weather -> wide)",
    "hp_modulation_index_winter": "mean |power change between consecutive quarter-hours while running| / the running level, winter (inverter heat pump 0.15-0.6; flat EV/boiler block < 0.1)",
    "hp_load_factor_winter": "mean daily power / daily maximum power, winter (heating 0.4-0.7; peaky household 0.2-0.3)",
    "hp_load_factor_winter_minus_summer": "the same, winter minus summer (positive for heating)",
    "hp_early_morning_share_winter": "share of the winter day's import between 02 h and 06 h (plain ~0.10; heat pump 0.15-0.25)",
    "hp_early_morning_share_winter_minus_summer": "the same, winter minus summer",
    "hp_peak_winter_minus_summer_kw": "99th percentile of the power, winter minus summer [kW]: extra capacity installed for heating (heat pump 1-4; direct electric 5-10; EV ~0)",
    "hp_daily_energy_persistence_winter": "correlation of the day's winter import anomaly (day minus its month's mean) with the previous day's anomaly (cold spells last days: heating 0.3-0.7; routine ~0)",
    "hp_energy_cold_slope_kwh_per_degc": "heating energy signature: -(slope of the day's import [kWh] on the mean temperature [°C], over days below 15 °C) = extra kWh per degree colder "
                                         "= heat loss / COP.  Heat pump 0.5-3; direct electric heating 2-6; no electric heat ~0.05; no weather file -> 0",
    "n_days": "days of data", "n_summer_days": "May-Aug days (0 = summer features used all days)",
    "n_winter_days": "Nov-Feb days (0 = winter features used all days)",
    "mean_import_kw": "average import power [kW]: household 0.2-0.8, tens = business",
    "base_kw": "median daily base load [kW] (10th percentile of the day)",
    "n_cold_days": "days below 15 °C used by feature 21",
}

# Value used when a feature cannot be computed: always the value a house WITHOUT a heat pump would
# show, so a fill never looks like a detection.  A warning is printed if it is ever needed.
FILL_VALUES = {name: 0.0 for name in FEATURE_NAMES}
FILL_VALUES.update({"hp_winter_summer_energy_ratio": 1.0, "hp_base_winter_over_summer": 1.0})

VCOLS = [f"v{k}" for k in range(N_SLOTS)]
KEY = ["mp_id", "channel", "date"]

# ---- the meter files: known name; the column layout is read per file from its first data
# row, because the files differ (2023: MP ID ; meter id ; OBIS ; Datum ; PLZ ; 96 values ;
# 2024+: no meter-id column; all rows end with a trailing ';').  The last 96 fields are the
# values; OBIS code, date and meter id are recognised by their formats; MP ID is the first
# remaining column, PLZ the next.
FILE_GLOB = "LG_AIM2Hackerdays_kWh_*.csv"       # only these files are read (override with --pattern)
OBIS_IMPORT = "1-1:1.29.0*255"
OBIS_EXPORT = "1-1:2.29.0*255"
OBIS_RE = re.compile(r"^\d+-\d+:\d+\.\d+\.\d+")
DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
METER_RE = re.compile(r"^[A-Z]{2}\d{6,}")


def file_layout(path: str) -> dict:
    """{'n_lead': leading columns, 'roles': {'mp_id': i, 'obis': i, 'date': i, 'meter': i|None, 'plz': i|None}}"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        first = f.readline().rstrip("\r\n").split(";")
    fields = first[:-1] if first and first[-1].strip() == "" else first     # trailing ';' -> empty last field
    n_lead = len(fields) - N_SLOTS
    if n_lead < 3:
        raise ValueError(f"{path}: expected >= 3 leading columns + {N_SLOTS} values, got {len(fields)} fields")
    lead = [x.strip() for x in fields[:n_lead]]
    roles: dict = {"meter": None, "plz": None}
    for i, v in enumerate(lead):
        if "obis" not in roles and OBIS_RE.match(v):
            roles["obis"] = i
        elif "date" not in roles and DATE_RE.match(v):
            roles["date"] = i
        elif roles["meter"] is None and METER_RE.match(v):
            roles["meter"] = i
    if "obis" not in roles or "date" not in roles:
        raise ValueError(f"{path}: could not recognise OBIS / date in the first data row {lead}")
    rest = [i for i in range(n_lead) if i not in (roles["obis"], roles["date"], roles["meter"])]
    roles["mp_id"] = rest[0]
    roles["plz"] = rest[1] if len(rest) > 1 else None
    return {"n_lead": n_lead, "roles": roles}


def layout_names(lay: dict) -> List[str]:
    inv = {i: r for r, i in lay["roles"].items() if i is not None}
    return [inv.get(i, f"x{i}") for i in range(lay["n_lead"])] + VCOLS


# =============================================================================
# 1. reading the CSVs
# =============================================================================
def find_files(root: str, month: Optional[str] = None, max_files: Optional[int] = None,
               pattern: str = FILE_GLOB) -> List[str]:
    """The meter files below root, by NAME (FILE_GLOB), sorted.  `month` keeps only paths
    containing that text (e.g. "Juni 2023" or "2023"); `max_files` the first N."""
    files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if month:
        files = [f for f in files if month.lower() in f.lower()]
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"no files named {pattern} found below {root}" + (f" matching '{month}'" if month else ""))
    return files


def first_mp_ids(path: str, n: int, max_lines: int = 200_000) -> List[str]:
    """The first n distinct MP IDs in the file (reads only the head of the file)."""
    ids: List[str] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()                                   # header
        for i, line in enumerate(f):
            mp = line.split(";", 1)[0].strip()
            if mp and mp not in seen:
                seen.add(mp); ids.append(mp)
                if len(ids) >= n:
                    break
            if i >= max_lines:
                break
    return ids


def filter_lines(path: str, mp_ids: Iterable[str], cache_dir: Optional[str] = None) -> str:
    """Return ONLY the text lines of `path` whose first column is one of the MP IDs.
    Uses grep in byte mode (LC_ALL=C: several times faster than UTF-8 mode) when available,
    otherwise a plain Python scan.  Nothing else in the file is ever parsed.
    With cache_dir, the extracted lines are saved per (file, house set) and reused next time,
    so a second run on the same houses never reads the big files again."""
    ids = sorted(set(str(m) for m in mp_ids))
    if not ids:
        return ""
    cache_file = None
    if cache_dir:
        st = os.stat(path)
        key = hashlib.md5(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{','.join(ids)}".encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    text = None
    if shutil.which("grep"):
        env = {**os.environ, "LC_ALL": "C"}
        if len(ids) <= 50:                  # few houses: one anchored regex
            pattern = "^(" + "|".join(re.escape(m) for m in ids) + ");"
            r = subprocess.run(["grep", "-E", pattern, path], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
            if r.returncode in (0, 1):      # 1 = no match
                text = r.stdout
        else:                               # many houses: fixed-string set search, then exact first-column check
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
                tf.write("".join(f"{m};\n" for m in ids))
            try:
                r = subprocess.run(["grep", "-F", "-f", tf.name, path], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", env=env)
            finally:
                os.unlink(tf.name)
            if r.returncode in (0, 1):
                keep = set(ids)
                text = "".join(l + "\n" for l in r.stdout.splitlines() if l.split(";", 1)[0].strip() in keep)
    if text is None:
        keep = set(ids)
        out = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.split(";", 1)[0].strip() in keep:
                    out.append(line)
        text = "".join(out)
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def _parse(source, chunksize: Optional[int], skip_header: bool, lay: dict):
    """Parse rows with the file's layout: n_lead leading columns + 96 float values, ';'-separated."""
    names = layout_names(lay)
    dtypes = {c: "string" for c in names[:lay["n_lead"]]}
    dtypes.update({v: "float32" for v in VCOLS})
    return pd.read_csv(source, sep=";", header=None, skiprows=1 if skip_header else 0, names=names, index_col=False,
                       dtype=dtypes, na_values=["", " ", "-"], keep_default_na=False, chunksize=chunksize,
                       encoding="utf-8", encoding_errors="replace", engine="c")


def _post(lab: pd.DataFrame, vals: np.ndarray):
    """Shared by the CSV and the store readers: count & floor negatives (counters are >= 0),
    keep the raw daily max, apply the export noise floor."""
    lab = lab.reset_index(drop=True)
    lab["n_neg"] = (vals < 0).sum(axis=1)
    np.maximum(vals, np.float32(0.0), out=vals)
    # export noise floor: a few Wh per quarter-hour is meter creep / a motor switching off,
    # not generation -> keep the raw daily max for the record, then treat those values as 0
    is_exp = (lab["channel"] == "export").values
    lab["vmax_raw"] = np.nanmax(np.where(np.isfinite(vals), vals, -1.0), axis=1)
    if is_exp.any():
        ve = vals[is_exp]
        ve[ve < EXPORT_NOISE_KW] = 0.0
        vals[is_exp] = ve
    return lab, vals


def _finish(ch: pd.DataFrame):
    """CSV rows -> (labels, values): label import / export by OBIS code, kWh per 15 min -> kW."""
    obis = ch["obis"].astype(str).str.strip()
    lab = pd.DataFrame({
        "mp_id": ch["mp_id"].astype(str).str.strip(),
        "channel": np.where(obis == OBIS_IMPORT, "import", np.where(obis == OBIS_EXPORT, "export", "other")),
        "date": pd.to_datetime(ch["date"].astype(str).str.strip(), format="%d.%m.%Y", errors="coerce"),
    }, index=ch.index)
    sel = (lab["channel"] != "other").values & lab["date"].notna().values
    if not sel.any():
        return None
    vals = ch.loc[sel, VCOLS].to_numpy(dtype="float32") * np.float32(4.0)     # kWh / 15 min -> kW
    return _post(lab[sel].copy(), vals)


def iter_aew_csv(path: str, mp_ids: Optional[Iterable[str]] = None,
                 chunksize: int = 200_000, cache_dir: Optional[str] = None) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
    """Yield (labels[mp_id, channel, date, n_neg], values[n, 96] kW >= 0) from one monthly file.
    With mp_ids: only those houses' lines are extracted from the text and parsed (fast).
    Without: the whole file is streamed in chunks."""
    lay = file_layout(path)
    if mp_ids is not None:
        text = filter_lines(path, mp_ids, cache_dir)
        if not text.strip():
            return
        out = _finish(_parse(io.StringIO(text), None, skip_header=False, lay=lay))
        if out is not None:
            yield out
        return
    for ch in _parse(path, chunksize, skip_header=True, lay=lay):
        out = _finish(ch)
        if out is not None:
            yield out


# =============================================================================
# 1b. reading the binary store built by build_store.py  (aew_store/bucket=N/part.parquet)
# =============================================================================
N_BUCKETS = 256
GERMAN_MONTHS = {m: i + 1 for i, m in enumerate(
    ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember"])}


def bucket_of(mp_id: str) -> int:
    """Same rule as build_store.py: MP ID mod 256 (non-numeric -> 0)."""
    try:
        return int(str(mp_id).strip()) % N_BUCKETS
    except ValueError:
        return 0


def _month_to_dates(month: Optional[str]):
    """'Juni 2023' -> (2023-06-01, 2023-07-01); '2023' -> the year; None -> no filter."""
    if not month:
        return None
    t = month.strip().lower()
    yr = re.search(r"(20\d{2})", t)
    if not yr:
        print(f"warning: --month '{month}' has no year -> ignored for the store", flush=True)
        return None
    y = int(yr.group(1))
    mo = next((v for k, v in GERMAN_MONTHS.items() if k in t), None)
    if mo is None:
        return pd.Timestamp(y, 1, 1).date(), pd.Timestamp(y + 1, 1, 1).date()
    end = pd.Timestamp(y + (mo == 12), mo % 12 + 1, 1).date()
    return pd.Timestamp(y, mo, 1).date(), end


def store_buckets(store: str) -> List[str]:
    dirs = sorted(glob.glob(os.path.join(store, "bucket=*")), key=lambda d: int(d.rsplit("=", 1)[1]))
    if not dirs:
        raise SystemExit(f"{store} is not a store (no bucket=* folders); build it with build_store.py")
    return dirs


def read_store(store: str, mp_ids: Optional[Iterable[str]] = None, buckets: Optional[Iterable[str]] = None,
               month: Optional[str] = None):
    """Yield (labels, values) from the store.  mp_ids -> only those houses (touches only their
    buckets and, inside them, only their row groups); buckets -> whole bucket folders."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dates = _month_to_dates(month)
    if mp_ids is not None:
        ids = sorted(set(str(m) for m in mp_ids))
        by_bucket: Dict[int, List[str]] = {}
        for m in ids:
            by_bucket.setdefault(bucket_of(m), []).append(m)
        jobs = [(os.path.join(store, f"bucket={b}"), ids_b) for b, ids_b in sorted(by_bucket.items())]
    else:
        jobs = [(d, None) for d in (buckets if buckets is not None else store_buckets(store))]
    for d, ids_b in jobs:
        if not os.path.isdir(d):
            continue
        flt = None
        if ids_b is not None:
            flt = ds.field("mp_id").isin(ids_b)
        if dates is not None:
            f2 = (ds.field("date") >= dates[0]) & (ds.field("date") < dates[1])
            flt = f2 if flt is None else (flt & f2)
        tab = ds.dataset(d, format="parquet").to_table(filter=flt, columns=["mp_id", "channel", "date"] + VCOLS,
                                                       use_threads=False)
        if tab.num_rows == 0:
            continue
        lab = pd.DataFrame({
            "mp_id": tab.column("mp_id").to_pandas().astype(str),
            "channel": tab.column("channel").to_pandas().astype(str),
            "date": pd.to_datetime(tab.column("date").to_pandas()),
        })
        vals = np.array(tab.select(VCOLS).to_pandas().to_numpy(dtype="float32"), copy=True)   # writable copy
        yield _post(lab, vals)


def first_store_ids(store: str, n: int) -> List[str]:
    """The n smallest MP IDs found in the first buckets (deterministic 'first N houses')."""
    import pyarrow.parquet as pq
    ids: List[str] = []
    for d in store_buckets(store):
        col = pq.read_table(os.path.join(d, "part.parquet"), columns=["mp_id"]).column("mp_id").to_pylist()
        ids.extend(sorted(set(col)))
        if len(ids) >= n:
            break
    return ids[:n]

# =============================================================================
# 2. per-day reduction (calendar day; heat-pump features never cross midnight)
# =============================================================================
def store_ids(bucket_dirs: Iterable[str]) -> List[str]:
    """All MP IDs in these bucket folders (reads only the mp_id column)."""
    import pyarrow.parquet as pq
    ids: Set[str] = set()
    for d in bucket_dirs:
        ids.update(pq.read_table(os.path.join(d, "part.parquet"), columns=["mp_id"]).column("mp_id").to_pylist())
    return sorted(ids)


DAY_COLS = ["tot", "base", "vmax", "vp99", "run_hours", "n_cycles", "run_level", "run_jump",
            "early_tot", "n_valid"]


def reduce_days_hp(lab: pd.DataFrame, vals: np.ndarray) -> pd.DataFrame:
    """Per-day numbers all twenty-one features are built from (import only, kW, >= 0)."""
    imp = (lab["channel"] == "import").values
    lab = lab[imp].reset_index(drop=True)
    v_raw = vals[imp]
    n = len(lab)
    d = lab[["mp_id", "date"]].copy()
    if n == 0:
        for c in DAY_COLS:
            d[c] = np.array([], dtype=float)
        return d
    finite = np.isfinite(v_raw)
    v = np.where(finite, v_raw, np.float32(0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        d["n_valid"] = finite.sum(axis=1)
        d["tot"] = v.sum(axis=1)                                     # kW quarter-hours (kWh = tot/4)
        d["early_tot"] = v[:, EARLY_MASK].sum(axis=1)
        masked = np.where(finite, v_raw, np.nan)
        base = np.nanpercentile(masked, 10, axis=1)
        base = np.where(np.isfinite(base), base, 0.0)
        d["base"] = base
        d["vmax"] = v.max(axis=1)
        p99 = np.nanpercentile(masked, 99, axis=1)
        d["vp99"] = np.where(np.isfinite(p99), p99, 0.0)
        above = np.where(finite, v_raw - base[:, None], np.float32(0.0))
        run = (above >= RUN_KW) & finite
        n_run = run.sum(axis=1)
        d["run_hours"] = n_run / 4.0
        d["run_level"] = np.where(n_run > 0, (above * run).sum(axis=1) / np.maximum(n_run, 1), 0.0)
        # cycles = runs of at least MIN_RUN_Q consecutive quarter-hours
        starts = run & ~np.concatenate([np.zeros((n, 1), bool), run[:, :-1]], axis=1)
        run_id = np.cumsum(starts, axis=1) * run                     # 0 outside runs, 1..k inside
        max_runs = int(run_id.max())
        lengths = np.zeros((n, max_runs + 1), dtype=np.int32)
        rows = np.repeat(np.arange(n), N_SLOTS)
        np.add.at(lengths, (rows, run_id.ravel()), run.ravel().astype(np.int32))
        lengths[:, 0] = 0
        d["n_cycles"] = (lengths >= MIN_RUN_Q).sum(axis=1)
        # modulation: mean |change| between consecutive quarter-hours INSIDE the runs
        diff = np.abs(np.diff(v, axis=1))
        both = run[:, 1:] & run[:, :-1]
        n_both = both.sum(axis=1)
        d["run_jump"] = np.where(n_both > 0, (diff * both).sum(axis=1) / np.maximum(n_both, 1), 0.0)
    return d


# =============================================================================
# 2b. loading (store or CSVs) with this reduction
# =============================================================================
def _single_thread_blas() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def _reduce_bucket_hp(args) -> List[pd.DataFrame]:
    """Worker (module-level so it can be sent to another process):
    (store, bucket_dir, month, None) = the whole bucket; (..., ids) = only those houses."""
    store, d, month, ids = args
    if ids is None:
        return [reduce_days_hp(lab, vals) for lab, vals in read_store(store, buckets=[d], month=month)]
    return [reduce_days_hp(lab, vals) for lab, vals in read_store(store, mp_ids=ids, month=month)]


def _sum_meters_hp(labs, vs) -> pd.DataFrame:
    """Several meters under one MP ID: sum them per (channel, day), then reduce."""
    if not labs:
        return pd.DataFrame(columns=["mp_id", "date"])
    lab = pd.concat(labs, ignore_index=True)
    mat = pd.DataFrame(np.vstack(vs), columns=range(N_SLOTS))
    mat[["mp_id", "channel", "date"]] = lab[["mp_id", "channel", "date"]]
    summed = mat.groupby(["mp_id", "channel", "date"], sort=False)[list(range(N_SLOTS))].sum(min_count=1).reset_index()
    return reduce_days_hp(summed[["mp_id", "channel", "date"]], summed[list(range(N_SLOTS))].to_numpy(dtype="float32"))


def load_days_hp(root: Optional[str] = None, store: Optional[str] = None, verbose: bool = True,
                 month: Optional[str] = None, max_houses: Optional[int] = None,
                 mp_ids: Optional[Iterable[str]] = None, workers: int = 2,
                 max_files: Optional[int] = None, cache_dir: Optional[str] = None,
                 pattern: str = FILE_GLOB) -> pd.DataFrame:
    """Per-day table: one row per MP ID and calendar day.  From the binary store when `store` is
    given (fast), otherwise from the CSVs below `root`."""
    t0 = time.time()
    keep: Optional[List[str]] = sorted(set(str(m) for m in mp_ids)) if mp_ids is not None else None
    parts: List[pd.DataFrame] = []
    files: List[str] = []
    if store:
        if keep is None and max_houses:
            keep = first_store_ids(store, max_houses)
            if verbose:
                shown = keep if len(keep) <= 20 else keep[:10] + ["..."] + keep[-5:]
                print(f"first {len(keep)} MP IDs of the store: {shown}", flush=True)
        if keep is not None:
            by_bucket: Dict[int, List[str]] = {}
            for m in keep:
                by_bucket.setdefault(bucket_of(m), []).append(m)
            jobs = [(store, os.path.join(store, f"bucket={b}"), month, ids) for b, ids in sorted(by_bucket.items())]
        else:
            jobs = [(store, d, month, None) for d in store_buckets(store)]
        if len(jobs) <= 4 or workers <= 1:
            for job in jobs:
                parts.extend(_reduce_bucket_hp(job))
        else:
            with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1)),
                                     initializer=_single_thread_blas) as pool:
                for i, bparts in enumerate(pool.map(_reduce_bucket_hp, jobs)):
                    parts.extend(bparts)
                    if verbose and (i + 1) % 32 == 0:
                        print(f"  {i + 1}/{len(jobs)} buckets ({time.time() - t0:.0f} s)", flush=True)
        src = "store"
    else:
        if not root:
            raise SystemExit("give the CSV root folder, or --store aew_store (build it with build_store.py)")
        files = find_files(root, month, max_files, pattern)
        if keep is None and max_houses:
            keep = first_mp_ids(files[0], max_houses)
            if verbose:
                print(f"first {len(keep)} MP IDs of {os.path.basename(files[0])}: {sorted(keep)}", flush=True)
        for i, f in enumerate(files):
            if verbose:
                print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, root)}", flush=True)
            for lab, vals in iter_aew_csv(f, mp_ids=keep, cache_dir=cache_dir if keep is not None else None):
                parts.append(reduce_days_hp(lab, vals))
        src = "csv"
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mp_id", "date"])
    # several meters under one MP ID -> the same (mp_id, date) twice -> re-read and sum per day
    dup = days.duplicated(["mp_id", "date"], keep=False)
    if dup.any():
        multi = set(days.loc[dup, "mp_id"].unique())
        if verbose:
            print(f"{len(multi):,} MP IDs have several meters -> summing their meters per day ...", flush=True)
        days = days[~days["mp_id"].isin(multi)]
        labs, vs = [], []
        if store:
            for lab, vals in read_store(store, mp_ids=multi, month=month):
                labs.append(lab); vs.append(vals)
        else:
            for f in files:
                for lab, vals in iter_aew_csv(f, mp_ids=multi, cache_dir=cache_dir):
                    labs.append(lab); vs.append(vals)
        days = pd.concat([days, _sum_meters_hp(labs, vs)], ignore_index=True)
    if verbose:
        print(f"{src}: {len(days):,} house-days for {days['mp_id'].nunique():,} houses in {time.time() - t0:.1f} s", flush=True)
    return days


# =============================================================================
# 3. weather for feature 21 (same file as features_pv21.py, temperature column)
# =============================================================================
def daily_temperature(weather: pd.DataFrame, min_cover: float = 0.9) -> pd.Series:
    """Mean air temperature [°C] per local calendar day — the same key as the `date` column
    of the per-day table.  Days with less than min_cover of their quarter-hours are dropped."""
    t = weather["temperature"].dropna()
    key = t.index.tz_localize(None).normalize()
    grp = t.groupby(key)
    mean, cover = grp.mean(), grp.count() / N_SLOTS
    out = mean[cover >= min_cover].rename("temp_c")
    out.index.name = "date"
    return out


def load_temperature(days: pd.DataFrame, path: str = DEFAULT_WEATHER) -> pd.Series:
    """Mean temperature per calendar day.  Builds the weather file with meteo_aargau.py
    (MeteoSwiss download, needs internet) if it does not exist yet."""
    from meteo_aargau import build, load_weather
    start, end = pd.Timestamp(days["date"].min()).date(), pd.Timestamp(days["date"].max()).date()
    if not os.path.exists(path):
        print(f"weather file {path} not found -> downloading MeteoSwiss data for {start} .. {end} ...", flush=True)
        try:
            built = build(start, end, os.path.dirname(path) or ".")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"could not build the weather file ({e}).\n"
                f"Run features_hp21.py with --no-weather to skip feature 21, or download by hand:\n"
                f"  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bus/ogd-smn_bus_t_historical_2020-2029.csv\n"
                f"  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bus/ogd-smn_bus_t_recent.csv\n"
                f"  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bez/ogd-smn_bez_t_historical_2020-2029.csv\n"
                f"  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bez/ogd-smn_bez_t_recent.csv\n"
                f"into {os.path.join(os.path.dirname(path) or '.', 'raw')}/ and run "
                f"python meteo_aargau.py --start {start} --end {end}")
        if os.path.abspath(built) != os.path.abspath(path):
            os.replace(built, path)
    temp = daily_temperature(load_weather(path))
    cold = int((temp < HEAT_LIMIT_C).sum())
    print(f"weather: {len(temp):,} days with a mean temperature ({temp.index.min().date()} .. "
          f"{temp.index.max().date()}, {temp.min():.1f} .. {temp.max():.1f} °C), {cold:,} of them below "
          f"{HEAT_LIMIT_C:.0f} °C; meter data {start} .. {end}", flush=True)
    return temp


# =============================================================================
# 4. the twenty-one features
# =============================================================================
def _season_or_all(d: pd.DataFrame, months: Tuple[int, ...], houses: pd.Index):
    """Rows of the season; a house with NO day in that season contributes all of its days instead
    (its n_<season>_days column is then 0, which tells the model)."""
    sel = d[d["month"].isin(months)]
    n = sel.groupby("mp_id").size().reindex(houses).fillna(0).astype(int)
    missing = n.index[n == 0]
    if len(missing):
        sel = pd.concat([sel, d[d["mp_id"].isin(missing)]], ignore_index=True)
    return sel, n


def feature_table(days: pd.DataFrame, temp_daily: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per MP ID: the twenty-one features + data-quality columns.  Every feature is a
    computed number for every house."""
    days = days.copy()
    days["date"] = pd.to_datetime(days["date"])
    days["month"] = days["date"].dt.month
    days["kwh"] = days["tot"] / 4.0
    houses = pd.Index(days["mp_id"].unique(), name="mp_id")
    out = pd.DataFrame(index=houses)
    eps = ZERO_KW

    def R(x, fill=0.0) -> pd.Series:
        return x.reindex(houses).fillna(fill)

    W, n_w = _season_or_all(days, WINTER_MONTHS, houses)
    S, n_s = _season_or_all(days, SUMMER_MONTHS, houses)
    gA, gW, gS = days.groupby("mp_id"), W.groupby("mp_id"), S.groupby("mp_id")

    # ---------------- A. seasonal amplitude and baseline ----------------
    e_w, e_s = R(gW["kwh"].mean()), R(gS["kwh"].mean())
    out["hp_winter_summer_energy_ratio"] = e_w / e_s.clip(lower=eps)
    base_w, base_s = R(gW["base"].median()), R(gS["base"].median())
    out["hp_base_winter_minus_summer_kw"] = base_w - base_s
    out["hp_base_winter_over_summer"] = base_w / base_s.clip(lower=eps)
    monthly = days.groupby(["mp_id", "month"])["kwh"].mean()
    mg = monthly.groupby(level="mp_id")
    out["hp_monthly_energy_amplitude"] = R((mg.max() - mg.min()) / mg.mean().clip(lower=eps))
    out["hp_daily_energy_winter_kwh"] = e_w
    out["hp_daily_energy_spread_winter"] = R(
        (gW["kwh"].quantile(0.9) - gW["kwh"].quantile(0.1)) / gW["kwh"].median().clip(lower=eps)).clip(upper=SPREAD_MAX)

    # ---------------- B. running pattern ----------------
    out["hp_run_hours_winter"] = R(gW["run_hours"].mean())
    out["hp_run_hours_winter_minus_summer"] = R(gW["run_hours"].mean()) - R(gS["run_hours"].mean())
    out["hp_cycles_per_day_winter"] = R(gW["n_cycles"].mean())
    out["hp_cycle_length_h_winter"] = R(gW["run_hours"].sum() / gW["n_cycles"].sum().clip(lower=1))
    out["hp_cycles_per_day_summer"] = R(gS["n_cycles"].mean())
    lvl_w = R(gW["run_level"].mean())
    out["hp_level_winter_kw"] = lvl_w
    # the denominator is floored at RUN_KW (the running threshold): a house that almost never runs
    # has a median level near 0, and dividing by it would produce a meaningless huge number.
    out["hp_level_spread_winter"] = R(
        (gW["run_level"].quantile(0.9) - gW["run_level"].quantile(0.1))
        / gW["run_level"].median().clip(lower=RUN_KW)).clip(upper=SPREAD_MAX)
    out["hp_modulation_index_winter"] = R(gW["run_jump"].mean()) / lvl_w.clip(lower=eps)

    # ---------------- C. shape of the day ----------------
    lf_w = R((W["tot"] / N_SLOTS / W["vmax"].clip(lower=eps)).groupby(W["mp_id"]).mean())
    lf_s = R((S["tot"] / N_SLOTS / S["vmax"].clip(lower=eps)).groupby(S["mp_id"]).mean())
    out["hp_load_factor_winter"] = lf_w
    out["hp_load_factor_winter_minus_summer"] = lf_w - lf_s
    em_w = R((W["early_tot"] / W["tot"].clip(lower=eps)).groupby(W["mp_id"]).mean())
    em_s = R((S["early_tot"] / S["tot"].clip(lower=eps)).groupby(S["mp_id"]).mean())
    out["hp_early_morning_share_winter"] = em_w
    out["hp_early_morning_share_winter_minus_summer"] = em_w - em_s

    # ---------------- D. size and persistence of the winter load ----------------
    out["hp_peak_winter_minus_summer_kw"] = R(gW["vp99"].quantile(0.9)) - R(gS["vp99"].quantile(0.9))
    out["hp_daily_energy_persistence_winter"] = _lag1_persistence(W, houses)

    # ---------------- E. temperature: the heating energy signature ----------------
    n_cold = pd.Series(0, index=houses)
    if temp_daily is None:
        out["hp_energy_cold_slope_kwh_per_degc"] = 0.0
    else:
        okd = days.loc[days["n_valid"] >= 0.9 * N_SLOTS, ["mp_id", "date", "kwh"]].copy()
        okd["x"] = okd["date"].map(temp_daily)
        okd = okd[okd["x"].notna() & (okd["x"] < HEAT_LIMIT_C) & okd["kwh"].notna()]
        xy = pd.DataFrame({"mp_id": okd["mp_id"].values, "x": okd["x"].values.astype(float),
                           "y": okd["kwh"].values.astype(float)})
        xy["xx"], xy["xy"] = xy["x"] * xy["x"], xy["x"] * xy["y"]
        sums = xy.groupby("mp_id")[["x", "y", "xx", "xy"]].sum()
        n = xy.groupby("mp_id").size().astype(float)
        den = n * sums["xx"] - sums["x"] ** 2                              # n² · var(x)
        good = (n >= MIN_COLD_DAYS) & (den > MIN_TEMP_STD ** 2 * n * n)
        slope = ((n * sums["xy"] - sums["x"] * sums["y"]) / den.where(good, 1.0)).where(good, 0.0)
        out["hp_energy_cold_slope_kwh_per_degc"] = R(-slope)
        n_cold = R(n)

    # ---------------- data quality ----------------
    out["n_days"] = R(gA.size()).astype(int)
    out["n_summer_days"] = n_s.astype(int)              # 0 = the summer features used all days
    out["n_winter_days"] = n_w.astype(int)
    out["mean_import_kw"] = R(gA["tot"].sum()) / R(gA["n_valid"].sum()).clip(lower=1)
    out["base_kw"] = R(gA["base"].median())
    out["n_cold_days"] = n_cold.astype(int)
    return out


def _lag1_persistence(W: pd.DataFrame, houses: pd.Index) -> pd.Series:
    """Correlation of a house's daily winter energy ANOMALY with the previous calendar day's.
    The anomaly is the day minus the mean of that house's days in the same month, which removes
    the seasonal trend — otherwise every house looks persistent just because winter is winter.
    Closed form over all (yesterday, today) pairs."""
    d = W[["mp_id", "date", "month", "kwh"]].copy()
    d["kwh"] = d["kwh"] - d.groupby(["mp_id", "month"])["kwh"].transform("mean")
    d = d.sort_values(["mp_id", "date"])
    same = d["mp_id"].values[1:] == d["mp_id"].values[:-1]
    consec = (d["date"].values[1:] - d["date"].values[:-1]) == np.timedelta64(1, "D")
    ok = same & consec
    if not ok.any():
        return pd.Series(0.0, index=houses)
    p = pd.DataFrame({"mp_id": d["mp_id"].values[1:][ok],
                      "x": d["kwh"].values[:-1][ok].astype(float),
                      "y": d["kwh"].values[1:][ok].astype(float)})
    p["xx"], p["yy"], p["xy"] = p["x"] ** 2, p["y"] ** 2, p["x"] * p["y"]
    s = p.groupby("mp_id")[["x", "y", "xx", "yy", "xy"]].sum()
    n = p.groupby("mp_id").size().astype(float)
    num = n * s["xy"] - s["x"] * s["y"]
    den = np.sqrt((n * s["xx"] - s["x"] ** 2).clip(lower=0) * (n * s["yy"] - s["y"] ** 2).clip(lower=0))
    r = (num / den.where(den > 0, np.nan)).where(n >= 5)
    return r.reindex(houses).fillna(0.0).clip(-1.0, 1.0)


# =============================================================================
# 5. Excel copy
# =============================================================================
def write_xlsx(tab: pd.DataFrame, path: str) -> bool:
    """Excel copy: houses as rows, features as columns, frozen header, filters, colour scales,
    plus a 'legend' sheet.  Returns False if openpyxl is missing."""
    try:
        import openpyxl
        from openpyxl.formatting.rule import ColorScaleRule
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("(no Excel copy: pip install openpyxl)", flush=True)
        return False
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        tab.reset_index().to_excel(xw, sheet_name="features", index=False, float_format="%.4f")
        pd.DataFrame({"feature": list(FEATURE_DESC), "meaning": list(FEATURE_DESC.values())}
                     ).to_excel(xw, sheet_name="legend", index=False)
        ws = xw.sheets["features"]
        ws.freeze_panes = "B2"; ws.auto_filter.ref = ws.dimensions
        n_rows = len(tab) + 1
        for j, col in enumerate(tab.reset_index().columns, start=1):
            letter = get_column_letter(j)
            ws.column_dimensions[letter].width = max(12, min(34, len(str(col)) * 0.9))
            if col in FEATURE_NAMES and n_rows > 2:
                ws.conditional_formatting.add(f"{letter}2:{letter}{n_rows}",
                    ColorScaleRule(start_type="min", start_color="F7FCF5", end_type="max", end_color="238B45"))
        for cell in ws[1]:
            cell.alignment = openpyxl.styles.Alignment(text_rotation=60, vertical="bottom")
        ws.row_dimensions[1].height = 150
        lg = xw.sheets["legend"]; lg.column_dimensions["A"].width = 44; lg.column_dimensions["B"].width = 100
    return True


def fill_missing(tab: pd.DataFrame) -> pd.DataFrame:
    """Replace every NaN by its FILL_VALUES entry -> a table of numbers only."""
    tab = tab.copy()
    for c, v in FILL_VALUES.items():
        if c in tab.columns:
            tab[c] = tab[c].fillna(v)
    return tab


# =============================================================================
# 6. command line
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder containing the monthly AEW CSVs (not needed with a store)")
    ap.add_argument("--store", help=f"binary store built by build_store.py (default: ./{DEFAULT_STORE} if it exists)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"output CSV (default {DEFAULT_OUT})")
    ap.add_argument("--month", help='only files whose path contains this text, e.g. "Juni 2023" or "2023"')
    ap.add_argument("--max-files", type=int, help="read only the first N monthly files")
    ap.add_argument("--max-houses", type=int, help="the first N MP IDs")
    ap.add_argument("--mp-ids", help="comma-separated MP IDs to keep, e.g. 53628,53701")
    ap.add_argument("--mp-ids-file", help=f"CSV/TXT with the MP IDs in the first column (default {DEFAULT_IDS_FILE} "
                                          "if it exists, unless --mp-ids / --max-houses is given; else all houses)")
    ap.add_argument("--pattern", default=FILE_GLOB, help=f"file name pattern of the meter files (default {FILE_GLOB})")
    ap.add_argument("--keep-nan", action="store_true", help="leave non-computable features empty instead of the neutral value")
    ap.add_argument("--xlsx", help="Excel copy path (default: same name as -o with .xlsx)")
    ap.add_argument("--no-xlsx", action="store_true", help="do not write the Excel copy")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"parallel processes (default {DEFAULT_WORKERS})")
    ap.add_argument("--cache-dir", default="aew_cache", help="folder for the extracted lines (default ./aew_cache; '' = off)")
    ap.add_argument("--batch-buckets", type=int, default=8, help="store, all houses: buckets processed together "
                    "(default 8 of 256; lower = less memory; 0 = all at once)")
    ap.add_argument("--weather", default=DEFAULT_WEATHER, help=f"15-min weather CSV for feature 21 (default {DEFAULT_WEATHER})")
    ap.add_argument("--no-weather", action="store_true", help="skip feature 21 (no download, no internet needed) -> 0")
    a = ap.parse_args(argv)

    store = a.store or (DEFAULT_STORE if os.path.isdir(DEFAULT_STORE) else None)
    if store and a.root and a.store is None:
        print(f"note: using the store ./{DEFAULT_STORE} instead of the CSVs (pass --store '' to force the CSVs)", flush=True)
    if a.store == "":
        store = None
    ids = [x.strip() for x in a.mp_ids.split(",")] if a.mp_ids else None
    ids_file = a.mp_ids_file or (DEFAULT_IDS_FILE if not (a.mp_ids or a.max_houses) and os.path.exists(DEFAULT_IDS_FILE) else None)
    if ids_file:
        if not os.path.exists(ids_file):
            raise SystemExit(f"MP ID file not found: {ids_file}  (edit DEFAULT_IDS_FILE at the top of the script, "
                             f"or pass --mp-ids-file / --mp-ids / --max-houses)")
        with open(ids_file, encoding="utf-8", errors="replace") as f:
            ids = (ids or []) + [ln.strip().split(";")[0].split(",")[0].strip() for ln in f
                                 if ln.strip() and not ln.lower().startswith("mp")]
        print(f"{len(ids):,} MP IDs read from {ids_file}", flush=True)

    kw = dict(month=a.month, workers=a.workers, cache_dir=a.cache_dir or None, pattern=a.pattern)
    if store and ids is None and not a.max_houses and a.batch_buckets > 0:
        # all houses: a few buckets at a time -> features -> next buckets (one row per house kept)
        buckets, tabs, temp, t0 = store_buckets(store), [], None, time.time()
        for i in range(0, len(buckets), a.batch_buckets):
            group = buckets[i:i + a.batch_buckets]
            days = load_days_hp(None, store, verbose=False, mp_ids=store_ids(group), **kw)
            if len(days):
                if temp is None and not a.no_weather:
                    temp = load_temperature(days, a.weather)
                tabs.append(feature_table(days, temp))
            del days
            print(f"  buckets {i + 1}-{i + len(group)} of {len(buckets)}: {sum(len(t) for t in tabs):,} houses done "
                  f"({time.time() - t0:.0f} s)", flush=True)
        tab = pd.concat(tabs)
    else:
        days = load_days_hp(a.root, store, max_houses=a.max_houses, mp_ids=ids, max_files=a.max_files, **kw)
        temp = None if a.no_weather else load_temperature(days, a.weather)
        tab = feature_table(days, temp)

    n_missing = int(tab[FEATURE_NAMES].isna().sum().sum())
    if n_missing:
        print(f"warning: {n_missing} feature values could not be computed" + ("" if a.keep_nan else " -> filled"), flush=True)
    if not a.keep_nan:
        tab = fill_missing(tab)
    tab.to_csv(a.out, float_format="%.4f")
    xlsx = a.xlsx if a.xlsx else os.path.splitext(a.out)[0] + ".xlsx"
    ok = False if a.no_xlsx else write_xlsx(tab, xlsx)
    print(f"\n{len(tab):,} houses -> {a.out}" + (f"  +  {xlsx}" if ok else "") + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 60,
                           "display.float_format", "{:.3f}".format):
        print(tab.T if len(tab) <= 30 else tab[FEATURE_NAMES].describe().T[["count", "mean", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":
    main()