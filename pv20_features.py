"""
pv20_features.py — TWENTY physically-motivated PV / battery main features, computed only
from the AEW smart-meter CSVs (no weather, no external data).

Run (inside the Renku session, from ~/work/mechenergy):
    FAST PATH — after a one-time  python build_store.py ../aew-data/test-blob/input_data -o aew_store :
        python pv20_features.py --store aew_store --mp-ids 53628,45390 -o two_houses.csv     (milliseconds)
        python pv20_features.py --store aew_store -o all_houses.csv                          (all 89k houses: minutes the
                                                             first time, then ~1 min from the cached per-day table)
        (if ./aew_store exists it is used automatically, even when a CSV root is given)
    CSV PATH — no store yet:
    ten houses, every month that exists (one or two whole years):
        python pv20_features.py ../aew-data/test-blob/input_data --max-houses 10 -o ten_houses.csv
    named houses (e.g. ones labelled PV):
        python pv20_features.py ../aew-data/test-blob/input_data --mp-ids 53628,45390,49688 -o pv.csv
    all labelled houses at once (one MP ID per line in the file; one slow pass, then cached):
        python pv20_features.py ../aew-data/test-blob/input_data --mp-ids-file labelled_ids.txt -o labelled.csv
    only 2023:
        python pv20_features.py ../aew-data/test-blob/input_data --max-houses 10 --month 2023
    every house of one month (slow but bounded memory):
        python pv20_features.py ../aew-data/test-blob/input_data --month "Juni 2023" -o juni2023.csv

HOW THE FILES ARE READ (this is what makes the ten-house case fast)
    When a house list is given (--mp-ids, or --max-houses which takes the first N MP IDs
    of the first file), each monthly file is first filtered as TEXT: only the lines that
    start with one of those MP IDs are kept (`grep -E '^(id1|id2|...);'`, Python fallback),
    and only those few lines are parsed into numbers.  Nothing else is ever converted.
    Without a house list the file is streamed in chunks and every row is reduced at once
    to a handful of per-day numbers, so the 96-column matrix never sits in memory.

Input files: only files named LG_AIM2Hackerdays_kWh_*.csv are read (change with --pattern).
Layout of a data row (';'-separated, fixed positions, 101 columns):
    MP ID ; meter id CHxxxx ; OBIS-Code ; Datum dd.mm.yyyy ; PLZ ; 00:15 ; 00:30 ; ... ; 24:00
    OBIS 1-1:1.29.0*255 = energy IMPORTED from the grid  (kWh per 15 min)
    OBIS 1-1:2.29.0*255 = energy EXPORTED to the grid    (kWh per 15 min)
The header line is skipped (it is one column short: it omits the meter id).

THE DATA ARE NON-NEGATIVE.  Import and export are two separate energy counters,
each >= 0 by construction.  Every feature is defined on these two non-negative
series directly; no signed "net" series is built.  Negative raw values are
floored at 0 and counted in n_negative_raw.

==============================================================================
THE TWENTY FEATURES   (summer = May-Aug, winter = Nov-Feb; a seasonal feature
needs >= 20 days in that season, an all-year feature >= 20 days in total, else NaN)
==============================================================================
A. EXPORT COUNTER — direct evidence (NaN when the meter has no export rows)

 1. export_kwh_total            [kWh, >= 0]
    Total energy ever exported.  A meter can only count export if something
    behind it pushes energy out: a generator (PV) or a storage device.
    0 = nothing behind the meter.  This is THE tell; the rest says what and how big.
 2. export_days_frac            [0..1]  share of all days with > 0.1 kWh exported.
    PV -> high (most days in summer, fewer in winter); battery-only -> depends on control.
 3. export_to_import_ratio_summer   [>= 0]  summer export / summer import.  PV -> 0.3 .. 3.
 4. export_to_import_ratio_winter   [>= 0]  same in winter.  PV -> small (weak sun);
    a battery trading with the grid -> similar to summer.
 5. export_winter_share         [0..1]  winter export per day / (summer + winter export per day).
    PV -> ~0.05-0.2 (sun is seasonal); grid-charged battery -> ~0.5 (no season).
 6. export_midday_share_summer  [0..1]  share of summer export between 9 h and 17 h.
    Sun -> 0.7-0.9.  Battery discharging in the evening -> low.
 7. export_night_share          [0..1]  share of ALL export between 20 h and 6 h.
    Sun -> ~0.  Anything here is not the sun: battery (or a CHP).
 8. export_mean_hour_summer     [h]  energy-weighted mean clock hour of summer export.
    PV -> ~13.5 (solar noon in Aargau, summer time).  Later = evening discharge.
 9. export_hour_std_summer      [h]  energy-weighted spread of that hour.
    PV -> ~2-3 h (a bell over the day); a short evening burst -> ~1 h.
10. export_max_kw               [kW]  95th percentile of the daily maximum export power
    ~ size of the inverter / PV system.

B. IMPORT COUNTER — consumption shape (works on every meter)

11. import_midday_night_ratio_summer  [>= 0]  mean import 11-15 h / mean import 01-05 h, summer.
    No PV -> >= 1 (people are awake).  PV -> << 1, down to 0.
12. import_midday_night_ratio_winter  [>= 0]  same in winter.  PV effect weak -> closer to
    a normal house; the summer/winter contrast is the PV signature.
13. import_zero_hours_midday_summer   [h, 0..8]  hours per summer day (9-17 h) with import <= 20 W.
    A house always draws something -> 0 h unless something covers the load: PV.
14. import_zero_hours_night           [h, 0..8]  hours per day (22-6 h) with import <= 20 W, all year.
    The sun cannot do this.  Zero import at night = a battery feeding the house
    (or an empty building).
15. import_midday_share_winter_minus_summer  [-1..1]  midday (11-15 h) share of the day's
    import, winter minus summer.  Out-at-work houses dip in both seasons -> 0;
    PV empties the summer midday only -> positive.
16. import_min_hour_offset_noon_summer  [h]  mean |hour of the daily import minimum - 13.5|
    (centre of the minimum plateau, hourly-smoothed).  PV -> < 1.5 h; night-minimum
    houses -> ~9 h.
17. import_summer_to_winter_ratio     [>= 0]  mean daily import summer / winter.
    PV pulls summer import down -> small (< 0.5); a plain house -> 0.7-1.0.
18. import_daylight_share_summer      [0..1]  share of the summer day's import in 9-17 h.
    Flat house -> 0.33; PV -> well below (< 0.15).
19. import_midday_spread_over_night_summer  [>= 0]  (95th - 5th percentile over summer days
    of the midday import) / mean night import.  PV: sunny days ~0, cloudy days normal
    -> spread about as big as the house's own load (~1); no PV -> steady (~0.2-0.5).
20. import_midday_p05_over_night_summer  [>= 0]  the 5 % lowest summer midday import
    (i.e. the sunniest days) / mean night import.  PV -> 0 on its best days even
    when the average is not; no PV -> ~1.

Export noise floor: the meter resolves 1 Wh per quarter-hour (= 0.004 kW).  Meter creep and
brief reverse flows when a motor switches off give a few Wh, never more; a real generator
exports hundreds of W at noon.  Export values below EXPORT_NOISE_KW (0.05 kW = 12.5 Wh per
quarter-hour, ~10x the noise floor) are therefore treated as 0.  has_real_export = 1 when the
house's typical daily maximum export reaches EXPORT_REAL_KW (0.1 kW); the export-shape
features 5-9 are NaN otherwise.  export_max_raw_kw keeps the raw maximum for the record.

Speed: with a house list, files are skimmed with grep in byte mode, --workers files at a
time, and the extracted lines are cached in --cache-dir (a repeat run on the same houses
reads only the cache).

Data-quality columns: n_days, n_summer_days, n_winter_days, has_export_register, has_real_export,
export_max_raw_kw,
mean_import_kw (average import power: household ~0.2-0.8 kW, tens of kW = business),
n_negative_raw (should be 0).
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

N_SLOTS = 96
SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
SOLAR_NOON_H = 13.5          # local clock hour of solar noon in Aargau on summer time (~13:28)
MIDDAY = (11, 15)            # hours
NIGHT = (1, 5)               # hours, the PV-free reference
DAYLIGHT = (9, 17)           # hours
ZERO_KW = 0.02               # import <= 20 W == "the house imports nothing"
EXPORT_NOISE_KW = 0.05       # export below 50 W (12.5 Wh per quarter-hour) is meter noise, not generation
EXPORT_REAL_KW = 0.10        # a house "really exports" if its typical daily max export reaches 100 W
MIN_DAYS = 20

SLOT_HOUR = (np.arange(N_SLOTS) + 0.5) / 4.0


def _mask(a: float, b: float) -> np.ndarray:
    """Quarter-hours whose centre lies in [a, b) — wraps past midnight if a > b."""
    return (SLOT_HOUR >= a) & (SLOT_HOUR < b) if a < b else (SLOT_HOUR >= a) | (SLOT_HOUR < b)


MID_MASK, NIGHT_MASK, DAY_MASK = _mask(*MIDDAY), _mask(*NIGHT), _mask(*DAYLIGHT)
LATE_MASK = _mask(20, 6)          # 20 h .. 06 h : export here is not the sun
NIGHTZ_MASK = _mask(22, 6)        # 22 h .. 06 h : zero import here is a battery

FEATURE_NAMES = [
    "export_kwh_total", "export_days_frac", "export_to_import_ratio_summer",
    "export_to_import_ratio_winter", "export_winter_share", "export_midday_share_summer",
    "export_night_share", "export_mean_hour_summer", "export_hour_std_summer", "export_max_kw",
    "import_midday_night_ratio_summer", "import_midday_night_ratio_winter",
    "import_zero_hours_midday_summer", "import_zero_hours_night",
    "import_midday_share_winter_minus_summer", "import_min_hour_offset_noon_summer",
    "import_summer_to_winter_ratio", "import_daylight_share_summer",
    "import_midday_spread_over_night_summer", "import_midday_p05_over_night_summer",
]

VCOLS = [f"v{k}" for k in range(N_SLOTS)]
KEY = ["mp_id", "channel", "date"]

# ---- the meter files: known name, known layout (nothing is guessed) --------------------
FILE_GLOB = "LG_AIM2Hackerdays_kWh_*.csv"       # only these files are read (override with --pattern)
LEAD_COLS = ["mp_id", "meter", "obis", "date", "plz"]   # the 5 columns before the 96 values
N_COLS = len(LEAD_COLS) + N_SLOTS                       # = 101 columns per data row
OBIS_IMPORT = "1-1:1.29.0*255"
OBIS_EXPORT = "1-1:2.29.0*255"


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


def _parse(source, chunksize: Optional[int], skip_header: bool):
    """Parse rows in the known layout: 5 leading columns + 96 float values, ';'-separated."""
    names = LEAD_COLS + VCOLS
    dtypes = {c: "string" for c in LEAD_COLS}
    dtypes.update({v: "float32" for v in VCOLS})
    return pd.read_csv(source, sep=";", header=None, skiprows=1 if skip_header else 0, names=names,
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
    if mp_ids is not None:
        text = filter_lines(path, mp_ids, cache_dir)
        if not text.strip():
            return
        out = _finish(_parse(io.StringIO(text), None, skip_header=False))
        if out is not None:
            yield out
        return
    for ch in _parse(path, chunksize, skip_header=True):
        out = _finish(ch)
        if out is not None:
            yield out


# =============================================================================
# 1b. reading the binary store built by build_store.py  (aew_store/bucket=N/part.parquet)
# =============================================================================
STORE_DEFAULT = "aew_store"
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
# 2. reduce every (meter, channel, day) row of 96 values to per-day numbers
# =============================================================================
_F = lambda m: m.astype(np.float32)
_W_MID, _W_NIGHT, _W_DAY, _W_LATE, _W_NIGHTZ = _F(MID_MASK), _F(NIGHT_MASK), _F(DAY_MASK), _F(LATE_MASK), _F(NIGHTZ_MASK)
_W_ONE = np.ones(N_SLOTS, dtype=np.float32)
_H, _H2 = SLOT_HOUR.astype(np.float32), (SLOT_HOUR ** 2).astype(np.float32)


def reduce_days(lab: pd.DataFrame, vals: np.ndarray) -> pd.DataFrame:
    """Per-day numbers all twenty features are built from (values are kW, >= 0).
    Window sums are matrix products with 0/1 weight vectors (one pass, no copies per window)."""
    finite = np.isfinite(vals)
    v = np.where(finite, vals, np.float32(0.0)).astype(np.float32, copy=False)
    fin = finite.astype(np.float32)
    d = lab[KEY + ["n_neg", "vmax_raw"]].copy() if "vmax_raw" in lab else lab[KEY + ["n_neg"]].assign(vmax_raw=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        d["tot"] = v @ _W_ONE                               # kW quarter-hours (kWh = tot/4)
        d["mid_tot"] = v @ _W_MID
        d["day_tot"] = v @ _W_DAY
        d["late_tot"] = v @ _W_LATE
        n_mid, n_night = fin @ _W_MID, fin @ _W_NIGHT
        d["mid_mean"] = np.where(n_mid > 0, d["mid_tot"] / n_mid, np.nan)
        d["night_mean"] = np.where(n_night > 0, (v @ _W_NIGHT) / n_night, np.nan)
        d["hw"] = v @ _H                                    # energy-weighted hour sums
        d["hw2"] = v @ _H2
        vmax = np.where(finite, vals, np.float32(-1.0)).max(axis=1)
        d["vmax"] = np.where(vmax < 0, np.nan, vmax)
        zero = ((v <= ZERO_KW) & finite).astype(np.float32)
        d["zero_q_mid"] = zero @ _W_DAY
        d["zero_q_night"] = zero @ _W_NIGHTZ
        d["n_valid"] = fin @ _W_ONE
        d["n_valid_day"] = fin @ _W_DAY
        d["n_valid_night"] = fin @ _W_NIGHTZ
        # hour of the daily minimum (centre of the minimum plateau of the hourly-smoothed curve)
        n = len(v)
        cnt = fin.reshape(n, 24, 4).sum(axis=2)
        hourly = np.where(cnt > 0, v.reshape(n, 24, 4).sum(axis=2) / np.maximum(cnt, 1), np.nan)
        lo = np.nanmin(hourly, axis=1, keepdims=True)
        tol = np.maximum(ZERO_KW, 0.05 * np.nanmean(hourly, axis=1, keepdims=True))
        at_min = hourly <= lo + tol
        idx = np.arange(24)
        centre = (at_min * idx).sum(axis=1) / np.maximum(at_min.sum(axis=1), 1) + 0.5
        argmin = np.nanargmin(np.where(np.isfinite(hourly), hourly, np.inf), axis=1) + 0.5
        wraps = at_min[:, 0] & at_min[:, 23]          # plateau crosses midnight (a night minimum)
        d["min_hour"] = np.where(wraps, argmin, centre)
    return d


def load_days(root: Optional[str] = None, verbose: bool = True, month: Optional[str] = None,
              max_files: Optional[int] = None, max_houses: Optional[int] = None,
              mp_ids: Optional[Iterable[str]] = None, pattern: str = FILE_GLOB, workers: int = 4,
              cache_dir: Optional[str] = None, store: Optional[str] = None,
              use_days_cache: bool = True) -> pd.DataFrame:
    """Return the per-day table (one row per MP ID, channel, day).
    From the binary store when `store` is given (fast), otherwise from the CSVs below `root`."""
    if store:
        return load_days_from_store(store, verbose, month, max_houses, mp_ids, workers, use_days_cache)
    if not root:
        raise SystemExit("give the CSV root folder, or --store aew_store (build it with build_store.py)")
    files = find_files(root, month, max_files, pattern)
    keep: Optional[Set[str]] = set(str(m) for m in mp_ids) if mp_ids is not None else None
    if keep is None and max_houses:
        keep = set(first_mp_ids(files[0], max_houses))
        if verbose:
            print(f"first {len(keep)} MP IDs of {os.path.basename(files[0])}: {sorted(keep)}")

    def one_file(f: str) -> Tuple[str, List[pd.DataFrame], int, Set[str]]:
        parts, n_rows, ids = [], 0, set()
        for lab, vals in iter_aew_csv(f, mp_ids=keep, cache_dir=cache_dir if keep is not None else None):
            parts.append(reduce_days(lab, vals))
            n_rows += len(lab); ids.update(lab["mp_id"].unique())
        return f, parts, n_rows, ids

    parts: List[pd.DataFrame] = []
    n_workers = max(1, workers) if keep is not None else 1        # the chunked full read stays sequential
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, (f, fparts, n_rows, ids) in enumerate(pool.map(one_file, files)):
            if verbose:
                print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, root)} ... {n_rows:,} rows, {len(ids):,} houses", flush=True)
            parts.extend(fparts)
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["n_neg"])
    return _fix_multimeter(days, verbose, lambda ids: [_exact_days_for(f, ids, cache_dir) for f in files])


def load_days_from_store(store: str, verbose: bool = True, month: Optional[str] = None,
                         max_houses: Optional[int] = None, mp_ids: Optional[Iterable[str]] = None,
                         workers: int = 4, use_days_cache: bool = True) -> pd.DataFrame:
    """Per-day table from the binary store.  A house list touches only its buckets / row groups;
    without a list every bucket is processed (`workers` in parallel)."""
    keep: Optional[List[str]] = sorted(set(str(m) for m in mp_ids)) if mp_ids is not None else None
    if keep is None and max_houses:
        keep = first_store_ids(store, max_houses)
        if verbose:
            print(f"first {len(keep)} MP IDs of the store: {keep}")
    t0 = time.time()
    days_cache = os.path.join(store, "_days.parquet")          # per-day table of ALL houses, no month filter
    if keep is None and not month and use_days_cache and os.path.exists(days_cache):
        days = pd.read_parquet(days_cache)
        if verbose:
            print(f"store: per-day table for all houses loaded from {days_cache} ({len(days):,} rows, {time.time() - t0:.1f} s)"
                  f"  [--refresh-days to recompute]", flush=True)
        return days
    if keep is not None:
        parts = [reduce_days(lab, vals) for lab, vals in read_store(store, mp_ids=keep, month=month)]
        if verbose:
            n = sum(len(p) for p in parts)
            print(f"store: {n:,} meter-days for {len(keep)} houses in {time.time() - t0:.2f} s", flush=True)
    else:
        buckets = store_buckets(store)
        parts = []
        # the reduction is CPU-bound -> separate processes, one bucket at a time each
        with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1)),
                                 initializer=_single_thread_blas) as pool:
            for i, bparts in enumerate(pool.map(_reduce_bucket, [(store, d, month) for d in buckets])):
                parts.extend(bparts)
                if verbose and (i + 1) % 32 == 0:
                    print(f"  {i + 1}/{len(buckets)} buckets ({time.time() - t0:.0f} s)", flush=True)
        if verbose:
            print(f"store: all {len(buckets)} buckets reduced in {time.time() - t0:.0f} s", flush=True)
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["n_neg"])
    days = _fix_multimeter(days, verbose, lambda ids: [_exact_days_from_store(store, ids, month)])
    if keep is None and not month and use_days_cache:
        days.to_parquet(days_cache, index=False)
        if verbose:
            print(f"store: per-day table saved to {days_cache} (next all-houses run skips the reduction)", flush=True)
    return days


def _single_thread_blas() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def _reduce_bucket(args) -> List[pd.DataFrame]:
    """Worker for the all-houses store run (module-level so it can be sent to another process)."""
    store, d, month = args
    return [reduce_days(lab, vals) for lab, vals in read_store(store, buckets=[d], month=month)]


def _fix_multimeter(days: pd.DataFrame, verbose: bool, exact_fn) -> pd.DataFrame:
    """Several meters under one MP ID -> the same (mp_id, channel, date) appears more than once.
    Sums would be additive but zero-hour counts / means are not, so those houses are re-read
    with their meters summed per day (exact_fn(ids) -> list of per-day tables)."""
    dup = days.duplicated(KEY, keep=False)
    if not dup.any():
        return days
    multi = set(days.loc[dup, "mp_id"].unique())
    if verbose:
        print(f"{len(multi):,} MP IDs have several meters -> summing their meters per day ...", flush=True)
    days = days[~days["mp_id"].isin(multi)]
    return pd.concat([days] + exact_fn(multi), ignore_index=True)


def _exact_days_from_store(store: str, mp_ids: Set[str], month: Optional[str]) -> pd.DataFrame:
    labs, vs = [], []
    for lab, vals in read_store(store, mp_ids=mp_ids, month=month):
        labs.append(lab); vs.append(vals)
    return _sum_meters(labs, vs)


def _sum_meters(labs, vs) -> pd.DataFrame:
    if not labs:
        return pd.DataFrame(columns=KEY + ["n_neg"])
    lab = pd.concat(labs, ignore_index=True)
    mat = pd.DataFrame(np.vstack(vs), columns=range(N_SLOTS))
    mat[KEY] = lab[KEY]
    summed = mat.groupby(KEY, sort=False)[list(range(N_SLOTS))].sum(min_count=1).reset_index()
    summed = summed.merge(lab.groupby(KEY, sort=False)[["n_neg", "vmax_raw"]].agg({"n_neg": "sum", "vmax_raw": "max"}).reset_index(), on=KEY)
    return reduce_days(summed[KEY + ["n_neg", "vmax_raw"]], summed[list(range(N_SLOTS))].to_numpy(dtype="float32"))


def _exact_days_for(path: str, mp_ids: Set[str], cache_dir: Optional[str] = None) -> pd.DataFrame:
    labs, vs = [], []
    for lab, vals in iter_aew_csv(path, mp_ids=mp_ids, cache_dir=cache_dir):
        labs.append(lab); vs.append(vals)
    return _sum_meters(labs, vs)


# =============================================================================
# 3. the twenty features, vectorised over all houses from the per-day table
# =============================================================================
def _valid(s: pd.Series, n: pd.Series) -> pd.Series:
    return s.where(n.reindex(s.index).fillna(0) >= MIN_DAYS)


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0)


def feature_table(days: pd.DataFrame) -> pd.DataFrame:
    days = days.copy()
    days["month"] = pd.to_datetime(days["date"]).dt.month
    I = days[days["channel"] == "import"]
    E = days[days["channel"] == "export"]
    houses = pd.Index(I["mp_id"].unique(), name="mp_id")
    out = pd.DataFrame(index=houses)

    Is, Iw = I[I["month"].isin(SUMMER_MONTHS)], I[I["month"].isin(WINTER_MONTHS)]
    Es, Ew = E[E["month"].isin(SUMMER_MONTHS)], E[E["month"].isin(WINTER_MONTHS)]
    n_all, n_s, n_w = I.groupby("mp_id").size(), Is.groupby("mp_id").size(), Iw.groupby("mp_id").size()
    has_exp = houses.isin(E["mp_id"].unique())
    g = lambda d, c: d.groupby("mp_id")[c]

    def R(s: pd.Series) -> pd.Series:               # align to the house index
        return s.reindex(houses)

    # ---------------- A. export counter ----------------
    exp_tot_all, exp_tot_s, exp_tot_w = g(E, "tot").sum(), g(Es, "tot").sum(), g(Ew, "tot").sum()
    out["export_kwh_total"] = _valid(R(exp_tot_all / 4.0), n_all)
    out["export_days_frac"] = _valid(R((E["tot"] / 4.0 > 0.1).groupby(E["mp_id"]).mean()), n_all)
    out["export_to_import_ratio_summer"] = _valid(R(_ratio(exp_tot_s, g(Is, "tot").sum())), n_s)
    out["export_to_import_ratio_winter"] = _valid(R(_ratio(exp_tot_w, g(Iw, "tot").sum())), n_w)
    per_day_s, per_day_w = _ratio(exp_tot_s, g(Es, "tot").size()), _ratio(exp_tot_w, g(Ew, "tot").size())
    both = per_day_s.add(per_day_w, fill_value=0)
    f5 = _ratio(per_day_w.reindex(both.index).fillna(0), both)
    out["export_winter_share"] = _valid(_valid(R(f5), n_s), n_w)
    out["export_midday_share_summer"] = _valid(R(_ratio(g(Es, "day_tot").sum(), exp_tot_s)), n_s)
    out["export_night_share"] = _valid(R(_ratio(g(E, "late_tot").sum(), exp_tot_all)), n_all)
    mean_h = _ratio(g(Es, "hw").sum(), exp_tot_s)
    var_h = _ratio(g(Es, "hw2").sum(), exp_tot_s) - mean_h ** 2
    out["export_mean_hour_summer"] = _valid(R(mean_h), n_s)
    out["export_hour_std_summer"] = _valid(R(np.sqrt(var_h.clip(lower=0))), n_s)
    out["export_max_kw"] = _valid(R(g(E, "vmax").quantile(0.95)), n_all)
    for c in FEATURE_NAMES[:10]:                    # no export rows at all -> NaN, not 0
        out.loc[~has_exp, c] = np.nan
    # shape-of-export features are meaningless when there is no real export -> NaN
    real = out["export_max_kw"] >= EXPORT_REAL_KW
    for c in ("export_winter_share", "export_midday_share_summer", "export_night_share",
              "export_mean_hour_summer", "export_hour_std_summer"):
        out.loc[~real, c] = np.nan

    # ---------------- B. import counter ----------------
    night_s, night_w = g(Is, "night_mean").mean(), g(Iw, "night_mean").mean()
    out["import_midday_night_ratio_summer"] = _valid(R(_ratio(g(Is, "mid_mean").mean(), night_s)), n_s)
    out["import_midday_night_ratio_winter"] = _valid(R(_ratio(g(Iw, "mid_mean").mean(), night_w)), n_w)
    ok = Is[Is["n_valid_day"] >= 0.9 * DAY_MASK.sum()]
    out["import_zero_hours_midday_summer"] = _valid(R(g(ok, "zero_q_mid").mean() / 4.0), g(ok, "zero_q_mid").size())
    ok = I[I["n_valid_night"] >= 0.9 * NIGHTZ_MASK.sum()]
    out["import_zero_hours_night"] = _valid(R(g(ok, "zero_q_night").mean() / 4.0), g(ok, "zero_q_night").size())

    def _share(d: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
        okd = d[(d["tot"] > 0) & (d["n_valid"] >= 0.9 * N_SLOTS)]
        gg = (okd[col] / okd["tot"]).groupby(okd["mp_id"])
        return gg.mean(), gg.size()
    sw, nw_ok = _share(Iw, "mid_tot")
    ss, ns_ok = _share(Is, "mid_tot")
    out["import_midday_share_winter_minus_summer"] = R(_valid(sw, nw_ok) - _valid(ss, ns_ok))
    ok = Is[Is["n_valid"] >= 0.9 * N_SLOTS]
    out["import_min_hour_offset_noon_summer"] = _valid(R((ok["min_hour"] - SOLAR_NOON_H).abs().groupby(ok["mp_id"]).mean()),
                                                       g(ok, "min_hour").size())
    out["import_summer_to_winter_ratio"] = _valid(_valid(R(_ratio(g(Is, "tot").mean(), g(Iw, "tot").mean())), n_s), n_w)
    sd, nd_ok = _share(Is, "day_tot")
    out["import_daylight_share_summer"] = R(_valid(sd, nd_ok))
    mm = g(Is, "mid_mean")
    out["import_midday_spread_over_night_summer"] = _valid(R(_ratio(mm.quantile(0.95) - mm.quantile(0.05), night_s)), n_s)
    out["import_midday_p05_over_night_summer"] = _valid(R(_ratio(mm.quantile(0.05), night_s)), n_s)

    # ---------------- data quality ----------------
    out["n_days"] = R(n_all).fillna(0).astype(int)
    out["n_summer_days"] = R(n_s).fillna(0).astype(int)
    out["n_winter_days"] = R(n_w).fillna(0).astype(int)
    out["has_export_register"] = has_exp.astype(int)
    out["has_real_export"] = np.where(has_exp, real.astype(int), np.nan)        # typical daily max >= EXPORT_REAL_KW
    out["export_max_raw_kw"] = R(g(E, "vmax_raw").max())                       # before the noise floor was applied
    tot, nv = g(I, "tot").sum(), g(I, "n_valid").sum()
    out["mean_import_kw"] = R(_ratio(tot, nv))
    out["n_negative_raw"] = R(days.groupby("mp_id")["n_neg"].sum()).fillna(0).astype(int)
    return out


# =============================================================================
# 4. single-house helpers (used by pv20_inspect.py)
# =============================================================================
def load_house(root: str, mp_id: str, month: Optional[str] = None,
               max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """{"import": DataFrame[date x 96 kW], "export": ..., "n_negative_raw": int} for one MP ID."""
    labs, vs = [], []
    for f in find_files(root, month, max_files):
        for lab, vals in iter_aew_csv(f, mp_ids=[str(mp_id)]):
            labs.append(lab); vs.append(vals)
    if not labs:
        raise SystemExit(f"MP ID {mp_id} not found in the selected files")
    lab = pd.concat(labs, ignore_index=True)
    mat = pd.DataFrame(np.vstack(vs), columns=range(N_SLOTS))
    mat[["channel", "date"]] = lab[["channel", "date"]]
    house: Dict[str, pd.DataFrame] = {"n_negative_raw": int(lab["n_neg"].sum())}
    for ch, gch in mat.groupby("channel"):
        house[ch] = gch.groupby("date")[list(range(N_SLOTS))].sum(min_count=1).sort_index()
    return house


def house_features(house: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    parts = []
    for ch in ("import", "export"):
        if ch in house:
            m = house[ch]
            lab = pd.DataFrame({"mp_id": "x", "channel": ch, "date": pd.to_datetime(m.index), "n_neg": 0})
            parts.append(reduce_days(lab, m.to_numpy(dtype="float32")))
    days = pd.concat(parts, ignore_index=True)
    days.loc[days.index[0], "n_neg"] = int(house.get("n_negative_raw", 0))
    return feature_table(days).iloc[0].to_dict()


# =============================================================================
# 5. command line
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder containing the monthly AEW CSVs (not needed with a store)")
    ap.add_argument("--store", help=f"binary store built by build_store.py (default: ./{STORE_DEFAULT} if it exists)")
    ap.add_argument("-o", "--out", default="pv20_features.csv")
    ap.add_argument("--month", help='only files whose path contains this text, e.g. "Juni 2023" or "2023"')
    ap.add_argument("--max-files", type=int, help="read only the first N monthly files")
    ap.add_argument("--max-houses", type=int, help="the first N MP IDs of the first file, followed through ALL files")
    ap.add_argument("--mp-ids", help="comma-separated MP IDs to keep, e.g. 53628,53701")
    ap.add_argument("--mp-ids-file", help="text file with one MP ID per line (e.g. all labelled houses)")
    ap.add_argument("--pattern", default=FILE_GLOB, help=f"file name pattern of the meter files (default {FILE_GLOB})")
    ap.add_argument("--refresh-days", action="store_true", help="store: recompute the all-houses per-day table instead of using _days.parquet")
    ap.add_argument("--workers", type=int, default=4, help="files skimmed in parallel when a house list is given (default 4)")
    ap.add_argument("--cache-dir", default="aew_cache", help="folder for the extracted lines (default ./aew_cache; '' = off)")
    a = ap.parse_args(argv)

    store = a.store or (STORE_DEFAULT if os.path.isdir(STORE_DEFAULT) else None)
    if store and a.root and a.store is None:
        print(f"note: using the store ./{STORE_DEFAULT} instead of the CSVs (pass --store '' to force the CSVs)", flush=True)
    if a.store == "":
        store = None
    ids = [x.strip() for x in a.mp_ids.split(",")] if a.mp_ids else None
    if a.mp_ids_file:
        with open(a.mp_ids_file) as f:
            ids = (ids or []) + [ln.strip().split(";")[0].split(",")[0] for ln in f if ln.strip() and not ln.lower().startswith("mp")]
    days = load_days(a.root, month=a.month, max_files=a.max_files, max_houses=a.max_houses, mp_ids=ids,
                     pattern=a.pattern, workers=a.workers, cache_dir=a.cache_dir or None, store=store,
                     use_days_cache=not a.refresh_days)
    tab = feature_table(days)
    tab.to_csv(a.out, float_format="%.4f")
    print(f"\n{len(tab):,} houses -> {a.out}\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 60,
                           "display.float_format", "{:.3f}".format):
        if len(tab) <= 30:
            print(tab.T)                                  # houses as columns: easier to read 20 features
        else:
            print(tab[FEATURE_NAMES].describe().T[["count", "mean", "min", "25%", "50%", "75%", "max"]])
    n_nan = tab[FEATURE_NAMES].isna().all()
    if n_nan.any():
        print("\nall-NaN features (no summer / no winter days in the selected files, or no export rows):",
              ", ".join(n_nan[n_nan].index))


if __name__ == "__main__":
    main()
