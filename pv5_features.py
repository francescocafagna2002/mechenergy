"""
pv5_features.py — FIVE physically-motivated PV main features, computed only from
the AEW smart-meter CSVs (no weather, no external data).

Run (inside the Renku session):
    python pv5_features.py /home/renku/work/aew-data/test-blob/input_data -o pv5_features.csv

Quick "does it work" run — one month, ten houses:
    python pv5_features.py ../aew-data/test-blob/input_data --month "Juni 2023" --max-houses 10 -o quick.csv
    (pick a summer month: with a single month only the summer features can be computed;
     feature 5 needs winter days too and will be NaN.  --max-files 1 takes the first file instead.)

Input file layout (one CSV per month, ';'-separated):
    MP ID ; <meter id CHxxxx> ; OBIS-Code ; Datum dd.mm.yyyy ; PLZ ; 00:15 ; 00:30 ; ... ; 24:00
    OBIS 1-1:1.29.0*255 = energy IMPORTED from the grid  (kWh per 15 min)
    OBIS 1-1:2.29.0*255 = energy EXPORTED to the grid    (kWh per 15 min)
The header line has one column fewer than the data rows (it omits the meter id),
so the parser never trusts header positions: the LAST 96 columns are the values,
the leading columns are recognised by their content.

THE DATA ARE NON-NEGATIVE.  Import and export are two separate energy counters;
each is >= 0 by construction and a house without an export register simply has
no export rows.  Every feature below is therefore defined on these two
non-negative series directly.  We never build a signed "net = import - export"
series.  A negative raw value cannot occur physically: it is floored at 0 and
counted in the n_negative_raw output column.

Internal representation: one "day matrix" per house and channel
    rows = dates, 96 columns = quarter-hours, values in kW (kWh x 4), all >= 0.
    Column k covers [k*15 min, (k+1)*15 min); its centre is (k + 0.5) / 4 h.

Output: one CSV row per MP ID with the five features plus data-quality columns.

------------------------------------------------------------------------------
THE FIVE FEATURES  (summer = May-Aug, winter = Nov-Feb, >= 20 days else NaN)
------------------------------------------------------------------------------
Two features read the EXPORT counter (direct evidence, NaN if the meter has no
export register); three read only the IMPORT counter (consumption), so they
also work for meters that record nothing but consumption.

1. export_to_import_ratio_summer            [>= 0]
       summer exported kWh / summer imported kWh.
       Only a generator on the house side of the meter can push energy out.
       No PV -> exactly 0.  PV -> typically 0.3 ... 3.

2. export_midday_share_summer               [0 ... 1]
       share of the summer exported energy that falls between 9 h and 17 h.
       Sunlight is concentrated around solar noon (~13:28 local summer time in
       Aargau), so PV export is too: PV -> roughly 0.7 ... 0.9.  Export spread
       evenly over the day, or at night, is NOT the sun (battery, CHP, data
       error).  NaN when the meter never exports.

3. import_midday_night_ratio_summer         [>= 0]
       mean import 11-15 h / mean import 01-05 h on the average summer day.
       Night import is a PV-free reference for the size of the house.
       Without PV people use at least as much at lunchtime as at night
       -> ratio >= 1.  With PV the sun covers the house at midday and the
       import counter drops towards 0 -> ratio well below 1, down to 0.

4. import_zero_hours_midday_summer          [hours, 0 ... 8]
       average number of hours per summer day, between 9 h and 17 h, during
       which the import counter reads (almost) zero (<= 0.02 kW).
       A house without generation ALWAYS draws something (fridge, standby,
       router) so its import never touches zero -> 0 h.  A PV house imports
       nothing for hours around noon on most summer days -> several hours.
       This is the "positive-only" way of saying "the load minimum sits at
       solar noon": the zeros can only be produced by on-site generation.

5. import_midday_share_winter_minus_summer  [-1 ... 1]
       share(season) = imported energy 11-15 h / imported energy of the whole
                       day, averaged over the season's days   (0 ... 1)
       feature = share(winter) - share(summer)
       A flat house has share ~ 4/24 = 0.17 in every season.  A household
       where everyone is out at work has a lower share, but the SAME lower
       share in winter and summer -> difference ~ 0.  PV drives the summer
       midday share to ~0 while the winter share stays near normal (short,
       weak sun) -> difference clearly positive.  Being a share of the
       day's own energy it does not depend on house size, on winter heating,
       or on any signed quantity.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

N_SLOTS = 96
SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
MIDDAY = (11, 15)            # hours, features 3 and 5
NIGHT = (1, 5)               # hours, feature 3
EXPORT_WINDOW = (9, 17)      # hours, feature 2
ZERO_WINDOW = (9, 17)        # hours, feature 4
ZERO_KW = 0.02               # import <= 20 W  ==  "the house imports nothing" (0.005 kWh per 15 min)
MIN_DAYS = 20                # minimum days in a season before a seasonal feature is trusted

SLOT_HOUR = (np.arange(N_SLOTS) + 0.5) / 4.0        # centre hour of each quarter-hour


def _mask(window: Tuple[int, int]) -> np.ndarray:
    return (SLOT_HOUR >= window[0]) & (SLOT_HOUR < window[1])


MID_MASK, NIGHT_MASK = _mask(MIDDAY), _mask(NIGHT)
EXPORT_MASK, ZERO_MASK = _mask(EXPORT_WINDOW), _mask(ZERO_WINDOW)

FEATURE_NAMES = [
    "export_to_import_ratio_summer",
    "export_midday_share_summer",
    "import_midday_night_ratio_summer",
    "import_zero_hours_midday_summer",
    "import_midday_share_winter_minus_summer",
]


# =============================================================================
# 1. reading the CSVs  ->  {mp_id: {"import": day-matrix, "export": day-matrix}}
# =============================================================================
DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
OBIS_RE = re.compile(r"^\d+-\d+:\d+\.\d+\.\d+")
METER_RE = re.compile(r"^[A-Z]{2}\d{6,}")


def find_files(root: str) -> List[str]:
    """Every .csv below root (any depth), sorted."""
    return sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))


def read_aew_csv(path: str) -> pd.DataFrame:
    """One monthly file -> table: mp_id, channel, date, v0..v95 (kW, >= 0)."""
    df = pd.read_csv(path, sep=";", header=None, skiprows=1, dtype=str,
                     encoding="utf-8", encoding_errors="replace", keep_default_na=False)
    n_lead = df.shape[1] - N_SLOTS
    if n_lead < 3:
        raise ValueError(f"{path}: expected >=3 leading columns + {N_SLOTS} values, got {df.shape[1]}")

    # recognise the leading columns by content (header is unreliable)
    lead = list(range(n_lead))
    first = df.iloc[0, :n_lead].astype(str).str.strip()
    roles: Dict[str, int] = {}
    for c in lead:
        v = first[c]
        if "date" not in roles and DATE_RE.match(v):
            roles["date"] = c
        elif "obis" not in roles and OBIS_RE.match(v):
            roles["obis"] = c
        elif "meter" not in roles and METER_RE.match(v):
            roles["meter"] = c
    rest = [c for c in lead if c not in roles.values()]
    roles["mp_id"] = rest[0]                     # first unrecognised numeric column = MP ID
    if "date" not in roles or "obis" not in roles:
        raise ValueError(f"{path}: could not find date / OBIS columns in first row {list(first)}")

    obis = df[roles["obis"]].str.strip()
    channel = np.where(obis.str.match(r"^\d+-\d+:1\.29\."), "import",
              np.where(obis.str.match(r"^\d+-\d+:2\.29\."), "export", "other"))
    out = pd.DataFrame({
        "mp_id": df[roles["mp_id"]].str.strip(),
        "channel": channel,
        "date": pd.to_datetime(df[roles["date"]].str.strip(), format="%d.%m.%Y", errors="coerce"),
    })
    vals = df.iloc[:, n_lead:].replace({"": np.nan, " ": np.nan, "-": np.nan})
    vals = vals.apply(lambda s: pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce"))
    vals = vals.astype("float32") * 4.0          # kWh per 15 min -> mean kW
    vals.columns = [f"v{k}" for k in range(N_SLOTS)]
    out = pd.concat([out, vals], axis=1)
    return out[(out["channel"] != "other") & out["date"].notna()]


def load_houses(root: str, verbose: bool = True, month: Optional[str] = None,
                max_files: Optional[int] = None, max_houses: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """All files below root -> {mp_id: {"import": DataFrame[date x 96 kW], "export": ...,
    "n_negative_raw": int}}.  Several meters per house are summed per channel and day.
    Quick-test knobs: `month` keeps only files whose path contains that text (e.g. "Juni 2023"),
    `max_files` reads only the first N files, `max_houses` keeps only the first N MP IDs seen."""
    files = find_files(root)
    if month:
        files = [f for f in files if month.lower() in f.lower()]
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"no .csv files found below {root}" + (f" matching '{month}'" if month else ""))
    parts = []
    for i, f in enumerate(files):
        if verbose:
            print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, root)}")
        parts.append(read_aew_csv(f))
    long = pd.concat(parts, ignore_index=True)
    if max_houses:
        keep = long["mp_id"].drop_duplicates().iloc[:max_houses]
        long = long[long["mp_id"].isin(keep)]
    vcols = [f"v{k}" for k in range(N_SLOTS)]
    # Import and export are meter COUNTERS: every value is >= 0 by construction.
    # A negative number can only be a data/parsing error -> count it, then floor at 0.
    neg = (long[vcols] < 0).sum(axis=1)
    long[vcols] = long[vcols].clip(lower=0.0)
    houses: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (mp_id, ch), g in long.groupby(["mp_id", "channel"], sort=False):
        mat = g.groupby("date")[vcols].sum(min_count=1).sort_index()
        mat.columns = range(N_SLOTS)
        h = houses.setdefault(str(mp_id), {})
        h[ch] = mat
        h["n_negative_raw"] = h.get("n_negative_raw", 0) + int(neg.loc[g.index].sum())
    return houses


# =============================================================================
# 2. helpers
# =============================================================================
def _season(mat: pd.DataFrame, months: Tuple[int, ...]) -> pd.DataFrame:
    return mat[mat.index.month.isin(months)]


def _median_import_kw(house) -> float:
    """Median of the strictly positive import values = typical size of the house."""
    x = house["import"].values.ravel()
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.median(x)) if len(x) else np.nan


def _midday_share(imp_season: pd.DataFrame) -> float:
    """Mean over days of (import 11-15 h) / (import of the whole day); days with no import skipped."""
    if len(imp_season) < MIN_DAYS:
        return np.nan
    x = imp_season.values
    day_total = np.nansum(x, axis=1)
    mid_total = np.nansum(x[:, MID_MASK], axis=1)
    ok = (day_total > 0) & (np.isfinite(x).sum(axis=1) >= 0.9 * N_SLOTS)
    if ok.sum() < MIN_DAYS:
        return np.nan
    return float(np.mean(mid_total[ok] / day_total[ok]))


# =============================================================================
# 3. the five features
# =============================================================================
def export_to_import_ratio_summer(house) -> float:
    """[>= 0]  summer export kWh / summer import kWh.  NaN without export register."""
    if "export" not in house:
        return np.nan
    imp, exp = _season(house["import"], SUMMER_MONTHS), _season(house["export"], SUMMER_MONTHS)
    if len(imp) < MIN_DAYS:
        return np.nan
    i, e = np.nansum(imp.values), np.nansum(exp.values)      # both sums >= 0
    return float(e / i) if i > 0 else np.nan


def export_midday_share_summer(house) -> float:
    """[0..1]  share of summer exported energy between 9 h and 17 h.  NaN if nothing is exported."""
    if "export" not in house:
        return np.nan
    exp = _season(house["export"], SUMMER_MONTHS)
    if len(exp) < MIN_DAYS:
        return np.nan
    total = np.nansum(exp.values)
    if total <= 0:
        return np.nan
    return float(np.nansum(exp.values[:, EXPORT_MASK]) / total)


def import_midday_night_ratio_summer(house) -> float:
    """[>= 0]  mean import 11-15 h / mean import 01-05 h on the average summer day."""
    imp = _season(house["import"], SUMMER_MONTHS)
    if len(imp) < MIN_DAYS:
        return np.nan
    prof = imp.mean(axis=0, skipna=True).values
    night = np.nanmean(prof[NIGHT_MASK])
    if not np.isfinite(night) or night <= 0:
        return np.nan
    return float(np.nanmean(prof[MID_MASK]) / night)


def import_zero_hours_midday_summer(house) -> float:
    """[hours]  mean hours per summer day, 9-17 h, with import <= ZERO_KW."""
    imp = _season(house["import"], SUMMER_MONTHS)
    if len(imp) < MIN_DAYS:
        return np.nan
    win = imp.values[:, ZERO_MASK]                       # days x 32 quarter-hours
    complete = np.isfinite(win).sum(axis=1) >= 0.9 * win.shape[1]
    if complete.sum() < MIN_DAYS:
        return np.nan
    zero_q = np.nansum(win[complete] <= ZERO_KW, axis=1)      # zero quarter-hours per day
    return float(zero_q.mean() / 4.0)


def import_midday_share_winter_minus_summer(house) -> float:
    """[-1..1]  midday share of daily import in winter minus in summer (PV -> positive)."""
    sw = _midday_share(_season(house["import"], WINTER_MONTHS))
    ss = _midday_share(_season(house["import"], SUMMER_MONTHS))
    return float(sw - ss) if np.isfinite(sw) and np.isfinite(ss) else np.nan


FEATURES = {
    "export_to_import_ratio_summer": export_to_import_ratio_summer,
    "export_midday_share_summer": export_midday_share_summer,
    "import_midday_night_ratio_summer": import_midday_night_ratio_summer,
    "import_zero_hours_midday_summer": import_zero_hours_midday_summer,
    "import_midday_share_winter_minus_summer": import_midday_share_winter_minus_summer,
}


def house_features(house) -> Dict[str, float]:
    """All five features + data-quality columns for one house. Never raises."""
    row: Dict[str, float] = {}
    for name, fn in FEATURES.items():
        try:
            row[name] = float(fn(house))
        except Exception:
            row[name] = np.nan
    imp = house["import"]
    row["n_days"] = len(imp)
    row["n_summer_days"] = len(_season(imp, SUMMER_MONTHS))
    row["n_winter_days"] = len(_season(imp, WINTER_MONTHS))
    row["has_export_register"] = int("export" in house)
    row["median_import_kw"] = _median_import_kw(house)
    row["n_negative_raw"] = int(house.get("n_negative_raw", 0))   # should be 0; >0 = suspicious input
    return row


def feature_table(houses) -> pd.DataFrame:
    rows = {mp: house_features(h) for mp, h in houses.items() if "import" in h}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("mp_id")


# =============================================================================
# 4. command line
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing the monthly AEW CSVs (searched recursively)")
    ap.add_argument("-o", "--out", default="pv5_features.csv", help="output CSV (default pv5_features.csv)")
    ap.add_argument("--month", help='only files whose path contains this text, e.g. "Juni 2023"')
    ap.add_argument("--max-files", type=int, help="read only the first N monthly files")
    ap.add_argument("--max-houses", type=int, help="keep only the first N MP IDs (quick test)")
    a = ap.parse_args(argv)

    houses = load_houses(a.root, month=a.month, max_files=a.max_files, max_houses=a.max_houses)
    tab = feature_table(houses)
    tab.to_csv(a.out, float_format="%.4f")
    print(f"\n{len(tab)} houses -> {a.out}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        if len(tab) <= 30:
            print(tab)
        else:
            print(tab[FEATURE_NAMES].describe().T[["count", "mean", "min", "25%", "50%", "75%", "max"]])
    n_nan = tab[FEATURE_NAMES].isna().all()
    if n_nan.any():
        print("\nall-NaN features (expected when the selected files contain no summer / no winter days,"
              " or no export register):", ", ".join(n_nan[n_nan].index))


if __name__ == "__main__":
    main()
