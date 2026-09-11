"""
ev20_features.py — TWENTY physically-motivated EV-CHARGER main features from the AEW
smart-meter data (import counter only, no external data).  Same structure, options,
output files and "every feature is computed for every house" rules as pv20_features.py,
whose data-reading layer it reuses (keep both files in the same folder).

Run (inside the Renku session, from ~/work/mechenergy):
    NO ARGUMENTS:  python ev20_features.py
        -> DEFAULTS block below: store aew_store, MP IDs from DEFAULT_IDS_FILE (first column,
           header skipped), output DEFAULT_OUT (+ .xlsx).
    python ev20_features.py --store aew_store --mp-ids 53628,45390 -o ev_two.csv
    python ev20_features.py --store aew_store --max-houses 10000 --workers 2 -o ev_10000.csv
    python ev20_features.py --store aew_store -o ev_all.csv            (all houses)

PHYSICS OF EV HOME CHARGING (what the features look for)
    A wallbox draws a nearly FLAT block of power at a FIXED level — 11 kW (3-phase 16 A, the
    Swiss standard), 7.4 kW, 3.7 kW (1-phase 16 A) or 2.3 kW (plain socket) — switching on
    abruptly, running 1-8 h, switching off abruptly (sometimes a short taper at the end).
    Sessions come a few days per week, mostly evening / night, always at the same level (same
    car, same charger), and move 10-40 kWh each — far more than any appliance.
    Confounders and how they differ:
      electric boiler  flat block too, but EVERY night at the SAME clock time, all year, 2-4 kW, 5-10 kWh
      heat pump        modulates / cycles rather than sitting flat; several kW; winter only
      dryer / oven     <= 2 kW, < 2 h, daytime
    A "day" here runs from NOON to NOON so that a night session is not cut at midnight.

PER-DAY REDUCTION (import, noon-to-noon window of 96 quarter-hours)
    base      = 10th percentile of the window (the house's base load that day)
    high      = quarter-hours with import >= base + STEP_KW (2 kW)
    block     = longest run of high quarter-hours; a "plateau day" has a block >= MIN_RUN_H (1 h)
    for the block: median level above base (kW), duration (h), energy above base (kWh), share of
    the block within +-10 % of its level, start clock hour, number of runs >= 1 h, largest quarter-hour up-step,
    daily max above base, hours high, energy above base while high, total energy.

==============================================================================
THE TWENTY FEATURES  (every one a computed number for every house; a house with
no plateau day is 0 in all of them except ev_gap_days_median = its number of days)
==============================================================================
 1. ev_plateau_days_frac            share of days with a block >= 1 h at >= 2 kW above base.
                                    EV 0.1-0.6 (a few days a week); boiler ~1; none 0.
 2. ev_plateau_level_kw             median block level above base [kW]. EV 3.7 / 7.4 / 11; dryer ~2; boiler 2-4.
 3. ev_plateau_level_p90_kw         90th percentile of the level: the big days (11 kW wallbox vs 2-3 kW appliances).
 4. ev_level_consistency            share of plateau days whose level is within +-15 % of the median level.
                                    EV high (same charger every time); mixed appliances low.
 5. ev_plateau_duration_h           median block duration [h]. EV 1-6; dryer 1-2; boiler 2-4.
 6. ev_plateau_duration_p90_h       the long sessions (EV up to 8 h+).
 7. ev_plateau_energy_kwh           median energy above base in the block [kWh]. EV 10-40; dryer 2-3; boiler 5-10.
 8. ev_plateau_energy_p90_kwh       the biggest sessions.
 9. ev_flat_plateau_days_frac       share of days with a FLAT block (>= 80 % of its core — the block minus
                                    its last 45 min, so an end-of-charge taper is tolerated — within +-10 %
                                    of its median level).
                                    EV and boiler flat; heat pump / thermostat cycling not.
10. ev_step_up_kw                   median largest quarter-hour jump on plateau days [kW]. EV: the whole level in one step.
11. ev_start_evening_night_frac     share of plateau days whose block starts 17-07 h. EV high.
12. ev_start_hour_spread_h          circular std of the start hour [h]. Ripple-controlled boiler ~0; EV 1-3 h.
13. ev_day_plateau_days_frac        share of days with a block starting 07-17 h (daytime / PV-surplus charging, dryers).
14. ev_weekend_minus_weekday_frac   plateau-day frequency on weekends minus weekdays. Boiler ~0; EV usually != 0.
15. ev_gap_days_median              median days between plateau days. Boiler 1; EV 2-4; none = n_days.
16. ev_plateau_days_winter_minus_summer  plateau-day frequency Nov-Feb minus May-Aug. Heat pump >> 0; EV ~0; PV-surplus < 0.
17. ev_max_over_base_kw             99th percentile of the daily peak above base [kW]. Plain house 2-3; EV >= 4.
18. ev_high_hours_per_day           mean hours per day at >= base + 2 kW, all days.
19. ev_high_energy_share            energy consumed at >= base + 2 kW / total import. EV 0.2-0.5; others ~0.
20. ev_sessions_per_plateau_day     mean number of distinct >= 1 h blocks on plateau days (two cars, EV + boiler -> > 1).

Data-quality columns: n_days, n_summer_days, n_winter_days, mean_import_kw, base_kw (median daily base).
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# reading layer shared with the PV script (same folder)
from features_pv20 import (N_SLOTS, SUMMER_MONTHS, WINTER_MONTHS, ZERO_KW, bucket_of, find_files, first_mp_ids,
                           first_store_ids, iter_aew_csv, read_store, store_buckets, _single_thread_blas)

# =============================================================================
# DEFAULTS — what  python ev20_features.py  does when run with no arguments.
# =============================================================================
DEFAULT_STORE = "aew_store"
DEFAULT_IDS_FILE = "mp_ids_para_features.csv"    # CSV/TXT with the MP IDs in the FIRST column, header on line 1
DEFAULT_OUT = "labelled_features_ev.csv"
DEFAULT_WORKERS = 2

STEP_KW = 2.0            # "high" = at least this much above the day's base load
MIN_RUN_Q = 4            # a block must last >= 4 quarter-hours (1 h)
FLAT_TOL = 0.10          # a block sample is "at level" if within +-10 % of the block's median level
FLAT_SHARE = 0.80        # a block is flat if >= 80 % of its core samples are at level (core = block minus its last 45 min)
LEVEL_TOL = 0.15         # +-15 % around the median level = "same level"
DAY_SHIFT = 48           # the day window starts at noon (slot 48)

SLOT_HOUR = (np.arange(N_SLOTS) + 0.5) / 4.0
WIN_HOUR = (SLOT_HOUR + 12.0) % 24.0            # clock hour of each slot in the noon-to-noon window

FEATURE_NAMES = [
    "ev_plateau_days_frac", "ev_plateau_level_kw", "ev_plateau_level_p90_kw", "ev_level_consistency",
    "ev_plateau_duration_h", "ev_plateau_duration_p90_h", "ev_plateau_energy_kwh", "ev_plateau_energy_p90_kwh",
    "ev_flat_plateau_days_frac", "ev_step_up_kw", "ev_start_evening_night_frac", "ev_start_hour_spread_h",
    "ev_day_plateau_days_frac", "ev_weekend_minus_weekday_frac", "ev_gap_days_median",
    "ev_plateau_days_winter_minus_summer", "ev_max_over_base_kw", "ev_high_hours_per_day",
    "ev_high_energy_share", "ev_sessions_per_plateau_day",
]

FEATURE_DESC = {
    "ev_plateau_days_frac": "share of days with a >= 1 h block at >= 2 kW above base (EV 0.1-0.6, boiler ~1, none 0)",
    "ev_plateau_level_kw": "median block level above base [kW] (EV 3.7 / 7.4 / 11; dryer ~2; boiler 2-4)",
    "ev_plateau_level_p90_kw": "90th percentile of the block level [kW]",
    "ev_level_consistency": "share of plateau days with level within +-15 % of the median level (EV high)",
    "ev_plateau_duration_h": "median block duration [h] (EV 1-6)",
    "ev_plateau_duration_p90_h": "90th percentile of the block duration [h]",
    "ev_plateau_energy_kwh": "median energy above base in the block [kWh] (EV 10-40; dryer 2-3; boiler 5-10)",
    "ev_plateau_energy_p90_kwh": "90th percentile of the block energy [kWh]",
    "ev_flat_plateau_days_frac": "share of days with a flat block (>= 80 % of it within +-10 % of its median level): EV and boiler yes, heat pump no",
    "ev_step_up_kw": "median largest quarter-hour up-step on plateau days [kW]",
    "ev_start_evening_night_frac": "share of plateau days whose block starts 17-07 h",
    "ev_start_hour_spread_h": "circular std of the block start hour [h] (boiler ~0; EV 1-3)",
    "ev_day_plateau_days_frac": "share of days with a block starting 07-17 h",
    "ev_weekend_minus_weekday_frac": "plateau-day frequency weekend minus weekday (boiler ~0)",
    "ev_gap_days_median": "median days between plateau days (boiler 1; EV 2-4; none = n_days)",
    "ev_plateau_days_winter_minus_summer": "plateau-day frequency Nov-Feb minus May-Aug (heat pump >> 0; EV ~0)",
    "ev_max_over_base_kw": "99th percentile of the daily peak above base [kW] (EV >= 4)",
    "ev_high_hours_per_day": "mean hours per day at >= base + 2 kW",
    "ev_high_energy_share": "energy at >= base + 2 kW / total import (EV 0.2-0.5)",
    "ev_sessions_per_plateau_day": "mean number of distinct >= 1 h blocks on plateau days",
    "n_days": "days of data", "n_summer_days": "May-Aug days", "n_winter_days": "Nov-Feb days",
    "mean_import_kw": "average import power [kW]: household 0.2-0.8, tens = business",
    "base_kw": "median daily base load [kW] (10th percentile of the day)",
}


# =============================================================================
# 1. per-day reduction on noon-to-noon windows (vectorised over all days of a chunk)
# =============================================================================
def _noon_windows(lab: pd.DataFrame, vals: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Rows = calendar days of import.  Build windows [noon of day d, noon of day d+1) by
    joining the second half of day d with the first half of the NEXT calendar day of the same
    house (NaN if that day is missing)."""
    order = np.lexsort((lab["date"].values, lab["mp_id"].values))
    lab = lab.iloc[order].reset_index(drop=True)
    vals = vals[order]
    n = len(lab)
    nxt = np.full((n, DAY_SHIFT), np.nan, dtype=np.float32)
    if n > 1:
        same = (lab["mp_id"].values[1:] == lab["mp_id"].values[:-1])
        consec = (pd.to_datetime(lab["date"]).values[1:] - pd.to_datetime(lab["date"]).values[:-1]) == np.timedelta64(1, "D")
        ok = same & consec
        nxt[:-1][ok] = vals[1:][ok, :DAY_SHIFT]
    win = np.concatenate([vals[:, DAY_SHIFT:], nxt], axis=1)          # slot 0 = 12:00, slot 48 = 00:00
    return lab, win


def reduce_days_ev(lab: pd.DataFrame, vals: np.ndarray) -> pd.DataFrame:
    """Per-window numbers all twenty features are built from."""
    imp = (lab["channel"] == "import").values
    lab, win = _noon_windows(lab[imp].reset_index(drop=True), vals[imp])
    n = len(lab)
    d = lab[["mp_id", "date"]].copy()
    if n == 0:
        for c in ["base", "tot", "n_valid", "vmax_over_base", "hours_high", "e_high", "n_runs", "blk_len",
                  "blk_level", "blk_energy", "blk_flat_share", "blk_start_h", "step_up"]:
            d[c] = np.array([], dtype=float)
        return d
    finite = np.isfinite(win)
    v = np.where(finite, win, np.float32(0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        d["n_valid"] = finite.sum(axis=1)
        d["tot"] = v.sum(axis=1)                                            # kW quarter-hours
        base = np.nanpercentile(np.where(finite, win, np.nan), 10, axis=1)
        base = np.where(np.isfinite(base), base, 0.0)
        d["base"] = base
        above = np.where(finite, win - base[:, None], np.float32(0.0))
        d["vmax_over_base"] = above.max(axis=1)
        high = (above >= STEP_KW) & finite
        d["hours_high"] = high.sum(axis=1) / 4.0
        d["e_high"] = (above * high).sum(axis=1) / 4.0                    # kWh above base while high
        # runs of high quarter-hours
        starts = high & ~np.concatenate([np.zeros((n, 1), bool), high[:, :-1]], axis=1)
        run_id = np.cumsum(starts, axis=1) * high                          # 0 outside runs, 1..k inside
        max_runs = int(run_id.max()) if n else 0
        lengths = np.zeros((n, max_runs + 1), dtype=np.int32)
        rows = np.repeat(np.arange(n), N_SLOTS)
        np.add.at(lengths, (rows, run_id.ravel()), high.ravel().astype(np.int32))
        lengths[:, 0] = 0
        d["n_runs"] = (lengths >= MIN_RUN_Q).sum(axis=1)
        longest = lengths.argmax(axis=1)                                    # run id of the longest run (0 if none)
        blk = (run_id == longest[:, None]) & (longest[:, None] > 0)
        blk_len = blk.sum(axis=1)
        d["blk_len"] = blk_len
        safe_len = np.maximum(blk_len, 1)
        lvl = np.nanmedian(np.where(blk, above, np.nan), axis=1)          # median level: robust to an end taper
        lvl = np.where(blk_len > 0, lvl, 0.0)
        d["blk_level"] = lvl
        d["blk_energy"] = np.where(blk_len > 0, (above * blk).sum(axis=1) / 4.0, 0.0)
        # flatness on the block's CORE: drop its last 3 quarter-hours when it is >= 2 h (end-of-charge taper)
        last = np.where(blk, np.arange(N_SLOTS)[None, :], -1).max(axis=1)
        core_end = np.where(blk_len >= 8, last - 3, last)
        core = blk & (np.arange(N_SLOTS)[None, :] <= core_end[:, None])
        core_len = np.maximum(core.sum(axis=1), 1)
        within = (np.abs(above - lvl[:, None]) <= FLAT_TOL * np.maximum(lvl[:, None], 1e-6)) & core
        d["blk_flat_share"] = np.where(blk_len > 0, within.sum(axis=1) / core_len, 0.0)   # share of the core at its level
        first = np.where(blk, np.arange(N_SLOTS)[None, :], N_SLOTS).min(axis=1)
        d["blk_start_h"] = np.where(blk_len > 0, WIN_HOUR[np.minimum(first, N_SLOTS - 1)], np.nan)
        diff = np.diff(v, axis=1)
        d["step_up"] = diff.max(axis=1) if N_SLOTS > 1 else 0.0
    return d


# =============================================================================
# 2. loading (store or CSVs) with this reduction
# =============================================================================
def _reduce_bucket_ev(args) -> List[pd.DataFrame]:
    store, d, month, ids = args
    if ids is None:
        return [reduce_days_ev(lab, vals) for lab, vals in read_store(store, buckets=[d], month=month)]
    return [reduce_days_ev(lab, vals) for lab, vals in read_store(store, mp_ids=ids, month=month)]


def _sum_meters_ev(labs, vs) -> pd.DataFrame:
    if not labs:
        return pd.DataFrame(columns=["mp_id", "date"])
    lab = pd.concat(labs, ignore_index=True)
    mat = pd.DataFrame(np.vstack(vs), columns=range(N_SLOTS))
    mat[["mp_id", "channel", "date"]] = lab[["mp_id", "channel", "date"]]
    summed = mat.groupby(["mp_id", "channel", "date"], sort=False)[list(range(N_SLOTS))].sum(min_count=1).reset_index()
    return reduce_days_ev(summed[["mp_id", "channel", "date"]], summed[list(range(N_SLOTS))].to_numpy(dtype="float32"))


def load_days_ev(root: Optional[str], store: Optional[str], verbose: bool = True, month: Optional[str] = None,
                 max_houses: Optional[int] = None, mp_ids: Optional[Iterable[str]] = None,
                 workers: int = 2, max_files: Optional[int] = None) -> pd.DataFrame:
    t0 = time.time()
    keep: Optional[List[str]] = sorted(set(str(m) for m in mp_ids)) if mp_ids is not None else None
    parts: List[pd.DataFrame] = []
    if store:
        if keep is None and max_houses:
            keep = first_store_ids(store, max_houses)
        if keep is not None:
            by_bucket: Dict[int, List[str]] = {}
            for m in keep:
                by_bucket.setdefault(bucket_of(m), []).append(m)
            jobs = [(store, os.path.join(store, f"bucket={b}"), month, ids) for b, ids in sorted(by_bucket.items())]
        else:
            jobs = [(store, d, month, None) for d in store_buckets(store)]
        if len(jobs) <= 4 or workers <= 1:
            for job in jobs:
                parts.extend(_reduce_bucket_ev(job))
        else:
            with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1)),
                                     initializer=_single_thread_blas) as pool:
                for i, bparts in enumerate(pool.map(_reduce_bucket_ev, jobs)):
                    parts.extend(bparts)
                    if verbose and (i + 1) % 32 == 0:
                        print(f"  {i + 1}/{len(jobs)} buckets ({time.time() - t0:.0f} s)", flush=True)
        src = "store"
    else:
        if not root:
            raise SystemExit("give the CSV root folder, or --store aew_store")
        files = find_files(root, month, max_files)
        if keep is None and max_houses:
            keep = first_mp_ids(files[0], max_houses)
        labs, vs = [], []
        for i, f in enumerate(files):
            if verbose:
                print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, root)}", flush=True)
            for lab, vals in iter_aew_csv(f, mp_ids=keep):
                if keep is not None:                       # few houses: join the months, reduce once
                    labs.append(lab); vs.append(vals)      # (noon-to-noon windows cross month ends)
                else:
                    parts.append(reduce_days_ev(lab, vals))
        if labs:
            parts.append(reduce_days_ev(pd.concat(labs, ignore_index=True), np.vstack(vs)))
        elif keep is None and verbose:
            print("note: CSV path without a house list reduces month by month; the window of the last day of each"
                  " month misses its next morning (use the store for exact results)", flush=True)
        src = "csv"
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mp_id", "date"])
    # several meters under one MP ID -> the same (mp_id, date) twice -> re-read summed per day
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
                for lab, vals in iter_aew_csv(f, mp_ids=multi):
                    labs.append(lab); vs.append(vals)
        days = pd.concat([days, _sum_meters_ev(labs, vs)], ignore_index=True)
    if verbose:
        print(f"{src}: {len(days):,} house-days for {days['mp_id'].nunique():,} houses in {time.time() - t0:.1f} s", flush=True)
    return days


# =============================================================================
# 3. the twenty features
# =============================================================================
def _circ_std_h(hours: pd.Series) -> float:
    if len(hours) == 0:
        return 0.0
    ang = hours.values / 24.0 * 2 * np.pi
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return float(np.sqrt(-2 * np.log(max(R, 1e-12))) * 24.0 / (2 * np.pi))


def feature_table(days: pd.DataFrame) -> pd.DataFrame:
    days = days.copy()
    days["date"] = pd.to_datetime(days["date"])
    days["month"] = days["date"].dt.month
    days["weekend"] = days["date"].dt.dayofweek >= 5
    days["plateau"] = days["blk_len"] >= MIN_RUN_Q
    days["flat"] = days["plateau"] & (days["blk_flat_share"] >= FLAT_SHARE)
    sh = days["blk_start_h"]
    days["start_eve_night"] = days["plateau"] & ((sh >= 17) | (sh < 7))
    days["start_day"] = days["plateau"] & (sh >= 7) & (sh < 17)
    houses = pd.Index(days["mp_id"].unique(), name="mp_id")
    out = pd.DataFrame(index=houses)
    g = days.groupby("mp_id")
    P = days[days["plateau"]]
    gp = P.groupby("mp_id")

    def R(x, fill=0.0) -> pd.Series:
        return x.reindex(houses).fillna(fill)

    n_all = R(g.size())
    n_p = R(gp.size())
    out["ev_plateau_days_frac"] = n_p / n_all.clip(lower=1)
    out["ev_plateau_level_kw"] = R(gp["blk_level"].median())
    out["ev_plateau_level_p90_kw"] = R(gp["blk_level"].quantile(0.9))
    med_level = P["mp_id"].map(gp["blk_level"].median())
    same = ((P["blk_level"] - med_level).abs() <= LEVEL_TOL * med_level).groupby(P["mp_id"]).mean()
    out["ev_level_consistency"] = R(same)
    out["ev_plateau_duration_h"] = R(gp["blk_len"].median()) / 4.0
    out["ev_plateau_duration_p90_h"] = R(gp["blk_len"].quantile(0.9)) / 4.0
    out["ev_plateau_energy_kwh"] = R(gp["blk_energy"].median())
    out["ev_plateau_energy_p90_kwh"] = R(gp["blk_energy"].quantile(0.9))
    out["ev_flat_plateau_days_frac"] = R(g["flat"].sum()) / n_all.clip(lower=1)
    out["ev_step_up_kw"] = R(gp["step_up"].median())
    out["ev_start_evening_night_frac"] = R(gp["start_eve_night"].mean())
    out["ev_start_hour_spread_h"] = R(gp["blk_start_h"].agg(_circ_std_h)).abs()
    out["ev_day_plateau_days_frac"] = R(g["start_day"].sum()) / n_all.clip(lower=1)
    we = days[days["weekend"]].groupby("mp_id")["plateau"].mean()
    wd = days[~days["weekend"]].groupby("mp_id")["plateau"].mean()
    out["ev_weekend_minus_weekday_frac"] = R(we) - R(wd)

    def _gap(sub: pd.DataFrame) -> float:
        dts = np.sort(sub["date"].values)
        return float(np.median(np.diff(dts).astype("timedelta64[D]").astype(float))) if len(dts) > 1 else np.nan
    gaps = gp.apply(_gap)
    span = R(g["date"].max() - g["date"].min()).dt.days.astype(float) + 1.0 if len(days) else n_all
    out["ev_gap_days_median"] = gaps.reindex(houses).fillna(span)
    win = days[days["month"].isin(WINTER_MONTHS)].groupby("mp_id")["plateau"].mean()
    sum_ = days[days["month"].isin(SUMMER_MONTHS)].groupby("mp_id")["plateau"].mean()
    all_ = g["plateau"].mean()
    out["ev_plateau_days_winter_minus_summer"] = R(win, np.nan).fillna(all_) - R(sum_, np.nan).fillna(all_)
    out["ev_max_over_base_kw"] = R(g["vmax_over_base"].quantile(0.99))
    out["ev_high_hours_per_day"] = R(g["hours_high"].mean())
    out["ev_high_energy_share"] = R(g["e_high"].sum()) / (R(g["tot"].sum()) / 4.0).clip(lower=ZERO_KW)
    out["ev_sessions_per_plateau_day"] = R(gp["n_runs"].mean())

    out["n_days"] = n_all.astype(int)
    out["n_summer_days"] = R(days[days["month"].isin(SUMMER_MONTHS)].groupby("mp_id").size()).astype(int)
    out["n_winter_days"] = R(days[days["month"].isin(WINTER_MONTHS)].groupby("mp_id").size()).astype(int)
    out["mean_import_kw"] = R(g["tot"].sum()) / R(g["n_valid"].sum()).clip(lower=1)
    out["base_kw"] = R(g["base"].median())
    return out


# =============================================================================
# 4. Excel copy
# =============================================================================
def write_xlsx(tab: pd.DataFrame, path: str) -> bool:
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
                    ColorScaleRule(start_type="min", start_color="FFF5EB", end_type="max", end_color="D94801"))
        for cell in ws[1]:
            cell.alignment = openpyxl.styles.Alignment(text_rotation=60, vertical="bottom")
        ws.row_dimensions[1].height = 150
        lg = xw.sheets["legend"]; lg.column_dimensions["A"].width = 40; lg.column_dimensions["B"].width = 100
    return True


# =============================================================================
# 5. command line
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder with the monthly CSVs (not needed with a store)")
    ap.add_argument("--store", help=f"binary store from build_store.py (default ./{DEFAULT_STORE} if it exists)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    ap.add_argument("--month", help='e.g. "Juni 2023" or "2023"')
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--max-houses", type=int)
    ap.add_argument("--mp-ids", help="comma-separated MP IDs")
    ap.add_argument("--mp-ids-file", help=f"CSV/TXT with MP IDs in the first column (default {DEFAULT_IDS_FILE})")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--xlsx"); ap.add_argument("--no-xlsx", action="store_true")
    a = ap.parse_args(argv)

    store = a.store if a.store is not None else (DEFAULT_STORE if os.path.isdir(DEFAULT_STORE) else None)
    if a.store == "":
        store = None
    ids = [x.strip() for x in a.mp_ids.split(",")] if a.mp_ids else None
    ids_file = a.mp_ids_file or (DEFAULT_IDS_FILE if not (a.mp_ids or a.max_houses) else None)
    if ids_file:
        if not os.path.exists(ids_file):
            raise SystemExit(f"MP ID file not found: {ids_file}  (edit DEFAULT_IDS_FILE at the top of the script, "
                             f"or pass --mp-ids-file / --mp-ids / --max-houses)")
        with open(ids_file, encoding="utf-8", errors="replace") as f:
            ids = (ids or []) + [ln.strip().split(";")[0].split(",")[0].strip() for ln in f
                                 if ln.strip() and not ln.lower().startswith("mp")]
        print(f"{len(ids):,} MP IDs read from {ids_file}", flush=True)

    days = load_days_ev(a.root, store, month=a.month, max_houses=a.max_houses, mp_ids=ids,
                        workers=a.workers, max_files=a.max_files)
    tab = feature_table(days)
    n_missing = int(tab[FEATURE_NAMES].isna().sum().sum())
    if n_missing:
        print(f"warning: {n_missing} feature values could not be computed -> 0", flush=True)
        tab[FEATURE_NAMES] = tab[FEATURE_NAMES].fillna(0.0)
    tab.to_csv(a.out, float_format="%.4f")
    xlsx = a.xlsx if a.xlsx else os.path.splitext(a.out)[0] + ".xlsx"
    ok = False if a.no_xlsx else write_xlsx(tab, xlsx)
    print(f"\n{len(tab):,} houses -> {a.out}" + (f"  +  {xlsx}" if ok else "") + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 60,
                           "display.float_format", "{:.3f}".format):
        print(tab.T if len(tab) <= 30 else tab[FEATURE_NAMES].describe().T[["count", "mean", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":
    main()