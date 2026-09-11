"""
ev20_features.py — TWENTY physically-motivated EV-CHARGER main features, computed only from the
AEW smart-meter CSVs (import counter only, no weather, no external data).  Same structure, options,
output files and "every feature is computed for every house" rules as pv20_features.py, whose
data-reading layer it imports (keep both files in the same folder).

Run (inside the Renku session, from ~/work/mechenergy):
    NO ARGUMENTS:  python ev20_features.py
        -> uses the DEFAULTS block below: store aew_store, MP IDs from DEFAULT_IDS_FILE (first
           column, header skipped), output DEFAULT_OUT (+ .xlsx).  Edit those four lines once.
    FAST PATH — after a one-time  python build_store.py ../aew-data/test-blob/input_data -o aew_store :
        python ev20_features.py --store aew_store --mp-ids 53628,45390 -o ev_two_houses.csv       (milliseconds)
        python ev20_features.py --store aew_store --max-houses 10000 --workers 2 -o ev_houses_10000.csv
        python ev20_features.py --store aew_store -o ev_all_houses.csv                            (all 89k houses: minutes the
                                                             first time, then ~1 min from the cached per-day table)
        (if ./aew_store exists it is used automatically, even when a CSV root is given)
    CSV PATH — no store yet:
    ten houses, every month that exists (one or two whole years):
        python ev20_features.py ../aew-data/test-blob/input_data --max-houses 10 -o ev_ten_houses.csv
    named houses:
        python ev20_features.py ../aew-data/test-blob/input_data --mp-ids 53628,45390,49688 -o ev.csv
    all labelled houses at once (one MP ID per line in the file; one slow pass, then cached):
        python ev20_features.py ../aew-data/test-blob/input_data --mp-ids-file labelled_ids.txt -o ev_labelled.csv
    only 2023:
        python ev20_features.py ../aew-data/test-blob/input_data --max-houses 10 --month 2023
    every house of one month (slow but bounded memory):
        python ev20_features.py ../aew-data/test-blob/input_data --month "Juni 2023" -o ev_juni2023.csv

    The all-houses per-day table is cached as aew_store/_ev_days.parquet (the PV script uses
    _days.parquet, so the two never overwrite each other); --refresh-days recomputes it.

NOON-TO-NOON WINDOWS AND HOW THE DATA ARE READ
    A window needs the next calendar day of the same house.  Store: a bucket holds every date of
    its houses, so each bucket is reduced in one go.  CSVs: the monthly files are read in
    chronological order and a day whose next day has not been read yet (end of a chunk or of a
    month) is held back until it has, so no window is cut at a file boundary.  With --month the
    last selected day has no next morning (its window is half empty) — same as a house's last day.

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
import inspect
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# 1. reading the CSVs / the store: shared with the PV script (same folder)
from pv20_features import (N_SLOTS, SUMMER_MONTHS, WINTER_MONTHS, ZERO_KW, bucket_of, find_files, first_mp_ids,
                           first_store_ids, iter_aew_csv, read_store, store_buckets, _single_thread_blas)
try:
    from pv20_features import FILE_GLOB
except ImportError:                          # older copies of pv20_features.py do not define it
    FILE_GLOB = "LG_AIM2Hackerdays_kWh_*.csv"


def _call(fn, *args, **kw):
    """Call a pv20_features function with only the keyword options that version accepts
    (older copies of find_files / iter_aew_csv have no pattern / cache_dir)."""
    params = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in kw.items() if k in params})

# =============================================================================
# DEFAULTS — what  python ev20_features.py  does when run with no arguments.
# Edit these four lines to your files; any command-line option still overrides them.
# =============================================================================
DEFAULT_STORE = "aew_store"                 # the binary store folder (built by build_store.py)
DEFAULT_IDS_FILE = "labelled_houses.csv"    # CSV/TXT with the MP IDs in the FIRST column, header on line 1
DEFAULT_OUT = "ev_features.csv"             # output table (an .xlsx copy is written next to it)
DEFAULT_WORKERS = 2                         # parallel processes (use the number of cores)

DAYS_CACHE = "_ev_days.parquet"   # per-day table of all houses inside the store (PV uses _days.parquet)

STEP_KW = 2.0            # "high" = at least this much above the day's base load
MIN_RUN_Q = 4            # a block must last >= 4 quarter-hours (1 h)
FLAT_TOL = 0.10          # a block sample is "at level" if within +-10 % of the block's median level
FLAT_SHARE = 0.80        # a block is flat if >= 80 % of its core samples are at level (core = block minus its last 45 min)
LEVEL_TOL = 0.15         # +-15 % around the median level = "same level"
DAY_SHIFT = 48           # the day window starts at noon (slot 48)

SLOT_HOUR = (np.arange(N_SLOTS) + 0.5) / 4.0
WIN_HOUR = (SLOT_HOUR + 12.0) % 24.0            # clock hour of each slot in the noon-to-noon window

KEY = ["mp_id", "date"]                          # one row of the per-day table (import only)

FEATURE_NAMES = [
    "ev_plateau_days_frac", "ev_plateau_level_kw", "ev_plateau_level_p90_kw", "ev_level_consistency",
    "ev_plateau_duration_h", "ev_plateau_duration_p90_h", "ev_plateau_energy_kwh", "ev_plateau_energy_p90_kwh",
    "ev_flat_plateau_days_frac", "ev_step_up_kw", "ev_start_evening_night_frac", "ev_start_hour_spread_h",
    "ev_day_plateau_days_frac", "ev_weekend_minus_weekday_frac", "ev_gap_days_median",
    "ev_plateau_days_winter_minus_summer", "ev_max_over_base_kw", "ev_high_hours_per_day",
    "ev_high_energy_share", "ev_sessions_per_plateau_day",
]


# =============================================================================
# 2. reduce every (house, day) of import to per-window numbers  (noon-to-noon windows)
# =============================================================================
def _import_rows(lab: pd.DataFrame, vals: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Keep only the import counter: the EV features never look at export."""
    imp = (lab["channel"] == "import").values
    return lab.loc[imp, KEY].reset_index(drop=True), vals[imp]


def _noon_windows(lab: pd.DataFrame, vals: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows = calendar days of import.  Build windows [noon of day d, noon of day d+1) by
    joining the second half of day d with the first half of the NEXT calendar day of the same
    house (NaN if that day is missing).  Returns (labels, values, windows), sorted by house and date."""
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
    return lab, vals, win


def _reduce_windows(lab: pd.DataFrame, win: np.ndarray) -> pd.DataFrame:
    """Per-window numbers all twenty features are built from (one row per house-day of `lab`)."""
    n = len(lab)
    d = lab[KEY].copy()
    if n == 0:
        for c in ["base", "tot", "n_valid", "vmax_over_base", "hours_high", "e_high", "n_runs", "blk_len",
                  "blk_level", "blk_energy", "blk_flat_share", "blk_start_h", "step_up"]:
            d[c] = np.array([], dtype=float)
        return d
    finite = np.isfinite(win)
    v = np.where(finite, win, np.float32(0.0))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                    # all-NaN days in nanmedian / nanpercentile
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
        # run lengths per day: a bincount over (day, run id) — same numbers as np.add.at, faster on older numpy
        flat_id = (run_id + (np.arange(n) * (max_runs + 1))[:, None]).ravel()
        lengths = np.bincount(flat_id, minlength=n * (max_runs + 1)).reshape(n, max_runs + 1)
        lengths[:, 0] = 0
        d["n_runs"] = (lengths >= MIN_RUN_Q).sum(axis=1)
        longest = lengths.argmax(axis=1)                                    # run id of the longest run (0 if none)
        blk = (run_id == longest[:, None]) & (longest[:, None] > 0)
        blk_len = blk.sum(axis=1)
        d["blk_len"] = blk_len
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


def reduce_days(lab: pd.DataFrame, vals: np.ndarray) -> pd.DataFrame:
    """(labels, values) with ALL the days of the houses involved -> per-window table."""
    lab, vals = _import_rows(lab, vals)
    lab, _, win = _noon_windows(lab, vals)
    return _reduce_windows(lab, win)


class _NoonStream:
    """CSV data arrive in pieces (monthly files, chunks of a file) but a window needs the NEXT day too.
    push() reduces every day whose next day is already known and holds the others back;
    flush() reduces held-back days that can no longer be completed (their morning stays NaN)."""

    def __init__(self) -> None:
        self.lab: Optional[pd.DataFrame] = None
        self.vals: Optional[np.ndarray] = None
        self.last_date: Optional[pd.Timestamp] = None

    def push(self, lab: pd.DataFrame, vals: np.ndarray) -> Optional[pd.DataFrame]:
        lab, vals = _import_rows(lab, vals)
        if len(lab) == 0:
            return None
        newest = pd.to_datetime(lab["date"]).max()
        self.last_date = newest if self.last_date is None else max(self.last_date, newest)
        if self.lab is not None and len(self.lab):
            lab = pd.concat([self.lab, lab], ignore_index=True)
            vals = np.vstack([self.vals, vals])
        lab, vals, win = _noon_windows(lab, vals)
        # does (same house, date + 1) exist among the rows read so far?
        day = pd.to_datetime(lab["date"]).values.astype("datetime64[D]").astype(np.int64)
        key = pd.factorize(lab["mp_id"])[0].astype(np.int64) * 1_000_000 + day
        has_next = np.isin(key + 1, key)
        self.lab, self.vals = lab[~has_next].reset_index(drop=True), vals[~has_next]
        return _reduce_windows(lab[has_next].reset_index(drop=True), win[has_next]) if has_next.any() else None

    def flush(self, before: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Reduce the held-back days dated before `before` (all of them if None)."""
        if self.lab is None or not len(self.lab):
            return None
        done = np.ones(len(self.lab), bool) if before is None else (pd.to_datetime(self.lab["date"]) < before).values
        if not done.any():
            return None
        lab, vals = self.lab[done].reset_index(drop=True), self.vals[done]
        self.lab, self.vals = self.lab[~done].reset_index(drop=True), self.vals[~done]
        lab, _, win = _noon_windows(lab, vals)
        return _reduce_windows(lab, win)


def _chronological(files: List[str]) -> List[str]:
    """Monthly files sorted by the date of their first data row (names like 'Juni 2023' do not sort by time)."""
    def first_date(path: str) -> pd.Timestamp:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.readline()
                field = next(x.strip() for x in f.readline().split(";") if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", x.strip()))
            return pd.to_datetime(field, format="%d.%m.%Y")
        except Exception:                    # no date found -> read that file last
            return pd.Timestamp.max
    return sorted(files, key=first_date)


def load_days(root: Optional[str] = None, verbose: bool = True, month: Optional[str] = None,
              max_files: Optional[int] = None, max_houses: Optional[int] = None,
              mp_ids: Optional[Iterable[str]] = None, pattern: str = FILE_GLOB, workers: int = 4,
              cache_dir: Optional[str] = None, store: Optional[str] = None,
              use_days_cache: bool = True) -> pd.DataFrame:
    """Return the per-window table (one row per MP ID and day).
    From the binary store when `store` is given (fast), otherwise from the CSVs below `root`."""
    if store:
        return load_days_from_store(store, verbose, month, max_houses, mp_ids, workers, use_days_cache)
    if not root:
        raise SystemExit("give the CSV root folder, or --store aew_store (build it with build_store.py)")
    files = _call(find_files, root, month, max_files, pattern=pattern)
    keep: Optional[Set[str]] = set(str(m) for m in mp_ids) if mp_ids is not None else None
    if keep is None and max_houses:
        keep = set(first_mp_ids(files[0], max_houses))          # same houses as pv20_features.py picks
        if verbose:
            print(f"first {len(keep)} MP IDs of {os.path.basename(files[0])}: {sorted(keep)}")
    files = _chronological(files)

    def read_one(f: str) -> list:                                  # a house list makes a file small: read it whole
        return list(_call(iter_aew_csv, f, mp_ids=keep, cache_dir=cache_dir))

    stream, parts = _NoonStream(), []
    n_workers = max(1, workers) if keep is not None else 1        # the chunked full read stays sequential
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for b in range(0, len(files), n_workers):                  # a few files at a time, fed to the stream IN ORDER
            batch = files[b:b + n_workers]
            loaded = pool.map(read_one, batch) if keep is not None else (iter_aew_csv(f) for f in batch)
            for i, (f, pieces) in enumerate(zip(batch, loaded), start=b):
                n_rows, ids = 0, set()
                for lab, vals in pieces:
                    parts.append(stream.push(lab, vals))
                    n_rows += len(lab); ids.update(lab["mp_id"].unique())
                parts.append(stream.flush(before=stream.last_date))   # earlier days cannot get a morning any more
                if verbose:
                    print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, root)} ... {n_rows:,} rows, {len(ids):,} houses", flush=True)
    parts.append(stream.flush())
    parts = [p for p in parts if p is not None and len(p)]
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY)
    return _fix_multimeter(days, verbose, lambda ids: [_exact_days_for(files, ids, cache_dir)])


def load_days_from_store(store: str, verbose: bool = True, month: Optional[str] = None,
                         max_houses: Optional[int] = None, mp_ids: Optional[Iterable[str]] = None,
                         workers: int = 4, use_days_cache: bool = True) -> pd.DataFrame:
    """Per-window table from the binary store.  A house list touches only its buckets / row groups;
    without a list every bucket is processed (`workers` in parallel)."""
    keep: Optional[List[str]] = sorted(set(str(m) for m in mp_ids)) if mp_ids is not None else None
    if keep is None and max_houses:
        keep = first_store_ids(store, max_houses)
        if verbose:
            shown = keep if len(keep) <= 20 else keep[:10] + ["..."] + keep[-5:]
            print(f"first {len(keep)} MP IDs of the store: {shown}")
    t0 = time.time()
    days_cache = os.path.join(store, DAYS_CACHE)               # per-window table of ALL houses, no month filter
    if keep is None and not month and use_days_cache and os.path.exists(days_cache):
        days = pd.read_parquet(days_cache)
        if verbose:
            print(f"store: per-day table for all houses loaded from {days_cache} ({len(days):,} rows, {time.time() - t0:.1f} s)"
                  f"  [--refresh-days to recompute]", flush=True)
        return days
    if keep is not None:
        by_bucket: Dict[int, List[str]] = {}
        for m in keep:
            by_bucket.setdefault(bucket_of(m), []).append(m)
        jobs = [(store, os.path.join(store, f"bucket={b}"), month, ids) for b, ids in sorted(by_bucket.items())]
        if len(jobs) <= 4 or workers <= 1:
            parts = [p for job in jobs for p in _reduce_bucket(job)]
        else:                                      # many buckets -> one process per bucket
            parts = []
            with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1)),
                                     initializer=_single_thread_blas) as pool:
                for i, bparts in enumerate(pool.map(_reduce_bucket, jobs)):
                    parts.extend(bparts)
                    if verbose and (i + 1) % 16 == 0:
                        print(f"  {i + 1}/{len(jobs)} buckets ({time.time() - t0:.0f} s)", flush=True)
        if verbose:
            n = sum(len(p) for p in parts)
            print(f"store: {n:,} house-days for {len(keep):,} houses in {time.time() - t0:.1f} s", flush=True)
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
    parts = [p for p in parts if len(p)]
    days = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY)
    days = _fix_multimeter(days, verbose, lambda ids: [_exact_days_from_store(store, ids, month)])
    if keep is None and not month and use_days_cache:
        days.to_parquet(days_cache, index=False)
        if verbose:
            print(f"store: per-day table saved to {days_cache} (next all-houses run skips the reduction)", flush=True)
    return days


def _reduce_bucket(args) -> List[pd.DataFrame]:
    """Worker for the store run (module-level so it can be sent to another process):
    (store, bucket_dir, month) = the whole bucket; (store, bucket_dir, month, ids) = only those houses.
    A bucket holds every date of its houses, so the noon-to-noon windows are exact."""
    store, d, month = args[:3]
    ids = args[3] if len(args) > 3 else None
    if ids is None:
        return [reduce_days(lab, vals) for lab, vals in read_store(store, buckets=[d], month=month)]
    return [reduce_days(lab, vals) for lab, vals in read_store(store, mp_ids=ids, month=month)]


def _fix_multimeter(days: pd.DataFrame, verbose: bool, exact_fn) -> pd.DataFrame:
    """Several meters under one MP ID -> the same (mp_id, date) appears more than once.
    Those houses are re-read with their meters summed per quarter-hour (exact_fn(ids) -> list of tables)."""
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


def _exact_days_for(files: List[str], mp_ids: Set[str], cache_dir: Optional[str] = None) -> pd.DataFrame:
    """CSV version: ALL files at once (the windows cross month ends; these are only a few houses)."""
    labs, vs = [], []
    for f in files:
        for lab, vals in _call(iter_aew_csv, f, mp_ids=mp_ids, cache_dir=cache_dir):
            labs.append(lab); vs.append(vals)
    return _sum_meters(labs, vs)


def _sum_meters(labs, vs) -> pd.DataFrame:
    if not labs:
        return pd.DataFrame(columns=KEY)
    lab = pd.concat(labs, ignore_index=True)
    mat = pd.DataFrame(np.vstack(vs), columns=range(N_SLOTS))
    mat[["mp_id", "channel", "date"]] = lab[["mp_id", "channel", "date"]]
    summed = mat.groupby(["mp_id", "channel", "date"], sort=False)[list(range(N_SLOTS))].sum(min_count=1).reset_index()
    return reduce_days(summed[["mp_id", "channel", "date"]], summed[list(range(N_SLOTS))].to_numpy(dtype="float32"))


# =============================================================================
# 3. the twenty features, vectorised over all houses from the per-window table
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
    # no plateau day in the whole selection -> an empty apply returns a DataFrame and the assignment crashed
    gaps = gp[["date"]].apply(_gap) if len(P) else pd.Series(dtype=float)
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


def write_xlsx(tab: pd.DataFrame, path: str) -> bool:
    """Excel copy of the feature table: houses as rows, features as columns, frozen header, filters,
    colour scales per feature, plus a 'legend' sheet.  Returns False if openpyxl is missing."""
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
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions
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
        lg = xw.sheets["legend"]
        lg.column_dimensions["A"].width = 40; lg.column_dimensions["B"].width = 100
    return True


# Value used when a feature cannot be computed: the value a house WITHOUT an EV shows (0 everywhere),
# so a fill never looks like a detection.  Only a safety net (a warning is printed if it was needed).
FILL_VALUES = {name: 0.0 for name in FEATURE_NAMES}


def fill_missing(tab: pd.DataFrame) -> pd.DataFrame:
    """Replace every NaN by its FILL_VALUES entry -> a table of numbers only."""
    tab = tab.copy()
    for c, v in FILL_VALUES.items():
        if c in tab.columns:
            tab[c] = tab[c].fillna(v)
    return tab


# =============================================================================
# 4. single-house helper (house = pv20_features.load_house(root, mp_id))
# =============================================================================
def house_features(house: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    m = house["import"]
    lab = pd.DataFrame({"mp_id": "x", "channel": "import", "date": pd.to_datetime(m.index)})
    return feature_table(reduce_days(lab, m.to_numpy(dtype="float32"))).iloc[0].to_dict()


# =============================================================================
# 5. command line
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder containing the monthly AEW CSVs (not needed with a store)")
    ap.add_argument("--store", help=f"binary store built by build_store.py (default: ./{DEFAULT_STORE} if it exists)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"output CSV (default {DEFAULT_OUT})")
    ap.add_argument("--month", help='only files whose path contains this text, e.g. "Juni 2023" or "2023"')
    ap.add_argument("--max-files", type=int, help="read only the first N monthly files")
    ap.add_argument("--max-houses", type=int, help="the first N MP IDs of the first file, followed through ALL files")
    ap.add_argument("--mp-ids", help="comma-separated MP IDs to keep, e.g. 53628,53701")
    ap.add_argument("--mp-ids-file", help=f"CSV/TXT with the MP IDs in the first column (default {DEFAULT_IDS_FILE} "
                                           "unless --mp-ids / --max-houses is given)")
    ap.add_argument("--pattern", default=FILE_GLOB, help=f"file name pattern of the meter files (default {FILE_GLOB})")
    ap.add_argument("--keep-nan", action="store_true", help="leave non-computable features empty instead of 0")
    ap.add_argument("--xlsx", help="Excel copy path (default: same name as -o with .xlsx)")
    ap.add_argument("--no-xlsx", action="store_true", help="do not write the Excel copy")
    ap.add_argument("--refresh-days", action="store_true", help=f"store: recompute the all-houses per-day table instead of using {DAYS_CACHE}")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"parallel processes / files (default {DEFAULT_WORKERS})")
    ap.add_argument("--cache-dir", default="aew_cache", help="folder for the extracted lines (default ./aew_cache; '' = off)")
    a = ap.parse_args(argv)

    store = a.store or (DEFAULT_STORE if os.path.isdir(DEFAULT_STORE) else None)
    if store and a.root and a.store is None:
        print(f"note: using the store ./{DEFAULT_STORE} instead of the CSVs (pass --store '' to force the CSVs)", flush=True)
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
    days = load_days(a.root, month=a.month, max_files=a.max_files, max_houses=a.max_houses, mp_ids=ids,
                     pattern=a.pattern, workers=a.workers, cache_dir=a.cache_dir or None, store=store,
                     use_days_cache=not a.refresh_days)
    tab = feature_table(days)
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
        if len(tab) <= 30:
            print(tab.T)                                  # houses as columns: easier to read 20 features
        else:
            print(tab[FEATURE_NAMES].describe().T[["count", "mean", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":
    main()