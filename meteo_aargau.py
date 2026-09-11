"""
meteo_aargau.py — canton-average global irradiance (and air temperature) for Aargau from
MeteoSwiss Open Data, on the same 15-min local-time grid as the AEW meter files.

Source: MeteoSwiss (SwissMetNet open data, 10-minute values).  Cite as "Source: MeteoSwiss".
    https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/<abbr>/ogd-smn_<abbr>_t_historical_2020-2029.csv
    https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/<abbr>/ogd-smn_<abbr>_t_recent.csv
    ';'-separated, Windows-1252, timestamps 'dd.mm.yyyy HH:MM' in UTC, marking the END of the 10 min.
    'historical' = start of decade .. 31 Dec last year; 'recent' = 1 Jan this year .. yesterday.

The two SwissMetNet stations inside canton AG:  BUS Buchs/Aarau (387 m),  BEZ Beznau (326 m).

    gre000z0  global radiation (horizontal), W/m², mean of the 10 min ending at the timestamp
    tre200s0  air temperature 2 m, °C, value at the timestamp

CANTON AVERAGE = plain mean of the two stations at each time step (if one station has a gap,
the other one alone is used; gaps > 1 h at both stations stay empty).

Output  data/weather/aargau_weather_15min.csv
    timestamp    interval START, Europe/Zurich (same convention as the meter slots: slot k = k/4 h)
    irradiance   canton mean global irradiance over the quarter-hour, W/m²
    temperature  canton mean air temperature over the quarter-hour, °C  (not used yet)

Run once in the Renku session terminal, from ~/work/mechenergy (needs internet, ~1 min):
    python meteo_aargau.py --start 2023-01-01 --end 2026-09-10
pv20_features.py also calls this automatically if the weather file is missing.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
STATIONS = ["BUS", "BEZ"]                  # Buchs/Aarau, Beznau
TZ_LOCAL = "Europe/Zurich"
DEFAULT_OUT_DIR = os.path.join("data", "weather")
DEFAULT_FILE = os.path.join(DEFAULT_OUT_DIR, "aargau_weather_15min.csv")
COLS = {"gre000z0": "irr", "tre200s0": "temp"}


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------
def _file_names(abbr: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    a, this_year = abbr.lower(), pd.Timestamp.now(tz="UTC").year
    names = [f"ogd-smn_{a}_t_historical_{d}-{d + 9}.csv"
             for d in range((start.year // 10) * 10, end.year + 1, 10) if d < this_year]
    if end.year >= this_year:
        names.append(f"ogd-smn_{a}_t_recent.csv")
    return names


def download(abbr: str, start, end, raw_dir: str, refresh: bool = False) -> list[str]:
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for name in _file_names(abbr, pd.Timestamp(start), pd.Timestamp(end)):
        path = os.path.join(raw_dir, name)
        if refresh or not os.path.exists(path):
            url = f"{BASE}/{abbr.lower()}/{name}"
            print(f"  downloading {url}", file=sys.stderr, flush=True)
            try:
                urllib.request.urlretrieve(url, path + ".part")
                os.replace(path + ".part", path)
            except Exception as e:  # noqa: BLE001
                print(f"  ! could not download {name}: {e}", file=sys.stderr)
                continue
        paths.append(path)
    return paths


def read_station(paths: list[str]) -> pd.DataFrame:
    """10-min table of one station: UTC index (as published), columns irr, temp."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=";", encoding="cp1252",
                         usecols=lambda c: c in ("reference_timestamp", *COLS))
        df["reference_timestamp"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(COLS.values()))
    df = (pd.concat(frames).drop_duplicates("reference_timestamp", keep="last")
          .set_index("reference_timestamp").sort_index().rename(columns=COLS))
    for c in COLS.values():
        if c not in df:
            df[c] = np.nan
    return df[list(COLS.values())].astype(float)


# ----------------------------------------------------------------------------
# 10-min station values -> 15-min local grid
# ----------------------------------------------------------------------------
def _interp(s: pd.Series, grid: pd.DatetimeIndex, max_gap_s: float = 3600.0) -> np.ndarray:
    """Time-linear interpolation onto grid; points between samples > max_gap apart stay NaN."""
    s = s.dropna()
    if s.empty:
        return np.full(len(grid), np.nan)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts = np.asarray((s.index - epoch) / pd.Timedelta("1s"), dtype=float)     # unit-safe (pandas 2 and 3)
    tg = np.asarray((grid - epoch) / pd.Timedelta("1s"), dtype=float)
    out = np.interp(tg, ts, s.to_numpy(), left=np.nan, right=np.nan)
    j = np.searchsorted(ts, tg, side="right")
    gap = ts[np.clip(j, 0, len(ts) - 1)] - ts[np.clip(j - 1, 0, len(ts) - 1)]
    out[gap > max_gap_s] = np.nan
    return out


def to_quarter_hours(df10: pd.DataFrame, start, end) -> pd.DataFrame:
    """Per-minute interpolation, then the mean of each quarter-hour, labelled by its START in local time.
    Irradiance (a 10-min mean ending at the timestamp) is first placed at the middle of its 10 min."""
    t0 = pd.Timestamp(start, tz=TZ_LOCAL).tz_convert("UTC")
    t1 = (pd.Timestamp(end, tz=TZ_LOCAL) + pd.Timedelta(days=1)).tz_convert("UTC")
    grid = pd.date_range(t0, t1, freq="1min", inclusive="left")
    irr = df10["irr"].copy()
    irr.index = irr.index - pd.Timedelta("5min")
    m = pd.DataFrame({"irr": _interp(irr, grid), "temp": _interp(df10["temp"], grid)}, index=grid)
    q = m.resample("15min", label="left", closed="left").mean()
    q.index = q.index.tz_convert(TZ_LOCAL)
    return q


def build(start, end, out_dir: str = DEFAULT_OUT_DIR, refresh: bool = False, verbose: bool = True) -> str:
    """Download both stations, average them, write out_dir/aargau_weather_15min.csv; returns the path."""
    per = {}
    for abbr in STATIONS:
        df10 = read_station(download(abbr, start, end, os.path.join(out_dir, "raw"), refresh))
        if df10.empty:
            print(f"  ! no data for station {abbr}", file=sys.stderr)
            continue
        per[abbr] = to_quarter_hours(df10, start, end)
    if not per:
        raise RuntimeError("no MeteoSwiss station data could be downloaded (is there internet access?)")
    idx = next(iter(per.values())).index
    res = pd.DataFrame({
        "irradiance": pd.concat([d["irr"] for d in per.values()], axis=1).mean(axis=1).clip(lower=0),
        "temperature": pd.concat([d["temp"] for d in per.values()], axis=1).mean(axis=1),
    }, index=idx).round(2)
    res.index.name = "timestamp"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "aargau_weather_15min.csv")
    res.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S%z")
    if verbose:
        miss = res["irradiance"].isna().mean()
        print(f"wrote {path}: {len(res):,} quarter-hours from {', '.join(per)} "
              f"({miss:.2%} irradiance missing)", flush=True)
    return path


# ----------------------------------------------------------------------------
# reading back
# ----------------------------------------------------------------------------
def load_weather(path: str = DEFAULT_FILE) -> pd.DataFrame:
    """The saved 15-min table with a tz-aware Europe/Zurich index."""
    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ_LOCAL)
    return df


def daily_irradiation(weather: pd.DataFrame, hours=(9, 17), min_cover: float = 0.9) -> pd.Series:
    """kWh/m² of global irradiation per local calendar day inside [hours[0], hours[1]) local clock time
    (quarter-hours starting 09:00 ... 16:45 for the default).  Days with < min_cover of those
    quarter-hours present are dropped.  Index: naive dates (datetime64, midnight)."""
    w = weather["irradiance"]
    h = w.index.hour + w.index.minute / 60.0
    w = w[(h >= hours[0]) & (h < hours[1])]
    day = w.index.tz_localize(None).normalize()               # local calendar day
    grp = w.groupby(day)
    n_expected = (hours[1] - hours[0]) * 4
    kwh = grp.sum(min_count=1) / 1000.0 / 4.0            # W/m² per quarter-hour -> kWh/m²
    cover = grp.count() / n_expected
    out = kwh[cover >= min_cover].rename("irr_kwh_m2")
    out.index.name = "date"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aargau canton-average irradiance/temperature from MeteoSwiss")
    ap.add_argument("--start", required=True, help="first local day, e.g. 2023-01-01")
    ap.add_argument("--end", required=True, help="last local day (inclusive), e.g. 2026-09-10")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"output folder (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--refresh", action="store_true", help="download again even if raw files exist")
    a = ap.parse_args(argv)
    path = build(a.start, a.end, a.out, a.refresh)
    w = load_weather(path)
    d = daily_irradiation(w)
    print(f"daily 9-17 h irradiation: {len(d)} days, mean {d.mean():.2f} kWh/m², "
          f"min {d.min():.2f}, max {d.max():.2f}")


if __name__ == "__main__":
    main()