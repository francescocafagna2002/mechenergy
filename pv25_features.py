#!/usr/bin/env python3
"""
pv25_features.py
================

25 model features for AEW PV detection:

    20 ORIGINAL electrical PV/battery features from pv20_features.py
    + 5 MeteoSwiss weather-interaction features
    = 25 model features

IMPORTANT
---------
This file IMPORTS pv20_features.py rather than copying/reimplementing its
20 features. Therefore the original 20 feature definitions remain exactly
the same.

Place these files in the same directory:
    pv20_features.py
    pv25_features.py

CSV test mode (no binary store required):
    python pv25_features.py \
        /home/renku/work/aew-data/test-blob/input_data \
        --gigi "/home/renku/work/aew-data/test-blob/input_data/HackDays2026 - GIGI.csv" \
        --max-houses 3 \
        -o pv25_quick.csv

Fast store mode (later, when aew_store exists):
    python pv25_features.py \
        --store aew_store \
        --gigi "/home/renku/work/aew-data/test-blob/input_data/HackDays2026 - GIGI.csv" \
        --max-houses 3 \
        -o pv25_quick.csv

LOCATION
--------
Locations may span several cantons. Only PLZ, Ort and Kanton are read
from GIGI, and canton is never a model feature.

GIGI is used ONLY for:
    PLZ
    Ort
    Kanton (location only)

Technology columns such as PV, Batterie/Speicher, WärmePumpe, Ladestation
are NEVER read into the feature calculations.

The electrical features are indexed by MP ID. GIGI itself contains GP Nr,
not MP ID, so for the MP-ID -> PLZ bridge this script uses the PLZ already
available in raw meter rows (or in the optional store). GIGI supplies the Ort
name for that PLZ. This avoids any label leakage.

The nearest WEATHER station is chosen by physical distance and is not
artificially restricted to AG: a station just outside the cantonal border
can be closer and meteorologically more representative.

THE 5 WEATHER FEATURES
----------------------
1. weather_export_irradiance_corr_summer
   Pearson correlation between daily grid export and daily global
   irradiation in May-Aug.
   PV expectation: positive.

2. weather_sunny_cloudy_export_contrast_summer
   (mean export on top-quartile irradiation days -
    mean export on bottom-quartile irradiation days)
   / (sum of both means)
   Range [-1, 1]. PV expectation: positive.

3. weather_export_per_irradiation_summer
   Total summer grid export [kWh] divided by total global irradiation
   [kWh/m2] over common valid days.
   This is a weather-normalised apparent PV-size indicator.

4. weather_midday_import_irradiance_corr_summer
   Correlation between grid import during 09:00-17:00 and MeteoSwiss
   irradiation during 09:00-17:00.
   With PV self-consumption this often becomes negative.

5. weather_winter_import_temperature_corr
   Correlation between total daily grid import and mean air temperature
   during Nov-Feb.
   Strong negative correlation is a heat-pump/electric-heating clue and
   therefore useful as an anti-confounder.

Why these five?
---------------
They add NEW physical information rather than more percentiles of the same
meter curve:
    solar availability -> export response
    sunny-vs-cloudy contrast
    weather-normalised export magnitude
    solar availability -> import suppression
    temperature -> heating-load confounder

METEOSWISS
----------
SwissMetNet hourly observations are downloaded through the official
MeteoSwiss/FSDI STAC collection:
    ch.meteoschweiz.ogd-smn

Relevant hourly parameters:
    gre000h0 = global radiation, hourly mean [W/m2]
    tre200h0 = air temperature 2 m, hourly mean [degC]

Reference timestamps are UTC and mark the END of the hourly interval.
They are converted to Europe/Zurich before local-day / daytime aggregation.

All geocoding, station metadata, STAC metadata and downloaded station files
are cached under .pv25_weather_cache by default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pv20_features as pv20
except ImportError as exc:
    raise SystemExit(
        "pv20_features.py must be in the same folder as pv25_features.py"
    ) from exc


# =============================================================================
# 0. CONSTANTS
# =============================================================================



MIN_WEATHER_DAYS = 20
SUNNY_Q = 0.75
CLOUDY_Q = 0.25

SUMMER_MONTHS = pv20.SUMMER_MONTHS
WINTER_MONTHS = pv20.WINTER_MONTHS

WEATHER_FEATURE_NAMES = [
    "weather_export_irradiance_corr_summer",
    "weather_sunny_cloudy_export_contrast_summer",
    "weather_export_per_irradiation_summer",
    "weather_midday_import_irradiance_corr_summer",
    "weather_winter_import_temperature_corr",
]

MODEL_FEATURE_NAMES = list(pv20.FEATURE_NAMES) + WEATHER_FEATURE_NAMES

assert len(pv20.FEATURE_NAMES) == 20
assert len(WEATHER_FEATURE_NAMES) == 5
assert len(MODEL_FEATURE_NAMES) == 25

GEOADMIN_SEARCH = (
    "https://api3.geo.admin.ch/rest/services/ech/SearchServer"
)

SMN_COLLECTION = "ch.meteoschweiz.ogd-smn"
SMN_COLLECTION_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    + SMN_COLLECTION
)
SMN_ITEMS_URL = SMN_COLLECTION_URL + "/items"

SMN_META_STATIONS = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
    "ogd-smn_meta_stations.csv"
)


# =============================================================================
# 1. BASIC HTTP CACHE
# =============================================================================

def _http_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AEW-Hackdays-PV25/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _cached_download(
    url: str,
    path: Path,
    refresh: bool = False,
    timeout: int = 90,
) -> Path:
    if path.exists() and not refresh:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _http_bytes(url, timeout=timeout)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)

    return path


def _cached_json_url(
    url: str,
    path: Path,
    refresh: bool = False,
    timeout: int = 90,
):
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    data = _http_bytes(url, timeout=timeout)
    obj = json.loads(data.decode("utf-8"))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return obj


# =============================================================================
# 2. GIGI: PLZ -> ORT ONLY
# =============================================================================

def _norm_col(value: str) -> str:
    s = str(value).strip().lower()
    for old, new in {
        "ä": "a",
        "ö": "o",
        "ü": "u",
        "é": "e",
        "è": "e",
        "à": "a",
    }.items():
        s = s.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "", s)


def _read_semicolon_csv(path: str) -> pd.DataFrame:
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=";",
                dtype=str,
                keep_default_na=False,
                usecols=lambda c: _norm_col(c) in {"plz", "ort", "kanton"},
                encoding=enc,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error or RuntimeError(f"Cannot read {path}")


def read_gigi_plz_ort(path: Optional[str]) -> Dict[str, str]:
    """
    Read ONLY PLZ and Ort from GIGI.

    No label/technology column is used.
    """
    if not path:
        return {}

    df = _read_semicolon_csv(path)
    cols = {_norm_col(c): c for c in df.columns}

    if "plz" not in cols:
        raise ValueError(
            f"GIGI: no PLZ column found. Columns={list(df.columns)}"
        )

    plz = df[cols["plz"]].astype(str).str.strip()

    if "ort" in cols:
        ort = df[cols["ort"]].astype(str).str.strip()
    else:
        ort = pd.Series("", index=df.index)

    canton = df[cols["kanton"]].str.strip() if "kanton" in cols else pd.Series("", index=df.index)
    loc = pd.DataFrame({"plz": plz, "ort": ort, "canton": canton})
    loc = loc[loc["plz"].str.fullmatch(r"\d{4}", na=False)]

    result: Dict[str, str] = {}

    for postal_code, group in loc.groupby("plz", sort=False):
        candidates = group["ort"].replace("", np.nan).dropna()

        if len(candidates):
            mode = candidates.mode()
            result[str(postal_code)] = str(
                mode.iloc[0] if len(mode) else candidates.iloc[0]
            )
        else:
            result[str(postal_code)] = ""

    return result


def read_gigi_cantons(path):
    if not path:
        return {}
    df = _read_semicolon_csv(path)
    cols = {_norm_col(c): c for c in df}
    if "kanton" not in cols:
        return {}
    result = {}
    for plz, group in df.groupby(cols["plz"]):
        values = set(group[cols["kanton"]].str.strip()) - {""}
        result[str(plz).strip()] = next(iter(values)) if len(values) == 1 else ""
    return result


@contextmanager
def raw_csv_layout():
    """Support four/five leading columns and trailing delimiters; keep PV20 formulas."""
    original = pv20._parse
    def parse(source, chunksize, skip_header):
        if isinstance(source, (str, os.PathLike)):
            with open(source, encoding="utf-8", errors="replace") as stream:
                if skip_header:
                    stream.readline()
                first = stream.readline()
        else:
            pos = source.tell()
            if skip_header:
                source.readline()
            first = source.readline()
            source.seek(pos)
        fields = next(csv.reader([first], delimiter=";")) if first else []
        if len(fields) > 2 and fields[1].strip() in (pv20.OBIS_IMPORT, pv20.OBIS_EXPORT):
            leading = ["mp_id", "obis", "date", "plz"]
        elif len(fields) > 2 and fields[2].strip() in (pv20.OBIS_IMPORT, pv20.OBIS_EXPORT):
            leading = pv20.LEAD_COLS
        else:
            raise ValueError("Unrecognized AEW row layout/OBIS: " + repr(fields[:5]))
        names = leading + pv20.VCOLS
        dtypes = {c: "string" for c in leading}
        dtypes.update({c: "float32" for c in pv20.VCOLS})
        return pd.read_csv(
            source, sep=";", header=None, names=names, index_col=False,
            usecols=range(len(names)), skiprows=1 if skip_header else 0,
            dtype=dtypes, na_values=["", " ", "-"], keep_default_na=False,
            chunksize=chunksize, encoding="utf-8", encoding_errors="replace", engine="c",
        )
    pv20._parse = parse
    try:
        yield
    finally:
        pv20._parse = original


# =============================================================================
# 3. MP ID -> PLZ FROM THE EXISTING BINARY STORE
# =============================================================================

def mpid_plz_from_store(
    store: str,
    mp_ids: Iterable[str],
) -> Dict[str, str]:
    """
    Read only mp_id + plz from the Parquet buckets needed by these houses.

    This does NOT reload the 96 time-series columns.
    """
    import pyarrow.dataset as ds

    ids = sorted(set(str(x) for x in mp_ids))
    by_bucket: Dict[int, List[str]] = {}

    for mp in ids:
        try:
            b = int(mp) % pv20.N_BUCKETS
        except ValueError:
            b = 0
        by_bucket.setdefault(b, []).append(mp)

    out: Dict[str, str] = {}

    for b, ids_b in by_bucket.items():
        folder = os.path.join(store, f"bucket={b}")

        if not os.path.isdir(folder):
            continue

        dataset = ds.dataset(folder, format="parquet")

        available = set(dataset.schema.names)
        if "plz" not in available:
            raise RuntimeError(
                "The aew_store has no 'plz' column. "
                "Rebuild it with the supplied build_store.py or provide "
                "a GP->MPID mapping if you want to use GIGI alone."
            )

        tab = dataset.to_table(
            filter=ds.field("mp_id").isin(ids_b),
            columns=["mp_id", "plz"],
            use_threads=False,
        )

        if tab.num_rows == 0:
            continue

        df = tab.to_pandas()
        df["mp_id"] = df["mp_id"].astype(str)
        df["plz"] = df["plz"].astype(str).str.strip()

        for mp, group in df.groupby("mp_id", sort=False):
            vals = group["plz"]
            vals = vals[vals.str.fullmatch(r"\d{4}", na=False)]

            if len(vals):
                mode = vals.mode()
                out[str(mp)] = str(
                    mode.iloc[0] if len(mode) else vals.iloc[0]
                )

    return out



# =============================================================================
# 3b. MP ID -> PLZ DIRECTLY FROM RAW AEW CSVs
# =============================================================================

def mpid_plz_from_csv(
    root: str,
    mp_ids: Iterable[str],
    pattern: str = pv20.FILE_GLOB,
    cache_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Read ONLY enough raw CSV text to recover PLZ for selected MP IDs.

    AEW fixed row layout:
        MP ID ; meter ; OBIS ; date ; PLZ ; 96 values

    For a small quick-test set (e.g. 3 houses), this uses pv20.filter_lines()
    so only matching text lines are extracted. It does NOT parse all 96 columns.
    """
    ids = sorted(set(str(x).strip() for x in mp_ids if str(x).strip()))
    if not ids:
        return {}

    files = pv20.find_files(root, pattern=pattern)
    out: Dict[str, str] = {}

    for path in files:
        missing = [mp for mp in ids if mp not in out]
        if not missing:
            break

        text = pv20.filter_lines(path, ids, cache_dir=cache_dir)
        if not text.strip():
            continue

        for line in text.splitlines():
            parts = line.split(";")
            if len(parts) < 5:
                continue

            mp = parts[0].strip()
            plz_index = 3 if parts[1].strip() in (pv20.OBIS_IMPORT, pv20.OBIS_EXPORT) else 4
            plz = parts[plz_index].strip()

            if mp in missing and re.fullmatch(r"\d{4}", plz):
                out[mp] = plz

    return out


# =============================================================================
# 4. GEO.ADMIN: PLZ + ORT + OBSERVED CANTON -> LAT/LON
# =============================================================================

def geocode_plz(
    plz: str,
    ort: str,
    cache_dir: Path,
    refresh: bool = False,
    canton: str = "",
) -> Tuple[float, float]:
    """
    Geocode a location using official geo.admin.ch search.

    Use the observed canton if unambiguous; otherwise use the Swiss postal code.
    """
    safe_key = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        f"{plz}_{ort}_{canton or 'CH'}",
    ).strip("_")

    cache = cache_dir / "geocode" / f"{safe_key}.json"

    if cache.exists() and not refresh:
        obj = json.loads(cache.read_text(encoding="utf-8"))
        return float(obj["lat"]), float(obj["lon"])

    query = " ".join(
        x for x in (str(plz), str(ort), canton) if x
    )

    params = {
        "searchText": query,
        "type": "locations",
        "origins": "zipcode,gazetteer",
        "limit": 20,
        "sr": 4326,
    }

    url = GEOADMIN_SEARCH + "?" + urllib.parse.urlencode(params)
    payload = json.loads(_http_bytes(url, 30).decode("utf-8"))

    results = payload.get("results", [])

    if not any(str(plz) in str(r.get("attrs", {}).get("label", "")) for r in results):
        params["origins"] = "zipcode"
        params["searchText"] = str(plz)
        url = GEOADMIN_SEARCH + "?" + urllib.parse.urlencode(params)
        payload = json.loads(_http_bytes(url, 30).decode("utf-8"))
        results = payload.get("results", [])

    if not results:
        raise RuntimeError(
            f"geo.admin.ch: no location found for {plz} {ort}"
        )

    chosen = None

    # Prefer a result containing the exact PLZ and, if available, Ort.
    for result in results:
        attrs = result.get("attrs", {})
        haystack = (
            str(attrs.get("label", ""))
            + " "
            + str(attrs.get("detail", ""))
        ).lower()

        if str(plz) not in haystack:
            continue

        if ort and ort.lower() not in haystack:
            continue

        chosen = result
        break

    if chosen is None:
        for result in results:
            attrs = result.get("attrs", {})
            haystack = (
                str(attrs.get("label", ""))
                + " "
                + str(attrs.get("detail", ""))
            )
            if str(plz) in haystack:
                chosen = result
                break

    if chosen is None:
        raise RuntimeError(f"No exact postal-code match for {plz} {ort}")

    attrs = chosen.get("attrs", {})

    lat = attrs.get("lat")
    lon = attrs.get("lon")

    if lat is None or lon is None:
        raise RuntimeError(
            f"geo.admin result has no lat/lon for {plz} {ort}"
        )

    result = {
        "plz": str(plz),
        "ort": str(ort),
        "canton": canton,
        "lat": float(lat),
        "lon": float(lon),
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return float(lat), float(lon)


# =============================================================================
# 5. METEOSWISS STATION METADATA
# =============================================================================

def _find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    norm_to_real = {_norm_col(c): c for c in columns}

    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_to_real:
            return norm_to_real[key]

    return None


def load_station_metadata(
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    path = _cached_download(
        SMN_META_STATIONS,
        cache_dir / "meteoswiss" / "ogd-smn_meta_stations.csv",
        refresh=refresh,
    )

    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        encoding="cp1252",
    )

    abbr_col = _find_column(
        df.columns,
        ["station_abbr", "station_abbreviation"],
    )
    name_col = _find_column(
        df.columns,
        ["station_name"],
    )
    lat_col = _find_column(
        df.columns,
        [
            "station_coordinates_wgs84_lat",
            "station_wgs84_lat",
            "latitude",
        ],
    )
    lon_col = _find_column(
        df.columns,
        [
            "station_coordinates_wgs84_lon",
            "station_wgs84_lon",
            "longitude",
        ],
    )

    if not all((abbr_col, lat_col, lon_col)):
        raise RuntimeError(
            "Could not identify station abbreviation / WGS84 coordinates "
            f"in MeteoSwiss metadata. Columns={list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "station": df[abbr_col].astype(str).str.strip(),
            "name": (
                df[name_col].astype(str).str.strip()
                if name_col
                else df[abbr_col].astype(str).str.strip()
            ),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    return out[
        out["station"].ne("")
        & out["lat"].notna()
        & out["lon"].notna()
    ].copy()


def haversine_km(
    lat: float,
    lon: float,
    other_lat: np.ndarray,
    other_lon: np.ndarray,
) -> np.ndarray:
    radius = 6371.0088

    p1 = np.radians(lat)
    p2 = np.radians(other_lat)
    dp = np.radians(other_lat - lat)
    dl = np.radians(other_lon - lon)

    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def rank_stations(
    stations: pd.DataFrame,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    result = stations.copy()

    result["distance_km"] = haversine_km(
        lat,
        lon,
        result["lat"].to_numpy(dtype=float),
        result["lon"].to_numpy(dtype=float),
    )

    return result.sort_values(
        "distance_km",
        ascending=True,
    ).reset_index(drop=True)


# =============================================================================
# 6. STAC: DISCOVER & DOWNLOAD HOURLY STATION FILES
# =============================================================================

def load_stac_items(
    cache_dir: Path,
    refresh: bool = False,
) -> List[dict]:
    """
    Retrieve the collection items and follow STAC pagination if required.
    """
    cache = cache_dir / "meteoswiss" / "stac_items.json"

    if cache.exists() and not refresh:
        obj = json.loads(cache.read_text(encoding="utf-8"))
        return obj["features"]

    url = SMN_ITEMS_URL + "?limit=500"
    all_features: List[dict] = []
    seen_urls = set()

    while url and url not in seen_urls:
        seen_urls.add(url)

        payload = json.loads(_http_bytes(url, 90).decode("utf-8"))
        all_features.extend(payload.get("features", []))

        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break

        url = next_url

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"features": all_features}),
        encoding="utf-8",
    )

    return all_features


def _station_item_score(feature: dict, station: str) -> int:
    station_l = station.lower()

    feature_id = str(feature.get("id", "")).lower()
    properties = feature.get("properties", {})
    assets = feature.get("assets", {})

    text = " ".join(
        [feature_id]
        + [str(x).lower() for x in properties.values()]
        + [
            str(k).lower() + " " + str(v.get("href", "")).lower()
            for k, v in assets.items()
        ]
    )

    score = 0

    if feature_id == station_l:
        score += 100

    if re.search(rf"(^|[^a-z0-9]){re.escape(station_l)}([^a-z0-9]|$)", text):
        score += 20

    if f"/{station_l}/" in text:
        score += 40

    return score


def station_hourly_asset_urls(
    station: str,
    stac_items: List[dict],
) -> List[str]:
    """
    Dynamically discover hourly historical/recent CSV assets for one station.

    We do not hardcode MeteoSwiss filenames; the STAC response remains the
    source of truth.
    """
    candidates = []

    for feature in stac_items:
        score = _station_item_score(feature, station)

        if score <= 0:
            continue

        for key, asset in feature.get("assets", {}).items():
            href = str(asset.get("href", ""))
            typ = str(asset.get("type", "")).lower()
            text = (str(key) + " " + href).lower()

            if not href:
                continue

            if ".csv" not in text and "text/csv" not in typ:
                continue

            # Hourly data is denoted by granularity "h". Accept common
            # MeteoSwiss naming forms without assuming one exact filename.
            hourly = (
                re.search(r"(^|[_/.-])h([_/.-]|$)", text) is not None
                or "hourly" in text
            )

            if not hourly:
                continue

            # For the challenge period we want the archived historical data
            # and the current-year recent data. "now" is unnecessary.
            if (
                "historical" not in text
                and "recent" not in text
                and "_h_" not in text
            ):
                continue

            candidates.append((score, href))

    # Highest-scoring station matches first; remove duplicates.
    candidates.sort(key=lambda x: -x[0])

    urls = []
    seen = set()

    for _, href in candidates:
        if href not in seen:
            seen.add(href)
            urls.append(href)

    return urls


def _read_meteoswiss_csv(path: Path) -> pd.DataFrame:
    for enc in ("cp1252", "utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Cannot decode MeteoSwiss CSV {path}")


def _normalise_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}

    for c in df.columns:
        clean = str(c).replace("'", "").strip()
        renamed[c] = clean

    return df.rename(columns=renamed)


def _weather_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "reference_timestamp",
        "reference_ts",
        "ReferenceTS",
        "REFERENCE_TS",
        "timestamp",
        "time",
    ]

    c = _find_column(df.columns, candidates)

    if c:
        return c

    # Last fallback: any column containing "reference" and "ts/time".
    for col in df.columns:
        n = _norm_col(col)
        if "reference" in n and ("ts" in n or "time" in n):
            return col

    return None


def _parse_weather_timestamp(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    ts = pd.to_datetime(
        s,
        format="%d.%m.%Y %H:%M",
        errors="coerce",
        utc=True,
    )

    missing = ts.isna()

    if missing.any():
        alt = pd.to_datetime(
            s[missing],
            errors="coerce",
            utc=True,
            dayfirst=True,
        )
        ts.loc[missing] = alt

    return ts


def download_station_hourly(
    station: str,
    cache_dir: Path,
    stac_items: List[dict],
    refresh: bool = False,
    years=None,
) -> pd.DataFrame:
    urls = station_hourly_asset_urls(station, stac_items)

    if not urls:
        raise RuntimeError(
            f"No hourly STAC CSV assets found for station {station}"
        )

    frames = []

    for url in urls:
        if "_h_now" in url:
            continue
        span = re.search(r"historical_(d{4})-(d{4})", url)
        if span and years and not any(int(span[1]) <= y <= int(span[2]) for y in years):
            continue
        name = os.path.basename(
            urllib.parse.urlsplit(url).path
        )

        if not name:
            name = hashlib.md5(url.encode()).hexdigest() + ".csv"

        path = _cached_download(
            url,
            cache_dir / "meteoswiss" / station.lower() / name,
            refresh=refresh,
            timeout=180,
        )

        frame = _normalise_weather_columns(
            _read_meteoswiss_csv(path)
        )

        ts_col = _weather_timestamp_column(frame)

        if ts_col is None:
            continue

        frame["_ts_utc"] = _parse_weather_timestamp(frame[ts_col])
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"Downloaded hourly files for {station}, but no timestamps parsed"
        )

    # Concatenating files with slightly different parameter columns is fine.
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data[data["_ts_utc"].notna()].copy()

    # Keep one observation per timestamp.
    data = (
        data.sort_values("_ts_utc")
        .drop_duplicates("_ts_utc", keep="last")
    )

    return data


# =============================================================================
# 7. HOURLY METEOSWISS -> LOCAL DAILY WEATHER
# =============================================================================

def weather_to_daily(weather: pd.DataFrame) -> pd.DataFrame:
    """
    gre000h0: hourly mean global radiation [W/m2].
    One hourly mean * 1 h / 1000 -> kWh/m2 for that interval.

    MeteoSwiss hourly timestamps are UTC and indicate interval END.
    """
    radiation_col = _find_column(
        weather.columns,
        ["gre000h0"],
    )
    temperature_col = _find_column(
        weather.columns,
        ["tre200h0"],
    )

    if not radiation_col:
        raise RuntimeError("Station has no gre000h0 hourly global radiation")

    w = weather.copy()

    w["radiation_wm2"] = pd.to_numeric(
        w[radiation_col],
        errors="coerce",
    ).clip(lower=0)

    if temperature_col:
        w["temperature_c"] = pd.to_numeric(
            w[temperature_col],
            errors="coerce",
        )
    else:
        w["temperature_c"] = np.nan

    local = (w["_ts_utc"] - pd.Timedelta(minutes=30)).dt.tz_convert("Europe/Zurich")

    w["date"] = pd.to_datetime(local.dt.date)

    # Use interval centres so local midnight belongs to the preceding day.
    w["interval_end_hour"] = (
        local.dt.hour
        + local.dt.minute / 60.0
    )

    day = w.groupby("date").agg(
        irradiation_kwh_m2=("radiation_wm2", lambda x: x.sum(min_count=1) / 1000.0),
        temperature_c=("temperature_c", "mean"),
        radiation_hours=("radiation_wm2", "count"),
        temperature_hours=("temperature_c", "count"),
    )

    midday = w[
        (w["interval_end_hour"] >= 9.0)
        & (w["interval_end_hour"] < 17.0)
    ]

    mid = (
        midday.groupby("date")["radiation_wm2"]
        .sum(min_count=1)
        / 1000.0
    )

    day["midday_irradiation_kwh_m2"] = mid.reindex(day.index)

    # An incomplete daylight window must not appear to be low irradiation.
    mid_hours = midday.groupby("date")["radiation_wm2"].count()
    day.loc[mid_hours.reindex(day.index, fill_value=0) < 8, "midday_irradiation_kwh_m2"] = np.nan

    # Avoid using incomplete weather days, including 23/25-hour DST days.
    expected_hours = pd.Series([
        (pd.Timestamp(d + pd.Timedelta(days=1)).tz_localize("Europe/Zurich")
         - pd.Timestamp(d).tz_localize("Europe/Zurich")).total_seconds() / 3600
        for d in day.index
    ], index=day.index)
    day.loc[
        day["radiation_hours"] < expected_hours,
        ["irradiation_kwh_m2", "midday_irradiation_kwh_m2"],
    ] = np.nan

    day.loc[
        day["temperature_hours"] < expected_hours,
        "temperature_c",
    ] = np.nan

    return day.sort_index()


# =============================================================================
# 8. CHOOSE NEAREST STATION THAT ACTUALLY HAS RADIATION
# =============================================================================

class WeatherRepository:
    def __init__(
        self,
        cache_dir: str,
        gigi_path: Optional[str] = None,
        refresh: bool = False,
        dates=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.dates = pd.DatetimeIndex(dates).unique() if dates is not None else None
        self.years = set(self.dates.year) if self.dates is not None else None

        self.plz_to_ort = read_gigi_plz_ort(gigi_path)
        self.plz_to_canton = read_gigi_cantons(gigi_path)
        self.stations = load_station_metadata(
            self.cache_dir,
            refresh=refresh,
        )
        self.stac_items = load_stac_items(
            self.cache_dir,
            refresh=refresh,
        )

        self._plz_choice: Dict[str, dict] = {}
        self._station_daily: Dict[str, pd.DataFrame] = {}
        self._station_errors = {}

    def _station_daily_data(self, station: str) -> pd.DataFrame:
        if station in self._station_errors:
            raise RuntimeError(self._station_errors[station])
        if station not in self._station_daily:
            hourly = download_station_hourly(
                station,
                self.cache_dir,
                self.stac_items,
                refresh=self.refresh,
                years=self.years,
            )
            self._station_daily[station] = weather_to_daily(hourly)

        return self._station_daily[station]

    def for_plz(
        self,
        plz: str,
    ) -> Tuple[pd.DataFrame, dict]:
        plz = str(plz).strip()

        if plz in self._plz_choice:
            info = self._plz_choice[plz]
            return (
                self._station_daily_data(info["station"]),
                info,
            )

        ort = self.plz_to_ort.get(plz, "")

        lat, lon = geocode_plz(
            plz,
            ort,
            self.cache_dir,
            refresh=self.refresh,
            canton=self.plz_to_canton.get(plz, ""),
        )

        ranked = rank_stations(
            self.stations,
            lat,
            lon,
        )

        errors = []

        # Check nearest stations until one has usable hourly irradiation.
        for _, station_row in ranked.iterrows():
            station = str(station_row["station"]).strip()

            try:
                daily = self._station_daily_data(station)

                if (
                    "irradiation_kwh_m2" in daily.columns
                    and daily["irradiation_kwh_m2"].reindex(self.dates if self.dates is not None else daily.index).notna().sum()
                    >= MIN_WEATHER_DAYS
                ):
                    info = {
                        "plz": plz,
                        "ort": ort,
                        "canton": self.plz_to_canton.get(plz, ""),
                        "lat": lat,
                        "lon": lon,
                        "station": station,
                        "station_name": str(station_row["name"]),
                        "station_distance_km": float(
                            station_row["distance_km"]
                        ),
                    }

                    self._plz_choice[plz] = info
                    return daily, info

                errors.append(f"{station}: insufficient irradiation in meter period")
            except Exception as exc:
                self._station_errors[station] = str(exc)
                errors.append(f"{station}: {exc}")

        raise RuntimeError(
            f"No nearby station with usable irradiation for {plz} {ort}. "
            + " | ".join(errors[:5])
        )


# =============================================================================
# 9. DAILY ELECTRICITY TABLE FROM pv20 REDUCED DAYS
# =============================================================================

def electricity_for_house(
    days: pd.DataFrame,
    mp_id: str,
) -> pd.DataFrame:
    h = days[days["mp_id"].astype(str) == str(mp_id)].copy()

    if h.empty:
        return pd.DataFrame()

    h["date"] = pd.to_datetime(h["date"])

    imp = h[h["channel"] == "import"].copy()
    exp = h[h["channel"] == "export"].copy()

    if imp.empty:
        return pd.DataFrame()

    # pv20 reduce_days stores sums in kW-quarter-hours.
    # Divide by 4 to obtain kWh.
    e = imp.groupby("date").agg(
        import_kwh=("tot", "sum"),
        midday_import_kwh=("day_tot", "sum"),
        n_valid=("n_valid", "sum"),
        n_valid_day=("n_valid_day", "sum"),
    )

    e["import_kwh"] /= 4.0
    e["midday_import_kwh"] /= 4.0

    # Do not silently treat badly incomplete electrical days as valid.
    e.loc[
        e["n_valid"] < pv20.N_SLOTS,
        ["import_kwh", "midday_import_kwh"],
    ] = np.nan

    e.loc[e["n_valid_day"] < pv20.DAY_MASK.sum(), "midday_import_kwh"] = np.nan

    if not exp.empty:
        x = exp.groupby("date").agg(
            export_kwh=("tot", "sum"),
            n_valid_export=("n_valid", "sum"),
        )
        x["export_kwh"] /= 4.0
        x.loc[
            x["n_valid_export"] < pv20.N_SLOTS,
            "export_kwh",
        ] = np.nan

        e = e.join(
            x[["export_kwh"]],
            how="left",
        )
    else:
        e["export_kwh"] = np.nan

    return e.sort_index()


# =============================================================================
# 10. FIVE WEATHER FEATURES
# =============================================================================

def _pearson(
    a: pd.Series,
    b: pd.Series,
    min_days: int = MIN_WEATHER_DAYS,
) -> float:
    pair = pd.concat([a, b], axis=1).dropna()

    if len(pair) < min_days:
        return np.nan

    x = pair.iloc[:, 0].to_numpy(dtype=float)
    y = pair.iloc[:, 1].to_numpy(dtype=float)

    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def _bounded_contrast(
    high_value: float,
    low_value: float,
) -> float:
    if not np.isfinite(high_value) or not np.isfinite(low_value):
        return np.nan

    den = high_value + low_value

    if den <= 0:
        return np.nan

    return float((high_value - low_value) / den)


def five_weather_features(
    electric: pd.DataFrame,
    weather: pd.DataFrame,
    has_export_register: bool,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    out = {name: np.nan for name in WEATHER_FEATURE_NAMES}

    joined = electric.join(
        weather[
            [
                "irradiation_kwh_m2",
                "midday_irradiation_kwh_m2",
                "temperature_c",
            ]
        ],
        how="inner",
    )

    summer = joined[
        joined.index.month.isin(SUMMER_MONTHS)
    ].copy()

    winter = joined[
        joined.index.month.isin(WINTER_MONTHS)
    ].copy()

    diagnostics = {
        "weather_common_summer_days": int(
            summer[["midday_import_kwh", "midday_irradiation_kwh_m2"]].dropna().shape[0]
        ),
        "weather_common_winter_days": int(
            winter[["import_kwh", "temperature_c"]].dropna().shape[0]
        ),
    }

    # 1) Daily export should follow available solar energy.
    if has_export_register:
        out["weather_export_irradiance_corr_summer"] = _pearson(
            summer["export_kwh"],
            summer["irradiation_kwh_m2"],
        )

        # 2) Sunny-vs-cloudy export contrast.
        valid_rad = summer[["export_kwh", "irradiation_kwh_m2"]].dropna()["irradiation_kwh_m2"]

        if len(valid_rad) >= MIN_WEATHER_DAYS:
            q_hi = float(valid_rad.quantile(SUNNY_Q))
            q_lo = float(valid_rad.quantile(CLOUDY_Q))

            sunny_export = summer.loc[
                summer["irradiation_kwh_m2"] >= q_hi,
                "export_kwh",
            ].dropna()

            cloudy_export = summer.loc[
                summer["irradiation_kwh_m2"] <= q_lo,
                "export_kwh",
            ].dropna()

            if len(sunny_export) >= 5 and len(cloudy_export) >= 5:
                out[
                    "weather_sunny_cloudy_export_contrast_summer"
                ] = _bounded_contrast(
                    float(sunny_export.mean()),
                    float(cloudy_export.mean()),
                )

        # 3) Export normalised by measured irradiation.
        valid_export = summer[
            ["export_kwh", "irradiation_kwh_m2"]
        ].dropna()

        if len(valid_export) >= MIN_WEATHER_DAYS:
            irr = float(valid_export["irradiation_kwh_m2"].sum())
            exported = float(valid_export["export_kwh"].sum())

            if irr > 0:
                out[
                    "weather_export_per_irradiation_summer"
                ] = exported / irr

    # 4) Daytime import suppression with sun.
    out[
        "weather_midday_import_irradiance_corr_summer"
    ] = _pearson(
        summer["midday_import_kwh"],
        summer["midday_irradiation_kwh_m2"],
    )

    # 5) Heating / heat-pump anti-confounder.
    out[
        "weather_winter_import_temperature_corr"
    ] = _pearson(
        winter["import_kwh"],
        winter["temperature_c"],
    )

    return out, diagnostics


# =============================================================================
# 11. BUILD 20 + 5 FEATURE TABLE
# =============================================================================

def add_weather_features(
    base_features: pd.DataFrame,
    days: pd.DataFrame,
    weather_repo: WeatherRepository,
    store: Optional[str] = None,
    root: Optional[str] = None,
    pattern: str = pv20.FILE_GLOB,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    out = base_features.copy()

    for name in WEATHER_FEATURE_NAMES:
        out[name] = np.nan

    out["plz"] = ""
    out["ort"] = ""
    out["weather_station"] = ""
    out["weather_station_distance_km"] = np.nan
    out["weather_common_summer_days"] = 0
    out["weather_common_winter_days"] = 0
    out["weather_error"] = ""

    ids = [str(x) for x in out.index]

    if store:
        plz_map = mpid_plz_from_store(
            store,
            ids,
        )
    elif root:
        plz_map = mpid_plz_from_csv(
            root,
            ids,
            pattern=pattern,
            cache_dir=cache_dir,
        )
    else:
        raise RuntimeError("Need either --store or a CSV root to recover PLZ")

    for i, mp_id in enumerate(ids, start=1):
        plz = plz_map.get(mp_id, "")

        print(
            f"[weather {i}/{len(ids)}] MP {mp_id} PLZ {plz or '?'}",
            flush=True,
        )

        out.loc[mp_id, "plz"] = plz

        if not plz:
            out.loc[mp_id, "weather_error"] = "No PLZ for MP ID"
            print(f"MP {mp_id}: No PLZ for MP ID", flush=True)
            continue

        try:
            weather, info = weather_repo.for_plz(plz)

            electric = electricity_for_house(
                days,
                mp_id,
            )

            wf, diagnostics = five_weather_features(
                electric=electric,
                weather=weather,
                has_export_register=bool(
                    base_features.loc[mp_id, "has_export_register"]
                ),
            )

            for name, value in wf.items():
                out.loc[mp_id, name] = value

            out.loc[mp_id, "ort"] = info.get("ort", "")
            out.loc[mp_id, "weather_station"] = (
                f"{info.get('station', '')} "
                f"{info.get('station_name', '')}"
            ).strip()

            out.loc[
                mp_id,
                "weather_station_distance_km",
            ] = info.get("station_distance_km", np.nan)

            for name, value in diagnostics.items():
                out.loc[mp_id, name] = value

        except Exception as exc:
            out.loc[mp_id, "weather_error"] = str(exc)
            print(f"MP {mp_id}: weather error: {exc}", flush=True)

    return out


# =============================================================================
# 12. CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Keep the useful pv20 CLI options.
    ap.add_argument(
        "root",
        nargs="?",
        help="raw AEW CSV root (normally unnecessary when --store is used)",
    )
    ap.add_argument(
        "--store",
        default=None,
        help="optional binary store built by build_store.py",
    )
    ap.add_argument(
        "--gigi",
        help=(
            "HackDays2026 - GIGI.csv; only PLZ and Ort are used, "
            "never PV/technology labels"
        ),
    )
    ap.add_argument(
        "-o",
        "--out",
        default="pv25_features.csv",
    )
    ap.add_argument(
        "--month",
        help='e.g. "2023" or "Juni 2023"',
    )
    ap.add_argument(
        "--max-houses",
        type=int,
        help="first N MP IDs (CSV: from first meter file; store: from store)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        help="CSV mode only: read only first N monthly meter files",
    )
    ap.add_argument(
        "--pattern",
        default=pv20.FILE_GLOB,
        help=f"CSV meter filename pattern (default {pv20.FILE_GLOB})",
    )
    ap.add_argument(
        "--cache-dir",
        default="aew_cache",
        help="CSV mode extracted-line cache (default: aew_cache)",
    )
    ap.add_argument(
        "--mp-ids",
        help="comma-separated MP IDs, e.g. 53628,45390",
    )
    ap.add_argument(
        "--mp-ids-file",
        help="text file containing one MP ID per line",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--refresh-days",
        action="store_true",
        help="recompute pv20 store _days.parquet cache",
    )
    ap.add_argument(
        "--weather-cache",
        default=".pv25_weather_cache",
    )
    ap.add_argument(
        "--refresh-weather",
        action="store_true",
    )
    ap.add_argument(
        "--no-weather",
        action="store_true",
        help="debug: output only the original 20 features",
    )

    a = ap.parse_args(argv)
    started = time.perf_counter()

    store = a.store if a.store and os.path.isdir(a.store) else None

    if store is None and not a.root:
        raise SystemExit(
            "Give either a CSV root folder or an existing --store folder."
        )

    if a.store and store is None:
        print(
            f"note: store '{a.store}' not found -> using raw CSV mode",
            flush=True,
        )

    ids = (
        [x.strip() for x in a.mp_ids.split(",") if x.strip()]
        if a.mp_ids
        else None
    )

    if a.mp_ids_file:
        with open(a.mp_ids_file, encoding="utf-8") as f:
            file_ids = [
                line.strip().split(";")[0].split(",")[0]
                for line in f
                if line.strip()
                and not line.lower().startswith("mp")
            ]

        ids = (ids or []) + file_ids

    # -------------------------------------------------------------------------
    # Exact original pv20 pipeline.
    # -------------------------------------------------------------------------
    with raw_csv_layout():
        days = pv20.load_days(
            root=a.root,
            month=a.month,
            max_files=a.max_files,
            max_houses=a.max_houses,
            mp_ids=ids,
            pattern=a.pattern,
            workers=a.workers,
            cache_dir=(a.cache_dir or None),
            store=store,
            use_days_cache=not a.refresh_days,
        )

    if days.empty:
        raise SystemExit("No valid meter rows found; check CSV layout, OBIS codes and selected MP IDs.")

    base = pv20.feature_table(days)

    if a.no_weather:
        result = base
    else:
        try:
            repo = WeatherRepository(
                cache_dir=a.weather_cache, gigi_path=a.gigi,
                refresh=a.refresh_weather, dates=days["date"],
            )
            result = add_weather_features(
                base_features=base, days=days, weather_repo=repo,
                store=store, root=a.root, pattern=a.pattern, cache_dir=a.cache_dir or None,
            )
        except Exception as exc:
            print(f"Weather initialization/mapping error: {exc}", flush=True)
            result = base.copy()
            for name in WEATHER_FEATURE_NAMES:
                result[name] = np.nan
            result["weather_error"] = str(exc)

    result.to_csv(
        a.out,
        float_format="%.5f",
    )

    print()
    print("=" * 78)
    print("PV25 FEATURE EXTRACTION COMPLETE")
    print("=" * 78)
    print(f"Houses                    : {len(result):,}")
    print(f"Original electrical       : {len(pv20.FEATURE_NAMES)}")
    print(f"MeteoSwiss weather        : {0 if a.no_weather else len(WEATHER_FEATURE_NAMES)}")
    print(
        f"TOTAL model features      : "
        f"{len(pv20.FEATURE_NAMES) + (0 if a.no_weather else len(WEATHER_FEATURE_NAMES))}"
    )
    print(f"Output                    : {a.out}")
    print(f"Runtime seconds           : {time.perf_counter() - started:.2f}")

    if not a.no_weather:
        print(f"Weather cache             : {a.weather_cache}")

        errors = (
            result["weather_error"]
            if "weather_error" in result
            else pd.Series(dtype=str)
        )
        errors = errors[
            errors.fillna("").astype(str).str.len() > 0
        ]

        print(f"Weather mapping errors    : {len(errors)}")

    if len(result) <= 10:
        display_cols = [c for c in MODEL_FEATURE_NAMES if c in result]

        for extra in (
            "plz",
            "ort",
            "weather_station",
            "weather_station_distance_km",
            "weather_common_summer_days",
            "weather_common_winter_days",
            "weather_error",
        ):
            if extra in result.columns:
                display_cols.append(extra)

        with pd.option_context(
            "display.width",
            300,
            "display.max_columns",
            None,
            "display.max_rows",
            100,
            "display.float_format",
            "{:.3f}".format,
        ):
            print()
            print(result[display_cols].T)

    for mp, row in result.iterrows():
        if len(result) <= 10:
            print(f"MP {mp} NaN features: " + ", ".join(c for c in MODEL_FEATURE_NAMES if c in row and pd.isna(row[c])))

    if not a.no_weather:
        weather_nan = result[WEATHER_FEATURE_NAMES].isna().all()

        if weather_nan.any():
            print(
                "\nWeather features all-NaN:",
                ", ".join(
                    weather_nan[weather_nan].index
                ),
            )


if __name__ == "__main__":
    main()
