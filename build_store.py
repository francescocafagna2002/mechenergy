"""
build_store.py — ONE-TIME conversion of the AEW monthly CSVs into a binary store
organised by house, so that any house can be read in milliseconds afterwards.

    python build_store.py ../aew-data/test-blob/input_data -o aew_store
    python build_store.py ../aew-data/test-blob/input_data -o aew_store --month 2023   # subset

Needs:  pip install pyarrow  (+ duckdb optionally).  Memory: DuckDB runs with 2 threads and a
1.5 GB cap and falls back to the pandas path for a file if it still runs out of memory;
--no-duckdb uses the pandas path throughout (peak ~1 GB, same speed on a small session).

What it does
------------
1. Reads every  LG_AIM2Hackerdays_kWh_*.csv  below the root.  The layout is read per file
   from its first data row (the files differ: 2023 rows are MP ID ; meter id ; OBIS ; Datum ; PLZ ;
   96 values ; — 2024+ rows have no meter-id column — all end with a trailing ';'),
   keeps the import (1-1:1.29.0*255) and export (1-1:2.29.0*255) rows, converts kWh per
   15 min to kW (x4) and writes Parquet files partitioned by  bucket = MP ID mod 256
   (about 350 houses per bucket), streaming file by file.
2. Compacts each bucket: rows sorted by (mp_id, channel, date) and written in small row
   groups, so a reader asking for one house touches only that house's row groups.

Store layout
------------
    aew_store/bucket=0/part.parquet ... bucket=255/part.parquet
    columns: mp_id (string), channel ('import' | 'export'), date (date), plz (string),
             meter (string), v0 .. v95 (float32, kW, RAW: negatives NOT floored, meters NOT summed)
    aew_store/_manifest.json  — source files, row counts, build time

Size: roughly 10 % of the CSVs (binary + snappy).  Build time: one pass over the CSVs
(disk-bound: ~ the time of one feature run today), then never again.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import List, Optional

N_SLOTS = 96
N_BUCKETS = 256
FILE_GLOB = "LG_AIM2Hackerdays_kWh_*.csv"
OBIS_IMPORT = "1-1:1.29.0*255"
OBIS_EXPORT = "1-1:2.29.0*255"
VCOLS = [f"v{k}" for k in range(N_SLOTS)]
STORE_COLS = ["mp_id", "channel", "date", "plz", "meter"] + VCOLS


def bucket_of(mp_id: str) -> int:
    """Bucket of a house = MP ID mod 256 (non-numeric IDs go to bucket 0). Same rule in the reader."""
    try:
        return int(str(mp_id).strip()) % N_BUCKETS
    except ValueError:
        return 0


def find_files(root: str, month: Optional[str] = None, pattern: str = FILE_GLOB) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if month:
        files = [f for f in files if month.lower() in f.lower()]
    if not files:
        raise SystemExit(f"no files named {pattern} below {root}")
    return files



# -----------------------------------------------------------------------------
# per-file layout: the files are NOT uniform (2023 files have a meter-id column, 2024+ do not,
# rows end with a trailing ';').  The layout is read from the first DATA row of each file:
# the last 96 fields are the values, the leading fields are recognised by their format.
# -----------------------------------------------------------------------------
import re as _re
_OBIS_RE = _re.compile(r"^\d+-\d+:\d+\.\d+\.\d+")
_DATE_RE = _re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
_METER_RE = _re.compile(r"^[A-Z]{2}\d{6,}")


def file_layout(path: str) -> dict:
    """{'n_fields': fields per data row (incl. trailing empty), 'n_lead': leading columns,
        'roles': {'mp_id': i, 'obis': i, 'date': i, 'meter': i|None, 'plz': i|None}}"""
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
        if "obis" not in roles and _OBIS_RE.match(v):
            roles["obis"] = i
        elif "date" not in roles and _DATE_RE.match(v):
            roles["date"] = i
        elif roles["meter"] is None and _METER_RE.match(v):
            roles["meter"] = i
    if "obis" not in roles or "date" not in roles:
        raise ValueError(f"{path}: could not recognise OBIS / date in the first data row {lead}")
    rest = [i for i in range(n_lead) if i not in (roles["obis"], roles["date"], roles["meter"])]
    roles["mp_id"] = rest[0]
    roles["plz"] = rest[1] if len(rest) > 1 else None
    return {"n_fields": len(first), "n_lead": n_lead, "roles": roles}


def layout_names(lay: dict) -> List[str]:
    """Column names for pandas: leading columns named by role (unknown ones x0, x1, ...), then v0..v95."""
    inv = {i: r for r, i in lay["roles"].items() if i is not None}
    return [inv.get(i, f"x{i}") for i in range(lay["n_lead"])] + VCOLS


# -----------------------------------------------------------------------------
# step 1a: DuckDB conversion (fast path)
# -----------------------------------------------------------------------------
def convert_duckdb(files: List[str], tmp_dir: str, threads: Optional[int], memory_limit: str) -> None:
    """One COPY per monthly file, few threads, capped memory: the partitioned write keeps a
    buffer per bucket per thread, so this is what keeps a 3-4 GB session alive."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads or min(2, os.cpu_count() or 1))}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{os.path.join(tmp_dir, '_duckdb_tmp')}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET partitioned_write_flush_threshold = 20000")     # rows buffered per bucket before flushing
    con.execute("SET partitioned_write_max_open_files = 300")        # >= number of buckets

    vals_sql = ", ".join(f"CAST(v{k} * 4 AS FLOAT) AS v{k}" for k in range(N_SLOTS))   # kWh/15min -> kW

    for i, f in enumerate(files):
        t = time.time()
        print(f"  [{i + 1}/{len(files)}] {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)} ...", end=" ", flush=True)
        lay = file_layout(f)
        names = layout_names(lay) + (["trailing"] if lay["n_fields"] > lay["n_lead"] + N_SLOTS else [])
        cols = {n: ("DOUBLE" if n.startswith("v") else "VARCHAR") for n in names}
        cols_sql = "{" + ", ".join(f"'{k}': '{v}'" for k, v in cols.items()) + "}"
        meter_sql = "TRIM(meter)" if lay["roles"]["meter"] is not None else "''"
        plz_sql = "TRIM(plz)" if lay["roles"]["plz"] is not None else "''"
        f_sql = f.replace("'", "''")
        sql = f"""
        COPY (
            SELECT COALESCE(TRY_CAST(TRIM(mp_id) AS BIGINT) % {N_BUCKETS}, 0) AS bucket,
                   TRIM(mp_id) AS mp_id,
                   CASE TRIM(obis) WHEN '{OBIS_IMPORT}' THEN 'import' ELSE 'export' END AS channel,
                   CAST(strptime(TRIM(date), '%d.%m.%Y') AS DATE) AS date,
                   {plz_sql} AS plz,
                   {meter_sql} AS meter,
                   {vals_sql}
            FROM read_csv('{f_sql}', delim=';', header=false, skip=1, columns={cols_sql},
                          null_padding=true, strict_mode=false, decimal_separator='.', quote='')
            WHERE TRIM(obis) IN ('{OBIS_IMPORT}', '{OBIS_EXPORT}')
              AND TRY_STRPTIME(TRIM(date), '%d.%m.%Y') IS NOT NULL
        ) TO '{tmp_dir}' (FORMAT PARQUET, PARTITION_BY (bucket), COMPRESSION ZSTD,
                          OVERWRITE_OR_IGNORE, FILENAME_PATTERN 'f{i:03d}_{{uuid}}')
        """
        try:
            con.execute(sql)
        except duckdb.OutOfMemoryException:
            print("DuckDB out of memory -> this file via pandas ...", end=" ", flush=True)
            _convert_one_pandas(f, tmp_dir, 10_000 + i * 1000)
        print(f"{time.time() - t:.0f} s", flush=True)
    con.close()


# -----------------------------------------------------------------------------
# step 1b: pandas conversion (fallback, ~10x slower)
# -----------------------------------------------------------------------------
def _convert_one_pandas(f: str, tmp_dir: str, part_base: int, chunksize: int = 250_000,
                        max_buffer_rows: int = 600_000) -> int:
    """Convert ONE monthly file (pandas, streaming): returns the number of meter-days written.
    Rows are kept per bucket in memory and flushed every ~max_buffer_rows rows (~400 MB)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    def flush(buf: dict, part_no: int) -> None:
        for b, frames in buf.items():
            g = pd.concat(frames, ignore_index=True)
            d = os.path.join(tmp_dir, f"bucket={int(b)}")
            os.makedirs(d, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(g, preserve_index=False),
                           os.path.join(d, f"data_{part_no}.parquet"), compression="snappy")
        buf.clear()

    lay = file_layout(f)
    names = layout_names(lay)
    dtypes = {c: "string" for c in names[:lay["n_lead"]]}
    dtypes.update({v: "float32" for v in VCOLS})
    has_meter, has_plz = lay["roles"]["meter"] is not None, lay["roles"]["plz"] is not None
    buf: dict = {}
    buffered = n_seen = n_kept = 0
    part_no = part_base
    for ch in pd.read_csv(f, sep=";", header=None, skiprows=1, names=names, dtype=dtypes, index_col=False,
                          na_values=["", " ", "-"], keep_default_na=False, chunksize=chunksize,
                          encoding="utf-8", encoding_errors="replace", engine="c"):
        obis = ch["obis"].astype(str).str.strip()
        keep = np.array(obis.isin([OBIS_IMPORT, OBIS_EXPORT]).values, copy=True)
        date = pd.to_datetime(ch["date"].astype(str).str.strip(), format="%d.%m.%Y", errors="coerce")
        keep &= date.notna().values
        n_seen += len(ch); n_kept += int(keep.sum())
        if n_seen >= 1000 and n_kept == 0:
            raise SystemExit(f"\nABORT: none of the first {n_seen} rows of {f} passed the OBIS/date filter.\n"
                             f"layout: {lay}\nfirst row as parsed: {ch.iloc[0, :6].tolist()}\n"
                             f"run  python build_store.py ROOT --check  to see the raw layout.")
        if not keep.any():
            continue
        ch, obis, date = ch[keep], obis[keep], date[keep]
        vals = ch[VCOLS].to_numpy(dtype="float32") * np.float32(4.0)         # kWh/15min -> kW
        out = pd.DataFrame({
            "mp_id": ch["mp_id"].astype(str).str.strip().values,
            "channel": np.where(obis.values == OBIS_IMPORT, "import", "export"),
            "date": date.dt.date.values,
            "plz": ch["plz"].astype(str).str.strip().values if has_plz else "",
            "meter": ch["meter"].astype(str).str.strip().values if has_meter else "",
        })
        out = pd.concat([out, pd.DataFrame(vals, columns=VCOLS)], axis=1)
        out["bucket"] = out["mp_id"].map(bucket_of)
        for b, g in out.groupby("bucket", sort=False):
            buf.setdefault(int(b), []).append(g.drop(columns="bucket"))
        buffered += len(out)
        if buffered >= max_buffer_rows:
            flush(buf, part_no); part_no += 1; buffered = 0
    if buf:
        flush(buf, part_no)
    return n_kept


def _job(args):
    f, tmp_dir, part_base, max_buffer_rows = args
    t = time.time()
    n = _convert_one_pandas(f, tmp_dir, part_base, max_buffer_rows=max_buffer_rows)
    return f, n, time.time() - t


def convert_pandas(files: List[str], tmp_dir: str, chunksize: int = 250_000, max_buffer_rows: int = 600_000,
                   part_offset: int = 0, jobs: int = 1, progress=None) -> None:
    """All files via the pandas path; `jobs` files in parallel processes (each ~1 GB peak)."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(f, tmp_dir, part_offset + i * 1000, max_buffer_rows) for i, f in enumerate(files)]
    t0 = time.time()
    done = 0
    if jobs <= 1:
        results = map(_job, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(_job, tasks)
    for f, n, secs in results:
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (len(files) - done)
        msg = (f"  [{done}/{len(files)}] {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)} ... "
               f"{n:,} meter-days, {secs:.0f} s | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")
        print(msg, flush=True)
        if progress:
            progress(msg)
    if jobs > 1:
        pool.shutdown()


# -----------------------------------------------------------------------------
# step 2: compact each bucket — sorted by house, small row groups
# -----------------------------------------------------------------------------
def compact(tmp_dir: str, store_dir: str, row_group_size: int = 16384) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(store_dir, exist_ok=True)
    stats = {"buckets": 0, "rows": 0, "houses": 0}
    dirs = sorted(glob.glob(os.path.join(tmp_dir, "bucket=*")), key=lambda d: int(d.rsplit("=", 1)[1]))
    for i, d in enumerate(dirs):
        b = int(d.rsplit("=", 1)[1])
        tab = pq.ParquetDataset(d).read()
        tab = tab.select([c for c in STORE_COLS if c in tab.column_names])
        # types: mp_id/channel/plz/meter as string, date as date32, values float32
        tab = tab.cast(pa.schema([
            pa.field("mp_id", pa.string()), pa.field("channel", pa.string()), pa.field("date", pa.date32()),
            pa.field("plz", pa.string()), pa.field("meter", pa.string()),
            *[pa.field(v, pa.float32()) for v in VCOLS]]))
        tab = tab.sort_by([("mp_id", "ascending"), ("channel", "ascending"), ("date", "ascending")])
        out_dir = os.path.join(store_dir, f"bucket={b}")
        os.makedirs(out_dir, exist_ok=True)
        pq.write_table(tab, os.path.join(out_dir, "part.parquet"), compression="snappy",
                       row_group_size=row_group_size, write_statistics=True)
        n_h = len(set(tab.column("mp_id").to_pylist()))
        stats["buckets"] += 1; stats["rows"] += tab.num_rows; stats["houses"] += n_h
        print(f"  bucket {b:3d}: {tab.num_rows:>9,} rows, {n_h:>5,} houses   [{i + 1}/{len(dirs)}]", flush=True)
    return stats


def check_file(path: str, n: int = 2000) -> None:
    """Print how the first rows of a file are read and how many survive the filters (seconds)."""
    import pandas as pd
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n"); first = fh.readline().rstrip("\r\n")
    print(f"file: {path}\n  header: {len(header.split(';'))} fields | {header[:70]} ... {header[-20:]}")
    print(f"  row 1 : {len(first.split(';'))} fields | {first[:70]} ... {first[-20:]}")
    lay = file_layout(path)
    print(f"  layout: {lay['n_lead']} leading columns, roles {lay['roles']}"
          f"{'  (trailing ; per line)' if lay['n_fields'] > lay['n_lead'] + N_SLOTS else ''}")
    names = layout_names(lay)
    df = pd.read_csv(path, sep=";", header=None, skiprows=1, names=names, dtype=str, index_col=False,
                     nrows=n, encoding="utf-8", encoding_errors="replace", engine="c")
    meter = df["meter"][0] if "meter" in df else None
    plz = df["plz"][0] if "plz" in df else None
    print(f"  parsed row 1: mp_id={df['mp_id'][0]!r} obis={df['obis'][0]!r} date={df['date'][0]!r} "
          f"meter={meter!r} plz={plz!r} v0={df['v0'][0]!r} v95={df['v95'][0]!r}")
    obis = df["obis"].astype(str).str.strip()
    ok_obis = obis.isin([OBIS_IMPORT, OBIS_EXPORT])
    ok_date = pd.to_datetime(df["date"].astype(str).str.strip(), format="%d.%m.%Y", errors="coerce").notna()
    print(f"  of the first {len(df)} rows: OBIS import/export = {int(ok_obis.sum())}, valid date = {int(ok_date.sum())}, "
          f"both = {int((ok_obis & ok_date).sum())}  ->  "
          + ("OK" if (ok_obis & ok_date).any() else "PROBLEM: nothing would be kept -> send this output"))


def check_all(files: List[str]) -> None:
    """Layout of every file (fast: reads two lines each) + the full check on the first file of each year."""
    print(f"{len(files)} files")
    seen_years = set()
    for f in files:
        lay = file_layout(f)
        year = os.path.basename(os.path.dirname(os.path.dirname(f)))
        tag = f"{lay['n_lead']} lead cols, meter {'yes' if lay['roles']['meter'] is not None else 'no '}, plz {'yes' if lay['roles']['plz'] is not None else 'no '}"
        print(f"  {os.path.relpath(f, os.path.commonpath(files)):70s} {tag}")
        seen_years.add(year)
    print()
    done = set()
    for f in files:
        year = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if year in done:
            continue
        done.add(year)
        check_file(f)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with the monthly CSVs")
    ap.add_argument("-o", "--store", default="aew_store", help="output store folder (default aew_store)")
    ap.add_argument("--month", help="only files whose path contains this text (e.g. '2023')")
    ap.add_argument("--pattern", default=FILE_GLOB)
    ap.add_argument("--threads", type=int, help="DuckDB threads (default 2)")
    ap.add_argument("--memory-limit", default="1.5GB", help="DuckDB memory limit (default 1.5GB; spills to disk)")
    ap.add_argument("--no-duckdb", action="store_true", help="use the pandas path (bounded memory, ~1 GB) instead of DuckDB")
    ap.add_argument("--buffer-rows", type=int, default=600_000, help="pandas path: rows buffered before flushing (default 600k ≈ 0.8 GB peak)")
    ap.add_argument("--jobs", type=int, default=1, help="pandas path: files converted in parallel (each ~1 GB RAM; use ~cores, RAM/1.5GB)")
    ap.add_argument("--progress", default="build_progress.txt", help="text file updated after every file (default build_progress.txt)")
    ap.add_argument("--check", action="store_true", help="only inspect the files' layouts (and the filters on one file per year), then exit")
    a = ap.parse_args(argv)

    files = find_files(a.root, a.month, a.pattern)
    if a.check:
        check_all(files); return

    def progress(msg: str) -> None:
        with open(a.progress, "a") as pf:
            pf.write(time.strftime("%H:%M:%S ") + msg.strip() + "\n")
    with open(a.progress, "w") as pf:
        pf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} build started: {len(files)} files -> {os.path.abspath(a.store)}\n")
    store = os.path.abspath(a.store)
    tmp = store + "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()

    use_duckdb = not a.no_duckdb
    if use_duckdb:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            print("duckdb not installed (pip install duckdb) -> using the slower pandas path", flush=True)
            use_duckdb = False
    print(f"step 1: converting {len(files)} files with {'DuckDB' if use_duckdb else f'pandas, {a.jobs} in parallel'} ...", flush=True)
    if use_duckdb:
        convert_duckdb(files, tmp, a.threads, a.memory_limit)
    else:
        convert_pandas(files, tmp, max_buffer_rows=a.buffer_rows, jobs=a.jobs, progress=progress)
    t1 = time.time()
    print(f"step 1 done in {t1 - t0:.0f} s\nstep 2: compacting buckets (sort by house) ...", flush=True)
    progress(f"step 1 done in {(t1 - t0) / 60:.1f} min; step 2 (compacting 256 buckets) running ...")

    if os.path.exists(store):
        shutil.rmtree(store)
    stats = compact(tmp, store)
    shutil.rmtree(tmp, ignore_errors=True)
    stats.update({"source_files": files, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "seconds": round(time.time() - t0), "n_buckets": N_BUCKETS})
    with open(os.path.join(store, "_manifest.json"), "w") as f:
        json.dump(stats, f, indent=1)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(store, "**", "*.parquet"), recursive=True))
    msg = (f"done: {stats['houses']:,} houses, {stats['rows']:,} meter-days, {size / 1e9:.2f} GB in {store}"
           f"  ({(time.time() - t0) / 60:.1f} min total)")
    print("\n" + msg); progress(msg)


if __name__ == "__main__":
    main()