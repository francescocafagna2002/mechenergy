"""
list_store_ids.py — MP IDs of the houses in the first N buckets of the binary store
(aew_store/bucket=0, bucket=1, ...) written to a CSV.

Run (inside the Renku session, from ~/work/mechenergy):
    python list_store_ids.py                                  # buckets 0..29 -> ids_first_30_buckets.csv
    python list_store_ids.py --buckets 30 --max-houses 10000  # same, cut at exactly 10 000 houses

Output (one column, one row per house):
    mp_id
    10028
    ...
The file can be fed straight back:   python pv20_features.py --store aew_store --mp-ids-file ids_first_30_buckets.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import pyarrow.dataset as ds

STORE_DEFAULT = "/home/renku/work/mechenergy/aew_store"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE_DEFAULT, help=f"store folder (default {STORE_DEFAULT})")
    ap.add_argument("--buckets", type=int, default=30, help="how many buckets, starting at bucket=0 (default 30)")
    ap.add_argument("--max-houses", type=int, help="keep only the first N houses of the list (default: all)")
    ap.add_argument("-o", "--out", default="ids_first_30_buckets.csv")
    a = ap.parse_args()

    rows = []
    for b in range(a.buckets):
        d = os.path.join(a.store, f"bucket={b}")
        if not os.path.isdir(d):
            print(f"bucket={b}: missing, skipped", flush=True)
            continue
        # only the mp_id column is read, so this is fast even for big buckets
        col = ds.dataset(d, format="parquet").to_table(columns=["mp_id"]).column("mp_id")
        ids = sorted(set(str(m).strip() for m in col.to_pylist()))   # same order as pv20_features.first_store_ids
        rows += ids
        print(f"bucket={b}: {len(ids):,} houses   (total {len(rows):,})", flush=True)

    out = pd.DataFrame({"mp_id": rows})
    if a.max_houses:
        out = out.head(a.max_houses)
    out.to_csv(a.out, index=False)
    print(f"\n{len(out):,} MP IDs from buckets 0..{a.buckets - 1} -> {a.out}")


if __name__ == "__main__":
    main()