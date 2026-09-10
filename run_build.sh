#!/bin/bash
# Background build of the AEW binary store. Run from ~/work/mechenergy:
#     bash run_build.sh
# Then watch progress with:   cat build_progress.txt     or    tail -f build.log
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-4}                     # files converted in parallel: ~1 core and ~1.2 GB RAM each
nohup python build_store.py ../aew-data/test-blob/input_data -o aew_store --no-duckdb --jobs "$JOBS" \
      --progress build_progress.txt > build.log 2>&1 &
echo "started (pid $!). progress: build_progress.txt   full log: build.log"