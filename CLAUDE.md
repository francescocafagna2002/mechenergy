# MechEnergy

## Data — CRITICAL RULE
The dataset lives in `../aew-data` (~200 GB, read-only).
NEVER read the data files directly — they would blow up the context window.

To understand the data's structure and columns, read these files in the repo root instead:
- `DATA_MANIFEST.txt` — folder structure and file sizes
- `DATA_HEADERS.txt` — column headers (first line of each file only)

Only open an actual data file if I explicitly ask you to, and even then use
sampling (`head`, `shuf`), never the whole file.

## Folders
- `../aew-data` — input data, READ-ONLY. Do not write here.
- `../store` — write results and outputs here (persistent storage).
- `mechenergy/` (this repo) — code, versioned with Git.

## Workflow
- Results and generated files go in `../store`, not in the repo.
- Commit and push code changes to the shared repo, but let me review before pushing.
