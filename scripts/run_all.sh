#!/usr/bin/env bash
# End-to-end: download -> parse -> experiments -> report.  Run from the repo root.
# Skip the first two steps if data/processed/mit_cycles_raw.parquet already exists (it is committed).
set -euo pipefail
if [ ! -f data/processed/mit_cycles_raw.parquet ]; then
  python -m soh.data.download          # ~8.5 GB, resumable
  python -m soh.data.mit_loader        # ~5 min, 3 processes
fi
[ -f data/raw/nasa/B0005.mat ] || python -m soh.data.download --nasa-only
python -m soh.experiments              # ~35-45 min on 4 CPU cores (add --quick for a ~10 min smoke run)
python -m soh.report                   # writes reports/RESULTS.md
pytest -q
