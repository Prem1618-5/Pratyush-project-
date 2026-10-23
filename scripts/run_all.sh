#!/usr/bin/env bash
# End-to-end: download -> parse -> experiments -> report.  Run from the repo root.
# Skip the first two steps if data/processed/mit_cycles_raw.parquet already exists (it is committed).
set -euo pipefail
if [ ! -f data/processed/mit_cycles_raw.parquet ]; then
  python -m soh.data.download          # ~8.5 GB, resumable
  python -m soh.data.mit_loader        # ~5 min, 3 processes
fi
for c in B0005 B0006 B0007 B0018; do   # all four, so a half-finished unzip is repaired
  [ -f data/raw/nasa/$c.mat ] || { python -m soh.data.download --nasa-only; break; }
done
python -m soh.experiments              # ~20 min on 4 CPU cores (measured 18 min) (add --quick for a ~10 min smoke run)
python -m soh.report                   # writes reports/RESULTS.md
pytest -q
