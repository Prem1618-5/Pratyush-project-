"""Central configuration: paths, dataset constants and fixed project choices.

Every number here is a *project choice* (documented in docs/) or a dataset fact
from Severson et al. 2019 / the NASA PCoE README.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SOH_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# ---- MIT-Stanford-TRI (Severson 2019) -------------------------------------
MIT_RATED_AH = 1.1          # A123 APR18650M1A nominal capacity
MIT_EOL_SOH = 80.0          # EOL = first cycle with QD < 0.88 Ah
# Batch-3 cells were stopped AT 0.88 Ah (their minimum SOH is 80.0-80.15 %), so a strict
# '< 80 %' crossing never happens there. EOL detection therefore uses an 80.5 % alarm level.
EOL_ALARM_SOH = 80.5
MIT_FILES = {
    "2017-05-12": ("2017-05-12_batchdata_updated_struct_errorcorrect.mat",
                   "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
                   3_025_320_241),
    "2017-06-30": ("2017-06-30_batchdata_updated_struct_errorcorrect.mat",
                   "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
                   2_007_331_155),
    "2018-04-12": ("2018-04-12_batchdata_updated_struct_errorcorrect.mat",
                   "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
                   3_236_690_412),
}
# Official cleaning from the authors' "Load Data.ipynb"
BATCH1_DROP = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]       # never reach 80 %
BATCH1_STITCH = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]          # continued in batch 2 ...
BATCH2_STITCH = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]        # ... by these cells
BATCH3_DROP = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]  # noisy channels

# Qdlin is Q(V) on 1000 points, 3.5 V -> 2.0 V. Partial window for dQ(V) features.
QDLIN_V = (3.5, 2.0, 1000)
DQ_WINDOW_V = (3.3, 2.9)    # partial window; never the full window ending at 2.0 V (= label)
DQ_REF_CYCLE = 10           # Severson's reference cycle for dQ(V) = Q_k(V) - Q_10(V)

# ---- NASA PCoE ------------------------------------------------------------
NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_ZIP_BYTES = 209_708_670
NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
NASA_RATED_AH = 2.0

# ---- Cleaning choices (project choices, reported in RESULTS.md) -----------
QD_MIN_AH, QD_MAX_AH = 0.0, 1.2     # physically impossible outside this range
SPIKE_WINDOW = 11                   # rolling-median window (cycles)
SPIKE_K = 5.0                       # |QD - median| > max(K * MAD, floor) -> flagged spike
SPIKE_FLOOR_AH = 0.005              # 0.45 % SOH; flags ~0.24 % of cycles (OSBAD: ~0.47 % anomalies)
BASELINE_CYCLES = 10                # baseline = median of first 10 valid cycles

SEEDS = [0, 1, 2]
# Empirical leak threshold is CALIBRATED on known disguised labels (see features.leakage_audit):
# threshold = (lowest single-feature R^2 among definitional leaks, excluding QD/QC) - margin
LEAK_MARGIN = 0.01


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
