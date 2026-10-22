"""Stage 4: splitting. The CELL, not the cycle, is the unit of splitting.

Official Severson split over the 124 cells in loader order (batch1, batch2, batch3):
  train          = indices 1, 3, ..., 81          (41 cells)
  primary test   = indices 0, 2, ..., 82  + 83     (43 cells)
  secondary test = indices 84 .. 123               (40 cells, batch 3, cycled ~1 year later)
"""
import numpy as np

TRAIN_IDX = np.arange(1, 83, 2)
PRIMARY_IDX = np.hstack((np.arange(0, 84, 2), 83))
SECONDARY_IDX = np.arange(84, 124)


def official_split(df):
    split = np.full(len(df), "", dtype=object)
    ci = df["cell_index"].to_numpy()
    split[np.isin(ci, TRAIN_IDX)] = "train"
    split[np.isin(ci, PRIMARY_IDX)] = "test_primary"
    split[np.isin(ci, SECONDARY_IDX)] = "test_secondary"
    assert (split != "").all(), "every cell must belong to exactly one split"
    return split


def random_cell_split(cells, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.array(sorted(cells))
    test = rng.choice(cells, size=int(round(len(cells) * test_frac)), replace=False)
    return np.setdiff1d(cells, test), test


def random_row_split(n, test_frac=0.35, seed=0):
    """DELIBERATELY WRONG split (leakage demo only): neighbouring cycles of one cell
    land on both sides, so the model can 'remember' the cell."""
    rng = np.random.default_rng(seed)
    test = rng.random(n) < test_frac
    return ~test, test


def validation_cells(train_cells, n_val=8, seed=0):
    """Hold out a few TRAINING cells for early stopping (never a test cell)."""
    rng = np.random.default_rng(seed)
    train_cells = np.array(sorted(train_cells))
    val = rng.choice(train_cells, size=n_val, replace=False)
    return np.setdiff1d(train_cells, val), val
