"""Stage 2: cycle-level cleaning and labelling.

Rules (doc Table 5). Hard physical filters apply to every cell; the QD-spike flag
only removes rows from TRAINING (test labels are never cleaned or smoothed).
"""
import numpy as np
import pandas as pd

from soh import config


def _rolling_spike(s, window, k, floor):
    """Flag points far from a centred rolling median: |resid| > max(k * MAD, floor).
    The absolute floor matters: capacity is so smooth that its MAD is ~0, and K*MAD alone
    would flag ordinary measurement noise (tried: 31 % of rows flagged without a floor)."""
    med = s.rolling(window, center=True, min_periods=3).median()
    resid = s - med
    mad = 1.4826 * (resid - resid.median()).abs().median()
    return (resid.abs() > max(k * mad, floor)).fillna(False)


def drop_invalid_rows(df):
    """Rules 1-2, applied everywhere (training AND soh.predict). QD is optional (a BMS may not have it)."""
    log = {}
    # 1) cycle 1 is unavailable / zero in one batch -> cycle 2 is the first real cycle
    drop = df["cycle"] == 1
    log["drop_cycle1"] = int(drop.sum())
    # 2) physically impossible capacity for a 1.1 Ah cell
    bad_qd = (~df["QD"].between(config.QD_MIN_AH, config.QD_MAX_AH, inclusive="neither")
              if "QD" in df else pd.Series(False, index=df.index))
    log["drop_impossible_QD"] = int((bad_qd & ~drop).sum())
    return df[~(drop | bad_qd)].copy(), log


def clean_measurements(df):
    """Rules 4-5, applied everywhere (training AND soh.predict), so a deployed model sees inputs
    cleaned exactly like its training data. Bad values become NaN; add_derived carries the last
    good value forward."""
    df = df.copy()
    g = df.groupby("cell_id", sort=False)
    # 4) IR glitches: IR <= 0 or spikes -> NaN, then carry last good value (causal)
    ir_bad = (df["IR"] <= 0) | g["IR"].transform(lambda s: _rolling_spike(s, 11, 8, 0.002)).astype(bool)
    df["flag_ir_bad"] = ir_bad
    df.loc[ir_bad, "IR"] = np.nan
    # 5) temperature probe problems / chamber excursions
    t_bad = g["Tavg"].transform(lambda s: _rolling_spike(s, 11, 8, 2.0)).astype(bool)
    df["flag_temp_bad"] = t_bad
    df.loc[t_bad, ["Tavg", "Tmin", "Tmax"]] = np.nan
    return df


def soh_init(df, qd="QD", n=config.BASELINE_CYCLES):
    """Capacity relative to the cell when new = median of its first n cycles (robust to one noisy
    first measurement; same baseline length as the d_/r_ features)."""
    base = df.groupby("cell_id", sort=False)[qd].transform(lambda s: s.iloc[:n].median())
    return df[qd] / base * 100.0


def clean_mit(df, rated_ah=config.MIT_RATED_AH):
    df = df.sort_values(["cell_index", "cycle"]).reset_index(drop=True)
    log = {"rows_raw": len(df)}

    df, log_rows = drop_invalid_rows(df)
    log.update(log_rows)

    g = df.groupby("cell_id", sort=False)
    # 3) QD spikes (flag only; removed from training rows later). Regeneration survives K=5.
    df["flag_qd_outlier"] = g["QD"].transform(
        lambda s: _rolling_spike(s, config.SPIKE_WINDOW, config.SPIKE_K, config.SPIKE_FLOOR_AH)).astype(bool)
    log["flag_qd_outlier"] = int(df["flag_qd_outlier"].sum())
    df = clean_measurements(df)
    log["flag_ir_bad"] = int(df["flag_ir_bad"].sum())
    log["flag_temp_bad"] = int(df["flag_temp_bad"].sum())

    # labels
    df["soh"] = df["QD"] / rated_ah * 100.0
    df["soh_init"] = soh_init(df)
    log["rows_clean"] = len(df)
    log["cells"] = int(df["cell_id"].nunique())
    return df.reset_index(drop=True), log
