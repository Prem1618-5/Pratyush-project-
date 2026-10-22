"""Stage 3: derived features, the feature catalogue, feature sets and the leakage audit.

Golden rule: a feature must not secretly BE the label. Every MIT cell is discharged
identically (4C to 2.0 V), so anything measured on that discharge is capacity in disguise.
Each feature carries a prior tag from physics (safe / caution / excluded), and the
empirical audit below double-checks it on training cells only.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from soh import config

# name: (family, prior status, plain-language meaning / reason)
CATALOGUE = {
    "IR":            ("F1 resistance", "safe", "Internal resistance rises as electrodes/electrolyte age"),
    "Tavg":          ("F2 temperature", "safe", "Aged cells self-heat more for the same current"),
    "Tmin":          ("F2 temperature", "safe", "Cycle minimum can temperature"),
    "Tmax":          ("F2 temperature", "safe", "Cycle maximum can temperature"),
    "Trange":        ("F2 temperature", "safe", "Tmax - Tmin: heating swing within a cycle"),
    "d_IR":          ("F3 change vs new", "safe", "IR minus the cell's own early-life baseline"),
    "d_Tavg":        ("F3 change vs new", "safe", "Tavg minus early-life baseline"),
    "chg_v_start":   ("F4 1C charge segment", "caution", "Lowest voltage in the 1C phase (after relaxation): overpotential + true SOC"),
    "chg_t_cv":      ("F4 1C charge segment", "caution", "Duration of the 3.6 V constant-voltage taper (kinetics)"),
    "chg_q_cv":      ("F4 1C charge segment", "caution", "Charge accepted during the CV taper"),
    "chg_cv_tau":    ("F4 1C charge segment", "caution", "Time for CV current to halve from 1C (diffusion/resistance)"),
    "chg_dT":        ("F4 1C charge segment", "caution", "Temperature change over the 1C+CV segment"),
    "d_chg_v_start": ("F3 change vs new", "caution", "chg_v_start minus baseline"),
    "d_chg_t_cv":    ("F3 change vs new", "caution", "chg_t_cv minus baseline"),
    "d_chg_cv_tau":  ("F3 change vs new", "caution", "chg_cv_tau minus baseline"),
    "chg_ic_peak":   ("F5 charge IC", "caution", "Height of dQ/dV peak on the 1C segment (graphite staging)"),
    "chg_ic_peak_v": ("F5 charge IC", "caution", "Voltage of that peak; shifts with LLI/LAM"),
    "d_chg_ic_peak": ("F3 change vs new", "caution", "chg_ic_peak minus baseline"),
    "dq_var_log":    ("F6 partial dQ(V)", "caution", "log10 var of Q_k(V)-Q_10(V), 3.3-2.9 V (Severson's key feature)"),
    "dq_min_log":    ("F6 partial dQ(V)", "caution", "log10 |min| of the same curve difference"),
    "dq_mean":       ("F6 partial dQ(V)", "caution", "Mean of the curve difference in the partial window"),
    "chargetime":    ("F7 protocol", "ablation", "Total charge time: mostly encodes which of 72 policies was used"),
    "cycle":         ("F9 age", "ablation", "Cycle count: shortcut, lifetimes span 150-2,300 cycles"),
    # ---- label-equivalent: excluded by definition (kept only to show the audit catches them)
    "QD":            ("X1 label", "excluded", "IS the label"),
    "QC":            ("X2 charge capacity", "excluded", "Coulombic efficiency ~1 so QC ~ QD"),
    "dis_time":      ("X3 discharge time", "excluded", "Q = I*t with I fixed at 4.4 A"),
    "dis_energy":    ("X4 discharge energy", "excluded", "Capacity x nearly constant LFP voltage"),
    "chg_t_cc":      ("X* 1C CC duration", "excluded", "Charge starts from empty, fast phase stops at 0.88 Ah, so 1C time ~ (QC-0.88)/1.1 A"),
    "chg_q_cc":      ("X* 1C CC charge", "excluded", "= QC - 0.88 Ah - CV charge: capacity in disguise"),
}

BASELINE_COLS = ["IR", "Tavg", "chg_v_start", "chg_t_cv", "chg_cv_tau", "chg_ic_peak"]
ROLL_COLS = ["IR", "Tavg", "Tmax", "chg_v_start", "chg_t_cv", "chg_cv_tau", "chg_ic_peak",
             "chg_ic_peak_v", "chg_dT", "dq_var_log", "dq_min_log", "dq_mean"]
TRANSFER_COLS = ["r_IR", "r_chg_t_cv", "r_chg_cv_tau"]   # dimensionless, exist in MIT and NASA

for c in ROLL_COLS:
    fam, status, mean = CATALOGUE[c]
    CATALOGUE[f"{c}_rm5"] = ("F8 history", status, f"5-cycle causal rolling mean of {c}")


def add_derived(df, baseline_cycles=config.BASELINE_CYCLES):
    """Per-cell derived features. Every operation is grouped by cell, so no window spans two cells."""
    df = df.sort_values(["cell_id", "cycle"]).copy()
    g = df.groupby("cell_id", sort=False)
    feat_cols = [c for c in CATALOGUE if c in df.columns and c not in ("QD", "cycle")]
    # carry the last good measurement forward inside a cell (what a BMS would do)
    df[feat_cols] = g[feat_cols].ffill()
    if {"Tmax", "Tmin"} <= set(df.columns):
        df["Trange"] = df["Tmax"] - df["Tmin"]
    g = df.groupby("cell_id", sort=False)
    for c in [c for c in BASELINE_COLS if c in df.columns]:
        base = g[c].transform(lambda s: s.dropna().iloc[:baseline_cycles].median()
                              if s.notna().any() else np.nan)
        df[f"d_{c}"] = df[c] - base
        df[f"r_{c}"] = df[c] / base
    for c in [c for c in ROLL_COLS if c in df.columns]:
        df[f"{c}_rm5"] = g[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return df.sort_values(["cell_index", "cycle"]).reset_index(drop=True) if "cell_index" in df else df


# ---- Ablation ladder (doc Table 15) --------------------------------------
A1 = ["IR", "Tavg", "Tmin", "Tmax", "Trange", "d_IR", "d_Tavg"]
A2 = A1 + ["chg_v_start", "chg_t_cv", "chg_q_cv", "chg_cv_tau", "chg_dT", "chg_ic_peak",
           "chg_ic_peak_v", "d_chg_v_start", "d_chg_t_cv", "d_chg_cv_tau", "d_chg_ic_peak"]
A3 = A2 + ["dq_var_log", "dq_min_log", "dq_mean"]
A4 = A3 + [f"{c}_rm5" for c in ROLL_COLS]
FEATURE_SETS = {
    "A0_cycle_only": ["cycle"],
    "A1_IR_temp": A1,
    "A2_bms_charge": A2,
    "A3_plus_dQV": A3,
    "A4_plus_history": A4,
    "A5_A4+chargetime": A4 + ["chargetime"],
    "A6_A4+cycle": A4 + ["cycle"],
}
MAIN_SET = "A4_plus_history"        # headline model
BMS_SET = "A2_bms_charge"           # deployable without a full lab discharge


def clip_to_train(df, cols, train_mask, q=0.005):
    """Clip every feature to the [0.5, 99.5] percentile range of TRAINING rows.
    Fit on train only (no leakage); protects linear/neural models from garbage cycles
    (e.g. a 500-min CV phase) and from out-of-range values in a new batch."""
    df = df.copy()
    lo, hi = df.loc[train_mask, cols].quantile(q), df.loc[train_mask, cols].quantile(1 - q)
    df[cols] = df[cols].clip(lo, hi, axis=1)
    return df, {"lo": lo.to_dict(), "hi": hi.to_dict()}


def apply_audit(feature_sets, audit):
    """Drop any feature the audit marked as excluded from every feature set."""
    bad = set(audit.loc[audit["status"].str.startswith("EXCLUDED"), "feature"])
    return {k: [f for f in v if f not in bad] for k, v in feature_sets.items()}


def leakage_audit(train, label="soh", margin=config.LEAK_MARGIN):
    """Score each candidate feature ALONE on training cells.

    - spearman_pooled : monotonic correlation over all training rows
    - spearman_within : median correlation computed inside each cell (ageing, not cell offsets)
    - r2_single_cv    : cross-cell R^2 of a 1-feature tree (GroupKFold by cell). If one number
                        alone predicts SOH of unseen cells almost perfectly, it is the label.

    The threshold is calibrated, not guessed: discharge time/energy and the 1C-CC amount are
    label-equivalent BY DEFINITION, yet a 1-feature tree only reaches R^2 ~0.94 with them
    (measurement noise caps it). Any candidate scoring as high as the weakest known leak is
    treated as a leak too: threshold = min(R^2 of definitional leaks other than QD/QC) - margin.
    """
    rows, groups, y = [], train["cell_id"].to_numpy(), train[label].to_numpy()
    cands = [c for c in CATALOGUE if c in train.columns]
    for c in cands:
        x = train[c].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 10 else np.nan
        within = train[ok].groupby("cell_id").apply(
            lambda d: spearmanr(d[c], d[label]).statistic if d[c].nunique() > 2 else np.nan,
            include_groups=False)
        tree = DecisionTreeRegressor(min_samples_leaf=50, max_depth=8, random_state=0)
        pred = cross_val_predict(tree, x[ok, None], y[ok], groups=groups[ok], cv=GroupKFold(5))
        r2 = 1 - np.sum((y[ok] - pred) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        fam, prior, meaning = CATALOGUE[c]
        rows.append(dict(feature=c, family=fam, prior=prior, spearman_pooled=rho,
                         spearman_within=np.nanmedian(within), r2_single_cv=r2,
                         coverage=ok.mean(), meaning=meaning))
    a = pd.DataFrame(rows)
    known = a[(a["prior"] == "excluded") & ~a["feature"].isin(["QD", "QC"])]["r2_single_cv"]
    threshold = float(known.min() - margin)

    def status(r):
        if r["prior"] == "excluded":
            return "EXCLUDED (physics/definition)"
        if r["r2_single_cv"] >= threshold:
            return f"EXCLUDED (empirical: R2 >= {threshold:.3f})"
        return r["prior"]
    a.insert(a.columns.get_loc("meaning"), "status", a.apply(status, axis=1))
    a.attrs["threshold"] = threshold
    return a
