"""Stage 7: evaluation metrics (all in % SOH unless stated)."""
import numpy as np
import pandas as pd

from soh import config


def rmse(y, p):
    """The one RMSE used everywhere (report, figures, permutation importance, CLI)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "RMSE": rmse(y, p),
        "MAE": float(np.mean(np.abs(e))),
        "MAPE": float(np.mean(np.abs(e) / y) * 100),
        "R2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "MaxAE": float(np.max(np.abs(e))),
    }


def band_rmse(y, p, bands=((80, 85), (85, 90), (90, 95), (95, 200))):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {}
    for lo, hi in bands:
        m = (y >= lo) & (y < hi)
        name = f"RMSE_{lo}-{hi if hi < 200 else 'up'}"
        out[name] = rmse(y[m], p[m]) if m.sum() else np.nan
    return out


def first_crossing(cycles, values, threshold):
    idx = np.where(np.asarray(values) < threshold)[0]
    return float(np.asarray(cycles)[idx[0]]) if len(idx) else np.nan


def eol_errors(df, pred_col="pred", label="soh", eol=config.EOL_ALARM_SOH):
    """Per cell: (first cycle predicted SOH < alarm) - (first cycle true SOH < alarm), in cycles.
    Negative = the BMS would raise the end-of-life alarm early (safe side).
    Cells where only one of the two crosses have no error: full_report counts them as missed
    alarms (truth crossed, estimate did not) or false alarms (estimate crossed, truth did not)."""
    rows = []
    for cid, d in df.sort_values("cycle").groupby("cell_id"):
        t = first_crossing(d["cycle"], d[label], eol)
        p = first_crossing(d["cycle"], d[pred_col], eol)
        rows.append({"cell_id": cid, "true_eol": t, "pred_eol": p, "eol_error": p - t})
    return pd.DataFrame(rows)


def per_cell_metrics(df, pred_col="pred", label="soh"):
    rows = [{"cell_id": c, **regression_metrics(d[label], d[pred_col]), "n": len(d)}
            for c, d in df.groupby("cell_id")]
    return pd.DataFrame(rows)


def full_report(df, pred_col="pred", label="soh"):
    """All headline numbers for one model on one test set."""
    m = regression_metrics(df[label], df[pred_col])
    m.update(band_rmse(df[label], df[pred_col]))
    pc = per_cell_metrics(df, pred_col, label)
    m["cell_RMSE_mean"], m["cell_RMSE_std"] = pc["RMSE"].mean(), pc["RMSE"].std()
    m["cell_RMSE_median"], m["cell_RMSE_worst"] = pc["RMSE"].median(), pc["RMSE"].max()
    eol = eol_errors(df, pred_col, label)
    ok = eol["eol_error"].notna()
    m["EOL_abs_err_mean"] = float(eol.loc[ok, "eol_error"].abs().mean()) if ok.any() else np.nan
    m["EOL_err_signed_mean"] = float(eol.loc[ok, "eol_error"].mean()) if ok.any() else np.nan
    m["EOL_undefined_cells"] = int((~ok & eol["true_eol"].notna()).sum())   # missed alarms
    # false alarms: estimate crossed the alarm level, the cell never did (premature replacement)
    m["EOL_false_alarm_cells"] = int((eol["pred_eol"].notna() & eol["true_eol"].isna()).sum())
    return m
