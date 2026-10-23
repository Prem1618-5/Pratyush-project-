"""Battery-health monitoring CLI: estimate present capacity and SOH from per-cycle measurements.

    python -m soh.predict --demo                       # unseen secondary-test cell
    python -m soh.predict --input my_cycles.csv        # your own data
    python -m soh.predict --input my_cycles.csv --model main

Input CSV: one row per cycle with `cell_id`, `cycle` and the raw measurements the model
needs (for the default BMS model: IR, Tavg, Tmin, Tmax and the 1C/CV charge-segment
features produced by soh.data.mit_loader.charge_segment_features). Optional `QD` column
(measured capacity, Ah) is used only to report the error.

The input is cleaned with the SAME rules as the training data (soh.preprocess): cycle 1 and
impossible QD rows are dropped, IR glitches and temperature excursions become NaN and the
last good value is carried forward.

Output per cycle:  est_capacity_Ah, est_SOH_%, status
    SOH = present capacity / rated capacity x 100
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from soh import config, features, metrics, preprocess

# REPLACE uses the project's EOL alarm level (80.5 %), not 80 %: batch-3 cells stop at
# 80.0-80.15 % and would otherwise never be flagged (see config.EOL_ALARM_SOH).
STATUS = [(90.0, "HEALTHY"), (config.EOL_ALARM_SOH, "MONITOR"), (-np.inf, "REPLACE / SECOND-LIFE")]


def health_status(soh):
    for lo, name in STATUS:
        if soh >= lo:
            return name


def load_model(kind="bms"):
    return joblib.load(config.MODELS_DIR / f"soh_xgb_{kind}.joblib")


def estimate(cycles, kind="bms", smooth=5):
    """cycles: raw per-cycle measurements (one or many cells). Returns a copy with estimates."""
    bundle = load_model(kind)
    clip = joblib.load(config.MODELS_DIR / "train_clip_ranges.joblib")
    df, _ = preprocess.drop_invalid_rows(cycles.sort_values(["cell_id", "cycle"]))
    df = features.add_derived(preprocess.clean_measurements(df))
    feats = bundle["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"input lacks features: {missing}")
    X = df[feats].clip(pd.Series(clip["lo"])[feats], pd.Series(clip["hi"])[feats], axis=1)
    df["est_SOH_%"] = bundle["model"].predict(X)
    # a BMS would display a lightly smoothed value; raw per-cycle estimates are noisy
    df["est_SOH_smoothed_%"] = df.groupby("cell_id")["est_SOH_%"].transform(
        lambda s: s.rolling(smooth, min_periods=1).median())
    df["est_capacity_Ah"] = df["est_SOH_smoothed_%"] / 100 * bundle["rated_ah"]
    df["status"] = df["est_SOH_smoothed_%"].map(health_status)
    cols = ["cell_id", "cycle", "est_capacity_Ah", "est_SOH_%", "est_SOH_smoothed_%", "status"]
    if "QD" in df:
        df["measured_SOH_%"] = df["QD"] / bundle["rated_ah"] * 100
        cols.append("measured_SOH_%")
    return df[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CSV of per-cycle measurements")
    ap.add_argument("--demo", action="store_true", help="run on an unseen secondary-test cell")
    ap.add_argument("--model", default="bms", choices=["bms", "main"],
                    help="bms = charge/IR/temperature only (deployable); main = adds lab dQ(V) features")
    ap.add_argument("--out", default="soh_estimates.csv")
    a = ap.parse_args()
    if a.demo:
        raw = pd.read_parquet(config.PROCESSED_DIR / "mit_cycles_raw.parquet")
        cell = sorted(raw.loc[raw["batch"] == "2018-04-12", "cell_id"].unique())[0]
        cycles = raw[(raw["cell_id"] == cell) & (raw["cycle"] > 1)].drop(columns=["cell_index"])
    else:
        cycles = pd.read_csv(a.input)
    out = estimate(cycles, a.model)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(out.iloc[np.linspace(0, len(out) - 1, 12).astype(int)].round(3).to_string(index=False))
    if "measured_SOH_%" in out:
        rmse = metrics.rmse(out["measured_SOH_%"], out["est_SOH_%"])
        print(f"\nRMSE vs measured: {rmse:.2f} % SOH over {len(out)} cycles")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
