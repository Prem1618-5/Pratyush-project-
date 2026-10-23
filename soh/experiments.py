"""Run every experiment in the project plan and save results to reports/results/.

    python -m soh.experiments              # everything (~20 min on 4 CPU cores; measured 18 min)
    python -m soh.experiments --quick      # 1 seed, short deep training (smoke test)

Then build the written report with:  python -m soh.report
"""
import argparse
import io
import json
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost

from soh import config, explain, features, metrics, plots, preprocess, splits
from soh.data import nasa_loader
from soh.models import deep, tabular

RES = config.REPORTS_DIR / "results"
LABEL = "soh"


def _save_json(obj, name):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / name).write_text(json.dumps(obj, indent=2, default=float))


def _model(name, seed, xgb_params):
    """Untrained model; XGBoost gets the parameters tuned once on the main feature set."""
    return tabular.make_model(name, seed, **(xgb_params if name == "xgb" else {}))


def _agg(reports):
    """mean and std over seeds for every metric."""
    df = pd.DataFrame(reports)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=0))} for k in df.columns}


# ==========================================================================
# 1. data preparation (Stages 2-4)
# ==========================================================================
def prepare():
    raw = pd.read_parquet(config.PROCESSED_DIR / "mit_cycles_raw.parquet")
    df, clean_log = preprocess.clean_mit(raw)
    df = features.add_derived(df)
    df["split"] = splits.official_split(df)
    train_rows = (df["split"] == "train") & ~df["flag_qd_outlier"]
    audit = features.leakage_audit(df[train_rows])
    fsets = features.apply_audit(features.FEATURE_SETS, audit)
    used = sorted({f for v in fsets.values() for f in v} | set(features.TRANSFER_COLS))
    df, clip = features.clip_to_train(df, used, train_rows.to_numpy())
    joblib.dump(clip, config.MODELS_DIR / "train_clip_ranges.joblib")
    audit.to_csv(RES / "leakage_audit.csv", index=False)
    clean_log["leak_threshold"] = audit.attrs["threshold"]
    clean_log["rows_per_split"] = df["split"].value_counts().to_dict()
    clean_log["cells_per_split"] = df.groupby("split")["cell_id"].nunique().to_dict()
    _save_json({"cleaning": clean_log, "feature_sets": fsets}, "data_summary.json")
    df.to_parquet(config.PROCESSED_DIR / "mit_features.parquet", index=False)

    plots.capacity_fade(df)
    tr = df[train_rows]
    safe = audit[~audit["status"].str.startswith("EXCLUDED") & ~audit["feature"].str.endswith("_rm5")
                 & ~audit["prior"].eq("ablation")]
    top = safe.reindex(safe["spearman_pooled"].abs().sort_values(ascending=False).index)["feature"][:5]
    plots.feature_scatter(tr, list(top))
    return df, fsets, train_rows


def split_frames(df, train_rows):
    return (df[train_rows], df[df["split"] == "test_primary"], df[df["split"] == "test_secondary"])


# ==========================================================================
# 2. tabular models (Stage 5) + 3. evaluation (Stage 7)
# ==========================================================================
def fit_eval_tabular(name, feats, tr, tests, seeds, params):
    """Fit one model per seed; return aggregated metrics per test set + last predictions."""
    per_set, preds, model = {k: [] for k in tests}, {}, None
    for seed in seeds:
        model = tabular.make_model(name, seed, **params).fit(tr[feats], tr[LABEL])
        for k, te in tests.items():
            p = te[["cell_id", "cycle", LABEL]].copy()
            p["pred"] = model.predict(te[feats])
            per_set[k].append(metrics.full_report(p))
            preds[k] = p
    return {k: _agg(v) for k, v in per_set.items()}, preds, model


def run_tabular(df, fsets, train_rows, seeds):
    tr, prim, sec = split_frames(df, train_rows)
    tests = {"test_primary": prim, "test_secondary": sec}
    groups = tr["cell_id"].to_numpy()
    feats = fsets[features.MAIN_SET]
    out, tuned, store = {}, {}, {}
    plan = [("Dummy (mean)", "dummy", features.MAIN_SET), ("XGBoost", "xgb", "A0_cycle_only"),
            ("Ridge", "ridge", features.MAIN_SET), ("Random Forest", "rf", features.MAIN_SET),
            ("XGBoost", "xgb", features.MAIN_SET)]
    for label, name, fset in plan:
        f = fsets[fset]
        t0 = time.time()
        tune = tabular.tune(name, tr[f], tr[LABEL].to_numpy(), groups) if name != "dummy" else {}
        params = tune.get("params", {})
        tuned[f"{name}|{fset}"] = tune
        s = seeds if name in ("rf", "xgb") else seeds[:1]        # dummy/ridge are deterministic
        res, preds, model = fit_eval_tabular(name, f, tr, tests, s, params)
        key = f"{label} | {fset}"
        out[key] = res
        store[key] = (preds, model, f, params)
        print(f"[tabular] {key:40s} primary RMSE {res['test_primary']['RMSE']['mean']:.3f}  "
              f"secondary {res['test_secondary']['RMSE']['mean']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
    _save_json({"results": out, "tuning": tuned, "features": feats}, "tabular.json")
    return out, store, tuned


# ==========================================================================
# 4. deep models on cycle windows (Stage 6)
# ==========================================================================
def run_deep(df, fsets, train_rows, seeds, ks=(5, 10, 20, 30), epochs=40):
    tr, prim, sec = split_frames(df, train_rows)
    feats = fsets["A3_plus_dQV"]          # windows already carry history -> no rolling means
    fit_cells, val_cells = splits.validation_cells(tr["cell_id"].unique(), n_val=8, seed=0)
    out, footprint = {}, {}

    def run(name, k, seed_list):
        Xa, ya, _ = deep.build_windows(tr[tr["cell_id"].isin(fit_cells)], feats, k)
        Xv, yv, _ = deep.build_windows(tr[tr["cell_id"].isin(val_cells)], feats, k)
        per_set, last = {"test_primary": [], "test_secondary": []}, None
        for seed in seed_list:
            t0 = time.time()
            m = deep.train_net(name, Xa, ya, Xv, yv, seed=seed, epochs=epochs)
            for key, te in [("test_primary", prim), ("test_secondary", sec)]:
                Xt, _, idx = deep.build_windows(te, feats, k)
                p = te.loc[idx, ["cell_id", "cycle", LABEL]].copy()
                p["pred"] = deep.predict_net(m, Xt)
                per_set[key].append(metrics.full_report(p))
            print(f"[deep] {name} k={k} seed={seed}: {m['epochs']} epochs, val RMSE "
                  f"{m['val_rmse']:.3f}, primary {per_set['test_primary'][-1]['RMSE']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            last = m
        return {k_: _agg(v) for k_, v in per_set.items()}, last

    for name, label in [("lstm", "LSTM"), ("gru", "GRU"), ("cnn", "1D-CNN")]:
        res, m = run(name, 10, seeds)
        out[f"{label} (k=10) | A3 windows"] = res
        buf = io.BytesIO()
        torch.save(m["net"].state_dict(), buf)
        Xt, _, _ = deep.build_windows(prim[prim["cell_id"] == prim["cell_id"].iloc[0]], feats, 10)
        t0 = time.perf_counter()
        for i in range(200):
            deep.predict_net(m, Xt[i % len(Xt):i % len(Xt) + 1])
        footprint[f"{label} (k=10)"] = {"params": deep.n_params(m["net"]), "size_kb": buf.tell() / 1024,
                                        "ms_per_cycle": (time.perf_counter() - t0) / 200 * 1000,
                                        "needs_full_discharge": "yes (dQ(V) features)"}
    window_ablation = {}
    for k in ks:                                   # window-length sensitivity (one seed, LSTM)
        res, _ = run("lstm", k, seeds[:1])
        window_ablation[k] = res
    _save_json({"results": out, "window_ablation": window_ablation, "footprint": footprint,
                "features": feats, "val_cells": list(val_cells)}, "deep.json")
    return out, window_ablation, footprint


# ==========================================================================
# 5. ablation ladder, leakage demo, robustness (O4, O6)
# ==========================================================================
def run_ablation(df, fsets, train_rows, seeds, xgb_params):
    tr, prim, sec = split_frames(df, train_rows)
    tests = {"test_primary": prim, "test_secondary": sec}
    out = {}
    for fset, feats in fsets.items():
        res, _, _ = fit_eval_tabular("xgb", feats, tr, tests, seeds, xgb_params)
        out[fset] = {k: {m: res[k][m] for m in ("RMSE", "MAE", "R2")} for k in res}
        print(f"[ablation] {fset:20s} primary {res['test_primary']['RMSE']['mean']:.3f} "
              f"secondary {res['test_secondary']['RMSE']['mean']:.3f}", flush=True)

    # A7 / leakage demo: same pool of cells (train + primary), random ROWS vs whole CELLS
    pool = df[df["split"].isin(["train", "test_primary"]) & ~df["flag_qd_outlier"]].reset_index(drop=True)
    feats = fsets[features.MAIN_SET]
    leak = {}
    for name in ("rf", "xgb"):
        row_rmse, cell_rmse = [], []
        for seed in seeds:
            a, b = splits.random_row_split(len(pool), test_frac=0.5, seed=seed)
            m = _model(name, seed, xgb_params)
            m.fit(pool.loc[a, feats], pool.loc[a, LABEL])
            row_rmse.append(metrics.regression_metrics(pool.loc[b, LABEL], m.predict(pool.loc[b, feats]))["RMSE"])
            tr_c, te_c = splits.random_cell_split(pool["cell_id"].unique(), test_frac=0.5, seed=seed)
            a = pool["cell_id"].isin(tr_c)
            m = _model(name, seed, xgb_params)
            m.fit(pool.loc[a, feats], pool.loc[a, LABEL])
            cell_rmse.append(metrics.regression_metrics(pool.loc[~a, LABEL], m.predict(pool.loc[~a, feats]))["RMSE"])
        leak[name] = {"random_row_rmse": np.mean(row_rmse), "random_row_std": np.std(row_rmse),
                      "cell_wise_rmse": np.mean(cell_rmse), "cell_wise_std": np.std(cell_rmse),
                      "inflation": np.mean(cell_rmse) / np.mean(row_rmse)}
        print(f"[leakage] {name}: random-row {np.mean(row_rmse):.3f} vs cell-wise {np.mean(cell_rmse):.3f}")
    out["A7_A4_random_row_split"] = {"test_pool": {"RMSE": {"mean": leak["xgb"]["random_row_rmse"],
                                                             "std": leak["xgb"]["random_row_std"]}}}

    # robustness: 10 random 80/20 CELL splits over all 124 cells
    # same rule for every cell: spike rows leave TRAINING only; test cells keep every row
    rob = []
    for seed in range(10):
        tr_c, te_c = splits.random_cell_split(df["cell_id"].unique(), 0.2, seed=seed)
        a = df[df["cell_id"].isin(tr_c) & ~df["flag_qd_outlier"]]
        b = df[df["cell_id"].isin(te_c)]
        m = _model("xgb", seed, xgb_params).fit(a[feats], a[LABEL])
        rob.append(metrics.regression_metrics(b[LABEL], m.predict(b[feats]))["RMSE"])
    robust = {"rmse_each": rob, "mean": float(np.mean(rob)), "std": float(np.std(rob))}
    print(f"[robust] 10 random cell splits: RMSE {robust['mean']:.3f} +/- {robust['std']:.3f}")
    _save_json({"ablation": out, "leakage": leak, "robustness": robust}, "ablation.json")
    return out, leak, robust


# ==========================================================================
# 6. NASA: leave-one-battery-out and MIT -> NASA zero-shot (O5)
# ==========================================================================
NASA_FEATS = ["IR", "Re", "Rct", "d_IR", "r_IR", "chg_t_cv", "chg_cv_tau", "chg_dT",
              "d_chg_t_cv", "d_chg_cv_tau", "r_chg_t_cv", "r_chg_cv_tau"]
# chg_t_cc (CC 1.5 A up to 4.2 V from a discharged cell) ~ capacity / 1.5 A -> excluded


def run_nasa(df, train_rows, seeds, xgb_params):
    nasa = features.add_derived(nasa_loader.build_nasa_table())
    nasa.to_parquet(config.PROCESSED_DIR / "nasa_features.parquet", index=False)
    out = {"lobo": {}, "zero_shot": {}}
    for name in ("ridge", "rf", "xgb"):
        per = {}
        allp = []
        for cell in config.NASA_CELLS:
            a, b = nasa[nasa["cell_id"] != cell], nasa[nasa["cell_id"] == cell]
            m = _model(name, 0, xgb_params)
            m.fit(a[NASA_FEATS], a[LABEL])
            p = b[["cell_id", "cycle", LABEL]].copy()
            p["pred"] = m.predict(b[NASA_FEATS])
            per[cell] = metrics.regression_metrics(p[LABEL], p["pred"])["RMSE"]
            allp.append(p)
        out["lobo"][name] = per
        if name == "xgb":
            plots.nasa_plot(pd.concat(allp), "NASA leave-one-battery-out (XGBoost, trained on the other 3 cells)",
                            "08_nasa_lobo.png", label=LABEL)
    # zero-shot: train on MIT training cells with dimensionless features only, label soh_init
    tr = df[train_rows].dropna(subset=features.TRANSFER_COLS)
    prim = df[df["split"] == "test_primary"]
    for name in ("rf", "xgb"):
        m = _model(name, 0, xgb_params)
        m.fit(tr[features.TRANSFER_COLS], tr["soh_init"])
        p = nasa[["cell_id", "cycle", "soh_init"]].copy()
        p["pred"] = m.predict(nasa[features.TRANSFER_COLS])
        per = {c: metrics.regression_metrics(d["soh_init"], d["pred"])["RMSE"] for c, d in p.groupby("cell_id")}
        per["MIT_primary_sanity"] = metrics.regression_metrics(
            prim["soh_init"], m.predict(prim[features.TRANSFER_COLS]))["RMSE"]
        per["min_pred"] = float(p["pred"].min())
        out["zero_shot"][name] = per
        if name == "xgb":
            plots.nasa_plot(p, "MIT -> NASA zero-shot (XGBoost on dimensionless features, label = soh_init)",
                            "09_nasa_zero_shot.png", label="soh_init")
    out["nasa_label_range"] = {"soh_min": float(nasa[LABEL].min()), "soh_init_min": float(nasa["soh_init"].min()),
                               "mit_train_soh_init_min": float(tr["soh_init"].min())}
    print("[nasa]", json.dumps(out, default=float)[:400])
    _save_json(out, "nasa.json")
    return out


# ==========================================================================
# 7. explainability, footprint, deployable models (O6, O7)
# ==========================================================================
def run_explain_and_deploy(df, fsets, train_rows, xgb_params, store):
    tr, prim, sec = split_frames(df, train_rows)
    out = {"footprint": {}}
    figs = {}
    rng = np.random.default_rng(0)
    sample = prim.iloc[rng.choice(len(prim), 4000, replace=False)]
    for tag, fset in [("main", features.MAIN_SET), ("bms", features.BMS_SET)]:
        feats = fsets[fset]
        model = _model("xgb", 0, xgb_params).fit(tr[feats], tr[LABEL])
        path = config.MODELS_DIR / f"soh_xgb_{tag}.joblib"
        joblib.dump({"model": model, "features": feats, "feature_set": fset, "label": "SOH %",
                     "rated_ah": config.MIT_RATED_AH}, path)
        perm = explain.grouped_permutation_importance(model.predict, sample[feats], sample[LABEL].to_numpy())
        perm.to_csv(RES / f"permutation_importance_{tag}.csv", index=False)
        if tag == "main":
            mean_abs, figs["shap"] = explain.shap_figures(model, sample[feats].iloc[:2000])
            mean_abs.to_csv(RES / "shap_mean_abs.csv", header=["mean_abs_shap"])
        one = prim[feats].iloc[[100]]
        t0 = time.perf_counter()
        for _ in range(300):
            model.predict(one)
        booster = model.get_booster()
        out["footprint"][f"XGBoost {fset}"] = {
            "params": f"{booster.num_boosted_rounds()} trees x depth<={xgb_params.get('max_depth', 6)}",
            "size_kb": path.stat().st_size / 1024,
            "ms_per_cycle": (time.perf_counter() - t0) / 300 * 1000,
            "needs_full_discharge": "no (charge segment, IR, temperature)" if tag == "bms" else "yes (dQ(V) features)"}
        if tag == "bms":   # BMS dashboard demo on an unseen secondary-test cell
            cell = sec["cell_id"].unique()[5]
            c = sec[sec["cell_id"] == cell].copy()
            c["pred"] = model.predict(c[feats])
            figs["monitor"] = plots.health_monitor(c, cell)
    for key in ("Random Forest | A4_plus_history", "Ridge | A4_plus_history"):
        _, model, feats, _ = store[key]
        buf = io.BytesIO()
        joblib.dump(model, buf)
        one = prim[feats].iloc[[100]]
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict(one)
        m = model[-1]
        size = (f"{len(m.estimators_)} trees, mean depth {np.mean([e.get_depth() for e in m.estimators_]):.0f}"
                if hasattr(m, "estimators_") else f"{len(feats) + 1} coefficients")
        out["footprint"][key.split(" |")[0] + " A4"] = {"params": size, "size_kb": buf.tell() / 1024,
                                                        "ms_per_cycle": (time.perf_counter() - t0) / 100 * 1000,
                                                        "needs_full_discharge": "yes (dQ(V) features)"}
    _save_json(out, "explain_deploy.json")
    return out


def reproducibility():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "uncommitted"
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "torch": torch.__version__,
            "seeds": config.SEEDS, "commit": commit, "date": time.strftime("%Y-%m-%d %H:%M")}


def main(quick=False, skip_deep=False):
    config.ensure_dirs()
    RES.mkdir(parents=True, exist_ok=True)
    seeds = config.SEEDS[:1] if quick else config.SEEDS
    t0 = time.time()
    df, fsets, train_rows = prepare()
    print(f"[prepare] {df['cell_id'].nunique()} cells, {len(df):,} rows ({time.time() - t0:.0f}s)")

    tab, store, tuned = run_tabular(df, fsets, train_rows, seeds)
    xgb_params = tuned[f"xgb|{features.MAIN_SET}"]["params"]
    # "best" = lowest mean of primary and secondary RMSE: a model that is excellent on familiar
    # cells but fails on a new batch (Ridge here) should not be the one we showcase
    best_key = min(tab, key=lambda k: tab[k]["test_primary"]["RMSE"]["mean"] + tab[k]["test_secondary"]["RMSE"]["mean"])
    _save_json({"best_model": best_key}, "best_model.json")
    preds, _, _, _ = store[best_key]
    plots.pred_vs_true(preds, best_key)
    for split in ("test_primary", "test_secondary"):
        pc = metrics.per_cell_metrics(preds[split])
        pc.to_csv(RES / f"per_cell_{split}.csv", index=False)
        plots.cell_trajectories(preds[split], pc, best_key, split)
    band = pd.DataFrame({k.split(" |")[0]: {b: v["mean"] for b, v in tab[k]["test_primary"].items()
                                            if b.startswith("RMSE_")}
                         for k in tab if "A4" in k}).T
    plots.error_by_band(band)

    if not skip_deep:
        run_deep(df, fsets, train_rows, seeds, epochs=8 if quick else 40, ks=(10,) if quick else (5, 10, 20, 30))
    abl, leak, robust = run_ablation(df, fsets, train_rows, seeds, xgb_params)
    order = list(fsets)
    plots.bar_compare([o for o in order], {
        "primary test": [abl[o]["test_primary"]["RMSE"]["mean"] for o in order],
        "secondary test": [abl[o]["test_secondary"]["RMSE"]["mean"] for o in order]},
        "Ablation ladder (XGBoost): RMSE by feature set", "RMSE (% SOH)", "07_ablation.png")
    plots.bar_compare(["Random Forest", "XGBoost"], {
        "random-row split (WRONG)": [leak[m]["random_row_rmse"] for m in ("rf", "xgb")],
        "cell-wise split": [leak[m]["cell_wise_rmse"] for m in ("rf", "xgb")]},
        "Leakage demo: same cells, same model, different split", "RMSE (% SOH)", "07b_leakage.png")
    run_nasa(df, train_rows, seeds, xgb_params)
    run_explain_and_deploy(df, fsets, train_rows, xgb_params, store)
    _save_json(reproducibility(), "reproducibility.json")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-deep", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, skip_deep=a.skip_deep)
