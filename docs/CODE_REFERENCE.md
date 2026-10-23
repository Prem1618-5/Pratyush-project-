# Code reference: what every module and function does

Companion to [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md). The guide explains the *ideas* in reading
order; this file is the *lookup table*: inputs, outputs and the one rule each function enforces.
Measured numbers quoted here come from a full run (`python -m soh.experiments`) and are
reproduced in `reports/RESULTS.md`.

---

## Pipeline at a glance

```
download.py ──► mit_loader.py / nasa_loader.py ──► preprocess.py ──► features.py ──► splits.py
   raw .mat           per-cycle parquet              clean + label     derived + audit   cell-wise split
                                                                                            │
         report.py ◄── experiments.py ◄── metrics.py ◄── models/tabular.py, models/deep.py ◄┘
        RESULTS.md     reports/results/*.json            explain.py, plots.py, predict.py
```

One command runs everything: `bash scripts/run_all.sh` (download → parse → experiments → report → tests).

| Command | What it produces |
|---|---|
| `python -m soh.data.download [--nasa-only]` | `data/raw/*.mat` (MIT ≈ 8.3 GB, NASA ≈ 210 MB; resumable, size-checked) |
| `python -m soh.data.mit_loader` | `data/processed/mit_cycles_raw.parquet` (committed: 100,501 rows, 124 cells) |
| `python -m soh.data.nasa_loader` | `data/processed/nasa_cycles.parquet` (committed) |
| `python -m soh.experiments [--quick] [--skip-deep]` (18 min measured, 4 cores) | `reports/results/*.json/csv`, `reports/figures/*.png`, `models/*.joblib` |
| `python -m soh.report` | `reports/RESULTS.md` |
| `python -m soh.predict --demo` | per-cycle SOH estimates + status, `soh_estimates.csv` |
| `pytest -q` | 9 unit tests on synthetic data (no download needed) |

---

## `soh/config.py` — every constant in one place

| Name | Value | Why |
|---|---|---|
| `MIT_RATED_AH` | 1.1 | A123 APR18650M1A nominal capacity; SOH = QD / 1.1 × 100 |
| `MIT_EOL_SOH` | 80.0 | Standard end-of-life (0.88 Ah) |
| `EOL_ALARM_SOH` | 80.5 | Batch-3 cells stop *at* 0.88 Ah (verified min SOH 80.00–80.15 %), so a strict < 80 % crossing never happens there |
| `BATCH1_DROP`, `BATCH3_DROP`, `BATCH1/2_STITCH` | cell ids | Official cleaning from the authors' `Load Data.ipynb` (140 → 124 cells) |
| `QDLIN_V`, `DQ_WINDOW_V`, `DQ_REF_CYCLE` | (3.5, 2.0, 1000), (3.3, 2.9), 10 | Q(V) grid, the *partial* window for ΔQ(V), Severson's reference cycle |
| `NASA_CELLS`, `NASA_RATED_AH` | B0005/6/7/18, 2.0 | NASA PCoE cells and their rating |
| `QD_MIN_AH`, `QD_MAX_AH` | 0.0, 1.2 | Physically impossible capacity outside this range |
| `SPIKE_WINDOW`, `SPIKE_K`, `SPIKE_FLOOR_AH` | 11, 5.0, 0.005 | Spike rule; the floor keeps flags at 0.24 % (30.8 % without it) |
| `BASELINE_CYCLES` | 10 | "Cell when new" = median of the first 10 valid cycles |
| `SEEDS` | [0, 1, 2] | Every stochastic model reports mean ± std over these |
| `LEAK_MARGIN` | 0.01 | Audit threshold = weakest known leak's R² − margin |

`SOH_DATA_DIR` (env var) moves the data folder. `ensure_dirs()` creates all output folders.

---

## `soh/data/` — raw files to per-cycle tables

### `download.py`
- `_fetch(url, dest, expected_size)` — streams with HTTP `Range` so an interrupted download resumes; raises if the final size is wrong.
- `download_mit()`, `download_nasa()` — NASA's zip contains inner zips; only the `FY08Q4` one (B0005/6/7/18) is extracted.

### `mit_loader.py`
Reads MATLAB v7.3 (= HDF5) files with `h5py`. Every field is an object reference → `f[ref][()]`.

| Function | In → out | Rule it enforces |
|---|---|---|
| `charge_segment_features(I, V, t, Qc, T)` | one cycle's curves → 9 scalars (`chg_*`) | Uses only the 1C CC + 3.6 V CV phase that **every** policy shares. CC rows need `Qc > 0.8 Ah` (skips policies that start with a short 1C step). Start voltage / IC window begin at the voltage **minimum** (batch-3 relaxation fix). IC = dQ/dV on a 5 mV grid, Savitzky–Golay (9, 3). |
| `discharge_features(I, V, t)` | → `dis_time`, `dis_energy` | Label-equivalent on purpose: computed only so the audit can prove it. |
| `dq_features(qwin, cycles)` | Q(V) in 3.3–2.9 V per cycle → `dq_var_log`, `dq_min_log`, `dq_mean` | ΔQ = Q_k − Q_ref, ref = cycle 10 (or the next valid one). Never the full window to 2.0 V (that would be QD). |
| `_parse_cell`, `parse_batch` | HDF5 → DataFrame + Q(V) window + policy string | Curves become scalars immediately; 8 GB never sits in RAM. |
| `build_mit_table(processes=3)` | 3 batches → one table | Drops / stitches cells exactly like the official loader (stitched cycles are offset by the batch-1 length). Adds `cell_index` in official order — the split depends on it. |

### `nasa_loader.py`
- `_charge_features(d)` — CC 1.5 A to 4.2 V then CV to 20 mA: `chg_t_cc`, `chg_t_cv`, `chg_cv_tau` (time for current to halve from 1.5 A), `chg_dT`.
- `load_cell(cell)` — one row per **discharge** (where capacity is measured); features from the preceding charge and the latest valid EIS fit (`IR = Re + Rct`, failed fits outside 0–1 Ω dropped).
- `build_nasa_table()` — forward/back-fills impedance per cell; labels `soh` (vs 2.0 Ah) and `soh_init` (vs the cell's first capacity).

---

## `soh/preprocess.py` — cleaning and labels

- `_rolling_spike(s, window, k, floor)` — flags `|x − rolling median| > max(k·MAD, floor)`.
- `clean_mit(df)` → `(df, log)`. Order and verified counts on the 100,501 raw rows:

| Step | Effect | Rows |
|---|---|---|
| Drop cycle 1 | removed everywhere | 124 |
| Drop QD ∉ (0, 1.2) Ah | removed everywhere | 4 |
| `flag_qd_outlier` (window 11, K 5, floor 0.005 Ah) | **flag only**; dropped from training rows later | 243 (0.24 %) |
| `flag_ir_bad` (IR ≤ 0 or spike) | IR set to NaN, forward-filled later | 29 |
| `flag_temp_bad` (Tavg spike > 2 °C) | Tavg/Tmin/Tmax set to NaN | 85 |

Adds `soh` (vs 1.1 Ah) and `soh_init` (vs the cell's first clean cycle). Result: 100,373 rows.

---

## `soh/features.py` — derived features, feature sets, leakage audit

- `CATALOGUE` — `{name: (family, prior status, meaning)}`. Prior status is `safe`, `caution`, `ablation` or `excluded`. `_rm5` entries are generated for every `ROLL_COLS` member and inherit the parent's status.
- `add_derived(df)` — per cell only: forward-fill → `Trange` → `d_X` (minus baseline) and `r_X` (÷ baseline) for `BASELINE_COLS` → `X_rm5` causal 5-cycle rolling mean for `ROLL_COLS`.
- Feature sets (ablation ladder): `A0` cycle only · `A1` IR + temperature · `A2` + 1C charge segment (**BMS model**) · `A3` + partial ΔQ(V) · `A4` + `_rm5` history (**main model**) · `A5` A4 + chargetime · `A6` A4 + cycle.
- `TRANSFER_COLS` — `r_IR`, `r_chg_t_cv`, `r_chg_cv_tau`: dimensionless, exist in both MIT and NASA (zero-shot transfer).
- `clip_to_train(df, cols, train_mask, q=0.005)` — clips every feature to the 0.5–99.5 percentile of **training** rows.
- `leakage_audit(train)` — per feature on training cells: pooled Spearman, median within-cell Spearman, and cross-cell R² of a one-feature tree (`GroupKFold(5)` by cell). Threshold = min R² of the definitional leaks other than QD/QC − `LEAK_MARGIN`; anything at or above it is `EXCLUDED (empirical …)`.
- `apply_audit(feature_sets, audit)` — removes every excluded feature from every set.

---

## `soh/splits.py` — the cell is the unit

| Function | Use |
|---|---|
| `official_split(df)` | Severson split by `cell_index`: train = odd 1–81 (41), primary = even 0–82 + 83 (43), secondary = 84–123 (40, batch 3) |
| `random_cell_split(cells, test_frac, seed)` | Robustness check (10 random 80/20 cell splits) and leakage demo |
| `random_row_split(n, test_frac, seed)` | **Deliberately wrong**; only for the leakage demo |
| `validation_cells(train_cells, n_val=8)` | Early-stopping cells taken from the training cells |

---

## `soh/models/`

### `tabular.py`
- `make_model(name, seed, **params)` — `dummy`, `ridge` (impute → scale → Ridge), `rf` (impute → 300-tree forest), `xgb` (handles NaN natively, `tree_method="hist"`). Pre-processing lives inside the pipeline, so it is fit on training rows only.
- `tune(name, X, y, groups)` — `RandomizedSearchCV` with `GroupKFold(5)` on every 3rd training row; returns best params + inner-CV RMSE. Search ranges in `SEARCH_SPACES`, iterations in `N_ITER`.

### `deep.py`
- `build_windows(df, feats, k)` — one window of k cycles per row, built per cell (left-padded with the cell's first row), so windows never cross cells or splits.
- `RNN` (2-layer LSTM/GRU 64→32, dropout 0.2, last step → MLP), `CNN1D` (two conv + batch-norm blocks, global average pool). `make_net(name, n_feat)` picks one.
- `Scaler` — median impute + min–max, fit on training windows, output clipped to [−1, 2].
- `train_net(...)` — Adam, MSE on standardised labels, `ReduceLROnPlateau`, early stopping (patience 6) on validation **training** cells; returns the best state.
- `predict_net`, `n_params`.

---

## `soh/metrics.py` — all in % SOH

| Function | Returns |
|---|---|
| `regression_metrics(y, p)` | RMSE, MAE, MAPE, R², MaxAE |
| `band_rmse(y, p)` | RMSE inside true-SOH bands 80–85, 85–90, 90–95, ≥ 95 |
| `first_crossing(cycles, values, thr)` | first cycle below `thr`, else NaN |
| `eol_errors(df)` | per cell: predicted − true first crossing of 80.5 % (negative = early = safe) |
| `per_cell_metrics(df)` | regression metrics per cell |
| `full_report(df)` | all of the above plus per-cell mean / median / worst, mean \|EOL error\|, signed EOL error and `EOL_undefined_cells` (= **missed alarms**: truth crossed, estimate never did) |

---

## `soh/explain.py`
- `feature_group(name)` — strips `d_`, `r_`, `_rm5` so a feature and its copies form one group.
- `grouped_permutation_importance(predict, X, y)` — permutes a whole group at once on held-out cells; reports mean RMSE increase.
- `shap_figures(model, X)` — TreeSHAP beeswarm + one waterfall (a cycle near 85 % SOH); returns mean |SHAP| per feature.

## `soh/plots.py`
Static PNGs in `reports/figures/`, one categorical palette `C`, split colours `SPLIT_COLOR`. One function per figure:
`capacity_fade` (01), `feature_scatter` (02), `pred_vs_true` (03), `cell_trajectories` (04, best/median/worst cell),
`error_by_band` (05), `bar_compare` (07 ablation, 07b leakage), `nasa_plot` (08, 09), `health_monitor` (10).

## `soh/experiments.py` — the orchestrator

| Function | Stage | Writes |
|---|---|---|
| `prepare()` | clean → derive → split → audit (training cells, spikes removed) → clip | `data_summary.json`, `leakage_audit.csv`, `models/train_clip_ranges.joblib`, figs 01–02 |
| `run_tabular()` | Dummy, XGB on `A0`, Ridge / RF / XGB on `A4`; tuned, 3 seeds for RF/XGB | `tabular.json` |
| `main()` (after `run_tabular`) | picks the showcase model = lowest **primary + secondary** RMSE (Random Forest A4), so a model that fails on the new batch (Ridge) is not showcased | `best_model.json`, `per_cell_*.csv`, figs 03–05 |
| `run_deep()` | LSTM / GRU / 1D-CNN on `A3` windows (k = 10), plus LSTM k ∈ {5, 10, 20, 30} | `deep.json` |
| `run_ablation()` | XGB over every feature set; random-row vs cell-wise demo; 10 random cell splits | `ablation.json`, figs 07, 07b |
| `run_nasa()` | leave-one-battery-out on NASA; MIT → NASA zero-shot on `TRANSFER_COLS` | `nasa.json`, figs 08–09 |
| `run_explain_and_deploy()` | saves `soh_xgb_main/bms.joblib`, permutation importance, SHAP, footprint, monitor demo | `explain_deploy.json`, `permutation_importance_*.csv`, `shap_mean_abs.csv`, figs 06, 10 |
| `reproducibility()` | library versions, seeds, git commit | `reproducibility.json` |

`NASA_FEATS` deliberately omits NASA `chg_t_cc`: CC charging from empty at 1.5 A ≈ capacity / 1.5 A.

## `soh/report.py`
`build()` turns `reports/results/` into `reports/RESULTS.md` (sections 1–9: data, audit, main results R1, ablation R2, leakage R3, NASA R4, explainability, footprint R5, reproducibility). Helpers: `ms()` formats mean ± std, `md_table()` builds a Markdown table.

## `soh/predict.py` — the BMS-style CLI
- `estimate(cycles, kind="bms", smooth=5)` — `add_derived` → clip to training ranges → XGBoost → rolling-median smoothing (5 cycles) → `est_capacity_Ah`, `status`.
- `health_status(soh)` — `HEALTHY` ≥ 90 %, `MONITOR` 80–90 %, `REPLACE / SECOND-LIFE` < 80 %.
- Needs `models/soh_xgb_{bms,main}.joblib` and `models/train_clip_ranges.joblib` (created by `soh.experiments`).

---

## `tests/test_core.py` — one test per rule

| Test | Rule it pins down |
|---|---|
| `test_charge_segment_features_found` | the 1C CC + CV segment is found on a synthetic cycle |
| `test_cc_charge_tracks_capacity_ie_is_a_leak` | bigger cell → bigger 1C CC charge (why `chg_q_cc` is excluded) |
| `test_discharge_time_is_capacity` | discharge time is exact |
| `test_dq_reference_cycle_is_zero` | ΔQ(V) is 0 at cycle 10 and negative later |
| `test_official_split_sizes_and_disjoint` | 41 / 43 / 40, no overlap |
| `test_windows_never_mix_cells` | every deep-model window holds one cell |
| `test_metrics_and_eol` | RMSE/MAE arithmetic; optimistic model → late (positive) alarm |
| `test_audit_flags_disguised_label_and_keeps_noise_feature` | audit excludes a disguised label, keeps noise |
| `test_clip_uses_training_rows_only` | clipping limits come from training rows |
