# Data-driven State of Health (SOH) estimation of EV lithium-ion batteries

End-to-end machine-learning project that estimates the **present usable capacity** and
**SOH = present capacity / rated capacity × 100** of lithium-ion cells from measurable signals
(voltage, current, temperature, resistance, charge history), evaluated against **measured
reference capacity** on cells the model has never seen.

- **Data:** MIT–Stanford–Toyota fast-charging dataset (Severson et al. 2019): 124 LFP/graphite
  cells, 1.1 Ah, 100,373 clean cycles. Plus NASA PCoE B0005/6/7/18 (2 Ah) for a cross-chemistry test.
- **Models:** Ridge, Random Forest, XGBoost, LSTM, GRU, 1D-CNN.
- **Protocol:** official cell-wise split (41 train / 43 primary-test / 40 secondary-test cells),
  leakage audit, 3 seeds. Full numbers in **[reports/RESULTS.md](reports/RESULTS.md)**.

## Headline results (% SOH, unseen cells, mean ± std over 3 seeds)

| Model | Features | Primary test RMSE | Secondary test RMSE (later batch) |
|---|---|---|---|
| Dummy (predict training mean) | — | 5.14 | 4.01 |
| XGBoost | cycle count only | 4.06 | 3.53 |
| Ridge | A4 (all audited) | 1.14 | 5.88 |
| Random Forest | A4 | 1.20 ± 0.01 | 2.46 ± 0.07 |
| XGBoost | A4 | 1.22 ± 0.01 | 3.00 ± 0.15 |
| **XGBoost (BMS-feasible model)** | **A2: charge segment + IR + temperature** | **1.24 ± 0.00** | **1.88 ± 0.19** |
| LSTM (k = 10 cycles) | A3 windows | 1.09 ± 0.03 | 4.05 ± 0.30 |
| 1D-CNN (k = 10 cycles) | A3 windows | 0.97 ± 0.05 | 4.46 ± 0.66 |

**Deployed model:** XGBoost on A2. It needs no laboratory discharge, it equals the best tree models on
familiar cells and it is the most accurate on the later batch. Example on an unseen cell:
`python -m soh.predict --demo` → 1.24 % SOH RMSE over 1,007 cycles.

## How the problem statement is answered

| Requirement | Where / how |
|---|---|
| Relationship between voltage, current, temperature, charge-discharge history and capacity fade | Leakage audit with pooled and within-cell Spearman correlations and single-feature R² ([RESULTS §2](reports/RESULTS.md)); feature scatter plots; SHAP and permutation importance (§7); ablation ladder A0–A6 (§4) |
| Estimate the present usable capacity | `est_capacity_Ah` in `soh/predict.py` (= estimated SOH × 1.1 Ah) |
| Calculate SOH with ML | Six models compared (§3); label `soh = QD / 1.1 × 100` |
| Evaluate against measured reference capacity | RMSE / MAE / MAPE / R², per-SOH-band and per-cell errors, end-of-life-alarm error, NASA tests (§3–§6) |
| Use in battery-health monitoring | `soh.predict` CLI (capacity, SOH, HEALTHY / MONITOR / REPLACE status), deployment footprint (§8) and [discussion below](#use-in-a-battery-management-system) |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                  # 9 fast tests, no data needed
python -m soh.data.download --nasa-only    # NASA cells (~210 MB); the parsed MIT table is committed
python -m soh.experiments                  # all experiments, ~20 min on 4 CPU cores (--quick: ~10 min)
python -m soh.report                       # rebuilds reports/RESULTS.md
python -m soh.predict --demo               # battery-health monitoring demo
```

`data/processed/mit_cycles_raw.parquet` (18 MB, the output of parsing the 8.3 GB of raw files) is
committed, so you do **not** need the big download to rerun the modelling. To rebuild everything from raw
data, run `scripts/run_all.sh` (it downloads with resume and verifies file sizes).

## Repository map (read in this order — see [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md))

```
soh/config.py            constants, paths, every cleaning choice
soh/data/download.py     Stage 1  download (resumable, size-checked)
soh/data/mit_loader.py   Stage 1+3 HDF5 parsing, official 124-cell cleaning + stitching, per-cycle features
soh/data/nasa_loader.py  NASA B0005/6/7/18 in the same layout
soh/preprocess.py        Stage 2  cycle-level cleaning, labels (soh, soh_init)
soh/features.py          Stage 3  feature catalogue, derived features, calibrated leakage audit, ablation sets
soh/splits.py            Stage 4  official cell-wise split, leakage-demo splits
soh/models/tabular.py    Stage 5  Dummy, Ridge, RF, XGBoost + GroupKFold tuning
soh/models/deep.py       Stage 6  per-cell windows, LSTM / GRU / 1D-CNN
soh/metrics.py           Stage 7  metrics, SOH bands, per-cell, EOL-alarm error
soh/explain.py           SHAP + grouped permutation importance
soh/experiments.py       runs everything -> reports/results/*.json
soh/report.py            Stage 8  fills templates R1–R5 -> reports/RESULTS.md
soh/predict.py           battery-health monitoring CLI
models/                  deployable XGBoost models (main = A4, bms = A2) + training clip ranges
```

## Key findings

1. **The split matters more than the model.** Same 84 cells, same model: a random **row** split gives
   0.15 % RMSE and a **cell-wise** split 0.56–0.65 %, i.e. ~4× optimistic. Random-row numbers are leakage, not skill.
2. **Many signals are the label in disguise.** Every MIT cell is discharged identically, so discharge
   time and energy restate capacity. The same is true of the charge taken in the 1C phase (the fast phase
   always stops at 0.88 Ah, so the rest ≈ QC − 0.88). The empirical audit reproduces this: known
   disguised labels reach single-feature R² 0.93–0.95. The exclusion threshold is calibrated from them
   (R² ≥ 0.922), not guessed.
3. **Counting cycles is not enough.** Cycle count alone gives 4.1 % RMSE, against 1.2 % for health features.
   Lifetimes range from ~150 to 2,300 cycles.
4. **Batch shift is the real enemy.** Batch 3 (secondary test, cycled ~1 year later) rests only 5 s
   after fast charging, against minutes in batches 1–2. This shifts relaxation-sensitive charge features.
   Every model is worse there, and Ridge and the deep models collapse (4–6 % RMSE). Trees and the smaller
   BMS feature set degrade least. When some batch-3 cells are in training (10 random 80/20 cell splits),
   error falls to 0.52 ± 0.10 %, the case for fleet recalibration.
5. **Deep models win in-distribution but generalise worse.** 1D-CNN and LSTM beat the trees on the primary
   set (0.97 and 1.09 %) but lose badly on the later batch. This refines, rather than contradicts, the
   literature's "trees beat LSTMs".
6. **Safety caveat: models are optimistic near end of life.** On the later batch no model's raw estimate
   crosses the 80.5 % alarm level before the cell does (missed alarms; RESULTS §3). A BMS must not use
   these estimates alone for retirement decisions without a safety margin or periodic reference tests.
7. **Cross-chemistry transfer fails, as expected.** NASA leave-one-battery-out gives 8–9 % RMSE. MIT → NASA
   zero-shot gives ~11–12 %, and the trees floor at ~85 % because MIT labels never go as low as NASA's (57 %).

## Use in a battery management system

- **What a car can measure.** IR (from current steps), temperature and the charging segment are measured
  on-board. The lab discharge curve used by the dQ(V) features is not. Hence the deployed **A2 model**,
  which needs no discharge data, matches the best primary error and is best on the later batch.
- **One caveat on A2.** Its top features (`chg_ic_peak`, `chg_v_start`) are measured a fixed charge
  (0.88 Ah) after a *full* discharge. In an EV the charge starts from an arbitrary SOC, so these must be
  re-expressed over a fixed **voltage window** (partial-charge incremental capacity) before field use.
- **Compute is not the bottleneck.** XGBoost runs in ~2 ms per cycle in Python (≈3.5 MB with 1,000 trees;
  shrinking the tree count would cut that). The 1D-CNN is 10.6k parameters. An unlimited-depth Random
  Forest is ~730 MB, impractical on an MCU.
- **Monitoring logic.** `soh.predict` smooths the per-cycle estimate (5-cycle median) and maps it to
  HEALTHY (≥ 90 %), MONITOR (80–90 %) and REPLACE / second-life (< 80 %). Given finding 6, a
  production BMS should add an uncertainty margin and use occasional reference capacity checks.
- **Fleet loop.** The on-board model sends features plus occasional reference capacities to the cloud,
  the cloud retrains on the new batch, and the model is updated over the air. Finding 4 is the evidence:
  seeing the new batch in training halves the error.

## Limitations

One LFP chemistry at 30 °C. Identical full 4C discharges, lab cycling only, no calendar ageing.
Only 41 training cells. MIT labels stop near 75–80 % SOH. There is no uncertainty estimate, and SHAP
explains the model, not the electrochemistry. Cleaning thresholds and the audit margin are project
choices (all in `soh/config.py`).

## Deviations from the planning document (and why)

- `chg_t_cc` / `chg_q_cc` (1C-segment duration/charge) moved from *caution* to **excluded**: the data
  showed they equal QC − 0.88 Ah.
- The QD-spike rule gained an absolute 0.005 Ah floor: without it 31 % of rows were flagged.
- The IC window starts at the voltage minimum of the 1C phase, to remove a batch-3 relaxation artefact.
- Features are clipped to the training range (0.5–99.5 %, fit on training cells only).
- The EOL alarm level is 80.5 %, not 80 %, because batch-3 cells were stopped exactly at 0.88 Ah.
- The leakage demo uses a 50/50 split of the 84 batch-1/2 cells. The robustness check uses 10 random
  80/20 splits of all 124 cells.
- Deep models early-stop on 8 of the 41 training cells and are not refit on all 41. Hyper-parameter
  tuning uses every 3rd cycle for speed.
- The ablation reuses the XGBoost hyper-parameters tuned on A4 for every feature set.
- Not done: the stretch raw-curve CNN and LightGBM.

## Data and citation

Severson, K.A. et al. "Data-driven prediction of battery cycle life before capacity degradation."
*Nature Energy* 4, 383–391 (2019); data from data.matr.io (CC BY 4.0 per dos Reis et al.).
B. Saha and K. Goebel (2007), "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research
Center, Moffett Field, CA. Full reference list: the project planning document.
