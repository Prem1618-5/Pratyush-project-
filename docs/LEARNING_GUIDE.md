# Learning guide: read the code in this order

Each stage = one file, one key idea, one exercise. Run `pytest -q` any time; the tests in
`tests/test_core.py` are small, readable examples of every rule below.

**Companion files.** [`CODE_REFERENCE.md`](CODE_REFERENCE.md) documents every module and function
(inputs, outputs, the rule each enforces). `reports/RESULTS.md` holds the full measured results.

**"✅ Verified"** boxes below quote a full run (`python -m soh.experiments`, seeds 0/1/2, 4 CPU cores,
18 min). Earlier versions of this guide stated some of these as expectations; they are now measured.

---

## 1. Data: what a "cycle" is — `soh/data/mit_loader.py`

**Idea.** One MIT cycle = fast charge to 0.88 Ah (80 % SOC) → rest → **1C charge to 3.6 V → CV taper** →
4C discharge to 2.0 V. The `summary` struct has one number per cycle (QD, IR, Tavg, …); the `cycles`
struct has the curves (I, V, t, Qc, T, Qdlin). Units: current in **C-rate**, time in **minutes**.

**Why the code looks like this.** The files are MATLAB v7.3 = HDF5; every field is a *reference* you
dereference with `f[ref][()]`. 8 GB of curves won't fit in RAM, so each cycle is turned into ~15 scalars
immediately and the curve is discarded.

**Exercise.** Print one cycle (`I, V, Qc` every 12th sample) and find the four phases by eye.
Then explain why the 1C segment is the most "BMS-like" part of the data.

## 2. Cleaning — `soh/preprocess.py`

**Idea.** Hard physical filters (cycle 1, QD outside 0–1.2 Ah) apply everywhere. Spike *flags* remove
rows from **training only**; test labels are never smoothed (otherwise you grade yourself on an easier test).

**Lesson learned while building this.** A robust rule `|QD − rolling median| > 5·MAD` flagged **31 %** of
rows, because capacity is so smooth its MAD ≈ 0. An absolute floor (0.005 Ah) fixed it (0.24 % flagged).
Always print how many rows each rule touches.

> ✅ **Verified** on the 100,501 raw rows: without the floor 30.8 % flagged; with it **243 rows (0.24 %)**.
> Also: 124 cycle-1 rows and 4 impossible-QD rows dropped, 29 IR glitches and 85 temperature
> excursions set to NaN → 100,373 clean rows (train 28,139 · primary 31,034 · secondary 41,200).
> Batch-3 minimum SOH is 80.00–80.15 % — the reason the EOL alarm sits at 80.5 %.

**Exercise.** Set `SPIKE_FLOOR_AH = 0` in `config.py` and re-run `clean_mit`; count the flags.

## 3. Features and the leakage audit — `soh/features.py`

**Idea.** Every MIT cell is discharged identically, so **any discharge quantity is capacity in disguise**
(Q = I·t with I fixed). Same for the charge accepted in the 1C phase: the fast phase always stops at
0.88 Ah and starts from empty, so 1C charge ≈ QC − 0.88 ≈ QD − 0.88.

**The audit** fits a one-feature tree per candidate, tested on *unseen cells*. Known disguised labels
(discharge time/energy, 1C CC charge) score R² ≈ 0.93–0.95, so the threshold is **calibrated** from them
(R² ≥ 0.922 ⇒ excluded) instead of guessed.

**Lesson learned.** Batch 3 rests only 5 s after fast charging (batches 1–2: minutes). Its voltage was
still *falling* when the 1C phase began, which turned into a fake dQ/dV spike (24 vs < 4). Fix: start
the IC window at the voltage minimum. A protocol detail broke a feature; only plotting per batch revealed it.

> ✅ **Verified.** Calibrated threshold = **0.922**. Definitional leaks scored 0.932 (`dis_energy`),
> 0.943 (`dis_time`), 0.947 (`chg_t_cc`), 0.949 (`chg_q_cc`); QD/QC ≈ 1.00. No *candidate* feature was
> excluded empirically — the strongest honest ones are `d_chg_v_start` 0.890, `chg_ic_peak` 0.897.
> The algebra also holds in the data: `chg_q_cc` = QC − 0.88 − `chg_q_cv` to a median 0.003 Ah.
> After the IC-window fix, batch-3 `chg_ic_peak` has median 3.9 (max 5.6), in line with batches 1–2.

**Exercise.** Add `QD` to the `A2` feature list and re-run. RMSE drops about 3× (✅ measured with tuned
XGBoost: primary 1.24 → 0.39, secondary 1.74 → 0.73 % SOH): that is what leakage looks like. It does not
reach 0 because trees predict piecewise constants and many other features already compete for splits.

## 4. Splitting — `soh/splits.py`

**Idea.** Consecutive cycles of one cell are near-copies. Split by **cell**: official 41 / 43 / 40.
The secondary set (batch 3) was cycled a year later → also tests batch shift.

> ✅ **Verified** (RESULTS.md §5, same 84 cells, 50 % held out): random-row RMSE 0.15 (XGB) / 0.16 (RF)
> vs cell-wise 0.56 / 0.65 → the wrong split looks **~4× better** than it is. Over 10 random 80/20 cell
> splits of all 124 cells XGBoost scores 0.52 ± 0.10 (range 0.33–0.66): the official split is a
> harder-than-average draw, and the cell choice alone moves RMSE by 2×.

**Exercise.** `reports/RESULTS.md` §5 shows the random-row split vs cell-wise split. Explain the gap
in one sentence to a non-specialist.

## 5–6. Models — `soh/models/tabular.py`, `soh/models/deep.py`

**Idea.** Trees (RF, XGBoost) are the expected winners on tabular per-cycle features; Ridge tests
"is a straight line enough?"; LSTM/GRU/1D-CNN see windows of k cycles **from one cell only**.
Tuning uses `GroupKFold(groups=cell_id)` on training cells; deep early stopping uses 8 *training*
cells as validation. No test cell is touched before the final evaluation.

> ✅ **Verified** (% SOH RMSE, primary / secondary; mean of 3 seeds):
>
> | Model | Primary | Secondary |
> |---|---|---|
> | Dummy (mean) | 5.14 | 4.01 |
> | XGBoost, cycle count only (A0) | 4.06 | 3.53 |
> | Ridge (A4) | 1.14 | **5.88** |
> | Random Forest (A4) | 1.20 | 2.46 |
> | XGBoost (A4, main) | 1.22 | 3.00 |
> | XGBoost (A2, **BMS**) | 1.24 | **1.88** |
> | LSTM / GRU / 1D-CNN (k = 10) | 1.09 / 1.15 / **0.97** | 4.05 / 4.17 / 4.46 |
>
> What the numbers say, versus the expectations above:
> - Trees were *not* the clear winners on the primary test: every learned model lands at 1.0–1.2,
>   and the 1D-CNN is best. The expectation was wrong; the differences are small.
> - **Batch shift is the real problem.** On the secondary (batch 3) cells Ridge and the deep models
>   degrade 4–5×; trees degrade least. The simpler BMS feature set (A2) generalises *best* (1.88);
>   adding ΔQ(V) and `_rm5` history (A3, A4) helps primary slightly but hurts secondary.
> - The ablation ladder (RESULTS.md §4): IR + temperature alone 2.53 → + charge segment 1.24 →
>   + ΔQ(V) 1.20 → + history 1.22. The 1C charge segment carries almost all the signal.
> - Window length (LSTM, seed 0): k = 5/10/20/30 → primary 1.12 / 1.05 / 1.03 / 1.02; secondary
>   is noisy (3.6–4.4). Longer windows help primary a little.

**Exercise.** In `deep.py`, change `k` and compare with the window-length table in RESULTS.md.

## 7. Evaluation — `soh/metrics.py`

RMSE/MAE/MAPE/R², per-SOH-band RMSE (errors near 80 % matter most), per-cell RMSE (mean, worst),
and **EOL-alarm error**: first cycle predicted < 80.5 % minus first cycle measured < 80.5 %, in cycles.
Negative = early alarm (safe), positive = late alarm (dangerous).

> ✅ **Verified — and a warning.** Every model misses the alarm on **all 40 secondary cells** (1D-CNN: 37/40)
> and on 6–21 of 43 primary cells. Batch-3 cells end at 80.0–80.15 %, so the truth sits below 80.5 % for
> only a few cycles; a model that is even slightly optimistic near end of life never crosses it.
> Low RMSE does not mean safe alarms: errors are largest in the 80–85 % band (2.0–2.5 primary,
> 2.4–3.6 secondary) — exactly where the decision is made.

**Exercise.** Why is pooled R² misleading here? (Hint: it rewards getting *between-cell* offsets right.)

## 8. Explainability — `soh/explain.py`

SHAP explains **the model, not the chemistry**. Permutation importance is computed on held-out cells,
with each feature permuted *together with* its `d_`, `r_`, `_rm5` copies so correlated copies don't
split the credit.

> ✅ **Verified.** The IC peak height (`chg_ic_peak` group) dominates both models (RMSE +3.7 main, +2.4 BMS
> when permuted), then the 1C start voltage (+1.2 / +1.8). ΔQ(V) adds little (≤ 0.05) once the charge
> features are present, and IR matters only in the BMS model (+0.15).

## 9. Use in a BMS — `soh/predict.py`

`python -m soh.predict --demo` prints estimated capacity (Ah), SOH and a status
(HEALTHY ≥ 90 %, MONITOR 80–90 %, REPLACE/second-life < 80 %) for an unseen cell, using the
**BMS model** (charge segment + IR + temperature only; no lab discharge needed).

> ✅ **Verified.** The demo cell b3c0 (1,007 cycles, never seen in training) gets RMSE **1.24 % SOH**.
> Footprint: 1,000 trees, depth ≤ 6, 3.5 MB, ~2 ms per cycle in Python on one CPU core.

---

## Glossary (one line each)

| Term | Meaning |
|---|---|
| SOH | present capacity ÷ rated capacity × 100 |
| C-rate | current relative to capacity; 1C empties the cell in ~1 h, 4C in ~15 min |
| CC-CV | constant current to a voltage limit, then constant voltage while current tapers |
| LLI / LAM | loss of lithium inventory / loss of active material — the two ageing modes |
| IC (dQ/dV) | incremental capacity; its peaks are graphite staging steps that shrink with age |
| ΔQ(V) | Q_k(V) − Q_10(V): how the discharge curve shape changed since cycle 10 |
| Leakage | information about the test answer reaching the model during training |
| GroupKFold | cross-validation where a whole group (cell) is either in or out of a fold |
| Knee | the point where fade accelerates near end of life |
