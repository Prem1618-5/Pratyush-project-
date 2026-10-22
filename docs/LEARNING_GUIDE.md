# Learning guide: read the code in this order

Each stage = one file, one key idea, one exercise. Run `pytest -q` any time; the tests in
`tests/test_core.py` are small, readable examples of every rule below.

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

**Exercise.** Add `QD` to the `A2` feature list and re-run. Watch RMSE collapse to ~0: that is what leakage
looks like.

## 4. Splitting — `soh/splits.py`

**Idea.** Consecutive cycles of one cell are near-copies. Split by **cell**: official 41 / 43 / 40.
The secondary set (batch 3) was cycled a year later → also tests batch shift.

**Exercise.** `reports/RESULTS.md` §5 shows the random-row split vs cell-wise split. Explain the gap
in one sentence to a non-specialist.

## 5–6. Models — `soh/models/tabular.py`, `soh/models/deep.py`

**Idea.** Trees (RF, XGBoost) are the expected winners on tabular per-cycle features; Ridge tests
"is a straight line enough?"; LSTM/GRU/1D-CNN see windows of k cycles **from one cell only**.
Tuning uses `GroupKFold(groups=cell_id)` on training cells; deep early stopping uses 8 *training*
cells as validation. No test cell is touched before the final evaluation.

**Exercise.** In `deep.py`, change `k` and compare with the window-length table in RESULTS.md.

## 7. Evaluation — `soh/metrics.py`

RMSE/MAE/MAPE/R², per-SOH-band RMSE (errors near 80 % matter most), per-cell RMSE (mean, worst),
and **EOL-alarm error**: first cycle predicted < 80.5 % minus first cycle measured < 80.5 %, in cycles.
Negative = early alarm (safe), positive = late alarm (dangerous).

**Exercise.** Why is pooled R² misleading here? (Hint: it rewards getting *between-cell* offsets right.)

## 8. Explainability — `soh/explain.py`

SHAP explains **the model, not the chemistry**. Permutation importance is computed on held-out cells,
with each feature permuted *together with* its `d_`, `r_`, `_rm5` copies so correlated copies don't
split the credit.

## 9. Use in a BMS — `soh/predict.py`

`python -m soh.predict --demo` prints estimated capacity (Ah), SOH and a status
(HEALTHY ≥ 90 %, MONITOR 80–90 %, REPLACE/second-life < 80 %) for an unseen cell, using the
**BMS model** (charge segment + IR + temperature only; no lab discharge needed).

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
