"""Fast tests on synthetic data (no 8 GB download needed):  pytest -q"""
import numpy as np
import pandas as pd

from soh import features, metrics, splits
from soh.data.mit_loader import charge_segment_features, discharge_features, dq_features
from soh.models.deep import build_windows


def synthetic_cycle(cap=1.05):
    """fast charge to 0.88 Ah -> rest -> 1C CC to 3.6 V -> CV taper -> 4C discharge (units: C, min, Ah)."""
    I, V, t, Q = [], [], [], []
    tt, q = 0.0, 0.0
    for _ in range(100):                      # 5C fast charge to 0.88 Ah
        tt += 0.096; q += 5 * 1.1 * 0.096 / 60
        I.append(5.0); V.append(3.3 + 0.2 * q / 0.88); t.append(tt); Q.append(q)
    for _ in range(20):                       # rest
        tt += 0.25; I.append(0.0); V.append(3.35); t.append(tt); Q.append(q)
    while q < cap - 0.03:                     # 1C CC
        tt += 0.1; q += 1.1 * 0.1 / 60
        I.append(1.0); V.append(3.38 + 0.2 * (q - 0.88) / (cap - 0.91)); t.append(tt); Q.append(q)
    cur = 1.0
    while cur > 0.02:                         # CV taper
        tt += 0.1; cur *= 0.97; q += cur * 1.1 * 0.1 / 60
        I.append(cur); V.append(3.6); t.append(tt); Q.append(q)
    for _ in range(150):                      # 4C discharge
        tt += 0.1; I.append(-4.0); V.append(3.1); t.append(tt); Q.append(q)
    return [np.array(a, float) for a in (I, V, t, Q)] + [np.full(len(I), 30.0)]


def test_charge_segment_features_found():
    I, V, t, Q, T = synthetic_cycle()
    f = charge_segment_features(I, V, t, Q, T)
    assert 3.37 < f["chg_v_start"] < 3.4
    assert f["chg_t_cc"] > 5 and f["chg_t_cv"] > 5
    assert 0 < f["chg_cv_tau"] < f["chg_t_cv"]
    assert np.isfinite(f["chg_ic_peak"])


def test_cc_charge_tracks_capacity_ie_is_a_leak():
    """The 1C CC charge grows with capacity: why chg_q_cc is EXCLUDED as label-equivalent."""
    small = charge_segment_features(*synthetic_cycle(0.98))["chg_q_cc"]
    big = charge_segment_features(*synthetic_cycle(1.08))["chg_q_cc"]
    assert big > small


def test_discharge_time_is_capacity():
    I, V, t, Q, T = synthetic_cycle()
    assert abs(discharge_features(I, V, t)["dis_time"] - 149 * 0.1) < 1e-6


def test_dq_reference_cycle_is_zero():
    cycles = np.arange(2, 30)
    q = np.linspace(0, 1, 50)[None, :] * (1 - 0.001 * cycles[:, None])
    res = dq_features(q.astype(np.float32), cycles)
    ref_row = np.where(cycles == 10)[0][0]
    assert res[ref_row, 2] == 0 and res[-1, 2] < 0      # later cycles lose capacity


def test_official_split_sizes_and_disjoint():
    df = pd.DataFrame({"cell_index": np.arange(124)})
    s = splits.official_split(df)
    assert (s == "train").sum() == 41 and (s == "test_primary").sum() == 43 and (s == "test_secondary").sum() == 40
    assert len(set(splits.TRAIN_IDX) & set(splits.PRIMARY_IDX)) == 0


def test_windows_never_mix_cells():
    df = pd.DataFrame({"cell_id": np.repeat(["a", "b", "c"], 15), "cycle": np.tile(np.arange(15), 3),
                       "x": np.repeat([1.0, 2.0, 3.0], 15), "soh": 90.0})
    X, y, idx = build_windows(df, ["x"], k=5)
    assert len(X) == len(df)
    assert all(np.unique(w).size == 1 for w in X[:, :, 0])   # each window holds one cell's value only


def test_metrics_and_eol():
    m = metrics.regression_metrics([90, 80], [91, 79])
    assert abs(m["RMSE"] - 1) < 1e-9 and abs(m["MAE"] - 1) < 1e-9
    d = pd.DataFrame({"cell_id": "a", "cycle": np.arange(100), "soh": np.linspace(90, 78, 100)})
    d["pred"] = d["soh"] + 1                      # optimistic model -> late alarm
    e = metrics.eol_errors(d)
    assert e["eol_error"].iloc[0] > 0


def test_audit_flags_disguised_label_and_keeps_noise_feature():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(20):
        cyc = np.arange(200)
        soh = 100 - cyc * rng.uniform(0.02, 0.1)
        rows.append(pd.DataFrame({"cell_id": f"c{c}", "cycle": cyc, "soh": soh,
                                  "dis_time": soh * 0.14 + rng.normal(0, 0.05, 200),   # disguised label
                                  "chg_t_cc": soh * 0.1 + rng.normal(0, 0.02, 200),
                                  "chg_v_start": soh * 0.01 + rng.normal(0, 0.03, 200),  # label + more noise
                                  "IR": rng.normal(0.02, 0.001, 200)}))
    a = features.leakage_audit(pd.concat(rows)).set_index("feature")
    assert a.loc["dis_time", "status"].startswith("EXCLUDED")
    assert a.loc["IR", "status"] == "safe"


def test_clip_uses_training_rows_only():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 100.0]})
    out, _ = features.clip_to_train(df, ["x"], np.array([True, True, True, False]), q=0.0)
    assert out["x"].max() == 2.0
