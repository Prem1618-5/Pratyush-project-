"""Stage 1+3: parse the MIT-Stanford-TRI .mat (HDF5) files into one per-cycle table.

What happens here
-----------------
1. Read each batch file with h5py (MATLAB v7.3 = HDF5; every field is an object
   reference that must be dereferenced with f[ref][()]).
2. Reproduce the authors' official cleaning (124 cells) and the batch-1 -> batch-2
   stitching (5 cells were moved to the next batch mid-life).
3. For every cycle, read the within-cycle curves (I, V, t, Qc, T, Qdlin) once and
   turn them into a handful of scalar features, so the 8 GB of curves never sit in RAM.

Output: data/processed/mit_cycles.parquet  (one row per cell x cycle)

Units in the raw files: current I in C-rate (1C = 1.1 A), time t in minutes,
capacities in Ah, temperature in degC, IR in ohm.
"""
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from soh import config

QDLIN_GRID = np.linspace(*config.QDLIN_V)
WIN = np.where((QDLIN_GRID <= config.DQ_WINDOW_V[0]) & (QDLIN_GRID >= config.DQ_WINDOW_V[1]))[0]

CHG_KEYS = ["chg_v_start", "chg_t_cc", "chg_q_cc", "chg_t_cv", "chg_q_cv", "chg_cv_tau",
            "chg_ic_peak", "chg_ic_peak_v", "chg_dT"]
DIS_KEYS = ["dis_time", "dis_energy"]


# --------------------------------------------------------------------------
# per-cycle feature functions (pure numpy, unit-tested in tests/)
# --------------------------------------------------------------------------
def charge_segment_features(I, V, t, Qc, T):
    """Features from the *uniform* part of charging: 1C CC from 80 % SOC to 3.6 V, then CV.

    Every cell gets exactly this segment, whatever its fast-charge policy, so it is the
    closest thing in this dataset to what a BMS sees on a normal charge.
    """
    out = dict.fromkeys(CHG_KEYS, np.nan)
    if len(I) < 10:
        return out
    dis = np.where(I < -0.5)[0]
    end = dis[0] if len(dis) else len(I)
    I, V, t, Qc, T = I[:end], V[:end], t[:end], Qc[:end], T[:end]

    # 1C constant-current phase AFTER the fast charge (Qc > 0.8 Ah excludes policies that
    # start with a short 1C step, e.g. "1C(4%)-6C").
    cc = np.where((np.abs(I - 1.0) < 0.05) & (Qc > 0.8) & (V < 3.598))[0]
    if len(cc) < 3:
        return out
    breaks = np.where(np.diff(cc) > 1)[0]
    s, e = cc[0], (cc[breaks[0]] if len(breaks) else cc[-1])

    out["chg_t_cc"] = t[e] - t[s]
    out["chg_q_cc"] = Qc[e] - Qc[s]

    # CV phase at 3.6 V: current tapers from 1C to C/50
    after = np.arange(e + 1, len(I))
    cv = after[(V[after] >= 3.59) & (I[after] > 0.005)]
    if len(cv) >= 3:
        out["chg_t_cv"] = t[cv[-1]] - t[cv[0]]
        out["chg_q_cv"] = Qc[cv[-1]] - Qc[cv[0]]
        half = cv[I[cv] <= 0.5]
        out["chg_cv_tau"] = (t[half[0]] - t[cv[0]]) if len(half) else np.nan
        seg_end = cv[-1]
    else:
        seg_end = e
    out["chg_dT"] = T[seg_end] - T[s]          # signed temperature change over the segment

    # Batch 3 rests only 5 s after the fast charge (batches 1-2: minutes), so the voltage is still
    # RELAXING downwards when 1C starts. Start voltage/IC features at the voltage minimum, i.e.
    # after relaxation; otherwise the dip becomes a fake dQ/dV spike (seen: peak 24 vs <4).
    m = s + int(np.argmin(V[s:e + 1]))
    out["chg_v_start"] = V[m]

    # Incremental capacity dQ/dV on the rising 1C CC phase (5 mV grid, Savitzky-Golay smoothing)
    v, q = np.maximum.accumulate(V[m:e + 1]), Qc[m:e + 1]
    v, idx = np.unique(v, return_index=True)
    q = q[idx]
    if len(v) >= 5 and v[-1] - v[0] >= 0.03:
        grid = np.arange(v[0], v[-1], 0.005)
        if len(grid) >= 9:
            ic = savgol_filter(np.gradient(np.interp(grid, v, q), grid), 9, 3)
            j = int(np.argmax(ic))
            out["chg_ic_peak"], out["chg_ic_peak_v"] = ic[j], grid[j]
    return out


def discharge_features(I, V, t):
    """Label-equivalent quantities. Computed ONLY so the leakage audit can prove they are."""
    out = dict.fromkeys(DIS_KEYS, np.nan)
    d = np.where(I < -3.9)[0]                               # 4C constant-current discharge
    if len(d) < 3:
        return out
    out["dis_time"] = t[d[-1]] - t[d[0]]
    out["dis_energy"] = np.trapezoid(V[d] * -I[d] * config.MIT_RATED_AH, t[d]) / 60.0   # Wh
    return out


def dq_features(qwin, cycles):
    """Severson-style dQ(V) features in a PARTIAL window (3.3 -> 2.9 V).

    dQ_k(V) = Q_k(V) - Q_ref(V), ref = cycle 10 (or the nearest valid later cycle).
    Cycles BEFORE the reference stay NaN: they would need a curve that has not been measured yet.
    """
    n = len(cycles)
    res = np.full((n, 3), np.nan)
    ok = np.isfinite(qwin).all(1) & (np.abs(qwin).sum(1) > 0)
    cand = np.where(ok & (cycles >= config.DQ_REF_CYCLE))[0]
    if len(cand) == 0:
        return res
    ref = qwin[cand[0]]
    dq = qwin - ref
    with np.errstate(divide="ignore", invalid="ignore"):
        res[:, 0] = np.log10(np.var(dq, axis=1) + 1e-10)
        res[:, 1] = np.log10(np.abs(np.min(dq, axis=1)) + 1e-10)
        res[:, 2] = np.mean(dq, axis=1)
    res[~ok] = np.nan
    res[cycles < cycles[cand[0]]] = np.nan          # causal: no look-ahead to the reference cycle
    return res


# --------------------------------------------------------------------------
# HDF5 parsing
# --------------------------------------------------------------------------
def _vec(f, ref):
    return np.asarray(f[ref][()], dtype=float).ravel()


def _parse_cell(f, batch, i):
    summ = f[batch["summary"][i, 0]]
    cyc = f[batch["cycles"][i, 0]]
    s = {k: np.hstack(summ[k][0, :].tolist()) for k in
         ["cycle", "IR", "QCharge", "QDischarge", "Tavg", "Tmin", "Tmax", "chargetime"]}
    n = min(len(s["cycle"]), cyc["I"].shape[0])
    rows, qwin = [], np.full((n, len(WIN)), np.nan, dtype=np.float32)
    for j in range(n):
        I, V, t, Qc, T, Qdlin = (_vec(f, cyc[k][j, 0]) for k in ["I", "V", "t", "Qc", "T", "Qdlin"])
        r = {"cycle": int(s["cycle"][j]), "IR": s["IR"][j], "QC": s["QCharge"][j],
             "QD": s["QDischarge"][j], "Tavg": s["Tavg"][j], "Tmin": s["Tmin"][j],
             "Tmax": s["Tmax"][j], "chargetime": s["chargetime"][j]}
        if len(I) == len(V) == len(t) == len(Qc) == len(T) and len(I) > 10:
            r.update(charge_segment_features(I, V, t, Qc, T))
            r.update(discharge_features(I, V, t))
        if len(Qdlin) == len(QDLIN_GRID):
            qwin[j] = Qdlin[WIN]
        rows.append(r)
    policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()
    return pd.DataFrame(rows), qwin, policy


def parse_batch(batch_name):
    fname = config.RAW_DIR / config.MIT_FILES[batch_name][0]
    prefix = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3"}[batch_name]
    cells = {}
    with h5py.File(fname, "r") as f:
        batch = f["batch"]
        for i in range(batch["summary"].shape[0]):
            df, qwin, policy = _parse_cell(f, batch, i)
            cells[f"{prefix}c{i}"] = (df, qwin, policy)
            print(f"  {batch_name} {prefix}c{i}: {len(df)} cycles  ({policy})", flush=True)
    return batch_name, cells


def build_mit_table(processes=3):
    """Parse all batches, apply official cleaning + stitching, return the per-cycle table."""
    with Pool(processes) as pool:
        parsed = dict(pool.map(parse_batch, list(config.MIT_FILES)))
    b1, b2, b3 = parsed["2017-05-12"], parsed["2017-06-30"], parsed["2018-04-12"]
    raw_count = len(b1) + len(b2) + len(b3)

    for k in config.BATCH1_DROP:
        b1.pop(k)
    for k1, k2 in zip(config.BATCH1_STITCH, config.BATCH2_STITCH):
        df1, q1, pol = b1[k1]
        df2, q2, _ = b2.pop(k2)
        df2 = df2.copy()
        df2["cycle"] = df2["cycle"] + len(df1)              # offset, as in the official loader
        b1[k1] = (pd.concat([df1, df2], ignore_index=True), np.vstack([q1, q2]), pol)
    for k in config.BATCH3_DROP:
        b3.pop(k)

    frames = []
    for batch_name, cells in [("2017-05-12", b1), ("2017-06-30", b2), ("2018-04-12", b3)]:
        for cid, (df, qwin, pol) in cells.items():            # insertion order = official order
            df = df.copy()
            dq = dq_features(qwin, df["cycle"].to_numpy())
            df["dq_var_log"], df["dq_min_log"], df["dq_mean"] = dq.T
            df.insert(0, "cell_id", cid)
            df.insert(1, "batch", batch_name)
            df.insert(2, "policy", pol)
            frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    table["cell_index"] = table["cell_id"].map({c: i for i, c in enumerate(table["cell_id"].unique())})
    print(f"raw cells: {raw_count}  ->  cleaned cells: {table['cell_id'].nunique()}  "
          f"rows: {len(table):,}")
    return table


if __name__ == "__main__":
    config.ensure_dirs()
    tbl = build_mit_table()
    out = config.PROCESSED_DIR / "mit_cycles_raw.parquet"
    tbl.to_parquet(out, index=False)
    print("wrote", out)
