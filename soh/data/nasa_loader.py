"""Load NASA PCoE cells B0005/6/7/18 into the same per-cycle layout as MIT.

One row per *discharge* (that is where the reference capacity is measured). Features
come from the charge that preceded it (CC 1.5 A to 4.2 V, then CV to 20 mA) and from
the most recent EIS impedance test (Re + Rct is used as the resistance analogue of
MIT's pulse IR). Time is converted to minutes to match MIT.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

from soh import config


def _charge_features(d):
    I, V, t, T = (np.asarray(d[k], float) for k in
                  ["Current_measured", "Voltage_measured", "Time", "Temperature_measured"])
    t = t / 60.0
    out = dict(chg_t_cc=np.nan, chg_t_cv=np.nan, chg_cv_tau=np.nan, chg_dT=np.nan)
    on = np.where(I > 0.01)[0]
    if len(on) < 10:
        return out
    cv = on[V[on] >= 4.19]
    if len(cv) < 3:
        return out
    out["chg_t_cc"] = t[cv[0]] - t[on[0]]
    tail = cv[I[cv] > 0.02]
    if len(tail) >= 2:
        out["chg_t_cv"] = t[tail[-1]] - t[cv[0]]
    half = cv[I[cv] <= 0.75]                                  # current halves from 1.5 A
    out["chg_cv_tau"] = (t[half[0]] - t[cv[0]]) if len(half) else np.nan
    out["chg_dT"] = T[on[-1]] - T[on[0]]
    return out


def load_cell(cell):
    m = sio.loadmat(config.RAW_DIR / "nasa" / f"{cell}.mat", simplify_cells=True)[cell]["cycle"]
    rows, last_charge, last_imp, k = [], None, (np.nan, np.nan), 0
    for c in m:
        if c["type"] == "charge":
            last_charge = _charge_features(c["data"])
        elif c["type"] == "impedance":
            re, rct = float(np.real(c["data"]["Re"])), float(np.real(c["data"]["Rct"]))
            if 0 < re < 1 and 0 < rct < 1:                    # drop failed EIS fits
                last_imp = (re, rct)
        elif c["type"] == "discharge":
            k += 1
            d = c["data"]
            r = {"cell_id": cell, "cycle": k, "QD": float(d["Capacity"]),
                 "Re": last_imp[0], "Rct": last_imp[1], "IR": last_imp[0] + last_imp[1],
                 "Tmax_dis": float(np.max(d["Temperature_measured"]))}
            r.update(last_charge or {})
            rows.append(r)
    return pd.DataFrame(rows)


def build_nasa_table():
    df = pd.concat([load_cell(c) for c in config.NASA_CELLS], ignore_index=True)
    # Re/Rct are only measured every few cycles: carry the last value forward, back-fill the start
    df[["Re", "Rct", "IR"]] = df.groupby("cell_id")[["Re", "Rct", "IR"]].transform(lambda s: s.ffill().bfill())
    df["soh"] = df["QD"] / config.NASA_RATED_AH * 100
    first = df.groupby("cell_id")["QD"].transform("first")
    df["soh_init"] = df["QD"] / first * 100
    return df


if __name__ == "__main__":
    config.ensure_dirs()
    t = build_nasa_table()
    print(t.groupby("cell_id").agg(cycles=("cycle", "max"), soh_first=("soh", "first"),
                                   soh_last=("soh", "last")))
    t.to_parquet(config.PROCESSED_DIR / "nasa_cycles.parquet", index=False)
