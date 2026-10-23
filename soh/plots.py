"""Stage 8: report figures (static PNGs, light theme, one shared palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from soh import config, metrics  # noqa: E402

# Validated categorical palette (dataviz reference palette, light mode), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPLIT_COLOR = {"train": C[0], "test_primary": C[1], "test_secondary": C[2]}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def capacity_fade(df):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for split, d in df.groupby("split"):
        for i, (_, c) in enumerate(d.groupby("cell_id")):
            ax.plot(c["cycle"], c["soh"], color=SPLIT_COLOR[split], lw=0.6, alpha=0.55,
                    label=f"{split} ({d['cell_id'].nunique()} cells)" if i == 0 else None)
    ax.axhline(config.MIT_EOL_SOH, color=INK, ls="--", lw=0.8)
    ax.text(ax.get_xlim()[1], 80.3, "EOL 80 %", ha="right", va="bottom", color=MUTED)
    ax.set(xlabel="Cycle", ylabel="SOH (%) = QD / 1.1 Ah x 100",
           title="Capacity fade of all 124 MIT cells, coloured by split")
    ax.legend(loc="upper right")
    return _save(fig, "01_capacity_fade.png")


def feature_scatter(train, feats):
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), sharey=True)
    sub = train.iloc[::7]
    for ax, f in zip(np.atleast_1d(axes), feats):
        ax.scatter(sub[f], sub["soh"], s=2, alpha=0.25, color=C[0], rasterized=True)
        ax.set(xlabel=f)
    np.atleast_1d(axes)[0].set_ylabel("SOH (%)")
    fig.suptitle("Top features vs SOH (training cells, every 7th cycle)")
    return _save(fig, "02_feature_scatter.png")


def pred_vs_true(preds, model_name):
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.8))
    for ax, (split, d) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(d["soh"], d["pred"], s=2, alpha=0.2, color=SPLIT_COLOR[split], rasterized=True)
        lo, hi = 75, 103
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
        rmse = metrics.rmse(d["soh"], d["pred"])
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured SOH (%)", ylabel="Estimated SOH (%)",
               title=f"{split}: RMSE {rmse:.2f} % SOH")
    fig.suptitle(f"{model_name}: estimated vs measured SOH (held-out cells)")
    return _save(fig, "03_pred_vs_true.png")


def cell_trajectories(d, per_cell, model_name, split):
    pc = per_cell.sort_values("RMSE")
    picks = {"best": pc.iloc[0], "median": pc.iloc[len(pc) // 2], "worst": pc.iloc[-1]}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (tag, row) in zip(axes, picks.items()):
        c = d[d["cell_id"] == row["cell_id"]].sort_values("cycle")
        ax.plot(c["cycle"], c["soh"], color=INK, lw=1.4, label="measured")
        ax.plot(c["cycle"], c["pred"], color=SPLIT_COLOR[split], lw=1, label="estimated")
        ax.axhline(80, color=MUTED, ls="--", lw=0.7)
        ax.set(title=f"{tag}: {row['cell_id']}  RMSE {row['RMSE']:.2f}", xlabel="Cycle")
    axes[0].set_ylabel("SOH (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{model_name} on {split} cells: best / median / worst")
    return _save(fig, f"04_trajectories_{split}.png")


def error_by_band(band_table):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bands = [c for c in band_table.columns if c.startswith("RMSE_")]
    x = np.arange(len(bands))
    w = 0.8 / len(band_table)
    for i, (name, row) in enumerate(band_table.iterrows()):
        ax.bar(x + i * w, row[bands], w * 0.92, color=C[i], label=name)
    ax.set_xticks(x + w * (len(band_table) - 1) / 2, [b.replace("RMSE_", "") + " %" for b in bands])
    ax.set(xlabel="True SOH band", ylabel="RMSE (% SOH)", title="Where do errors grow? (primary test)")
    ax.legend()
    return _save(fig, "05_error_by_band.png")


def bar_compare(labels, series, title, ylabel, fname):
    """series: {legend_name: values}"""
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 3.4))
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + i * w, vals, w * 0.92, color=C[i], label=name)
        ax.bar_label(b, fmt="%.2f", fontsize=7, color=MUTED, padding=1)
    ax.set_xticks(x + w * (len(series) - 1) / 2, labels, rotation=20, ha="right")
    ax.set(ylabel=ylabel, title=title)
    if len(series) > 1:
        ax.legend()
    return _save(fig, fname)


def nasa_plot(nasa_pred, title, fname, label="soh_init"):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, (cid, d) in zip(axes, nasa_pred.groupby("cell_id")):
        ax.plot(d["cycle"], d[label], color=INK, lw=1.4, label="measured")
        ax.plot(d["cycle"], d["pred"], color=C[1], lw=1, label="estimated")
        ax.set(title=cid, xlabel="Discharge cycle")
    axes[0].set_ylabel(f"{label} (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle(title)
    return _save(fig, fname)


def health_monitor(c, cell_id, fname="10_health_monitor_demo.png"):
    """What a BMS dashboard would show: estimated SOH with status bands."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axhspan(90, 105, color="#e8f3e8", lw=0)
    alarm = config.EOL_ALARM_SOH                  # same REPLACE level as soh.predict
    ax.axhspan(alarm, 90, color="#fdf3dc", lw=0)
    ax.axhspan(60, alarm, color="#fbe4e3", lw=0)
    ax.plot(c["cycle"], c["soh"], color=INK, lw=1.4, label="measured (lab reference)")
    ax.plot(c["cycle"], c["pred"], color=C[0], lw=1, label="ML estimate (BMS-feasible model)")
    x0, x = c["cycle"].min(), c["cycle"].max()
    ax.text(x0, 101.5, "HEALTHY (>= 90 %)", color="#2f6b2f", fontsize=8)
    ax.text(x0, 88.3, f"MONITOR ({alarm:g}-90 %)", color="#8a5a00", fontsize=8)
    ax.text(x, 75.5, f"REPLACE / 2nd life (< {alarm:g} %)", color="#9b2c2c", fontsize=8, ha="right")
    ax.set(ylim=(74, 103), xlabel="Cycle", ylabel="SOH (%)",
           title=f"Battery-health monitoring demo, unseen cell {cell_id}")
    ax.legend(loc="lower center")
    return _save(fig, fname)
