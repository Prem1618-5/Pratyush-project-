"""Explainability: SHAP (TreeSHAP) and grouped permutation importance on held-out cells."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from soh import config, metrics  # noqa: E402


def feature_group(name):
    """Group a feature with its derived versions (d_X, r_X, X_rm5) so correlated copies
    are permuted together and do not split the credit."""
    base = name.replace("_rm5", "")
    for p in ("d_", "r_"):
        if base.startswith(p):
            base = base[len(p):]
    return base


def grouped_permutation_importance(predict, X, y, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base_rmse = metrics.rmse(y, predict(X))
    groups = {}
    for c in X.columns:
        groups.setdefault(feature_group(c), []).append(c)
    rows = []
    for g, cols in groups.items():
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            incs.append(metrics.rmse(y, predict(Xp)) - base_rmse)
        rows.append({"group": g, "features": ", ".join(cols), "rmse_increase": np.mean(incs),
                     "std": np.std(incs)})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def shap_figures(model, X, prefix="06"):
    """Beeswarm (global) + waterfall (one cycle) for a tree model."""
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: which features drive the SOH estimate (primary-test cycles)")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / f"{prefix}_shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    i = int(np.argmin(np.abs(sv.values.sum(1) + sv.base_values - 85)))   # a cycle near 85 % SOH
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title("SHAP waterfall: one cycle's estimate, feature by feature")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / f"{prefix}_shap_waterfall.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=X.columns).sort_values(ascending=False)
    return mean_abs, [f"figures/{prefix}_shap_beeswarm.png", f"figures/{prefix}_shap_waterfall.png"]
