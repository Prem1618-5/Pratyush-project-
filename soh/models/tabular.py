"""Stage 5: tabular baselines -- Dummy, Ridge, Random Forest, XGBoost.

Scalers/imputers live INSIDE sklearn Pipelines, so they are fit on training cells only.
Hyper-parameters are tuned with GroupKFold(5) over training cells (groups = cell_id),
so no test cell is ever seen while tuning.
"""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEARCH_SPACES = {   # practitioner starting ranges (doc Table 9), not sourced optima
    "ridge": {"m__alpha": loguniform(1e-3, 1e3)},
    "rf": {"m__max_depth": [None, 10, 20, 30], "m__min_samples_leaf": randint(1, 6),
           "m__max_features": ["sqrt", 0.5]},
    "xgb": {"n_estimators": [300, 600, 1000], "learning_rate": uniform(0.03, 0.07),
            "max_depth": randint(3, 9), "subsample": uniform(0.7, 0.2),
            "colsample_bytree": uniform(0.7, 0.2), "min_child_weight": randint(1, 20)},
}
N_ITER = {"ridge": 20, "rf": 10, "xgb": 20}


def make_model(name, seed=0, **params):
    if name == "dummy":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", DummyRegressor())])
    if name == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", Ridge(alpha=params.get("m__alpha", 1.0)))])
    if name == "rf":
        p = {k.replace("m__", ""): v for k, v in params.items()}
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed,
                                                     **p))])
    if name == "xgb":   # XGBoost handles missing values natively; trees need no scaling
        base = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5)
        base.update(params)
        return XGBRegressor(tree_method="hist", n_jobs=4, random_state=seed, **base)
    raise ValueError(name)


def tune(name, X, y, groups, seed=0, subsample_every=3):
    """Random search with GroupKFold over training cells. Rows are thinned (every 3rd cycle)
    to keep tuning fast; neighbouring cycles are near-duplicates, so little is lost."""
    if name not in SEARCH_SPACES:
        return {}
    idx = np.arange(0, len(y), subsample_every)
    search = RandomizedSearchCV(make_model(name, seed), SEARCH_SPACES[name], n_iter=N_ITER[name],
                                cv=GroupKFold(5), scoring="neg_root_mean_squared_error",
                                random_state=seed, n_jobs=1 if name != "ridge" else -1)
    search.fit(X.iloc[idx], y[idx], groups=groups[idx])
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return {"params": best, "cv_rmse": -search.best_score_}
