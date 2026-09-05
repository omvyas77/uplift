"""Uplift learners, in increasing order of sophistication.

The comparison between them is the story, not any single one of them. Five are
implemented here or wrapped from EconML; only the causal forest gives honest
confidence intervals on tau(x), which is why it is worth its runtime.

EconML's fit signature is `est.fit(Y, T, X=X, W=W)` - outcome first, treatment
second. Getting that backwards silently produces garbage, so the wrappers below
assert their shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression


def _base_clf(seed: int = 0, **kw) -> LGBMClassifier:
    params: dict[str, Any] = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=seed,
        n_jobs=-1,
    )
    params.update(kw)
    return LGBMClassifier(**params)


def _base_reg(seed: int = 0, **kw) -> LGBMRegressor:
    params: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=500,
        verbose=-1,
        random_state=seed,
        n_jobs=-1,
    )
    params.update(kw)
    return LGBMRegressor(**params)


def _check(X, w, y) -> None:
    if not (len(X) == len(w) == len(y)):
        raise ValueError(f"length mismatch: X={len(X)}, w={len(w)}, y={len(y)}")
    if set(np.unique(w)) - {0, 1}:
        raise ValueError("treatment must be binary 0/1")


# ---------- 1. S-learner ----------
class SLearner:
    """One model on [X, W]; tau_hat = f(X,1) - f(X,0).

    Known failure mode: when the treatment signal is weak relative to the
    covariates, the trees barely split on W and tau_hat collapses toward zero.
    `treatment_gain_share_` measures exactly that, so the pathology is checked
    rather than hoped away.
    """

    name = "s_learner"

    def __init__(self, seed: int = 0):
        self.model = _base_clf(seed)

    def fit(self, X, w, y):
        _check(X, w, y)
        self.model.fit(np.column_stack([X, w]), y)
        gains = self.model.booster_.feature_importance(importance_type="gain")
        self.treatment_gain_share_ = float(gains[-1] / gains.sum())
        return self

    def predict(self, X):
        n = len(X)
        p1 = self.model.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
        return p1 - p0

    def diagnostics(self) -> dict:
        return {"treatment_gain_share": getattr(self, "treatment_gain_share_", None)}


# ---------- 2. T-learner ----------
class TLearner:
    """Separate models per arm.

    Simple and strong, but on an 85/15 design the control model sees only 15% of
    the data, so it is much the noisier of the two - which is the whole reason
    the X-learner exists.
    """

    name = "t_learner"

    def __init__(self, seed: int = 0):
        self.m1, self.m0 = _base_clf(seed), _base_clf(seed + 1)

    def fit(self, X, w, y):
        _check(X, w, y)
        self.m1.fit(X[w == 1], y[w == 1])
        self.m0.fit(X[w == 0], y[w == 0])
        self.n_treated_, self.n_control_ = int((w == 1).sum()), int((w == 0).sum())
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]

    def diagnostics(self) -> dict:
        return {
            "n_treated": getattr(self, "n_treated_", None),
            "n_control": getattr(self, "n_control_", None),
        }


# ---------- 3. Class transformation (transformed outcome) ----------
class ClassTransformation:
    """Z = Y*W/e - Y*(1-W)/(1-e); E[Z|X] = tau(X).

    One model, and it targets uplift directly. Cheap, and a surprisingly hard
    baseline to beat. On an 85/15 design the (1-e) denominator is small, so the
    control-side contribution is high-variance - worth knowing before reading
    its Qini.
    """

    name = "class_transformation"

    def __init__(self, seed: int = 0):
        self.model = _base_reg(seed, n_estimators=400, num_leaves=63)

    def fit(self, X, w, y, propensity: float | np.ndarray | None = None):
        _check(X, w, y)
        e = w.mean() if propensity is None else propensity
        z = y * w / e - y * (1 - w) / (1 - e)
        self.model.fit(X, z)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def diagnostics(self) -> dict:
        return {}


# ---------- 4-6. EconML wrappers ----------
def fit_x_learner(X, w, y, seed: int = 0):
    """X-learner: imputes the effect within each arm, then blends by propensity.
    Designed for exactly the unbalanced-arm case this dataset has."""
    from econml.metalearners import XLearner

    _check(X, w, y)
    est = XLearner(
        models=_base_clf(seed),
        propensity_model=LogisticRegression(max_iter=1000),
        cate_models=_base_reg(seed),
    )
    est.fit(y, w, X=X)
    return est


def fit_dr_learner(X, w, y, seed: int = 0, cv: int = 3):
    """Doubly robust + cross-fitted. Consistent if EITHER the outcome model or
    the propensity model is right - the strongest guarantee of the set."""
    from econml.dr import DRLearner

    _check(X, w, y)
    est = DRLearner(
        model_propensity=LogisticRegression(max_iter=1000),
        model_regression=_base_reg(seed),
        model_final=_base_reg(seed),
        cv=cv,
        random_state=seed,
    )
    est.fit(y, w, X=X)
    return est


def fit_causal_forest(X, w, y, seed: int = 0, max_n: int = 400_000, cv: int = 3):
    """Causal forest.

    The only estimator here that gives honest confidence intervals on tau(x)
    via `effect_interval`. Subsampled because 14M rows will not finish; the cap
    is a runtime decision, stated rather than hidden.
    """
    from econml.dml import CausalForestDML

    _check(X, w, y)
    if len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, w, y = X[idx], w[idx], y[idx]
    est = CausalForestDML(
        model_y=_base_reg(seed, n_estimators=200),
        model_t=_base_clf(seed, n_estimators=200),
        discrete_treatment=True,
        n_estimators=300,
        min_samples_leaf=200,
        max_samples=0.4,
        cv=cv,
        random_state=seed,
    )
    est.fit(y, w, X=X)
    return est


class EconMLWrapper:
    """Uniform .predict(X) over EconML estimators, so the eval loop is uniform."""

    def __init__(self, est, name: str, n_train: int | None = None):
        self.est, self.name, self.n_train = est, name, n_train

    def predict(self, X):
        return np.asarray(self.est.effect(X)).ravel()

    def effect_interval(self, X, alpha: float = 0.05):
        return self.est.effect_interval(X, alpha=alpha)

    def diagnostics(self) -> dict:
        return {"n_train": self.n_train}
