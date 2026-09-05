"""The observational toolkit, to be run on the confounded slice and scored
against the RCT ground truth.

Report the overlap diagnostic BEFORE any estimate. If there is no overlap, no
amount of adjustment saves you, and the honest move is to say so rather than to
publish a number.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


def fit_propensity(
    X, w, seed: int = 0, cross_fit: bool = True, n_folds: int = 5, misspecified: bool = False
):
    """Cross-fitted propensity scores.

    Cross-fitting avoids the overfitting bias that makes in-sample propensity
    scores too extreme. `misspecified=True` fits linear-only logistic regression,
    which is deliberately wrong when selection is nonlinear - that is how the
    repo demonstrates what "doubly robust" actually buys.
    """
    if misspecified:
        model_fn = lambda: LogisticRegression(max_iter=1000)  # noqa: E731
    else:
        model_fn = lambda: LGBMClassifier(  # noqa: E731
            n_estimators=300,
            num_leaves=31,
            min_child_samples=200,
            verbose=-1,
            random_state=seed,
            n_jobs=-1,
        )
    if not cross_fit:
        return model_fn().fit(X, w).predict_proba(X)[:, 1]
    ps = np.zeros(len(w))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, w):
        ps[te] = model_fn().fit(X[tr], w[tr]).predict_proba(X[te])[:, 1]
    return ps


def check_overlap(ps, w, lo: float = 0.01, hi: float = 0.99) -> dict[str, float]:
    """Positivity / overlap diagnostic."""
    ps = np.asarray(ps)
    return {
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "frac_below": float((ps < lo).mean()),
        "frac_above": float((ps > hi).mean()),
        "treated_ps_p05": float(np.quantile(ps[w == 1], 0.05)),
        "control_ps_p95": float(np.quantile(ps[w == 0], 0.95)),
        # the single most useful summary: how much of the treated distribution
        # has no control counterpart at all
        "frac_treated_outside_control_support": float(
            (ps[w == 1] > np.quantile(ps[w == 0], 0.99)).mean()
        ),
    }


def naive_ate(y, w) -> float:
    y = np.asarray(y, dtype=float)
    return float(y[w == 1].mean() - y[w == 0].mean())


def ipw_ate(y, w, ps, clip: tuple[float, float] = (0.02, 0.98), stabilized: bool = True) -> float:
    """Inverse propensity weighting.

    Clipping trades a little bias for a large variance reduction when weights
    explode - always report the clip alongside the estimate.
    """
    y = np.asarray(y, dtype=float)
    ps = np.clip(ps, *clip)
    if stabilized:
        p = w.mean()
        w1 = w * p / ps
        w0 = (1 - w) * (1 - p) / (1 - ps)
        return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))
    return float(np.mean(w * y / ps) - np.mean((1 - w) * y / (1 - ps)))


def aipw_ate(
    y, w, X, ps, seed: int = 0, clip: tuple[float, float] = (0.02, 0.98), n_folds: int = 5
) -> tuple[float, float]:
    """Augmented IPW / doubly robust, cross-fitted.

    Consistent if EITHER the outcome model or the propensity model is correctly
    specified. The SE is the influence-function SE, which is the right one here.
    """
    y = np.asarray(y, dtype=float)
    ps = np.clip(ps, *clip)
    m1, m0 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, w):
        tr1, tr0 = tr[w[tr] == 1], tr[w[tr] == 0]
        f1 = LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed, n_jobs=-1)
        f0 = LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed, n_jobs=-1)
        m1[te] = np.asarray(f1.fit(X[tr1], y[tr1]).predict_proba(X[te]))[:, 1]
        m0[te] = np.asarray(f0.fit(X[tr0], y[tr0]).predict_proba(X[te]))[:, 1]
    scores = m1 - m0 + w * (y - m1) / ps - (1 - w) * (y - m0) / (1 - ps)
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores)))


def matching_ate(y, w, ps, k: int = 1, caliper: float | None = 0.05) -> float:
    """1:k nearest-neighbour matching on the propensity score, with a caliper.

    Matching on the SCORE rather than on X is the standard reduction; the
    caliper is what stops a treated unit with no comparable control from being
    matched to a wildly different one, which is the usual way matching quietly
    fails under poor overlap.
    """
    y = np.asarray(y, dtype=float)
    ps = np.asarray(ps).reshape(-1, 1)
    treated, control = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    nn = NearestNeighbors(n_neighbors=k).fit(ps[control])
    dist, ind = nn.kneighbors(ps[treated])
    if caliper is not None:
        ok = dist.mean(axis=1) <= caliper
        treated, ind = treated[ok], ind[ok]
    if len(treated) == 0:
        return float("nan")
    return float(y[treated].mean() - y[control[ind]].mean(axis=1).mean())


def dml_ate(y, w, X, seed: int = 0, cv: int = 3) -> tuple[float, float]:
    """Double machine learning via EconML's LinearDML - a second doubly-robust
    route, with a different final-stage. Included so the bias table shows one
    method that is not our own implementation."""
    from econml.dml import LinearDML
    from lightgbm import LGBMRegressor

    est = LinearDML(
        model_y=LGBMRegressor(n_estimators=200, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed),
        discrete_treatment=True,
        cv=cv,
        random_state=seed,
    )
    est.fit(np.asarray(y, dtype=float), w, X=None, W=X)
    eff = float(np.asarray(est.const_marginal_effect()).ravel()[0])
    lo, hi = est.const_marginal_effect_interval(alpha=0.05)
    se = float((np.asarray(hi).ravel()[0] - np.asarray(lo).ravel()[0]) / (2 * 1.96))
    return eff, se
