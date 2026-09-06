"""Variance reduction: CUPED, adapted honestly to a dataset with no pre-period.

The problem to confront out loud: textbook CUPED needs a PRE-EXPERIMENT
covariate, typically the same metric measured before the experiment started.
Criteo has no pre-period and no time column, so the textbook version is
unavailable.

The correct adaptation is CUPAC - Control Using Predictions As Covariates - a
cross-fitted prediction of the outcome from pre-treatment covariates. It is
valid because f0..f11 are pre-treatment and therefore independent of
assignment, which is the only property CUPED actually requires.

Expect a small number on this data, and report it with the reason: variance
reduction is bounded by rho^2, and 12 randomly-projected anonymized features do
not predict a 4.7% binary outcome well.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class CUPEDResult:
    theta: float
    correlation: float
    variance_reduction: float  # equals rho^2
    se_before: float
    se_after: float
    se_reduction: float
    effective_sample_multiplier: float

    def summary(self) -> str:
        return (
            f"theta={self.theta:.4f}  rho={self.correlation:.4f}  "
            f"var_red={100 * self.variance_reduction:.2f}%  "
            f"SE {self.se_before:.3e} -> {self.se_after:.3e} "
            f"({100 * self.se_reduction:.2f}% narrower)  "
            f"~= {self.effective_sample_multiplier:.3f}x sample"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def cuped_adjust(y: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """Y_adj = Y - theta*(X - mean(X)), theta = Cov(Y,X)/Var(X).

    Mean-preserving, so the ATE estimate is unchanged in expectation; only its
    variance shrinks.
    """
    y = np.asarray(y, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    theta = float(np.cov(y, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1))
    return y - theta * (covariate - covariate.mean()), theta


def feature_groups(X: np.ndarray) -> np.ndarray:
    """A group id per row, keyed on the exact feature vector.

    Needed because this dataset has 1.26M exact-duplicate rows. Plain KFold
    shuffles rows independently, so a duplicate's twin lands in another fold and
    the model MEMORISES its outcome instead of predicting it - which would
    inflate rho and therefore the reported variance reduction. Grouping on the
    feature vector keeps every copy of a row on the same side of the split.
    """
    _, groups = np.unique(np.ascontiguousarray(X), axis=0, return_inverse=True)
    return np.asarray(groups).ravel()


def build_cupac_covariate(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    n_estimators: int = 300,
    grouped: bool = True,
) -> np.ndarray:
    """Cross-fitted E[Y | X] fitted on CONTROL units only.

    Control-only + cross-fitting is what keeps the covariate free of any
    treatment information, so it remains a legitimate pre-treatment covariate.
    Fit it on the treated units and you would be smuggling the effect you are
    trying to measure into the adjustment.

    `grouped=True` additionally keeps every duplicate of a feature vector in the
    same fold - see `feature_groups`. It is the default because on this data
    plain KFold cannot distinguish prediction from memorisation.
    """
    pred = np.zeros(len(y))
    if grouped:
        splitter = GroupKFold(n_splits=n_folds).split(X, groups=feature_groups(X))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)
    for train_idx, test_idx in splitter:
        ctrl = train_idx[w[train_idx] == 0]
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            verbose=-1,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[ctrl], y[ctrl])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def evaluate_cuped(y: np.ndarray, w: np.ndarray, covariate: np.ndarray) -> CUPEDResult:
    y = np.asarray(y, dtype=float)
    y_adj, theta = cuped_adjust(y, covariate)
    rho = float(np.corrcoef(y, covariate)[0, 1])

    def se_of_diff(vals: np.ndarray) -> float:
        a, b = vals[w == 1], vals[w == 0]
        return float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))

    se_before, se_after = se_of_diff(y), se_of_diff(y_adj)
    var_red = float(1.0 - (y_adj.var(ddof=1) / y.var(ddof=1)))

    return CUPEDResult(
        theta=theta,
        correlation=rho,
        variance_reduction=var_red,
        se_before=se_before,
        se_after=se_after,
        se_reduction=1.0 - se_after / se_before,
        # a variance reduction of v is worth 1/(1-v) times the sample size
        effective_sample_multiplier=float(1.0 / (1.0 - var_red)) if var_red < 1 else float("inf"),
    )
