"""Average treatment effect: three estimators.

The comparison between them IS the variance-reduction story, so they all return
the same dataclass and print in the same format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class ATEResult:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    relative_lift: float
    relative_ci: tuple[float, float]
    baseline_rate: float
    n: int

    def summary(self) -> str:
        return (
            f"{self.method:<26} ATE={self.ate:+.6f} "
            f"[{self.ci_low:+.6f}, {self.ci_high:+.6f}]  "
            f"rel={100 * self.relative_lift:+.3f}% "
            f"[{100 * self.relative_ci[0]:+.3f}%, {100 * self.relative_ci[1]:+.3f}%]  "
            f"SE={self.se:.3e}  p={self.p_value:.3e}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _relative_ci(y1: np.ndarray, y0: np.ndarray, alpha: float = 0.05):
    """Delta-method CI for the RATIO of two means.

    The absolute-difference CI cannot simply be divided by the baseline: the
    baseline is estimated too, and ignoring its variance understates the width.
    """
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    ratio = m1 / m0
    se_log = np.sqrt(v1 / m1**2 + v0 / m0**2)  # delta method on log(m1/m0)
    z = stats.norm.ppf(1 - alpha / 2)
    return ratio - 1.0, (
        float(ratio * np.exp(-z * se_log) - 1.0),
        float(ratio * np.exp(z * se_log) - 1.0),
    )


def ate_difference_in_means(y: np.ndarray, w: np.ndarray, alpha: float = 0.05) -> ATEResult:
    """Neyman's estimator. The unbiased baseline every other method must beat."""
    y = np.asarray(y, dtype=float)
    y1, y0 = y[w == 1], y[w == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    rel, rel_ci = _relative_ci(y1, y0, alpha)
    return ATEResult(
        method="difference-in-means",
        ate=float(ate),
        se=float(se),
        ci_low=float(ate - z * se),
        ci_high=float(ate + z * se),
        p_value=float(2 * stats.norm.sf(abs(ate / se))),
        relative_lift=float(rel),
        relative_ci=rel_ci,
        baseline_rate=float(y0.mean()),
        n=len(y),
    )


def ate_lin_regression(
    y: np.ndarray, w: np.ndarray, X: np.ndarray, alpha: float = 0.05
) -> ATEResult:
    """Lin (2013): regress Y on W, centered X, and their INTERACTIONS, with
    HC-robust standard errors.

    Two details worth understanding rather than copying:

    * HC1 and not default OLS SEs, because with a binary outcome the error
      variance differs by arm and homoskedastic SEs are simply wrong.
    * Lin and not plain ANCOVA, because regressing Y ~ W + X without
      interactions can be LESS efficient than the simple difference when the
      treatment effect varies with X. The W x X_centered terms remove that risk.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(y)), w, Xc, w[:, None] * Xc])
    fit = sm.OLS(y, design).fit(cov_type="HC1")

    ate, se = float(fit.params[1]), float(fit.bse[1])
    z = stats.norm.ppf(1 - alpha / 2)
    baseline = float(y[w == 0].mean())
    return ATEResult(
        method="Lin regression-adjusted",
        ate=ate,
        se=se,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        p_value=float(fit.pvalues[1]),
        relative_lift=ate / baseline,
        relative_ci=((ate - z * se) / baseline, (ate + z * se) / baseline),
        baseline_rate=baseline,
        n=len(y),
    )


def ate_from_adjusted(
    y_adj: np.ndarray,
    w: np.ndarray,
    baseline_rate: float,
    method: str = "CUPAC-adjusted",
    alpha: float = 0.05,
) -> ATEResult:
    """ATE on a variance-reduced outcome. The adjustment is mean-preserving, so
    the point estimate is unchanged in expectation; only the SE shrinks.

    `baseline_rate` is passed in from the UNADJUSTED control mean, because the
    adjusted outcome is no longer on the probability scale and its control mean
    is not a rate.
    """
    a, b = y_adj[w == 1], y_adj[w == 0]
    ate = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    z = stats.norm.ppf(1 - alpha / 2)
    return ATEResult(
        method=method,
        ate=ate,
        se=se,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        p_value=float(2 * stats.norm.sf(abs(ate / se))),
        relative_lift=ate / baseline_rate,
        relative_ci=((ate - z * se) / baseline_rate, (ate + z * se) / baseline_rate),
        baseline_rate=baseline_rate,
        n=len(y_adj),
    )
