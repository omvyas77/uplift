"""Sequential testing and the peeking problem.

Criteo has no time dimension, so this data cannot be genuinely monitored over
time. Rather than invent a synthetic "day" column and quietly pretend, this
module SIMULATES the peeking problem on data generated under a known null.
Everything here is labelled as simulation, in the docstrings and in the printed
readout.

The demonstration: repeatedly testing a null experiment at alpha = 0.05
inflates the false-positive rate far above 5%; an alpha-spending boundary
brings it back.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def peeking_simulation(
    n_peeks: int = 10,
    n_per_peek: int = 10_000,
    p: float = 0.047,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """An A/A test peeked at repeatedly. The true effect is zero by construction,
    so every rejection is a false positive."""
    rng = np.random.default_rng(seed)
    naive_fp = 0
    for _ in range(n_sims):
        a = rng.binomial(1, p, n_peeks * n_per_peek)
        b = rng.binomial(1, p, n_peeks * n_per_peek)
        for k in range(1, n_peeks + 1):
            m = k * n_per_peek
            diff = a[:m].mean() - b[:m].mean()
            se = np.sqrt(a[:m].var(ddof=1) / m + b[:m].var(ddof=1) / m)
            if se > 0 and abs(diff / se) > 1.96:
                naive_fp += 1
                break
    return {
        "peeks": float(n_peeks),
        "n_per_peek": float(n_per_peek),
        "nominal_alpha": 0.05,
        "actual_false_positive_rate": naive_fp / n_sims,
    }


def obrien_fleming_boundaries(n_peeks: int, alpha: float = 0.05) -> np.ndarray:
    """O'Brien-Fleming alpha-spending boundaries on the z scale.

    Spends very little alpha early (so an early peek needs overwhelming
    evidence) and approaches the fixed-sample critical value at the final look.
    """
    t = np.arange(1, n_peeks + 1) / n_peeks
    z_final = stats.norm.ppf(1 - alpha / 2)
    return z_final / np.sqrt(t)


def peeking_simulation_corrected(
    n_peeks: int = 10,
    n_per_peek: int = 10_000,
    p: float = 0.047,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """The same A/A simulation, stopped against an O'Brien-Fleming boundary.
    The false-positive rate should return to roughly alpha."""
    rng = np.random.default_rng(seed)
    bounds = obrien_fleming_boundaries(n_peeks, alpha)
    fp = 0
    for _ in range(n_sims):
        a = rng.binomial(1, p, n_peeks * n_per_peek)
        b = rng.binomial(1, p, n_peeks * n_per_peek)
        for k in range(1, n_peeks + 1):
            m = k * n_per_peek
            diff = a[:m].mean() - b[:m].mean()
            se = np.sqrt(a[:m].var(ddof=1) / m + b[:m].var(ddof=1) / m)
            if se > 0 and abs(diff / se) > bounds[k - 1]:
                fp += 1
                break
    return {
        "peeks": float(n_peeks),
        "nominal_alpha": alpha,
        "actual_false_positive_rate": fp / n_sims,
        "final_boundary_z": float(bounds[-1]),
        "first_boundary_z": float(bounds[0]),
    }


def msprt_pvalue(y_a: np.ndarray, y_b: np.ndarray, tau: float = 0.01) -> np.ndarray:
    """Always-valid p-value via a mixture SPRT with a normal mixing prior of
    variance tau^2 on the effect. Valid at every sample size simultaneously, so
    it can be watched continuously without inflating the error rate."""
    n = min(len(y_a), len(y_b))
    k = np.arange(1, n + 1)
    diff = np.cumsum(y_a[:n]) / k - np.cumsum(y_b[:n]) / k
    var = (y_a[:n].var(ddof=1) + y_b[:n].var(ddof=1)) / 2.0
    sigma2 = 2 * var / k  # variance of the difference at each look

    lam = np.sqrt(sigma2 / (sigma2 + tau**2)) * np.exp(
        (tau**2 * diff**2) / (2 * sigma2 * (sigma2 + tau**2))
    )
    # an always-valid p-value is 1/running-max of the likelihood ratio
    return np.minimum(1.0, 1.0 / np.maximum.accumulate(lam))
