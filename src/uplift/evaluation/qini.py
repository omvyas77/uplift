"""Qini / AUUC, plus bootstrap confidence intervals.

Two implementations live here on purpose:

* `qini_coefficient` is written from scratch so the metric is understood and
  testable. Its three property tests - random ~ 0, oracle > random, reversal
  flips the sign - catch essentially every implementation bug.
* The numbers actually REPORTED come from scikit-uplift, via
  `sklift_qini` / `sklift_auuc`. Qini values are NOT comparable across
  implementations with different normalizations, so the repo states which one
  produced each number.

The framing that has to be right: tau is never observed for any individual, so
there is no per-unit error and no RMSE against ground truth. Evaluation is at
the group level - rank by predicted tau, then check whether the top-ranked
groups actually show a larger treated-versus-control gap.
"""

from __future__ import annotations

import numpy as np


def qini_curve(y: np.ndarray, w: np.ndarray, score: np.ndarray, tie_aware: bool = True):
    """Cumulative incremental outcome as units are treated in descending score order.

    At depth k:  Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k)

    The N_t/N_c factor rescales the control response to the treated group size,
    which is what makes the curve readable as "incremental outcomes gained".

    Tie handling matters more than it looks. A Qini curve is defined by
    THRESHOLDS on the score, so within a block of equal scores there is no
    defined ordering and the curve should interpolate linearly across the block.
    Evaluating it at every unit instead makes the result depend on the arbitrary
    within-block order, which shows up as an asymmetry: a score and its negation
    stop producing mirror-image coefficients. With `tau = 0.05 * (x > 0)` - two
    distinct values, so one enormous tie - that asymmetry is 0.025 on a
    coefficient of 0.15, and the `reversed ranking flips the sign` property test
    fails. Returning block boundaries only fixes it, and is a no-op on
    continuous scores.

    Returns (x, q) with the origin (0, 0) prepended so the integral is taken
    over the whole curve.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")  # stable: ties reproducible
    y, w, s = y[order], w[order], score[order]
    n_t, n_c = np.cumsum(w), np.cumsum(1 - w)
    y_t, y_c = np.cumsum(y * w), np.cumsum(y * (1 - w))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.nan_to_num(y_t - np.where(n_c > 0, y_c * n_t / np.maximum(n_c, 1), 0.0))
    x = np.arange(1, len(y) + 1, dtype=float)

    if tie_aware and len(s) > 1:
        keep = np.append(s[:-1] != s[1:], True)  # last unit of each tied block
        x, q = x[keep], q[keep]

    return np.concatenate([[0.0], x]), np.concatenate([[0.0], q])


def qini_coefficient(y, w, score) -> float:
    """Area between the Qini curve and the random line, normalized by the area
    between a perfect curve and the random line.

    Note the "perfect" curve is built from realized outcomes, which no model can
    achieve, so even an oracle ranking scores well below 1.0.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    x, q = qini_curve(y, w, score)
    oracle = np.where((w == 1) & (y == 1), 2.0, np.where((w == 0) & (y == 0), 1.0, 0.0))
    xp, q_perfect = qini_curve(y, w, oracle)
    xr = np.array([0.0, float(len(y))])
    area_random = np.trapezoid(q[-1] * xr / len(y), xr)
    denom = np.trapezoid(q_perfect, xp) - area_random
    if denom == 0:
        return 0.0
    return float((np.trapezoid(q, x) - area_random) / denom)


# ---------------- the reported numbers ----------------
def sklift_qini(y, uplift, treatment) -> float:
    from sklift.metrics import qini_auc_score

    return float(
        qini_auc_score(
            y_true=np.asarray(y), uplift=np.asarray(uplift), treatment=np.asarray(treatment)
        )
    )


def sklift_auuc(y, uplift, treatment) -> float:
    from sklift.metrics import uplift_auc_score

    return float(
        uplift_auc_score(
            y_true=np.asarray(y), uplift=np.asarray(uplift), treatment=np.asarray(treatment)
        )
    )


def uplift_at_k(y, uplift, treatment, k: float = 0.3) -> float:
    from sklift.metrics import uplift_at_k as _uak

    return float(
        _uak(
            y_true=np.asarray(y),
            uplift=np.asarray(uplift),
            treatment=np.asarray(treatment),
            strategy="overall",
            k=k,
        )
    )


# ---------------- uncertainty ----------------
def bootstrap_qini_ci(y, w, score, n_boot: int = 200, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI on the Qini coefficient.

    Not optional. Without it a leaderboard manufactures a winner out of noise,
    which on this data is the default outcome rather than an edge case.
    """
    rng = np.random.default_rng(seed)
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = sklift_qini(y[idx], score[idx], w[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi)


def paired_bootstrap_difference(y, w, score_a, score_b, n_boot: int = 200, seed: int = 0):
    """Resample ONCE per iteration and score both models on the same resample.

    Paired resampling removes the shared sampling noise, which is the only way
    to tell two similar models apart at this signal level. Returns
    (mean_diff, lo, hi, resolved) where `resolved` means the CI excludes zero.
    """
    rng = np.random.default_rng(seed)
    y, w = np.asarray(y), np.asarray(w)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = sklift_qini(y[idx], score_a[idx], w[idx]) - sklift_qini(
            y[idx], score_b[idx], w[idx]
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(diffs.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)
