"""Overlap (positivity) diagnostics.

These run BEFORE any estimate. If the confounded slice has no comparable units
in both arms, no estimator can recover the truth, and a bias table computed on
it is measuring the injection rather than the methods.

The signature to look for: naive sign-flipped, every adjusted estimator biased
in the same direction, weight-based methods worst, and a discard-based method
(matching) best. That pattern is what positivity failure looks like -
reweighting cannot fix a region where one arm simply has no units, while
discarding can.

The decisive number is the Kish effective sample size. An IPW estimate whose
weights concentrate on a handful of units is not estimating a population
quantity; it is reporting a few rows with very large weights.
"""

from __future__ import annotations

import numpy as np

# Thresholds used to turn the diagnostics into a verdict. They are conventions,
# not theorems, and are stated here so a reader can disagree with the specific
# numbers rather than having to reverse-engineer them.
ESS_DEAD = 0.10
ESS_MARGINAL = 0.50
CLIP_DEAD = 0.05
CLIP_MARGINAL = 0.01
TOP1PCT_DEAD = 0.30
TOP1PCT_MARGINAL = 0.10


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    Equals n when every weight is equal and falls toward 1 as the weight mass
    concentrates. IPW with ESS/n below ~0.1 is not estimating anything stable.
    """
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w**2))


def _top_share(x: np.ndarray, q: float = 0.01) -> float:
    """Share of total weight held by the heaviest q of units."""
    if x.size == 0 or x.sum() == 0:
        return 0.0
    k = max(1, int(len(x) * q))
    return float(np.sort(x)[-k:].sum() / x.sum())


def overlap_report(ps: np.ndarray, w: np.ndarray, clip: tuple[float, float] = (0.02, 0.98)) -> dict:
    """Positivity diagnostics for a propensity vector.

    `common_support_gap` is the sharpest single number: the 5th percentile of
    treated propensities minus the 95th percentile of control propensities. Above
    zero means the bulk of the two distributions do not overlap at all, and any
    comparison is extrapolation rather than adjustment.
    """
    ps = np.asarray(ps, dtype=float)
    w = np.asarray(w)
    lo, hi = clip
    t, c = w == 1, w == 0

    wt = 1.0 / np.clip(ps[t], lo, hi)
    wc = 1.0 / (1.0 - np.clip(ps[c], lo, hi))

    ess_t, ess_c = effective_sample_size(wt), effective_sample_size(wc)
    n_t, n_c = int(t.sum()), int(c.sum())

    return {
        "n_treated": n_t,
        "n_control": n_c,
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "ps_p01": float(np.quantile(ps, 0.01)),
        "ps_p99": float(np.quantile(ps, 0.99)),
        "frac_clipped_lo": float((ps < lo).mean()),
        "frac_clipped_hi": float((ps > hi).mean()),
        "frac_clipped_any": float(((ps < lo) | (ps > hi)).mean()),
        "treated_ps_p05": float(np.quantile(ps[t], 0.05)) if n_t else float("nan"),
        "control_ps_p95": float(np.quantile(ps[c], 0.95)) if n_c else float("nan"),
        "common_support_gap": (
            float(np.quantile(ps[t], 0.05) - np.quantile(ps[c], 0.95))
            if n_t and n_c
            else float("nan")
        ),
        "ess_treated": ess_t,
        "ess_control": ess_c,
        "ess_frac_treated": ess_t / n_t if n_t else 0.0,
        "ess_frac_control": ess_c / n_c if n_c else 0.0,
        "top1pct_weight_share_treated": _top_share(wt),
        "top1pct_weight_share_control": _top_share(wc),
    }


def overlap_verdict(report: dict) -> dict:
    """Reduce the report to healthy / marginal / dead, per criterion.

    Exists so the sweep can say WHERE overlap died without a human reading
    twenty numbers per row.
    """

    def grade(value: float, dead: float, marginal: float, higher_is_worse: bool) -> str:
        if higher_is_worse:
            if value > dead:
                return "dead"
            return "marginal" if value > marginal else "healthy"
        if value < dead:
            return "dead"
        return "marginal" if value < marginal else "healthy"

    grades = {
        "ess_frac_control": grade(
            report["ess_frac_control"], ESS_DEAD, ESS_MARGINAL, higher_is_worse=False
        ),
        "ess_frac_treated": grade(
            report["ess_frac_treated"], ESS_DEAD, ESS_MARGINAL, higher_is_worse=False
        ),
        "frac_clipped_any": grade(
            report["frac_clipped_any"], CLIP_DEAD, CLIP_MARGINAL, higher_is_worse=True
        ),
        "top1pct_weight_share_control": grade(
            report["top1pct_weight_share_control"],
            TOP1PCT_DEAD,
            TOP1PCT_MARGINAL,
            higher_is_worse=True,
        ),
        "common_support": "dead" if report["common_support_gap"] > 0 else "healthy",
    }
    order = {"healthy": 0, "marginal": 1, "dead": 2}
    worst = max(grades.values(), key=lambda g: order[g])
    return {"per_criterion": grades, "overall": worst, "usable": worst != "dead"}
