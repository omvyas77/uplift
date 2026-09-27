"""Inject selection bias into randomized data, to create an observational slice
whose true answer is known.

This is the core idea of the whole project. Every observational causal study has
the same weakness: you estimate an effect and have no way to check it. Holding a
randomized experiment removes that weakness - deliberately destroy the
randomization, run the observational toolkit on the wreckage, and compare each
estimate to the answer you already had.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfoundedSample:
    idx: np.ndarray  # indices kept from the RCT
    selection_score: np.ndarray  # e(X) for every ORIGINAL unit
    keep_prob: np.ndarray  # P(keep | X, W) for every original unit
    ground_truth_ate: float
    ground_truth_se: float
    ground_truth_att: float
    ground_truth_att_se: float
    strength: float
    nonlinear: bool
    n_kept: int
    n_original: int


def inject_confounding(
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray | None = None,
    strength: float = 1.0,
    nonlinear: bool = False,
    seed: int = 0,
) -> ConfoundedSample:
    """Create a confounded observational slice from randomized data.

    Mechanism: build a selection score e(X) from covariates, then keep a TREATED
    unit with probability e(X) and a CONTROL unit with probability 1 - e(X).
    Treatment is now correlated with X in the surviving sample, so a naive
    comparison is confounded - exactly like real observational data.

    Crucially, potential outcomes are untouched. Only WHO WE OBSERVE changes.

    The subtlety that makes the ground truth correct, and which an interviewer
    may well probe: at a 50/50 treatment share, s(X) = 0.5*e + 0.5*(1-e) = 0.5 is
    constant, so the slice has the same covariate distribution as the population
    and the truth is just the population ATE. Criteo is 85/15, so s(X) is NOT
    constant - the slice is an s(X)-reweighted population and its target
    estimand is the s(X)-weighted ATE. Using the plain population ATE as "truth"
    would build a bias into the bias table.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if coefs is None:
        coefs = np.zeros(X.shape[1])
        coefs[:4] = [1.2, -0.8, 0.6, 0.9]  # a few covariates drive selection
    score = Xs @ coefs
    if nonlinear:
        # Squares and an interaction. Without these the selection logit is
        # EXACTLY linear in X, and so is logit P(W=1 | X) in the kept sample:
        #   logit P(W=1|X,kept) = log(p_w/(1-p_w)) + strength * (Xs @ coefs)
        # which makes plain logistic regression the CORRECTLY SPECIFIED model,
        # not a misspecified one. Any "doubly robust survives misspecification"
        # demonstration built on the linear score is therefore vacuous - it was
        # comparing a correct linear model against an over-flexible GBM.
        score = score + 0.7 * (Xs[:, 0] ** 2 - 1.0) - 0.5 * Xs[:, 1] * Xs[:, 2]
    e = 1.0 / (1.0 + np.exp(-strength * score))

    keep_prob = np.where(w == 1, e, 1.0 - e)
    keep = rng.random(n) < keep_prob
    idx = np.flatnonzero(keep)

    # ---- ground truth for THIS slice, computed from the full RCT ----
    p_w = w.mean()
    targets = ground_truths(y, w, e, p_w)

    return ConfoundedSample(
        idx=idx,
        selection_score=e,
        keep_prob=keep_prob,
        ground_truth_ate=targets["ate"],
        ground_truth_se=targets["ate_se"],
        ground_truth_att=targets["att"],
        ground_truth_att_se=targets["att_se"],
        strength=float(strength),
        nonlinear=bool(nonlinear),
        n_kept=len(idx),
        n_original=int(n),
    )


def true_propensity_in_slice(e: np.ndarray, p_w: float) -> np.ndarray:
    """The EXACT propensity inside the confounded slice - no estimation needed.

    We built the selection, so we know it in closed form. A treated unit is kept
    with probability e(X) and a control with 1 - e(X), so by Bayes:

        P(W=1 | X, kept) = p_w*e / (p_w*e + (1-p_w)*(1-e))

    This matters more than it looks. Fitting a flexible propensity model to data
    where treatment is RANDOMIZED produces pure overfitting noise: on this
    dataset at strength 0 the model's AUC for predicting treatment is 0.506 -
    no signal - yet it emits scores from 0.47 to 0.99. Because the covariates
    strongly predict the OUTCOME, reweighting by 1/ps then wrecks the estimate:
    IPW returns -92.9% against truth with the estimated score and +3.9% with the
    true constant one.

    Having the oracle lets the sweep separate "the method failed" from "the
    nuisance estimate failed", which are very different findings.
    """
    e = np.asarray(e, dtype=float)
    num = p_w * e
    return num / (num + (1.0 - p_w) * (1.0 - e))


def ground_truths(y, w, e, p_w: float) -> dict[str, float]:
    """BOTH target estimands for the confounded slice, computed from the RCT.

    This exists because the estimators do not all target the same quantity, and
    scoring them against one number was silently comparing apples to oranges:

        s(X)   = P(kept | X)               -> reweights to the slice population
                                              ................ the ATE target
        s_t(X) = P(kept AND treated | X)   -> reweights to the slice's TREATED
                                              population ...... the ATT target

    IPW and AIPW estimate the ATE. Nearest-neighbour matching, which matches
    treated units to controls and averages over the TREATED, estimates the ATT.
    Under heterogeneous effects ATT != ATE, so matching scored against the ATE
    is being marked against the wrong answer key.

    Both are valid targets computed from the RCT, where W is independent of X.
    """
    s = p_w * e + (1.0 - p_w) * (1.0 - e)
    s_t = p_w * e
    ate, ate_se = weighted_diff_in_means(y, w, s)
    att, att_se = weighted_diff_in_means(y, w, s_t)
    return {"ate": ate, "ate_se": ate_se, "att": att, "att_se": att_se}


def weighted_diff_in_means(y, w, weights):
    """Weighted difference in means on the RCT, plus its standard error.

    Valid as the slice's target because W is independent of X in the RCT.
    """
    y = np.asarray(y, dtype=float)

    def wmean_var(vals, wt):
        wt = wt / wt.sum()
        m = np.sum(wt * vals)
        v = np.sum(wt**2 * (vals - m) ** 2)  # variance of a weighted mean
        return m, v

    m1, v1 = wmean_var(y[w == 1], weights[w == 1])
    m0, v0 = wmean_var(y[w == 0], weights[w == 0])
    return float(m1 - m0), float(np.sqrt(v1 + v0))
