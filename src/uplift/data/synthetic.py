"""Synthetic data with a KNOWN treatment effect.

This module is what lets CI verify the statistical correctness of every
estimator in the repo without downloading 311 MB. If a test asserts that AIPW
recovers the truth, the truth has to come from somewhere - it comes from here.

One hard-won detail: keep the outcome probability well away from 0 and 1. If
`p0 + tau` clips at the boundary, the realized effect stops matching the
nominal tau and the test fails for a reason that has nothing to do with the
estimator. Hence `tanh` for the heterogeneous part and a baseline near 0.05.
"""

from __future__ import annotations

import numpy as np

# The default coefficients `inject_confounding` uses to build its selection
# score. Exposed here so a test can make the BASELINE RISK depend on the same
# covariates that drive selection - which is what actually creates confounding.
SELECTION_COEFS = np.array([1.2, -0.8, 0.6, 0.9])


def make_rct(
    n: int = 200_000,
    n_features: int = 6,
    treatment_share: float = 0.85,
    baseline: float = 0.05,
    tau_scale: float = 0.04,
    heterogeneous: bool = True,
    baseline_spread: float = 0.02,
    baseline_coefs: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (X, w, y, true_ate) from a randomized trial with a known effect.

    Treatment is assigned independently of X - it is an RCT - so the population
    ATE is E[tau(X)], which we compute exactly rather than estimate.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    w = rng.binomial(1, treatment_share, n)

    if heterogeneous:
        # tanh keeps tau bounded so p0 + tau never clips at 0 or 1
        tau = tau_scale * (1.0 + np.tanh(X[:, 0] + 0.5 * X[:, 1])) / 2.0 * 2.0
    else:
        tau = np.full(n, tau_scale)

    # Covariate-driven baseline risk. WHICH covariates matters enormously for any
    # test that injects confounding afterwards: confounding only bites when the
    # covariates driving SELECTION also drive the OUTCOME. The default loads
    # feature 2 only, which is deliberately mild; pass `baseline_coefs` aligned
    # with SELECTION_COEFS to build a scenario where a naive comparison is badly
    # biased and an adjusted one has something real to fix.
    if baseline_coefs is None:
        bc = np.zeros(n_features)
        bc[2] = 1.0
    else:
        bc = np.asarray(baseline_coefs, dtype=float)
        if len(bc) != n_features:
            raise ValueError(f"baseline_coefs must have length {n_features}, got {len(bc)}")
    p0 = baseline + baseline_spread * np.tanh(X @ bc)
    p = np.clip(p0 + w * tau, 1e-6, 1 - 1e-6)
    y = rng.binomial(1, p)

    return X, w, y, float(tau.mean())


def make_noncompliance(
    n: int = 400_000,
    compliance: float = 0.5,
    baseline: float = 0.05,
    complier_effect: float = 0.04,
    treatment_share: float = 0.85,
    compliance_selection: float = 0.0,
    baseline_spread: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One-sided non-compliance: z -> d -> y, with the effect flowing ONLY
    through d. Returns (z, d, y, true_cace).

    `compliance_selection` controls whether WHO COMPLIES is related to who would
    have had the outcome anyway. At 0 compliance is pure coin-flipping, and the
    naive exposed-vs-control comparison happens to be unbiased for the CACE - so
    a test built on that default cannot demonstrate the `exposure` trap at all.

    Above 0 a latent u drives both compliance and baseline risk, which is the
    realistic case and the one this dataset has: users reachable by an ad
    impression are users who were actively browsing, and browsing users visit
    sites more anyway. Then E[Y | D=1] exceeds the control mean by more than the
    causal effect, and the naive comparison overstates - which is the thing worth
    testing.
    """
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, treatment_share, n)

    u = rng.normal(size=n)  # latent "browsing intensity"
    # centre the compliance logit so P(comply) averages ~`compliance`
    logit_c = np.log(compliance / (1 - compliance)) + compliance_selection * u
    p_comply = 1.0 / (1.0 + np.exp(-logit_c))
    d = z * rng.binomial(1, p_comply)

    p0 = np.clip(baseline + baseline_spread * np.tanh(u), 1e-6, 1 - 1e-6)
    y = rng.binomial(1, np.clip(p0 + complier_effect * d, 1e-6, 1 - 1e-6))
    return z, d, y, complier_effect
