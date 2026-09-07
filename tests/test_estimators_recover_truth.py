"""The highest-signal test file in the repo.

Every estimator that appears in the bias table is checked against a KNOWN
treatment effect on synthetic data. That is what makes the real bias table
credible, and it lets CI verify the causal code without touching the 14M-row
dataset.

Design note that took measurement to get right: the confounding has to actually
BITE, or the test proves nothing. Injecting selection on covariates that barely
affect the outcome gave a naive bias (0.006) the same size as the doubly-robust
estimator's own noise (0.005), so across seeds the DR estimator was sometimes
WORSE than naive and any "DR beats naive" assertion was a coin flip. The fix is
to make baseline risk depend on the same covariates that drive selection - see
`baseline_coefs` in `make_rct` - which is the definition of confounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.causal.confounding import inject_confounding
from uplift.causal.estimators import (
    aipw_ate,
    check_overlap,
    fit_propensity,
    ipw_ate,
    matching_ate,
    naive_ate,
)
from uplift.data.synthetic import SELECTION_COEFS, make_rct

# Aligns baseline risk with the covariates that drive selection, normalised so
# the tanh argument stays in a sane range.
CONFOUNDED_BASELINE = np.concatenate([SELECTION_COEFS / np.abs(SELECTION_COEFS).max(), np.zeros(2)])


def _scenario(n: int = 120_000, strength: float = 1.5, seed: int = 0):
    """An RCT, a confounded slice of it, and the truth for that slice."""
    X, w, y, _ = make_rct(
        n=n,
        treatment_share=0.85,
        baseline_spread=0.05,
        baseline_coefs=CONFOUNDED_BASELINE,
        seed=seed,
    )
    cs = inject_confounding(X, w, y, strength=strength, seed=100 + seed)
    return X, w, y, cs, X[cs.idx], w[cs.idx], y[cs.idx]


@pytest.mark.stat
def test_aipw_recovers_truth_under_injected_confounding():
    _, _, _, cs, Xo, wo, yo = _scenario(seed=0)
    truth = cs.ground_truth_ate

    naive = naive_ate(yo, wo)
    ps = fit_propensity(Xo, wo, seed=2, n_folds=3)
    dr, dr_se = aipw_ate(yo, wo, Xo, ps, seed=2, n_folds=3)

    # the confounding must actually bite, or the test proves nothing
    assert abs(naive - truth) > 3 * abs(dr - truth), (
        f"confounding did not bite: naive err {abs(naive - truth):.5f} "
        f"vs dr err {abs(dr - truth):.5f}"
    )
    # doubly robust must land within ~4 SEs of the truth
    assert abs(dr - truth) < 4 * max(dr_se, 1e-4)


@pytest.mark.stat
def test_every_adjusted_estimator_beats_naive():
    """IPW and matching, not just AIPW - the whole bias table, not one row."""
    _, _, _, cs, Xo, wo, yo = _scenario(seed=1)
    truth = cs.ground_truth_ate
    ps = fit_propensity(Xo, wo, seed=3, n_folds=3)

    naive_err = abs(naive_ate(yo, wo) - truth)
    for name, est in [
        ("ipw", ipw_ate(yo, wo, ps)),
        ("matching", matching_ate(yo, wo, ps)),
        ("aipw", aipw_ate(yo, wo, Xo, ps, seed=3, n_folds=3)[0]),
    ]:
        assert abs(est - truth) < naive_err, f"{name} did not beat the naive comparison"


@pytest.mark.stat
def test_no_confounding_means_no_bias():
    """Strength 0 keeps a random half, so the naive comparison is unbiased.

    This is the apparatus validating itself: if this row is not ~0, the bias
    table is measuring its own bug rather than the estimators.
    """
    X, w, y, _ = make_rct(n=200_000, treatment_share=0.85, seed=3)
    cs = inject_confounding(X, w, y, strength=0.0, seed=4)
    assert abs(naive_ate(y[cs.idx], w[cs.idx]) - cs.ground_truth_ate) < 0.005


def test_ground_truth_uses_the_selection_weighted_ate():
    """At 85/15 the slice is a REWEIGHTED population, so its target estimand is
    not the population ATE.

    s(X) = P(W=1)*e + P(W=0)*(1-e) is constant only at a 50/50 share. Using the
    plain population ATE as "truth" under 85/15 would build a bias straight into
    the bias table, so this pins the distinction.
    """
    X, w, y, _ = make_rct(
        n=200_000,
        treatment_share=0.85,
        baseline_spread=0.05,
        baseline_coefs=CONFOUNDED_BASELINE,
        seed=7,
    )
    population_ate = naive_ate(y, w)  # unbiased: this is still the RCT
    weighted = inject_confounding(X, w, y, strength=1.5, seed=8).ground_truth_ate
    assert abs(weighted - population_ate) > 2e-3, (
        "s(X) weighting should move the target under an 85/15 design"
    )

    balanced = make_rct(
        n=200_000,
        treatment_share=0.5,
        baseline_spread=0.05,
        baseline_coefs=CONFOUNDED_BASELINE,
        seed=7,
    )
    Xb, wb, yb, _ = balanced
    # at 50/50, s(X) is constant, so the weighted truth collapses to the ATE
    assert (
        abs(
            inject_confounding(Xb, wb, yb, strength=1.5, seed=8).ground_truth_ate
            - naive_ate(yb, wb)
        )
        < 2e-3
    )


@pytest.mark.stat
def test_overlap_degrades_as_confounding_strengthens():
    """The diagnostic that tells you when to STOP trusting an adjusted estimate."""
    prev = None
    for strength in (0.5, 2.0, 4.0):
        _, _, _, _, Xo, wo, _ = _scenario(n=60_000, strength=strength, seed=5)
        ov = check_overlap(fit_propensity(Xo, wo, seed=5, n_folds=3), wo)
        if prev is not None:
            assert ov["ps_min"] <= prev["ps_min"]
            assert (
                ov["frac_treated_outside_control_support"]
                >= prev["frac_treated_outside_control_support"]
            )
        prev = ov
    assert prev is not None and prev["ps_min"] < 0.01


@pytest.mark.stat
@pytest.mark.xfail(
    strict=True,
    reason=(
        "AIPW does NOT beat IPW here under a misspecified propensity - measured "
        "0/6 seeds at strength 1.5, with and without nonlinear selection "
        "(AIPW ~0.019 vs IPW ~0.007 mean abs error). Two things were ruled out "
        "and one is still open.\n"
        "RULED OUT - the selection score was linear in X, which makes "
        "logit P(W=1|X) in the kept slice exactly linear too, so the 'misspecified' "
        "logistic regression was in fact the CORRECTLY specified model. "
        "inject_confounding(nonlinear=True) now adds squares and an interaction. "
        "It did not change the verdict.\n"
        "STILL OPEN - the comparison is not like-for-like. ipw_ate is STABILIZED "
        "(self-normalized Hajek weights) while aipw_ate uses unnormalized "
        "Horvitz-Thompson weights in its correction term, so the stabilization may "
        "be doing the work rather than the double robustness. The fix to try is a "
        "self-normalized AIPW. Left strict so it XPASSes loudly once that lands, "
        "rather than being quietly deleted to keep the suite green."
    ),
)
def test_doubly_robust_survives_a_misspecified_propensity():
    """What double robustness is supposed to buy - and does not, here.

    Selection is nonlinear, so a linear-only propensity model is wrong by
    construction. IPW leans entirely on that model; AIPW also has the outcome
    model to fall back on, and textbook double robustness says it should
    therefore survive. On this setup it does not. See the xfail reason.
    """
    X, w, y, _ = make_rct(
        n=120_000,
        treatment_share=0.85,
        baseline_spread=0.05,
        baseline_coefs=CONFOUNDED_BASELINE,
        seed=6,
    )
    cs = inject_confounding(X, w, y, strength=1.5, nonlinear=True, seed=106)
    Xo, wo, yo = X[cs.idx], w[cs.idx], y[cs.idx]
    truth = cs.ground_truth_ate
    bad_ps = fit_propensity(Xo, wo, seed=6, n_folds=3, misspecified=True)

    ipw_err = abs(ipw_ate(yo, wo, bad_ps) - truth)
    dr_err = abs(aipw_ate(yo, wo, Xo, bad_ps, seed=6, n_folds=3)[0] - truth)
    assert dr_err < ipw_err, (
        f"doubly robust ({dr_err:.5f}) should beat IPW ({ipw_err:.5f}) "
        "when the propensity model is misspecified"
    )
