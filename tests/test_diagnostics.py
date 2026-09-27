"""Overlap diagnostics, the oracle propensity, and the estimand split.

These pin the machinery that turned the causal chapter from "my estimators
failed" into a diagnosis. Three things are checked because each one was wrong at
some point:

* the overlap report must flag disjoint support and weight concentration;
* the oracle propensity must be EXACT, since it is what separates "the method
  failed" from "the propensity estimate failed";
* matching must be scored against the ATT, not the ATE.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.causal.confounding import (
    ground_truths,
    inject_confounding,
    true_propensity_in_slice,
)
from uplift.causal.diagnostics import (
    effective_sample_size,
    overlap_report,
    overlap_verdict,
)
from uplift.causal.estimators import ipw_ate
from uplift.data.synthetic import make_rct


# ------------------------------------------------------------------ ESS
def test_equal_weights_give_full_effective_sample():
    assert effective_sample_size(np.ones(1000)) == pytest.approx(1000.0)


def test_one_dominant_weight_collapses_the_effective_sample():
    w = np.concatenate([[1e6], np.ones(999)])
    assert effective_sample_size(w) < 2.0


def test_ess_ignores_zero_weights():
    assert effective_sample_size(np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(2.0)


# ------------------------------------------------------------------ overlap
def test_overlap_report_flags_disjoint_support():
    """Treated near 1, control near 0: no comparable units anywhere."""
    rng = np.random.default_rng(0)
    ps = np.concatenate([rng.uniform(0.90, 0.99, 5000), rng.uniform(0.01, 0.10, 5000)])
    w = np.concatenate([np.ones(5000), np.zeros(5000)]).astype(int)
    rep = overlap_report(ps, w)
    assert rep["common_support_gap"] > 0
    assert overlap_verdict(rep)["per_criterion"]["common_support"] == "dead"
    assert not overlap_verdict(rep)["usable"]


def test_overlap_report_passes_a_healthy_design():
    """A randomized design has a constant propensity and perfect overlap."""
    rng = np.random.default_rng(1)
    n = 20_000
    w = rng.binomial(1, 0.85, n)
    ps = np.full(n, 0.85)
    rep = overlap_report(ps, w)
    # A constant propensity gives a gap of EXACTLY 0, which is the ideal case.
    assert rep["common_support_gap"] <= 0
    assert rep["ess_frac_treated"] == pytest.approx(1.0)
    assert rep["ess_frac_control"] == pytest.approx(1.0)
    assert overlap_verdict(rep)["overall"] == "healthy"


def test_clipping_fraction_is_reported_honestly():
    rng = np.random.default_rng(2)
    n = 10_000
    ps = np.clip(rng.normal(0.5, 0.3, n), 0.001, 0.999)
    w = rng.binomial(1, 0.5, n)
    rep = overlap_report(ps, w, clip=(0.02, 0.98))
    assert rep["frac_clipped_any"] == pytest.approx(rep["frac_clipped_lo"] + rep["frac_clipped_hi"])
    assert 0.0 < rep["frac_clipped_any"] < 1.0


# ------------------------------------------------------------------ oracle ps
def test_oracle_propensity_is_the_design_ratio_when_nothing_is_injected():
    """At strength 0 every e(X) is 0.5, so the slice keeps the design ratio."""
    e = np.full(100, 0.5)
    assert np.allclose(true_propensity_in_slice(e, 0.85), 0.85)


def test_oracle_propensity_moves_with_the_selection_score():
    e = np.array([0.1, 0.5, 0.9])
    ps = true_propensity_in_slice(e, 0.85)
    assert ps[0] < ps[1] < ps[2]
    assert np.all((ps > 0) & (ps < 1))


@pytest.mark.stat
def test_the_oracle_propensity_actually_recovers_the_truth():
    """The point of having it: IPW with the exact propensity must work, so any
    failure with an ESTIMATED one is nuisance estimation, not the method.

    The tolerance has to account for the ESTIMATOR's own sampling error, not
    just the truth's. A first version compared against 4x the truth's SE alone
    and failed at n=200k by 15% - which was Monte-Carlo noise, not bias. That
    mistake is the whole reason the bias table needs intervals: at these sample
    sizes and a 4% effect, the estimator's noise is the same size as the biases
    being reported.
    """
    X, w, y, _ = make_rct(n=600_000, treatment_share=0.85, seed=0)
    cs = inject_confounding(X, w, y, strength=1.0, seed=1)
    wo, yo = w[cs.idx], y[cs.idx]
    ps_true = true_propensity_in_slice(cs.selection_score[cs.idx], float(w.mean()))

    est = ipw_ate(yo, wo, ps_true, clip=(0.0, 1.0))
    # combined SE: the truth is an estimate and so is the estimator
    n_eff = len(yo)
    est_se = float(np.sqrt(yo.var(ddof=1) / n_eff)) * 3.0  # generous, IPW inflates variance
    tol = 4 * float(np.hypot(cs.ground_truth_se, est_se))
    assert abs(est - cs.ground_truth_ate) < tol, (
        f"IPW with the ORACLE propensity gave {est:.6f} against a truth of "
        f"{cs.ground_truth_ate:.6f} (tolerance {tol:.6f}); if this fails the "
        "estimator itself is wrong, not the nuisance model"
    )


# ------------------------------------------------------------------ estimands
def test_ate_and_att_targets_coincide_only_without_selection():
    """The targets agree when there is nothing to select on, and NOT merely at a
    balanced design.

    My first version of this test asserted they coincide at a 50/50 share,
    reasoning that s(X) = 0.5*e + 0.5*(1-e) is constant there. That is true for
    s, but the ATT target uses s_t(X) = p_w * e(X), which is e-weighted at ANY
    design ratio. So the two estimands differ whenever e varies and effects are
    heterogeneous - the design ratio is irrelevant.
    """
    X, w, y, _ = make_rct(n=200_000, treatment_share=0.5, seed=2)
    cs = inject_confounding(X, w, y, strength=0.0, seed=3)  # e == 0.5 everywhere
    assert abs(cs.ground_truth_att - cs.ground_truth_ate) < 1e-9


@pytest.mark.stat
def test_ate_and_att_targets_diverge_at_85_15():
    """The finding that made matching look like the best estimator: under an
    unbalanced design with heterogeneous effects, ATT != ATE, and matching
    targets the ATT."""
    X, w, y, _ = make_rct(n=300_000, treatment_share=0.85, seed=4)
    cs = inject_confounding(X, w, y, strength=1.5, seed=5)
    gap = abs(cs.ground_truth_att - cs.ground_truth_ate)
    assert gap > cs.ground_truth_se, (
        f"ATT-ATE gap {gap:.6f} is within the truth's own noise; "
        "the estimand distinction would not be demonstrable"
    )


def test_ground_truths_returns_both_targets_with_standard_errors():
    _, w, y, _ = make_rct(n=50_000, treatment_share=0.85, seed=6)
    e = np.full(len(y), 0.5)
    out = ground_truths(y, w, e, float(w.mean()))
    assert set(out) == {"ate", "ate_se", "att", "att_se"}
    assert out["ate_se"] > 0 and out["att_se"] > 0
    # at e = 0.5 the two weightings coincide
    assert out["ate"] == pytest.approx(out["att"], abs=1e-9)
