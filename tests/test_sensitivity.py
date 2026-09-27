"""Sensitivity analysis — E-values and negative controls, previously at 0%.

The negative-control panel is what independently confirmed the Criteo imbalance
finding by a different route than the SMD analysis (max |z| 26.18 on the raw
RCT), and the E-value is quoted in the causal report. Neither had a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.causal.sensitivity import (
    e_value,
    negative_control_outcome,
    negative_control_panel,
    risk_ratio,
    rosenbaum_bounds,
)


# ---------------------------------------------------------------------- E-value
def test_no_effect_needs_no_confounding_to_explain():
    """RR = 1 means there is nothing to explain away, so the E-value is 1."""
    assert np.isclose(e_value(1.0)["e_value_point"], 1.0)


def test_e_value_grows_with_the_effect():
    """A bigger association takes a stronger hidden confounder to overturn."""
    vals = [e_value(rr)["e_value_point"] for rr in (1.2, 2.0, 4.0)]
    assert vals == sorted(vals)
    assert all(v > 1 for v in vals)


def test_the_published_formula_is_what_is_implemented():
    """E = RR + sqrt(RR*(RR-1)), from VanderWeele & Ding. RR=2 gives 1+sqrt(2)."""
    assert np.isclose(e_value(2.0)["e_value_point"], 2 + np.sqrt(2))


def test_protective_and_harmful_effects_are_symmetric():
    """RR and 1/RR describe the same strength of association in opposite
    directions, so they must need the same confounder to explain away."""
    assert np.isclose(e_value(2.0)["e_value_point"], e_value(0.5)["e_value_point"])


def test_a_ci_touching_the_null_needs_no_confounding():
    """If the interval already includes 1, no unmeasured confounding is required
    to make the effect vanish - the E-value for the bound is 1 by definition."""
    assert np.isclose(e_value(1.5, rr_ci_bound=0.9)["e_value_ci"], 1.0)


def test_a_ci_clear_of_the_null_needs_real_confounding():
    out = e_value(2.0, rr_ci_bound=1.5)
    assert out["e_value_ci"] > 1.0
    # the bound is closer to the null, so it is easier to explain away
    assert out["e_value_ci"] < out["e_value_point"]


# ---------------------------------------------------------------------- risk ratio
@pytest.mark.stat
def test_risk_ratio_recovers_a_planted_ratio():
    rng = np.random.default_rng(0)
    n = 400_000
    w = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.where(w == 1, 0.10, 0.05))  # RR = 2
    rr, lo, hi = risk_ratio(y, w)
    assert abs(rr - 2.0) < 0.1
    assert lo < 2.0 < hi


# ------------------------------------------------------------- negative controls
def test_a_pre_treatment_covariate_shows_no_effect_in_an_rct():
    """Treatment cannot affect something measured before it. In a true RCT this
    check must pass - that is what makes its FAILURE informative elsewhere."""
    rng = np.random.default_rng(1)
    n = 100_000
    X = rng.normal(size=(n, 4))
    w = rng.binomial(1, 0.85, n)  # independent of X
    panel = negative_control_panel(X, w)
    assert panel["z"].abs().max() < 4
    assert not panel["fails_at_z3"].any()


def test_it_fires_loudly_when_treatment_is_correlated_with_a_covariate():
    """The confounded case: selection on X makes a pre-treatment covariate look
    'caused' by treatment."""
    rng = np.random.default_rng(2)
    n = 100_000
    X = rng.normal(size=(n, 4))
    w = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)  # treatment depends on X
    panel = negative_control_panel(X, w)
    assert panel["z"].abs().max() > 20
    assert panel.iloc[0]["feature"] == "f0"


def test_weighting_can_restore_balance():
    """Passing weights must actually change the answer - otherwise the
    'after IPW' column in the causal report would be meaningless."""
    rng = np.random.default_rng(3)
    n = 60_000
    X = rng.normal(size=(n, 2))
    w = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    unweighted = negative_control_outcome(X, w, 0)
    # weights that undo the selection: down-weight where treatment was likely
    ps = 1.0 / (1.0 + np.exp(-X[:, 0]))
    wts = np.where(w == 1, 1.0 / np.clip(ps, 0.05, 0.95), 1.0 / np.clip(1 - ps, 0.05, 0.95))
    weighted = negative_control_outcome(X, w, 0, weights=wts)
    assert abs(weighted["effect"]) < abs(unweighted["effect"])


def test_panel_is_sorted_worst_first():
    rng = np.random.default_rng(4)
    n = 40_000
    X = rng.normal(size=(n, 3))
    w = rng.binomial(1, 0.5, n)
    X[w == 1, 1] += 0.2
    panel = negative_control_panel(X, w)
    assert panel["feature"].iloc[0] == "f1"
    assert list(panel["z"].abs()) == sorted(panel["z"].abs(), reverse=True)


# ---------------------------------------------------------------------- Rosenbaum
def test_rosenbaum_gamma_one_means_no_hidden_bias():
    rows = {r["gamma"]: r for r in rosenbaum_bounds()}
    assert np.isclose(rows[1.0]["max_log_odds_shift"], 0.0)
    assert "no hidden bias" in rows[1.0]["interpretation"]


def test_rosenbaum_bound_grows_with_gamma():
    shifts = [r["max_log_odds_shift"] for r in rosenbaum_bounds((1.0, 1.5, 2.0))]
    assert shifts == sorted(shifts)
