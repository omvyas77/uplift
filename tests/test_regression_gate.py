"""The CI regression gate must fail when it should - and only then.

A gate nobody trusts is worse than no gate, so both directions are pinned: a
real regression must trip it, and a noise-level dip must not.
"""

from __future__ import annotations

import copy

import pytest

from uplift.evaluation.regression import TOLERANCE_SE, gate

BASELINE = {
    "dataset": {"rows": 13_979_592, "sha256": "abc"},
    "models": {
        "s_learner": {"qini": 0.08932, "qini_ci": [0.0779, 0.1005]},
        "t_learner": {"qini": 0.07118, "qini_ci": [0.0620, 0.0825]},
    },
}
# 95% CI width 0.0226 -> SE ~0.00577
RESULTS = {
    "dataset": {"rows": 13_979_592, "sha256": "abc"},
    "models": copy.deepcopy(BASELINE["models"]),
}


def test_unchanged_results_pass():
    ok, _ = gate(copy.deepcopy(RESULTS), BASELINE)
    assert ok


def test_real_regression_fails():
    r = copy.deepcopy(RESULTS)
    r["models"]["s_learner"]["qini"] -= 0.02
    ok, lines = gate(r, BASELINE)
    assert not ok
    assert any("s_learner" in line and line.startswith("FAIL") for line in lines)


def test_noise_level_dip_passes():
    """Qini's bootstrap SE here is ~0.006. A strict never-decrease rule would
    fail roughly half of all reruns on noise alone."""
    r = copy.deepcopy(RESULTS)
    r["models"]["s_learner"]["qini"] -= 0.003
    ok, _ = gate(r, BASELINE)
    assert ok


def test_the_tolerance_boundary_is_where_it_claims_to_be():
    se = (0.1005 - 0.0779) / (2 * 1.96)
    for delta, expected_ok in ((TOLERANCE_SE * se * 0.9, True), (TOLERANCE_SE * se * 1.1, False)):
        r = copy.deepcopy(RESULTS)
        r["models"]["s_learner"]["qini"] -= delta
        ok, _ = gate(r, BASELINE)
        assert ok is expected_ok, f"delta={delta:.5f} should give ok={expected_ok}"


def test_an_improvement_passes():
    r = copy.deepcopy(RESULTS)
    r["models"]["s_learner"]["qini"] += 0.05
    ok, _ = gate(r, BASELINE)
    assert ok


def test_a_silently_dropped_model_fails():
    """Deleting a model must not be a way to make the gate green."""
    r = copy.deepcopy(RESULTS)
    del r["models"]["t_learner"]
    ok, lines = gate(r, BASELINE)
    assert not ok
    assert any("missing" in line for line in lines)


@pytest.mark.parametrize("key,value", [("rows", 123), ("sha256", "different")])
def test_dataset_drift_fails(key, value):
    """Comparing metrics across different data is meaningless, so say so loudly."""
    r = copy.deepcopy(RESULTS)
    r["dataset"][key] = value
    ok, lines = gate(r, BASELINE)
    assert not ok
    assert any(f"dataset {key} changed" in line for line in lines)


def test_a_new_model_not_in_the_baseline_is_ignored():
    """Adding a model should not require regenerating the floor first."""
    r = copy.deepcopy(RESULTS)
    r["models"]["brand_new"] = {"qini": -0.5, "qini_ci": [-0.6, -0.4]}
    ok, _ = gate(r, BASELINE)
    assert ok
