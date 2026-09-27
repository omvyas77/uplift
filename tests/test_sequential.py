"""The peeking simulation.

This module was written, verified by hand once, and then never imported by
anything for weeks - while the README listed "a peeking simulation" under what
the project demonstrates. It is now wired into the experiment readout and pinned
here, so the claim is backed by something that runs.

Criteo has no time column, so none of this monitors the real experiment. Every
number below comes from data generated under a KNOWN NULL, which is the honest
way to show the peeking problem on a dataset that cannot be monitored.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.experiment.sequential import (
    msprt_pvalue,
    obrien_fleming_boundaries,
    peeking_simulation,
    peeking_simulation_corrected,
)


@pytest.mark.stat
def test_peeking_inflates_the_false_positive_rate():
    """The whole point: repeated testing of a NULL experiment at alpha=0.05
    rejects far more than 5% of the time."""
    out = peeking_simulation(n_peeks=10, n_per_peek=4000, n_sims=300, seed=1)
    assert out["actual_false_positive_rate"] > 0.10, out


@pytest.mark.stat
def test_alpha_spending_brings_it_back():
    naive = peeking_simulation(n_peeks=10, n_per_peek=4000, n_sims=300, seed=1)
    fixed = peeking_simulation_corrected(n_peeks=10, n_per_peek=4000, n_sims=300, seed=1)
    assert fixed["actual_false_positive_rate"] < naive["actual_false_positive_rate"]
    assert fixed["actual_false_positive_rate"] < 0.10


@pytest.mark.stat
def test_a_single_look_is_already_calibrated():
    """One peek is just a fixed-sample test, so it must sit near alpha. This is
    the control: without it, a low rate could mean the simulation is broken
    rather than that the correction works."""
    out = peeking_simulation(n_peeks=1, n_per_peek=20_000, n_sims=600, seed=2)
    assert 0.02 < out["actual_false_positive_rate"] < 0.09, out


def test_obrien_fleming_boundaries_are_decreasing_and_end_at_the_fixed_value():
    b = obrien_fleming_boundaries(5, alpha=0.05)
    assert np.all(np.diff(b) < 0), "boundaries must relax as information accrues"
    assert np.isclose(b[-1], 1.959963984540054)
    assert b[0] > 4, "an early look must demand overwhelming evidence"


def test_msprt_pvalue_is_bounded_and_monotone_safe():
    rng = np.random.default_rng(3)
    a, b = rng.binomial(1, 0.05, 20_000), rng.binomial(1, 0.05, 20_000)
    p = msprt_pvalue(a, b)
    assert p.shape == (20_000,)
    assert np.all((p >= 0) & (p <= 1))
    # always-valid p-values are built from a running max, so they never increase
    assert np.all(np.diff(p) <= 1e-12)


@pytest.mark.stat
def test_msprt_rarely_dips_below_alpha_under_the_null():
    """An always-valid p-value may be inspected at EVERY look without inflating
    the error rate - the property the naive test lacks."""
    rng = np.random.default_rng(4)
    crossings = 0
    trials = 60
    for _ in range(trials):
        a, b = rng.binomial(1, 0.05, 8_000), rng.binomial(1, 0.05, 8_000)
        if np.nanmin(msprt_pvalue(a, b)) < 0.05:
            crossings += 1
    assert crossings / trials < 0.15, f"{crossings}/{trials} crossed under the null"
