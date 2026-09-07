"""ITT, CACE, and the `exposure` trap.

The point of this file is the last test: conditioning on a post-treatment
variable overstates the effect. That is the single distinction the build guide
says is worth more than three extra models, and it is the one this dataset
punishes hardest, because compliance is only 3.6%.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.causal.iv import estimand_table, itt_and_cace
from uplift.data.synthetic import make_noncompliance


def test_cace_recovers_the_complier_effect():
    z, d, y, true_cace = make_noncompliance(n=400_000, compliance=0.5, seed=5)
    r = itt_and_cace(y, z, d)
    assert abs(r.first_stage - 0.5) < 0.01
    assert abs(r.cace - true_cace) < 0.005


def test_cace_exceeds_itt_under_partial_compliance():
    """CACE = ITT / first_stage, and the first stage is below 1, so CACE is
    always the larger of the two under partial compliance."""
    z, d, y, _ = make_noncompliance(n=300_000, compliance=0.5, seed=6)
    r = itt_and_cace(y, z, d)
    assert r.cace > r.itt
    assert np.isclose(r.cace, r.itt / r.first_stage, rtol=1e-9)


def test_lower_compliance_widens_the_gap():
    """The lower the first stage, the more CACE and ITT diverge - which is why
    the gap on the real data is 28x at 3.6% compliance."""
    ratios = []
    for compliance in (0.8, 0.4, 0.1):
        z, d, y, _ = make_noncompliance(
            n=400_000, compliance=compliance, complier_effect=0.04, seed=7
        )
        r = itt_and_cace(y, z, d)
        ratios.append(r.cace / r.itt)
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[2] > 8  # at 10% compliance CACE is ~10x the ITT


def test_cace_is_stable_as_compliance_falls():
    """The complier effect itself is a fixed quantity: only the ITT shrinks."""
    for compliance in (0.8, 0.4, 0.2):
        z, d, y, true_cace = make_noncompliance(
            n=600_000, compliance=compliance, complier_effect=0.04, seed=8
        )
        r = itt_and_cace(y, z, d)
        assert abs(r.cace - true_cace) < 0.008, f"failed at compliance={compliance}"


def test_itt_is_the_product_of_cace_and_compliance():
    z, d, y, true_cace = make_noncompliance(n=400_000, compliance=0.3, seed=9)
    r = itt_and_cace(y, z, d)
    assert abs(r.itt - true_cace * 0.3) < 0.004


def test_naive_exposed_comparison_overstates_the_effect():
    """THE test in this file.

    When who-complies is related to who-would-have-converted-anyway, comparing
    EXPOSED users to controls conditions on a post-treatment variable and
    overstates the effect. That is the trap the `exposure` column sets.
    """
    z, d, y, true_cace = make_noncompliance(
        n=600_000,
        compliance=0.5,
        complier_effect=0.04,
        compliance_selection=1.5,
        baseline_spread=0.04,
        seed=10,
    )
    r = itt_and_cace(y, z, d)
    assert abs(r.cace - true_cace) < 0.006, "CACE should still be roughly right"
    assert r.naive_exposed_vs_control > r.cace
    assert r.naive_bias_ratio > 1.2, f"expected a clear overstatement, got {r.naive_bias_ratio:.2f}"


def test_naive_is_unbiased_when_compliance_is_random():
    """The control that makes the previous test meaningful.

    With `compliance_selection=0` compliance is a coin flip independent of the
    outcome, and the naive exposed-vs-control comparison happens to be unbiased
    for the CACE. So a test suite built on the default generator would show no
    trap at all - the bias comes from SELECTION into exposure, not from
    non-compliance as such.
    """
    z, d, y, _ = make_noncompliance(n=600_000, compliance=0.5, seed=10)
    r = itt_and_cace(y, z, d)
    assert abs(r.naive_bias_ratio - 1.0) < 0.1


def test_irrelevant_instrument_raises():
    """A zero first stage means CACE is 0/0. Fail loudly rather than return inf."""
    rng = np.random.default_rng(11)
    n = 10_000
    z = rng.binomial(1, 0.85, n)
    d = np.zeros(n, dtype=int)  # nobody is ever exposed
    y = rng.binomial(1, 0.05, n)
    with pytest.raises(ValueError, match="first stage"):
        itt_and_cace(y, z, d)


def test_estimand_table_labels_the_invalid_comparison():
    """The table exists to say which number answers which question, and which
    one answers nothing."""
    z, d, y, _ = make_noncompliance(n=100_000, seed=12)
    rows = estimand_table(itt_and_cace(y, z, d))
    by_name = {r["estimand"]: r for r in rows}
    assert set(by_name) == {"ITT", "CACE", "exposed-vs-control"}
    assert "INVALID" in by_name["exposed-vs-control"]["valid"]
    assert by_name["exposed-vs-control"]["answers"].lower() == "nothing"
    assert "Always" in by_name["ITT"]["valid"]
