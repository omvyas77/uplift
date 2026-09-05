import numpy as np

from uplift.experiment.power import allocation_table, mde, required_n


def test_allocation_efficiency_matches_4r1minusr():
    for share in (0.5, 0.7, 0.85, 0.9):
        assert np.isclose(
            mde(1_000_000, 0.05, share).allocation_efficiency, 4 * share * (1 - share)
        )


def test_balanced_split_is_the_most_efficient():
    balanced = mde(1_000_000, 0.05, 0.5)
    skewed = mde(1_000_000, 0.05, 0.85)
    assert balanced.mde_absolute < skewed.mde_absolute
    assert balanced.allocation_efficiency == 1.0


def test_criteo_design_is_51_percent_efficient():
    """The headline design finding, guarded against a refactor."""
    r = mde(13_979_592, 0.046992, 0.85)
    assert np.isclose(r.allocation_efficiency, 0.51)
    assert 0.0094 < r.mde_relative < 0.0095


def test_mde_shrinks_with_n():
    assert mde(100_000, 0.05).mde_absolute > mde(10_000_000, 0.05).mde_absolute


def test_required_n_inverts_mde():
    """required_n and mde must be consistent: feed one into the other."""
    n = required_n(0.05, mde_relative=0.02, treatment_share=0.5)
    back = mde(n, 0.05, treatment_share=0.5)
    assert abs(back.mde_relative - 0.02) < 0.001


def test_allocation_table_reports_the_variance_penalty():
    rows = {r["treatment_share"]: r for r in allocation_table(13_979_592, 0.046992)}
    assert np.isclose(rows[0.5]["allocation_efficiency"], 1.0)
    assert np.isclose(rows[0.85]["n_multiple_for_same_mde"], 1 / 0.51)
