"""The Dagster definitions must load, and the gates must really be gates.

`orchestration/` was linted and type-checked in CI but never executed, so a
broken asset graph would have shipped green. `dagster definitions validate`
catches import-time errors; these tests additionally pin the two properties the
pipeline's safety actually depends on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dagster", reason="install the `orchestrate` extra")
pytest.importorskip("dagster_dbt", reason="install the `orchestrate` extra")


@pytest.fixture(scope="module")
def defs():
    from orchestration.definitions import defs as _defs

    return _defs


def test_definitions_load(defs):
    assert defs is not None


def test_the_expected_assets_exist(defs):
    keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    for expected in (
        "criteo_raw",
        "criteo_units",
        "uplift_models",
        "uplift_evaluation",
        "targeting_policy",
    ):
        assert any(expected in k for k in keys), f"missing asset {expected}; have {sorted(keys)}"


def test_srm_and_balance_checks_are_blocking():
    """If these stop blocking, a broken randomization could train a model.

    That is the single principle the orchestration layer exists to enforce, so
    it is asserted rather than assumed.
    """
    from orchestration.definitions import check_covariate_balance, check_no_srm

    for check in (check_no_srm, check_covariate_balance):
        specs = list(check.check_specs)
        assert specs, f"{check} declares no check specs"
        assert all(s.blocking for s in specs), f"{check} is not blocking"


def test_the_regression_check_is_not_blocking():
    """Deliberately non-blocking: a metric regression should be loud and visible
    but must not stop the policy artifact being rebuilt for inspection."""
    from orchestration.definitions import check_no_metric_regression

    assert all(not s.blocking for s in check_no_metric_regression.check_specs)


def test_the_schedule_is_weekly(defs):
    schedules = list(defs.schedules)
    assert schedules, "no schedule defined"
    assert any("* * 1" in s.cron_schedule for s in schedules)
