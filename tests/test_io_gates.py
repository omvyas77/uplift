"""The mart readers back BLOCKING asset checks, so they must fail closed.

A gate that reports "no SRM" because it could not find its input is worse than
a gate that crashes. These tests pin that behaviour.
"""

import duckdb
import pytest

from uplift.data.io import MartNotFoundError, resolve_mart


def test_missing_mart_raises_rather_than_returning_empty():
    con = duckdb.connect(":memory:")
    with pytest.raises(MartNotFoundError, match="mart_arm_summary"):
        resolve_mart(con, "mart_arm_summary")


def test_resolve_finds_a_mart_in_a_non_default_schema():
    """dbt writes marts to `main_marts`, not `main`. Hardcoding either is the
    bug this function exists to prevent."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA main_marts")
    con.execute("CREATE TABLE main_marts.mart_arm_summary AS SELECT 1 AS split")
    assert resolve_mart(con, "mart_arm_summary") == '"main_marts"."mart_arm_summary"'


def test_resolve_finds_views_too():
    con = duckdb.connect(":memory:")
    con.execute("CREATE VIEW mart_covariate_balance AS SELECT 1 AS smd")
    assert "mart_covariate_balance" in resolve_mart(con, "mart_covariate_balance")


def test_sampling_applies_after_the_split_filter():
    """`USING SAMPLE` trailing a WHERE samples the TABLE, not the filtered set.

    Asking for 100,000 train rows returned 59,758 - the train share of a
    100,000-row sample of everything - and asking for a sample of the control
    arm returned 15% of the request. Both silently. Found while building the
    A/A test, which asked for 80,000 control rows and got 12,078.
    """
    import duckdb

    from uplift.data.io import _sampled_query

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE units AS "
        "SELECT i AS unit_id, CASE WHEN i % 10 < 6 THEN 'train' ELSE 'test' END AS split "
        "FROM range(100000) t(i)"
    )
    n = len(con.execute(_sampled_query("unit_id", "units", "train", 20_000, 1)).df())
    assert n == 20_000, f"asked for 20,000 train rows, got {n}"

    # and the filter must still be applied
    splits = con.execute(_sampled_query("unit_id, split", "units", "train", 5_000, 1)).df()["split"]
    assert set(splits) == {"train"}


def test_sampling_without_a_split_is_unchanged():
    import duckdb

    from uplift.data.io import _sampled_query

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE units AS SELECT i AS unit_id FROM range(50000) t(i)")
    assert len(con.execute(_sampled_query("unit_id", "units", None, 7_000, 1)).df()) == 7_000
    # no sample requested -> everything
    assert len(con.execute(_sampled_query("unit_id", "units", None, None, 1)).df()) == 50_000
