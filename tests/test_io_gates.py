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
