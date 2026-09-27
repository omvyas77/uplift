"""Thin read helpers used by the CLI, the Dagster checks, and the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift.config import settings
from uplift.data.ingest import connect

FEATURES = [f"f{i}" for i in range(12)]


def table_exists(con, name: str) -> bool:
    return bool(
        con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [name]).fetchone()[
            0
        ]
        or con.execute(
            "SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [name]
        ).fetchone()[0]
    )


def load_split(
    split: str | None = None,
    outcome: str = "visit",
    sample: int | None = None,
    seed: int = 20260101,
    con=None,
    columns: str = "units",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split as (X, w, y).

    NOTE: `exposure` is deliberately NOT returned as a feature. It is a
    post-treatment variable and using it as a feature would break
    identification. `load_iv_split` returns it, for the IV module only.
    """
    close = con is None
    con = con or connect(read_only=True)
    try:
        where = f"WHERE split = '{split}'" if split else ""
        samp = f"USING SAMPLE {sample} ROWS (reservoir, {seed})" if sample else ""
        df = con.execute(
            f"SELECT {', '.join(FEATURES)}, treatment, {outcome} AS y FROM {columns} {where} {samp}"
        ).df()
    finally:
        if close:
            con.close()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    w = df["treatment"].to_numpy(dtype=np.int8)
    y = df["y"].to_numpy(dtype=np.int8)
    return X, w, y


def load_iv_split(
    split: str | None = None,
    outcome: str = "visit",
    sample: int | None = None,
    seed: int = 20260101,
    con=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, z, d, y) for the IV / CACE module - the ONLY place `exposure` is read."""
    close = con is None
    con = con or connect(read_only=True)
    try:
        where = f"WHERE split = '{split}'" if split else ""
        samp = f"USING SAMPLE {sample} ROWS (reservoir, {seed})" if sample else ""
        df = con.execute(
            f"SELECT {', '.join(FEATURES)}, treatment, exposure, {outcome} AS y "
            f"FROM units {where} {samp}"
        ).df()
    finally:
        if close:
            con.close()
    return (
        df[FEATURES].to_numpy(dtype=np.float32),
        df["treatment"].to_numpy(dtype=np.int8),
        df["exposure"].to_numpy(dtype=np.int8),
        df["y"].to_numpy(dtype=np.int8),
    )


class MartNotFoundError(RuntimeError):
    """A mart the pipeline gates on could not be found.

    Raised rather than returning empty, because the callers are BLOCKING asset
    checks. A gate that cannot read its input must fail closed: reporting "no
    SRM" because the table was missing is strictly worse than crashing.
    """


def resolve_mart(con, table: str) -> str:
    """Fully-qualified name of a dbt mart, whatever schema it landed in.

    dbt writes marts into `main_marts` (dbt_project.yml sets +schema: marts,
    and dbt-duckdb prefixes the target schema), not `main`. Hardcoding either
    one is how these helpers silently broke - so look it up instead.
    """
    rows = con.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() WHERE table_name = ?", [table]
    ).fetchall()
    if not rows:
        rows = con.execute(
            "SELECT schema_name, view_name FROM duckdb_views() WHERE view_name = ?", [table]
        ).fetchall()
    if not rows:
        raise MartNotFoundError(
            f"{table!r} not found in {settings.duckdb_path}. Run `make dbt-build` first."
        )
    schema, name = rows[0]
    return f'"{schema}"."{name}"'


def _read_mart(table: str, con=None) -> pd.DataFrame:
    close = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {resolve_mart(con, table)}").df()
    finally:
        if close:
            con.close()
    if df.empty:
        raise MartNotFoundError(f"{table!r} exists but is empty; the gate cannot be evaluated")
    return df


def read_arm_summary(con=None) -> pd.DataFrame:
    return _read_mart("mart_arm_summary", con)


def read_balance(con=None) -> pd.DataFrame:
    return _read_mart("mart_covariate_balance", con)
