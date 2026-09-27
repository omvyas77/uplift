"""Deterministic 60/20/20 train/valid/test buckets.

Two decisions worth stating:

* The bucket is a hash of the immutable `unit_id`, not a random shuffle, so the
  split survives a re-run, a re-clone, and a machine change.
* The split is deliberately NOT stratified by treatment. The assignment is
  already random; stratifying invites you to think of it as something you
  control. `assert_no_treatment_leakage_across_splits` checks the ratio is
  stable across splits instead.
"""

from __future__ import annotations

import duckdb

from uplift.logging import get_logger

log = get_logger(__name__)

# duckdb's hash() is deterministic within a major version, and duckdb is pinned
# in uv.lock. PORTABLE_BUCKET_SQL is the version-independent alternative -
# slower, still fine at this scale.
BUCKET_SQL = "mod(abs(hash(unit_id)), 100)"
PORTABLE_BUCKET_SQL = (
    "mod(abs(CAST(('0x' || substr(md5(CAST(unit_id AS VARCHAR)), 1, 8)) AS BIGINT)), 100)"
)

SPLIT_SQL = f"""
CREATE OR REPLACE TABLE units AS
SELECT
    *,
    {BUCKET_SQL} AS bucket,
    CASE
        WHEN {BUCKET_SQL} < 60 THEN 'train'
        WHEN {BUCKET_SQL} < 80 THEN 'valid'
        ELSE 'test'
    END AS split
FROM raw_criteo_units
"""


def build_splits(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, float]]:
    con.execute(SPLIT_SQL)
    rows = con.execute(
        """
        SELECT split, count(*) AS n, avg(treatment::DOUBLE) AS treatment_ratio,
               avg(visit::DOUBLE) AS visit_rate
        FROM units GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    out = {
        r[0]: {"n": int(r[1]), "treatment_ratio": float(r[2]), "visit_rate": float(r[3])}
        for r in rows
    }
    log.info("splits_built", **{k: v["n"] for k, v in out.items()})
    return out
