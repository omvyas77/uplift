"""Regenerate the committed 50k-row CI seed.

This is what makes CI free and fast: GitHub Actions builds the entire dbt
project against this file, in memory, in seconds, instead of downloading 311 MB.
"""

from __future__ import annotations

from uplift.config import settings
from uplift.data.ingest import connect
from uplift.logging import get_logger

log = get_logger(__name__)
SEED_PATH = settings.duckdb_path.parent.parent / "dbt" / "seeds" / "criteo_sample.csv"
N_ROWS = 50_000


# Features are rounded to 6 decimals in the seed. Full float64 text costs 12 MB
# for 50k rows, which is over the pre-commit large-file limit and buys nothing:
# the seed exists to exercise dbt's SQL, not to reproduce a statistic.
FEATURE_PRECISION = 6


def main() -> None:
    con = connect(read_only=True)
    feats = ", ".join(f"round(f{i}, {FEATURE_PRECISION}) AS f{i}" for i in range(12))
    con.execute(
        f"""
        COPY (
            SELECT unit_id, {feats}, treatment, exposure, visit, conversion
            FROM units
            USING SAMPLE {N_ROWS} ROWS (reservoir, {settings.random_seed})
        ) TO '{SEED_PATH}' (HEADER, DELIMITER ',')
        """
    )
    n, ratio = con.execute(
        f"SELECT count(*), avg(treatment::DOUBLE) FROM read_csv_auto('{SEED_PATH}')"
    ).fetchone()
    con.close()
    log.info("seed_written", path=str(SEED_PATH), rows=n, treatment_ratio=round(ratio, 4))


if __name__ == "__main__":
    main()
