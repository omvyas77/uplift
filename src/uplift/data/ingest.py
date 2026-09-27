"""csv.gz -> Parquet (immutable, with a stable unit_id) -> DuckDB.

DuckDB reads gzipped CSV directly, so there is no decompress-to-disk step and
no pandas in the path.

`unit_id` is assigned ONCE and persisted to Parquet, and that Parquet file is
then treated as immutable. Regenerating the ids on every run would silently
reshuffle the train/test split - a subtle and very real source of "my metrics
moved and I don't know why".
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from uplift.config import settings
from uplift.logging import get_logger

log = get_logger(__name__)

FEATURES = ", ".join(f"f{i}" for i in range(12))


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    settings.ensure_dirs()
    con = duckdb.connect(str(settings.duckdb_path), read_only=read_only)
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET threads={settings.duckdb_threads}")
    tmp = settings.data_dir / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {sql}")
    return int(row[0])


def ingest(csv_path: Path | None = None, progress: bool = True) -> dict[str, float]:
    settings.ensure_dirs()
    csv_path = csv_path or settings.criteo_csv_path
    parquet_path = settings.parquet_path

    con = connect()
    if progress:
        con.execute("PRAGMA enable_progress_bar")

    if not parquet_path.exists():
        log.info("csv_to_parquet", src=str(csv_path), dst=str(parquet_path))
        con.execute(
            f"""
            COPY (
                SELECT
                    row_number() OVER () - 1        AS unit_id,
                    {FEATURES},
                    CAST(treatment  AS TINYINT)     AS treatment,
                    CAST(exposure   AS TINYINT)     AS exposure,
                    CAST(visit      AS TINYINT)     AS visit,
                    CAST(conversion AS TINYINT)     AS conversion
                FROM read_csv_auto('{csv_path}', header=true)
            )
            TO '{parquet_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
            """
        )
    else:
        log.info("parquet_exists_skipping_convert", path=str(parquet_path))

    con.execute(
        f"CREATE OR REPLACE VIEW raw_criteo_units AS SELECT * FROM read_parquet('{parquet_path}')"
    )

    # --- assertions: fail loudly at ingest, not silently at model time ---
    row = con.execute(
        """
        SELECT count(*), avg(treatment), avg(visit), avg(conversion), avg(exposure)
        FROM raw_criteo_units
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("summary query returned no rows; the table is empty")
    n, ratio, visit_rate, conv_rate, exposure_rate = row

    log.info(
        "ingested",
        rows=n,
        treatment_ratio=round(ratio, 6),
        visit_rate=round(visit_rate, 6),
        conversion_rate=round(conv_rate, 6),
        exposure_rate=round(exposure_rate, 6),
    )

    if n != settings.expected_rows:
        raise ValueError(f"expected {settings.expected_rows} rows, got {n}")
    if abs(ratio - settings.designed_treatment_ratio) > 0.01:
        raise ValueError(
            f"treatment ratio {ratio:.4f} far from designed {settings.designed_treatment_ratio}"
        )

    # Log, do not raise. We want to DISCOVER what the data does and write the
    # finding into the README, not assume it and silently WHERE it away.
    bad_exposure = _scalar(
        con, "SELECT count(*) FROM raw_criteo_units WHERE exposure = 1 AND treatment = 0"
    )
    log.info("structural_check", exposed_but_untreated=bad_exposure)

    bad_conv = _scalar(
        con, "SELECT count(*) FROM raw_criteo_units WHERE conversion = 1 AND visit = 0"
    )
    log.info("structural_check", converted_without_visit=bad_conv)

    con.close()
    return {
        "rows": int(n),
        "treatment_ratio": float(ratio),
        "visit_rate": float(visit_rate),
        "conversion_rate": float(conv_rate),
        "exposure_rate": float(exposure_rate),
        "exposed_but_untreated": int(bad_exposure),
        "converted_without_visit": int(bad_conv),
    }
