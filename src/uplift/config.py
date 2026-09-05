"""Central configuration.

Everything tunable lives here so that no magic number is hardcoded in twelve
places, and so the two *business assumptions* at the bottom are visible rather
than buried inside a revenue calculation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UPLIFT_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    evals_dir: Path = REPO_ROOT / "evals"
    docs_dir: Path = REPO_ROOT / "docs"
    duckdb_path: Path = REPO_ROOT / "data" / "uplift.duckdb"

    # DuckDB is happy to use every byte you give it. This box has 8GB, so the
    # limit is deliberately well under half of RAM; raise it via the env var on
    # a bigger machine.
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 4

    # --- dataset facts, asserted at ingest time ---
    criteo_urls: tuple[str, ...] = (
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
        "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
        "https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz",
    )
    criteo_sha256: str = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
    criteo_filename: str = "criteo-research-uplift-v2.1.csv.gz"
    expected_rows: int = 13_979_592
    designed_treatment_ratio: float = 0.85

    # --- analysis conventions ---
    primary_outcome: str = "visit"
    secondary_outcome: str = "conversion"
    feature_cols: tuple[str, ...] = tuple(f"f{i}" for i in range(12))
    srm_alpha: float = 0.001  # industry convention, not 0.05
    srm_effect_size_gate: float = 0.005  # what the pipeline actually blocks on
    # Two balance thresholds, because this data needs two. 0.02 is what a clean
    # RCT at n=14M should satisfy and is what the WARN-level dbt test uses;
    # 0.10 is the matching-literature "balanced" rule of thumb and is what the
    # blocking gate uses. Criteo v2.1 sits between them - see docs/findings.md.
    smd_threshold: float = 0.02  # warn level
    smd_blocking_threshold: float = 0.10  # error level / Dagster gate
    random_seed: int = 20260101

    # --- business assumptions, surfaced not buried ---
    # Every dollar figure in this repo is a function of these two numbers. They
    # are exposed as dashboard sliders and restated above any monetary claim.
    value_per_conversion_usd: float = 25.0
    cost_per_treatment_usd: float = 0.01

    @property
    def criteo_csv_path(self) -> Path:
        return self.data_dir / self.criteo_filename

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / "criteo_units.parquet"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.evals_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
