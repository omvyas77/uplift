# Building **Uplift** — A Complete Step-by-Step Build Guide

*A causal experimentation & uplift-modeling platform on the Criteo incrementality data.*

**Target:** ~10 hrs/week × 10 weeks. Roughly 100 hours of work.
**Outcome:** a public repo, a live demo URL, a blog post, two résumé bullets, and the ability to answer causal-inference interview questions from lived experience rather than reading.

---

## Table of contents

- [0. Read this before you write any code](#0-read-this-before-you-write-any-code)
- [1. Locked technology decisions](#1-locked-technology-decisions)
- [2. Day 0 — environment setup](#2-day-0--environment-setup)
- [3. Repo scaffold](#3-repo-scaffold)
- [4. Phase 1 (weeks 1–2) — ingestion, DuckDB, dbt](#4-phase-1-weeks-12--ingestion-duckdb-dbt)
- [5. Phase 2 (weeks 3–4) — the experiment-analysis module](#5-phase-2-weeks-34--the-experiment-analysis-module)
- [6. Phase 3 (weeks 5–6) — uplift models and Qini/AUUC](#6-phase-3-weeks-56--uplift-models-and-qiniauuc)
- [7. Phase 4 (weeks 7–8) — quasi-experimental, IV, sensitivity](#7-phase-4-weeks-78--quasi-experimental-iv-sensitivity)
- [8. Phase 5 (weeks 9–10) — policy, API, dashboard, ops](#8-phase-5-weeks-910--policy-api-dashboard-ops)
- [9. README, blog post, résumé](#9-readme-blog-post-résumé)
- [Appendix A — command cheat sheet](#appendix-a--command-cheat-sheet)
- [Appendix B — pitfalls that make a repo look junior](#appendix-b--pitfalls-that-make-a-repo-look-junior)
- [Appendix C — verified dataset facts](#appendix-c--verified-dataset-facts)
- [Appendix D — weekly checklist](#appendix-d--weekly-checklist)

---

## 0. Read this before you write any code

### 0.1 What actually makes this project stand out

There are hundreds of "Criteo uplift modeling" repos on GitHub. Almost all of them are one notebook that trains a T-learner and plots a Qini curve. If you build that, you have added nothing.

Five things separate this build from those. Everything in this guide is organized around delivering them. If you are short on time, cut anything else before you cut these.

**① Validate your observational estimators against randomized ground truth.**
You have an RCT. Deliberately destroy the randomization by sampling units as a function of their covariates, then run IPW / matching / doubly-robust estimation on the corrupted slice and check whether you recover the number you already know from the RCT. Almost nobody does this, because it requires understanding that the RCT *is* the answer key. It produces a bias table — a genuinely rare artifact — and it lets you say "my doubly-robust estimator recovered the true effect to within 6% while the naive comparison was off by 80%" instead of "I ran propensity score matching."

**② Handle the `exposure` column correctly.**
Criteo ships four outcome-ish columns: `treatment`, `exposure`, `visit`, `conversion`. `exposure` records whether the user was *actually shown* the ad. It is a **post-treatment variable**. Conditioning on it — filtering to `exposure = 1`, or adding it as a feature — breaks randomization and inflates your effect estimate. The correct treatment of it is intention-to-treat as the headline number, plus a complier-average causal effect (CACE) via instrumental variables as a secondary estimand. This single distinction is worth more in an interview than three extra models.

**③ Do variance reduction honestly.**
Criteo has no pre-experiment period, so textbook CUPED does not directly apply. The right move is CUPAC (use a cross-fitted prediction of the outcome from pre-treatment covariates as the covariate) or Lin's regression-adjusted estimator. Then report the variance reduction you actually achieved, which on this data will be small, and explain why: variance reduction equals ρ² where ρ is the correlation between the covariate and the outcome, and 12 randomly-projected anonymized features do not predict a 4.7% binary outcome well. Reporting a small honest number with the reason beats claiming a big one.

**④ Turn model scores into an evaluated decision.**
Stop at "here is the Qini curve" and you have built a model. Go on to define a targeting policy, estimate that policy's value on held-out randomized data via inverse-propensity / doubly-robust off-policy evaluation, and put a confidence interval on it — now you have built a decision system. This is the difference between a data scientist and a modeler.

**⑤ Test the statistics, not just the code.**
Write tests that generate synthetic data with a *known* treatment effect and assert your estimators recover it. This is the highest-signal thing in the repo for anyone who reads code, it makes CI fast and free (no 14M-row download), and it will catch real bugs in your own implementations.

### 0.2 What you are deliberately *not* building

- **Not Kubernetes.** This is a batch analytics pipeline with a small serving surface. Docker Compose plus scheduled GitHub Actions is correct. Say so in the README. Pretending otherwise reads as cargo-culting.
- **Not a deep learning uplift model.** DragonNet/TARNet add nothing here and cost you a week.
- **Not a Spark cluster.** DuckDB handles 14M × 16 columns on a laptop, faster than Spark would.
- **Not real-time streaming.** There is no stream.

### 0.3 Honest scope warning

You will hit two walls, and knowing about them now saves a week each:

1. **The uplift signal in Criteo is weak.** Qini coefficients will be small and unstable. This is a property of the data (12 anonymized, randomly-projected features), not your code. Plan to report bootstrap confidence intervals on Qini from the start rather than discovering later that your "best model" is inside the noise band of your worst.
2. **`conversion` has a base rate near 0.3%.** Uplift modeling on it is extremely noisy. Use `visit` (4.7%) as your primary outcome — the Criteo paper itself recommends this — and treat conversion as a secondary analysis with explicit caveats.

---

## 1. Locked technology decisions

Decide these once, write them in the README, and stop relitigating them.

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Language | **Python 3.12** | CausalML publishes wheels for cp311/cp312 only. 3.13 will force a source build. 3.12 is the safe ceiling. |
| Env/deps | **uv** | Fast, lockfile-based, handles the scientific stack's pin conflicts far better than pip+venv. |
| Storage | **DuckDB** (local file) + **Parquet** | Free, embedded, out-of-core, reads gzipped CSV natively. BigQuery free tier is the swap-in if you want a cloud story. |
| Transform | **dbt-core + dbt-duckdb** | Gets you the analytics-engineering signal: staging→marts, tests, docs, lineage. |
| Metrics layer | **MetricFlow** (`dbt-metricflow[duckdb]`), fallback to a `mart_metrics` model + `metrics.yml` | See §4.7 — MetricFlow works on DuckDB but DuckDB is not on dbt's officially supported adapter list. Have the fallback ready. |
| Dataframes | **Polars** for ETL, **pandas** at the modeling boundary | Polars for the big scans; pandas because EconML/scikit-uplift expect numpy/pandas. Do not fight this. |
| Uplift metrics | **scikit-uplift** (`sklift`) | Canonical Qini/AUUC implementations plus the dataset fetcher. Use theirs as the reported numbers. |
| Causal models | **EconML 0.17.0** | Meta-learners, DR-learner, CausalForestDML, and the IV estimators you need for CACE — all in one package. |
| Causal models (optional) | **CausalML 0.17.0** | Adds uplift trees and its own metrics. Genuinely optional — see §2.4 on why you might skip it. |
| Base learners | **LightGBM** | Fast on 14M rows, handles the feature ranges without scaling. |
| Classical stats | **statsmodels**, **scipy.stats** | Robust standard errors, chi-square, power. |
| API | **FastAPI + Pydantic v2 + uvicorn** | |
| Dashboard | **Streamlit** | Metabase means running another container for no added signal. |
| Orchestration | **Dagster + dagster-dbt** | Asset-based model fits this DAG better than Airflow's task model, and asset *checks* are the natural home for your SRM/balance gates. |
| Containers | **Docker + Docker Compose** | |
| CI | **GitHub Actions** | |
| Lint/format/types | **ruff**, **mypy** | |
| Tests | **pytest** + **hypothesis** (optional) | |
| Deploy | **Hugging Face Space** (dashboard) or **Fly.io / Cloud Run** (API) | Free or near-free. You need a live URL. |

---

## 2. Day 0 — environment setup

### 2.1 Install uv and pin Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l

mkdir uplift && cd uplift
git init
uv python install 3.12
uv init --python 3.12 --name uplift
```

### 2.2 `pyproject.toml`

Create this at the repo root. Note the dependency *groups* — this is deliberate, see §2.4.

```toml
[project]
name = "uplift"
version = "0.1.0"
description = "Causal experimentation and uplift-modeling platform on randomized incrementality data"
requires-python = ">=3.12,<3.13"
dependencies = [
    "duckdb>=1.1",
    "polars>=1.0",
    "pandas>=2.2",
    "numpy>=1.26,<3",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "lightgbm>=4.5",
    "statsmodels>=0.14",
    "pyarrow>=17",
    "pydantic>=2.8",
    "pydantic-settings>=2.4",
    "typer>=0.12",
    "structlog>=24.1",
    "matplotlib>=3.9",
]

[project.optional-dependencies]
causal = [
    "econml>=0.17,<0.18",
    "scikit-uplift>=0.5.1",
]
# Optional and isolated on purpose - see section 2.4
causal-extra = [
    "causalml>=0.17,<0.18",
]
dbt = [
    "dbt-core>=1.9",
    "dbt-duckdb>=1.9",
]
metrics = [
    "dbt-metricflow[duckdb]",
]
serve = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "streamlit>=1.38",
    "plotly>=5.24",
]
orchestrate = [
    "dagster>=1.8",
    "dagster-webserver>=1.8",
    "dagster-dbt>=0.24",
]
dev = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.11",
    "types-requests",
    "ipykernel",
    "jupyterlab",
]

[project.scripts]
uplift = "uplift.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PD", "NPY", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
warn_unused_ignores = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
    "slow: requires the full dataset; excluded from CI",
    "stat: statistical test with a fixed seed and a tolerance",
]
```

### 2.3 Install and lock

```bash
uv sync --extra causal --extra dbt --extra serve --extra orchestrate --extra dev
uv run python -c "import econml, sklift, duckdb, lightgbm; print('ok')"
git add pyproject.toml uv.lock && git commit -m "chore: pin toolchain"
```

**Commit `uv.lock`.** Reproducibility is part of what you are demonstrating.

### 2.4 On CausalML — read before installing

EconML and CausalML both compile against numpy/scipy/scikit-learn and both pin them tightly. Installing them into one environment is the single most common way this project stalls on day one.

**You do not need CausalML.** EconML gives you S/T/X-learners, DR-learner, causal forests, and IV. scikit-uplift gives you Qini/AUUC, the uplift-by-percentile tables, the plots, and the dataset fetcher. Between them every deliverable in this guide is covered.

CausalML adds uplift *trees* (a genuinely different model class that splits directly on an uplift criterion) and its own metric implementations. That is a nice-to-have.

**Recommended:** build the whole project on `econml` + `scikit-uplift`. If you want uplift trees at the end, install CausalML into a *separate* environment and export predictions to Parquet:

```bash
uv venv .venv-causalml --python 3.12
uv pip install --python .venv-causalml causalml==0.17.0 pandas pyarrow
```

CausalML 0.17.0 requires Python ≥ 3.11 and its Linux wheels need glibc ≥ 2.28 (Ubuntu 20.04+, Debian 10+, RHEL 8+). On older Linux it builds from source, which needs a C++ toolchain and Cython.

### 2.5 Repo hygiene from commit one

`.gitignore`:

```gitignore
.venv/
.venv-*/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data: NEVER commit. Ship the download script instead.
data/
*.duckdb
*.duckdb.wal
*.csv
*.csv.gz
*.parquet

# dbt
dbt/target/
dbt/dbt_packages/
dbt/logs/
.user.yml

# Artifacts
artifacts/
models/*.pkl
models/*.joblib
!models/.gitkeep

.env
.DS_Store
.ipynb_checkpoints/
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: detect-private-key
```

```bash
uv run pre-commit install
```

The `check-added-large-files` hook is what stops you from accidentally committing a 311 MB CSV, which is the most common way these repos become unclonable.

---

## 3. Repo scaffold

Create this structure now, in full, with empty files. It stops you from drifting into notebook sprawl.

```
uplift/
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml
├── uv.lock
├── Makefile
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # lint + types + fast tests (every push)
│       ├── dbt.yml                  # dbt build + test on a seeded sample
│       └── eval.yml                 # weekly full eval + regression gate
│
├── src/uplift/
│   ├── __init__.py
│   ├── cli.py                       # typer entrypoint
│   ├── config.py                    # pydantic-settings
│   ├── logging.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # fetch + checksum the Criteo file
│   │   ├── ingest.py                # CSV.gz -> Parquet -> DuckDB
│   │   ├── splits.py                # deterministic train/valid/test buckets
│   │   └── synthetic.py             # known-ground-truth generator for tests + CI
│   │
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── srm.py                   # sample ratio mismatch
│   │   ├── balance.py               # standardized mean differences
│   │   ├── ate.py                   # diff-in-means, Lin estimator, robust SEs
│   │   ├── cuped.py                 # CUPED / CUPAC
│   │   ├── power.py                 # MDE, power, allocation efficiency
│   │   └── sequential.py            # always-valid p-values, peeking simulation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── learners.py              # S/T/X/DR learners, causal forest
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── qini.py                  # Qini/AUUC + bootstrap CIs
│   │   ├── calibration.py           # CATE calibration / GATES
│   │   └── report.py                # writes evals/results.json
│   │
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── confounding.py           # inject selection bias into the RCT
│   │   ├── estimators.py            # IPW, matching, AIPW, DML
│   │   ├── iv.py                    # ITT vs CACE using `exposure`
│   │   ├── did.py                   # diff-in-diff (Olist)
│   │   └── sensitivity.py           # E-value, Rosenbaum, negative controls
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── targeting.py             # threshold selection
│   │   └── ope.py                   # off-policy evaluation (IPW / DR)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   └── dashboard/
│       └── app.py
│
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── packages.yml
│   ├── models/
│   │   ├── sources.yml
│   │   ├── staging/
│   │   │   ├── stg_criteo__units.sql
│   │   │   └── stg_criteo__units.yml
│   │   ├── intermediate/
│   │   │   ├── int_units_with_splits.sql
│   │   │   └── int_cupac_covariate.sql
│   │   └── marts/
│   │       ├── fct_experiment_units.sql
│   │       ├── mart_arm_summary.sql
│   │       ├── mart_covariate_balance.sql
│   │       ├── mart_uplift_deciles.sql
│   │       ├── semantic_models.yml
│   │       └── schema.yml
│   ├── tests/
│   │   ├── assert_conversion_implies_visit.sql
│   │   ├── assert_exposure_implies_treatment.sql
│   │   ├── assert_treatment_ratio_in_range.sql
│   │   └── assert_covariate_balance.sql
│   ├── macros/
│   │   └── deterministic_bucket.sql
│   └── seeds/
│       └── criteo_sample.csv        # ~50k rows for CI. This one IS committed.
│
├── orchestration/
│   ├── __init__.py
│   └── definitions.py               # Dagster assets, checks, schedules
│
├── evals/
│   ├── results.json                 # current metrics (committed)
│   ├── baseline.json                # regression floor (committed)
│   └── history/                     # metrics over time
│
├── notebooks/                       # exploration ONLY, never the deliverable
│   └── 00_explore.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── test_srm.py
│   ├── test_cuped.py
│   ├── test_qini.py
│   ├── test_power.py
│   ├── test_estimators_recover_truth.py   # the important one
│   ├── test_iv.py
│   └── test_api.py
│
└── docs/
    ├── architecture.png
    ├── metric_definitions.md
    └── decisions.md                 # ADR-style: what you chose and why
```

```bash
mkdir -p src/uplift/{data,experiment,models,evaluation,causal,policy,api,dashboard}
mkdir -p dbt/{models/{staging,intermediate,marts},tests,macros,seeds}
mkdir -p orchestration evals/history notebooks tests docs .github/workflows
find src orchestration -type d -exec touch {}/__init__.py \;
git add -A && git commit -m "chore: scaffold"
```

### 3.1 `Makefile`

Write this now; you will use it constantly.

```makefile
.PHONY: help install download ingest dbt-build dbt-test experiment train eval api dashboard dagster test lint fmt docker-up clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:        ## sync the environment
	uv sync --extra causal --extra dbt --extra serve --extra orchestrate --extra dev

download:       ## fetch + verify the Criteo file
	uv run uplift download

ingest:         ## csv.gz -> parquet -> duckdb
	uv run uplift ingest

dbt-build:      ## run all dbt models
	cd dbt && uv run dbt deps && uv run dbt build

dbt-test:       ## dbt tests only
	cd dbt && uv run dbt test

experiment:     ## SRM, balance, ATE, CUPED, power
	uv run uplift experiment-report

train:          ## fit the uplift models
	uv run uplift train

eval:           ## evaluate + write evals/results.json
	uv run uplift evaluate

causal:         ## confounding-injection bias table + IV + sensitivity
	uv run uplift causal-report

api:            ## serve the targeting API
	uv run uvicorn uplift.api.main:app --reload --port 8000

dashboard:      ## serve the Streamlit dashboard
	uv run streamlit run src/uplift/dashboard/app.py

dagster:        ## local Dagster UI
	uv run dagster dev -m orchestration.definitions

test:           ## fast tests (no big data)
	uv run pytest -m "not slow"

test-all:       ## everything, including full-data tests
	uv run pytest

lint:
	uv run ruff check src tests orchestration
	uv run mypy src

fmt:
	uv run ruff format src tests orchestration
	uv run ruff check --fix src tests orchestration

docker-up:
	docker compose up --build

clean:
	rm -rf data/*.parquet data/*.duckdb dbt/target dbt/logs artifacts/*
```

### 3.2 `src/uplift/config.py`

Central config so nothing is hardcoded in twelve places.

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UPLIFT_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    evals_dir: Path = REPO_ROOT / "evals"
    duckdb_path: Path = REPO_ROOT / "data" / "uplift.duckdb"

    # --- dataset facts, asserted at ingest time ---
    criteo_urls: tuple[str, ...] = (
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
        "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
        "https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz",
    )
    criteo_sha256: str = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
    expected_rows: int = 13_979_592
    designed_treatment_ratio: float = 0.85

    # --- analysis conventions ---
    primary_outcome: str = "visit"
    secondary_outcome: str = "conversion"
    feature_cols: tuple[str, ...] = tuple(f"f{i}" for i in range(12))
    srm_alpha: float = 0.001          # industry convention, not 0.05
    smd_threshold: float = 0.02       # balance tolerance
    random_seed: int = 20260101

    # --- business assumptions, surfaced not buried ---
    value_per_conversion_usd: float = 25.0
    cost_per_treatment_usd: float = 0.01

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.evals_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
```

The two business assumptions at the bottom belong in config, exposed as dashboard sliders, and stated in the README. Every incremental-revenue number you quote depends on them; hiding them is how portfolio projects end up claiming implausible dollar figures.

---

## 4. Phase 1 (weeks 1–2) — ingestion, DuckDB, dbt

**Goal by end of week 2:** `make download && make ingest && make dbt-build` runs clean from a fresh clone, and `dbt test` passes including a randomization-balance check.

### 4.1 Understand the data before you model it

Four columns matter more than the twelve features:

| Column | Meaning | Status | How to use it |
|---|---|---|---|
| `treatment` | Randomly assigned to the ad campaign | **Pre-treatment / the randomization** | This is your `W`. Everything headline is ITT on this. |
| `exposure` | The user was actually *shown* an ad | **POST-treatment** | Never a feature. Never a filter. Use only as the compliance indicator in the IV/CACE module (§7.4). |
| `visit` | Visited the advertiser site | Primary outcome | Base rate ≈ 4.7%. Your `Y`. |
| `conversion` | Converted | Secondary outcome | Base rate ≈ 0.3%. Too rare for reliable uplift ranking; report with caveats. |
| `f0`–`f11` | Anonymized, randomly projected features | Pre-treatment | Your `X`. **No interpretability** — do not write "f3 is likely user age." |

Structural facts to verify (and to encode as dbt tests):
- Only treated users can be exposed, so `exposure = 1 ⟹ treatment = 1`.
- A conversion should imply a visit, so `conversion = 1 ⟹ visit = 1`. **Verify this empirically** — if it does not hold, that is a finding worth a paragraph in the README, not something to silently `WHERE` away.
- There is **no time column.** No dates, no user IDs, no sessions. This constrains you: no classic difference-in-differences, no genuine sequential monitoring. §4.6 and §7.5 handle this honestly.

### 4.2 `src/uplift/data/download.py`

```python
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import structlog

from uplift.config import settings

log = structlog.get_logger()
CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download_criteo(force: bool = False) -> Path:
    """Download the Criteo uplift file from the first mirror that works, then verify."""
    settings.ensure_dirs()
    dest = settings.data_dir / "criteo-research-uplift-v2.1.csv.gz"

    if dest.exists() and not force:
        if _sha256(dest) == settings.criteo_sha256:
            log.info("already_present", path=str(dest))
            return dest
        log.warning("checksum_mismatch_redownloading", path=str(dest))

    last_error: Exception | None = None
    for url in settings.criteo_urls:
        try:
            log.info("downloading", url=url)
            tmp = dest.with_suffix(".part")
            req = urllib.request.Request(url, headers={"User-Agent": "uplift-portfolio/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, CHUNK)
            tmp.replace(dest)
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror_failed", url=url, error=str(exc))
            last_error = exc
    else:
        raise RuntimeError(f"all mirrors failed; last error: {last_error}")

    digest = _sha256(dest)
    if digest != settings.criteo_sha256:
        raise ValueError(
            f"checksum mismatch\n  expected {settings.criteo_sha256}\n  got      {digest}"
        )
    log.info("verified", path=str(dest), sha256=digest)
    return dest
```

Three mirrors and a checksum, because a portfolio repo that fails on `make download` eighteen months from now is worse than no repo. The checksum also proves to a reader that you know which exact file version you used.

### 4.3 `src/uplift/data/ingest.py`

DuckDB reads gzipped CSV directly; there is no need to decompress to disk or load into pandas.

```python
from __future__ import annotations

import duckdb
import structlog

from uplift.config import settings

log = structlog.get_logger()

FEATURES = ", ".join(f"f{i}" for i in range(12))


def ingest(csv_path=None) -> None:
    """csv.gz -> Parquet (immutable, with a stable unit_id) -> DuckDB table."""
    settings.ensure_dirs()
    csv_path = csv_path or settings.data_dir / "criteo-research-uplift-v2.1.csv.gz"
    parquet_path = settings.data_dir / "criteo_units.parquet"

    con = duckdb.connect(str(settings.duckdb_path))
    con.execute("PRAGMA enable_progress_bar")
    # Bound memory so this works on a 16GB laptop.
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{settings.data_dir}/duckdb_tmp'")

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

    con.execute(
        f"CREATE OR REPLACE VIEW raw_criteo_units AS "
        f"SELECT * FROM read_parquet('{parquet_path}')"
    )

    # --- assertions: fail loudly at ingest, not silently at model time ---
    n, ratio, visit_rate, conv_rate = con.execute(
        """
        SELECT count(*), avg(treatment), avg(visit), avg(conversion)
        FROM raw_criteo_units
        """
    ).fetchone()

    log.info("ingested", rows=n, treatment_ratio=round(ratio, 6),
             visit_rate=round(visit_rate, 6), conversion_rate=round(conv_rate, 6))

    if n != settings.expected_rows:
        raise ValueError(f"expected {settings.expected_rows} rows, got {n}")
    if abs(ratio - settings.designed_treatment_ratio) > 0.01:
        raise ValueError(f"treatment ratio {ratio:.4f} far from designed {settings.designed_treatment_ratio}")

    bad_exposure = con.execute(
        "SELECT count(*) FROM raw_criteo_units WHERE exposure = 1 AND treatment = 0"
    ).fetchone()[0]
    log.info("structural_check", exposed_but_untreated=bad_exposure)

    bad_conv = con.execute(
        "SELECT count(*) FROM raw_criteo_units WHERE conversion = 1 AND visit = 0"
    ).fetchone()[0]
    log.info("structural_check", converted_without_visit=bad_conv)

    con.close()
```

`unit_id` is assigned **once** and persisted to Parquet. That Parquet file is then immutable. This is what makes your train/test split reproducible — regenerate the ids on every run and your split silently reshuffles, which is a subtle and very real source of "my metrics moved and I don't know why."

Log the two structural checks rather than raising. You want to *discover* what the data does and then write the finding into the README, not assume it.

### 4.4 Deterministic splits — `src/uplift/data/splits.py`

```python
SPLIT_SQL = """
CREATE OR REPLACE TABLE units AS
SELECT
    *,
    mod(abs(hash(unit_id)), 100) AS bucket,
    CASE
        WHEN mod(abs(hash(unit_id)), 100) < 60 THEN 'train'
        WHEN mod(abs(hash(unit_id)), 100) < 80 THEN 'valid'
        ELSE 'test'
    END AS split
FROM raw_criteo_units
"""
```

**60/20/20.** Train the learners, tune on valid, and touch `test` exactly once, at the end. With 14M rows a 20% test set is ~2.8M units, which is plenty of precision for the held-out Qini estimates.

Two notes:
- `hash()` is deterministic within a DuckDB major version. Pin `duckdb` in `uv.lock` and say so in the README. If you want version-independence, use `mod(abs(CAST(('0x' || substr(md5(CAST(unit_id AS VARCHAR)), 1, 8)) AS BIGINT)), 100)` instead — portable, slower, still fine at this scale.
- **Do not stratify the split by treatment.** The assignment is already random; stratifying invites you to think of it as something you control.

### 4.5 dbt setup

```bash
cd dbt
```

`dbt_project.yml`:

```yaml
name: uplift
version: "1.0.0"
config-version: 2
profile: uplift

model-paths: ["models"]
test-paths: ["tests"]
macro-paths: ["macros"]
seed-paths: ["seeds"]

target-path: "target"
clean-targets: ["target", "dbt_packages"]

models:
  uplift:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: table
      +schema: intermediate
    marts:
      +materialized: table
      +schema: marts

seeds:
  uplift:
    criteo_sample:
      +column_types:
        unit_id: BIGINT
        treatment: TINYINT
        exposure: TINYINT
        visit: TINYINT
        conversion: TINYINT
```

`profiles.yml` (keep it *in* the repo so a cloner does not have to configure `~/.dbt/`):

```yaml
uplift:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "../data/uplift.duckdb"
      threads: 4
      extensions: []
    ci:
      type: duckdb
      path: ":memory:"
      threads: 2
```

The `ci` target running in-memory is what lets GitHub Actions build the whole dbt project against the committed 50k-row seed in seconds.

`packages.yml`:

```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: [">=1.3.0", "<2.0.0"]
  - package: metaplane/dbt_expectations
    version: [">=0.10.0", "<0.11.0"]
```

```bash
uv run dbt deps
uv run dbt debug
```

### 4.6 Models

`models/sources.yml`:

```yaml
version: 2

sources:
  - name: raw
    schema: main
    description: >
      Criteo Uplift Prediction Dataset v2.1 (13,979,592 rows), assembled from
      randomized incrementality trials. Loaded by `uplift ingest`, not by dbt.
    tables:
      - name: units
        description: One row per randomized unit.
        columns:
          - name: unit_id
            description: Stable surrogate key assigned at ingestion.
            tests: [unique, not_null]
          - name: treatment
            description: Randomized assignment. THE instrument / the experiment.
            tests:
              - not_null
              - accepted_values: {values: [0, 1]}
          - name: exposure
            description: >
              POST-TREATMENT. Whether an ad was actually served. Never use as a
              feature or filter; used only as the compliance indicator for CACE.
            tests:
              - not_null
              - accepted_values: {values: [0, 1]}
          - name: visit
            description: Primary outcome. Base rate ~4.7%.
            tests:
              - not_null
              - accepted_values: {values: [0, 1]}
          - name: conversion
            description: Secondary outcome. Base rate ~0.3%; too rare for stable ranking.
            tests:
              - not_null
              - accepted_values: {values: [0, 1]}
```

Notice the `exposure` description. Documentation that encodes a *methodological* warning, not just a column type, is one of the cheapest ways to signal seniority to anyone browsing your `dbt docs`.

`models/staging/stg_criteo__units.sql`:

```sql
{{ config(materialized='view') }}

select
    unit_id,
    {% for i in range(12) %}
    cast(f{{ i }} as double) as f{{ i }},
    {% endfor %}
    cast(treatment  as tinyint) as is_treated,
    cast(exposure   as tinyint) as is_exposed,
    cast(visit      as tinyint) as visited,
    cast(conversion as tinyint) as converted
from {{ source('raw', 'units') }}
```

`models/intermediate/int_units_with_splits.sql`:

```sql
{{ config(materialized='table') }}

with base as (select * from {{ ref('stg_criteo__units') }})

select
    *,
    {{ deterministic_bucket('unit_id') }} as bucket,
    case
        when {{ deterministic_bucket('unit_id') }} < 60 then 'train'
        when {{ deterministic_bucket('unit_id') }} < 80 then 'valid'
        else 'test'
    end as split
from base
```

`macros/deterministic_bucket.sql`:

```sql
{% macro deterministic_bucket(id_column, n_buckets=100) %}
    mod(abs(hash({{ id_column }})), {{ n_buckets }})
{% endmacro %}
```

`models/marts/mart_arm_summary.sql` — the headline experiment table:

```sql
{{ config(materialized='table') }}

with per_arm as (
    select
        split,
        is_treated,
        count(*)                        as n_units,
        sum(visited)                    as n_visits,
        sum(converted)                  as n_conversions,
        sum(is_exposed)                 as n_exposed,
        avg(visited::double)            as visit_rate,
        avg(converted::double)          as conversion_rate,
        avg(is_exposed::double)         as exposure_rate
    from {{ ref('int_units_with_splits') }}
    group by 1, 2
)

select
    split,
    max(case when is_treated = 1 then n_units end)          as n_treatment,
    max(case when is_treated = 0 then n_units end)          as n_control,
    max(case when is_treated = 1 then visit_rate end)       as visit_rate_treatment,
    max(case when is_treated = 0 then visit_rate end)       as visit_rate_control,
    max(case when is_treated = 1 then visit_rate end)
      - max(case when is_treated = 0 then visit_rate end)   as visit_rate_lift_abs,
    (max(case when is_treated = 1 then visit_rate end)
      / nullif(max(case when is_treated = 0 then visit_rate end), 0)) - 1
                                                            as visit_rate_lift_rel,
    max(case when is_treated = 1 then conversion_rate end)  as conv_rate_treatment,
    max(case when is_treated = 0 then conversion_rate end)  as conv_rate_control,
    max(case when is_treated = 1 then exposure_rate end)    as compliance_rate,
    max(case when is_treated = 1 then n_units end)::double
      / nullif(sum(n_units), 0)                             as observed_treatment_ratio
from per_arm
group by split
```

`models/marts/mart_covariate_balance.sql` — the randomization check, in SQL:

```sql
{{ config(materialized='table') }}

with unpivoted as (
    {% for i in range(12) %}
    select 'f{{ i }}' as feature, is_treated, f{{ i }} as value
    from {{ ref('int_units_with_splits') }}
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
),

stats as (
    select
        feature,
        is_treated,
        count(*)      as n,
        avg(value)    as mean_value,
        var_samp(value) as var_value
    from unpivoted
    group by 1, 2
)

select
    feature,
    max(case when is_treated = 1 then mean_value end) as mean_treatment,
    max(case when is_treated = 0 then mean_value end) as mean_control,
    -- standardized mean difference, pooled-SD version
    (max(case when is_treated = 1 then mean_value end)
     - max(case when is_treated = 0 then mean_value end))
    / nullif(sqrt((
        max(case when is_treated = 1 then var_value end)
        + max(case when is_treated = 0 then var_value end)
      ) / 2.0), 0) as smd
from stats
group by feature
```

Under a valid randomization every `|smd|` should be tiny. This mart is what your dbt balance test reads.

### 4.7 Tests — this is where the analytics-engineering signal lives

Generic tests go in `models/marts/schema.yml`:

```yaml
version: 2

models:
  - name: int_units_with_splits
    columns:
      - name: unit_id
        tests: [unique, not_null]
      - name: split
        tests:
          - accepted_values: {values: ['train', 'valid', 'test']}
      - name: bucket
        tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 99

  - name: mart_arm_summary
    columns:
      - name: observed_treatment_ratio
        tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0.84
              max_value: 0.86
      - name: compliance_rate
        description: P(exposed | treated). The IV first stage.
        tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0.01
              max_value: 1.0

  - name: mart_covariate_balance
    columns:
      - name: smd
        description: >
          Standardized mean difference between arms. Under valid randomization
          this should be ~0 for every feature. A failure here means the
          randomization assumption underpinning every downstream number is broken.
        tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: -0.02
              max_value: 0.02
```

Singular tests go in `tests/` — a singular test passes when it returns **zero rows**.

`tests/assert_exposure_implies_treatment.sql`:

```sql
-- Only assigned-to-treatment units can be exposed to the ad.
-- A violation would mean the assignment/exposure logging is inconsistent
-- and the IV first stage is not what we think it is.
select unit_id
from {{ ref('int_units_with_splits') }}
where is_exposed = 1 and is_treated = 0
```

`tests/assert_conversion_implies_visit.sql`:

```sql
-- Funnel monotonicity: a conversion should imply a visit.
select unit_id
from {{ ref('int_units_with_splits') }}
where converted = 1 and visited = 0
```

`tests/assert_splits_are_balanced.sql`:

```sql
-- Hash bucketing should give ~60/20/20. Allow 1pp drift.
with s as (
    select split, count(*)::double / sum(count(*)) over () as frac
    from {{ ref('int_units_with_splits') }}
    group by 1
)
select * from s
where (split = 'train' and abs(frac - 0.60) > 0.01)
   or (split = 'valid' and abs(frac - 0.20) > 0.01)
   or (split = 'test'  and abs(frac - 0.20) > 0.01)
```

`tests/assert_no_treatment_leakage_across_splits.sql`:

```sql
-- The treatment ratio must be stable across splits; a drift here means the
-- split is correlated with assignment, which would bias held-out evaluation.
with s as (
    select split, avg(is_treated::double) as ratio
    from {{ ref('int_units_with_splits') }}
    group by 1
)
select * from s where abs(ratio - 0.85) > 0.005
```

```bash
uv run dbt build
uv run dbt docs generate && uv run dbt docs serve
```

**Take a screenshot of the dbt lineage graph.** It goes in the README.

### 4.8 The metrics layer

Define your metrics once, in YAML, so that "north-star metric" is a file rather than a claim.

`models/marts/semantic_models.yml`:

```yaml
version: 2

semantic_models:
  - name: experiment_units
    description: One row per randomized unit in the incrementality trial.
    model: ref('int_units_with_splits')
    defaults:
      agg_time_dimension: null
    entities:
      - name: unit
        type: primary
        expr: unit_id
    dimensions:
      - name: split
        type: categorical
      - name: is_treated
        type: categorical
    measures:
      - name: units
        agg: count
        expr: unit_id
      - name: visits
        agg: sum
        expr: visited
      - name: conversions
        agg: sum
        expr: converted
      - name: exposures
        agg: sum
        expr: is_exposed

metrics:
  - name: visit_rate
    label: "Visit rate (NORTH STAR)"
    description: >
      P(visit). The primary decision metric. Chosen over conversion rate because
      its 4.7% base rate gives usable precision, per the Criteo paper's own
      recommendation; conversion at ~0.3% is a guardrail, not a decision metric.
    type: ratio
    type_params:
      numerator: visits
      denominator: units

  - name: conversion_rate
    label: "Conversion rate (SECONDARY)"
    type: ratio
    type_params:
      numerator: conversions
      denominator: units

  - name: compliance_rate
    label: "Exposure rate among treated (IV FIRST STAGE)"
    description: P(exposed | treated). CACE = ITT / compliance_rate.
    type: ratio
    type_params:
      numerator: exposures
      denominator: units
    filter: "{{ Dimension('unit__is_treated') }} = 1"
```

```bash
uv pip install "dbt-metricflow[duckdb]"
cd dbt
uv run mf list metrics
uv run mf query --metrics visit_rate --group-by unit__is_treated
uv run mf query --metrics visit_rate --group-by unit__is_treated --explain
```

**Honest caveat, and put it in your README:** dbt's official documentation lists MetricFlow support for Snowflake, BigQuery, Databricks, Postgres, and Redshift. MetricFlow does ship a DuckDB SQL renderer and this works in practice, but DuckDB is not on the officially supported list. If it breaks on a version bump, fall back to a plain `mart_metrics.sql` model plus a `docs/metric_definitions.md` file. You lose the `mf` CLI; you keep the substance, which is that metrics are defined once, version-controlled, and testable. Knowing the difference between the substance and the tool is the point.

### 4.9 The CI seed

Create a small committed sample so CI never downloads 311 MB:

```python
# scripts/make_seed.py
import duckdb
from uplift.config import settings

con = duckdb.connect(str(settings.duckdb_path))
con.execute(
    f"""
    COPY (
        SELECT * FROM raw_criteo_units
        USING SAMPLE 50000 ROWS (reservoir, {settings.random_seed})
    ) TO 'dbt/seeds/criteo_sample.csv' (HEADER, DELIMITER ',')
    """
)
```

~50k rows is about 4 MB — comfortably under the pre-commit large-file limit and enough for every dbt test except the tight balance tolerances (loosen those to ±0.05 on the `ci` target using a dbt var).

### 4.10 Phase 1 done when

- [ ] `make download && make ingest && make dbt-build` works from a clean clone
- [ ] `dbt test` passes, including balance and split-stability tests
- [ ] You can state the observed treatment ratio, visit rate, conversion rate, and compliance rate to 4 decimals
- [ ] You know empirically whether `conversion ⟹ visit` holds in this data
- [ ] dbt docs lineage screenshot saved to `docs/`

---

## 5. Phase 2 (weeks 3–4) — the experiment-analysis module

**Goal by end of week 4:** `make experiment` prints a defensible experiment readout — ATE with a correct confidence interval, an SRM verdict, a balance table, the variance reduction you achieved, and the MDE this design supports.

This is the module that answers *"What is CUPED, how do you power an experiment, and how do you detect SRM?"* Build it before the fancy models. It is faster to build, and in product-DS interviews it comes up more often.

### 5.1 SRM — `src/uplift/experiment/srm.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    n_treatment: int
    n_control: int
    observed_ratio: float
    expected_ratio: float
    chi2: float
    p_value: float
    is_srm: bool
    abs_deviation: float
    ratio_ci: tuple[float, float]

    def summary(self) -> str:
        verdict = "SRM DETECTED" if self.is_srm else "no SRM"
        return (
            f"{verdict}: observed {self.observed_ratio:.6f} "
            f"(95% CI {self.ratio_ci[0]:.6f}–{self.ratio_ci[1]:.6f}) "
            f"vs designed {self.expected_ratio:.6f}; "
            f"chi2={self.chi2:.2f}, p={self.p_value:.3e}"
        )


def check_srm(
    n_treatment: int,
    n_control: int,
    expected_ratio: float = 0.85,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit against the DESIGNED allocation.

    alpha defaults to 0.001, not 0.05. SRM checks run on every experiment and
    every metric; at 0.05 you would raise a false alarm on 1 experiment in 20.
    """
    n = n_treatment + n_control
    observed = np.array([n_treatment, n_control], dtype=float)
    expected = np.array([n * expected_ratio, n * (1.0 - expected_ratio)])

    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    p = float(stats.chi2.sf(chi2, df=1))

    ratio = n_treatment / n
    lo, hi = stats.binomtest(n_treatment, n).proportion_ci(confidence_level=0.95)

    return SRMResult(
        n_treatment=n_treatment,
        n_control=n_control,
        observed_ratio=ratio,
        expected_ratio=expected_ratio,
        chi2=chi2,
        p_value=p,
        is_srm=p < alpha,
        abs_deviation=abs(ratio - expected_ratio),
        ratio_ci=(float(lo), float(hi)),
    )
```

**The interesting part, and the thing to write about.** Run this on Criteo and it will very likely report SRM. That is not a bug. At n ≈ 14M the test can detect a deviation of a few *thousandths* from 0.85, and the published design ratio ".85" is stated to two decimals. Your readout should therefore report the observed ratio with a CI alongside the p-value, and your README should say:

> The chi-square SRM test rejects against a designed ratio of exactly 0.85. The observed ratio is X.XXXX (95% CI ...). At n = 14M this test has power to detect deviations far smaller than the precision to which the design ratio is published, so the rejection is uninformative about data quality. This is the standard large-sample failure mode of significance testing: report the effect size, not only the p-value.

That paragraph demonstrates more statistical maturity than any model in the repo.

Add a calibration demonstration so you can *prove* the test works:

```python
def simulate_srm_calibration(
    n: int = 1_000_000,
    ratio: float = 0.85,
    n_sims: int = 1000,
    true_ratio: float | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Under H0 the rejection rate should equal alpha; under H1 it should be high."""
    rng = np.random.default_rng(seed)
    actual = ratio if true_ratio is None else true_ratio
    draws = rng.binomial(n, actual, size=n_sims)
    results = [check_srm(int(t), n - int(t), expected_ratio=ratio) for t in draws]
    return {
        "rejection_rate_at_0.001": float(np.mean([r.is_srm for r in results])),
        "rejection_rate_at_0.05": float(np.mean([r.p_value < 0.05 for r in results])),
    }
```

Call it twice — once with `true_ratio=None` (expect ≈ 0.001 and ≈ 0.05) and once with `true_ratio=0.8505` (expect near 1.0). Put both in the README. Now your SRM check is *validated*, not merely written.

### 5.2 Balance — `src/uplift/experiment/balance.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def standardized_mean_differences(
    df: pd.DataFrame, features: list[str], treatment_col: str = "is_treated"
) -> pd.DataFrame:
    """SMD per feature. |SMD| < 0.1 is the usual 'balanced' rule of thumb;
    for a true RCT at n=14M expect |SMD| < 0.01."""
    t = df[treatment_col].to_numpy().astype(bool)
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=float)
        m1, m0 = x[t].mean(), x[~t].mean()
        v1, v0 = x[t].var(ddof=1), x[~t].var(ddof=1)
        pooled = np.sqrt((v1 + v0) / 2.0)
        rows.append(
            {
                "feature": f,
                "mean_treatment": m1,
                "mean_control": m0,
                "smd": (m1 - m0) / pooled if pooled > 0 else 0.0,
                "var_ratio": v1 / v0 if v0 > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["balanced"] = out["smd"].abs() < 0.1
    return out.sort_values("smd", key=np.abs, ascending=False)
```

Report the variance ratio too, not just the mean difference. Two groups can have identical means and very different spreads; a mean-only balance table would call that balanced.

### 5.3 ATE — `src/uplift/experiment/ate.py`

Three estimators, because the comparison between them *is* the variance-reduction story.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class ATEResult:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    relative_lift: float
    relative_ci: tuple[float, float]
    baseline_rate: float
    n: int

    def summary(self) -> str:
        return (
            f"{self.method:<24} ATE={self.ate:+.6f} "
            f"[{self.ci_low:+.6f}, {self.ci_high:+.6f}]  "
            f"rel={100 * self.relative_lift:+.3f}% "
            f"[{100 * self.relative_ci[0]:+.3f}%, {100 * self.relative_ci[1]:+.3f}%]  "
            f"SE={self.se:.2e}  p={self.p_value:.3e}"
        )


def _relative_ci(y1: np.ndarray, y0: np.ndarray, alpha: float = 0.05):
    """Delta-method CI for the RATIO of two means. The absolute-difference CI
    cannot simply be divided by the baseline: the baseline is estimated too."""
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    ratio = m1 / m0
    se_log = np.sqrt(v1 / m1**2 + v0 / m0**2)      # delta method on log(m1/m0)
    z = stats.norm.ppf(1 - alpha / 2)
    return ratio - 1.0, (ratio * np.exp(-z * se_log) - 1.0, ratio * np.exp(z * se_log) - 1.0)


def ate_difference_in_means(y: np.ndarray, w: np.ndarray, alpha: float = 0.05) -> ATEResult:
    """Neyman's estimator. The unbiased baseline every other method must beat."""
    y1, y0 = y[w == 1], y[w == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    rel, rel_ci = _relative_ci(y1, y0, alpha)
    return ATEResult(
        method="difference-in-means",
        ate=float(ate), se=float(se),
        ci_low=float(ate - z * se), ci_high=float(ate + z * se),
        p_value=float(2 * stats.norm.sf(abs(ate / se))),
        relative_lift=float(rel), relative_ci=rel_ci,
        baseline_rate=float(y0.mean()), n=len(y),
    )


def ate_lin_regression(
    y: np.ndarray, w: np.ndarray, X: np.ndarray, alpha: float = 0.05
) -> ATEResult:
    """Lin (2013): regress Y on W, centered X, and their INTERACTIONS, with
    HC-robust SEs. Never less efficient than difference-in-means asymptotically,
    unlike plain ANCOVA which can be worse under effect heterogeneity."""
    Xc = X - X.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(y)), w, Xc, w[:, None] * Xc])
    fit = sm.OLS(y, design).fit(cov_type="HC1")

    ate, se = float(fit.params[1]), float(fit.bse[1])
    z = stats.norm.ppf(1 - alpha / 2)
    baseline = float(y[w == 0].mean())
    rel = ate / baseline
    return ATEResult(
        method="Lin regression-adjusted",
        ate=ate, se=se,
        ci_low=ate - z * se, ci_high=ate + z * se,
        p_value=float(fit.pvalues[1]),
        relative_lift=rel,
        relative_ci=((ate - z * se) / baseline, (ate + z * se) / baseline),
        baseline_rate=baseline, n=len(y),
    )
```

Two details worth understanding rather than copying:

- **Why HC1 and not default OLS SEs.** With a binary outcome the error variance differs by arm, so homoskedastic SEs are wrong. Robust SEs fix it. If an interviewer asks "how do you get valid standard errors here," this is the answer.
- **Why Lin and not plain ANCOVA.** Regressing `Y ~ W + X` without interactions can be *less* efficient than the simple difference when the treatment effect varies with X. Adding `W × X_centered` (Lin's estimator) removes that risk. That is a genuinely non-obvious point and a good thing to be able to explain.

### 5.4 CUPED / CUPAC — `src/uplift/experiment/cuped.py`

**The problem you must confront out loud:** CUPED needs a *pre-experiment* covariate, typically the same metric measured before the experiment started. Criteo has no pre-period and no time dimension. So the textbook version is unavailable.

**The correct adaptation:** use a cross-fitted prediction of the outcome from pre-treatment covariates as the covariate. This is CUPAC (Control Using Predictions As Covariates). It is valid because `f0`–`f11` are pre-treatment and therefore independent of assignment, which is the only property CUPED actually requires.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class CUPEDResult:
    theta: float
    correlation: float
    variance_reduction: float       # equals rho^2
    se_before: float
    se_after: float
    se_reduction: float
    effective_sample_multiplier: float


def cuped_adjust(y: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """Y_adj = Y - theta*(X - mean(X)),  theta = Cov(Y,X)/Var(X).

    The adjustment is mean-preserving, so the ATE estimate is unchanged in
    expectation; only its variance shrinks.
    """
    theta = float(np.cov(y, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1))
    return y - theta * (covariate - covariate.mean()), theta


def build_cupac_covariate(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Cross-fitted E[Y | X] fitted on CONTROL units only.

    Control-only + cross-fitting is what keeps the covariate free of any
    treatment information, so it stays a legitimate pre-treatment covariate.
    """
    pred = np.zeros(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        ctrl = train_idx[w[train_idx] == 0]
        model = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, verbose=-1, random_state=seed,
        )
        model.fit(X[ctrl], y[ctrl])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def evaluate_cuped(y: np.ndarray, w: np.ndarray, covariate: np.ndarray) -> CUPEDResult:
    y_adj, theta = cuped_adjust(y, covariate)
    rho = float(np.corrcoef(y, covariate)[0, 1])

    def se_of_diff(vals: np.ndarray) -> float:
        a, b = vals[w == 1], vals[w == 0]
        return float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))

    se_before, se_after = se_of_diff(y), se_of_diff(y_adj)
    var_red = 1.0 - (y_adj.var(ddof=1) / y.var(ddof=1))

    return CUPEDResult(
        theta=theta,
        correlation=rho,
        variance_reduction=float(var_red),
        se_before=se_before,
        se_after=se_after,
        se_reduction=1.0 - se_after / se_before,
        # a variance reduction of v is worth 1/(1-v) times the sample size
        effective_sample_multiplier=float(1.0 / (1.0 - var_red)) if var_red < 1 else float("inf"),
    )
```

**The key identity, verified:** variance reduction = ρ², where ρ = corr(Y, covariate). On synthetic data with ρ = 0.6, θ came out at 0.5992 and the measured variance reduction at 0.3598 — exactly ρ² = 0.36 — with the mean preserved to five decimals.

**What to expect on Criteo, and what to say about it.** With a 4.7% binary outcome and twelve randomly-projected features, ρ will be small, so the variance reduction will be a few percent at best. Report the actual number. Then explain it:

> CUPAC delivered X% variance reduction, equivalent to a Y% larger sample. This is modest because variance reduction is bounded by ρ² and the anonymized features predict the binary visit outcome only weakly (control-set R² ≈ Z). In a production setting the same machinery on a pre-period version of the metric typically achieves 30–50%, because a user's past behaviour predicts their future behaviour far better than 12 projected features do.

A small honest number plus the reason it is small is a better interview answer than a large number you cannot account for.

### 5.5 Power and allocation — `src/uplift/experiment/power.py`

This module contains the single most quotable finding in the whole project.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PowerResult:
    mde_absolute: float
    mde_relative: float
    n_total: int
    treatment_share: float
    baseline_rate: float
    power: float
    alpha: float
    allocation_efficiency: float
    n_equivalent_balanced: int


def mde(
    n_total: int,
    baseline_rate: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
) -> PowerResult:
    """Minimum detectable effect for a two-proportion test with unequal allocation."""
    n_t = n_total * treatment_share
    n_c = n_total * (1 - treatment_share)
    p = baseline_rate
    var = p * (1 - p) / n_t + p * (1 - p) / n_c

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    mde_abs = (z_a + z_b) * np.sqrt(var)

    # Efficiency of an r/(1-r) split relative to 50/50 is 4*r*(1-r).
    eff = 4 * treatment_share * (1 - treatment_share)

    return PowerResult(
        mde_absolute=float(mde_abs),
        mde_relative=float(mde_abs / p),
        n_total=n_total,
        treatment_share=treatment_share,
        baseline_rate=p,
        power=power,
        alpha=alpha,
        allocation_efficiency=float(eff),
        n_equivalent_balanced=int(n_total * eff),
    )


def required_n(
    baseline_rate: float,
    mde_relative: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    p = baseline_rate
    delta = p * mde_relative
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    per_unit_var = p * (1 - p) * (1 / treatment_share + 1 / (1 - treatment_share))
    return int(np.ceil(((z_a + z_b) ** 2 * per_unit_var) / delta**2))
```

**The finding, computed and verified on the real design parameters** (N = 13,979,592, p = 0.046992, r = 0.85, 80% power, α = 0.05 two-sided):

| Quantity | 85/15 (actual design) | 50/50 |
|---|---|---|
| Allocation efficiency `4r(1−r)` | **0.51** | 1.00 |
| Variance of the difference | **1.96×** higher | baseline |
| Relative MDE | **0.945%** | 0.675% |
| Sample needed for the same MDE | **1.96×** | baseline |

So: *Criteo's 85/15 split is 51% as statistically efficient as a balanced split. Half the sample is effectively wasted from a pure-precision standpoint.* And yet the design is correct — a business does not want to withhold ads from half its users just to tighten a confidence interval. The 85/15 allocation buys revenue with precision.

Being able to (a) quantify that trade-off and (b) explain why the "inefficient" design is nevertheless the right business call is exactly the judgment product-DS interviews are probing for. Put the table in your README.

### 5.6 Sequential testing — `src/uplift/experiment/sequential.py` (optional, high value)

Criteo has no time dimension, so you cannot genuinely monitor an experiment over time. You have two honest options; pick one and label it:

**Option A (recommended, ~3 hours).** Simulate the peeking problem rather than pretending to monitor real data. Show that repeatedly testing a *null* experiment at α = 0.05 inflates the false-positive rate far above 5%, then show that an always-valid procedure controls it.

```python
def peeking_simulation(n_peeks: int = 10, n_per_peek: int = 10_000,
                       p: float = 0.047, n_sims: int = 2000, seed: int = 0) -> dict:
    """A/A test peeked at repeatedly. True effect is zero by construction."""
    rng = np.random.default_rng(seed)
    naive_fp = 0
    for _ in range(n_sims):
        a = rng.binomial(1, p, n_peeks * n_per_peek)
        b = rng.binomial(1, p, n_peeks * n_per_peek)
        for k in range(1, n_peeks + 1):
            m = k * n_per_peek
            za = a[:m].mean() - b[:m].mean()
            se = np.sqrt(a[:m].var(ddof=1) / m + b[:m].var(ddof=1) / m)
            if abs(za / se) > 1.96:
                naive_fp += 1
                break
    return {"peeks": n_peeks, "nominal_alpha": 0.05,
            "actual_false_positive_rate": naive_fp / n_sims}
```

With 10 peeks expect roughly 15–25% rather than 5%. Then add an O'Brien-Fleming alpha-spending boundary or an mSPRT always-valid p-value and show the rate return to ≈ 5%.

**Option B.** Assign each unit a synthetic "day" and monitor. Fine *if* the README says the day is synthetic, in a sentence that a reader cannot miss. Do not quietly invent a time column; a reviewer who notices will assume the rest of the repo is equally loose.

### 5.7 Wire it up — `make experiment`

`uplift experiment-report` should read from DuckDB, run everything above, print a readout, and write `evals/experiment.json`. Target output:

```
EXPERIMENT READOUT — Criteo Uplift v2.1, primary outcome = visit
────────────────────────────────────────────────────────────────
Assignment    n_treatment=11,882,655  n_control=2,096,937  ratio=0.850000
SRM           <verdict + observed ratio with CI + p-value + interpretation>
Balance       max |SMD| = 0.00xx across f0-f11  (all < 0.02)
Compliance    P(exposed | treated) = 0.xxxx        <- IV first stage

ATE (visit)
  difference-in-means      ATE=+0.00xxxx [ ... ]  rel=+xx.xx%  SE=x.xxe-04
  Lin regression-adjusted  ATE=+0.00xxxx [ ... ]  rel=+xx.xx%  SE=x.xxe-04
  CUPAC-adjusted           ATE=+0.00xxxx [ ... ]  rel=+xx.xx%  SE=x.xxe-04

Variance reduction  CUPAC theta=x.xxx  rho=0.xxx  var_red=x.x%  ~= 1.0xx x sample
Design              MDE(80%) = 0.945% relative | allocation efficiency 0.51
                    -> the 85/15 split is 51% as efficient as 50/50
```

### 5.8 Tests for Phase 2

`tests/test_cuped.py`:

```python
import numpy as np
from uplift.experiment.cuped import cuped_adjust


def test_variance_reduction_equals_rho_squared():
    rng = np.random.default_rng(0)
    n, target_rho = 200_000, 0.6
    x = rng.normal(size=n)
    y = target_rho * x + rng.normal(size=n) * np.sqrt(1 - target_rho**2)

    y_adj, theta = cuped_adjust(y, x)
    rho = np.corrcoef(y, x)[0, 1]

    observed = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    assert abs(observed - rho**2) < 1e-3
    assert abs(theta - target_rho) < 0.02


def test_cuped_preserves_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    y = 0.4 * x + rng.normal(size=50_000)
    y_adj, _ = cuped_adjust(y, x)
    assert abs(y_adj.mean() - y.mean()) < 1e-9
```

`tests/test_srm.py`:

```python
from uplift.experiment.srm import check_srm, simulate_srm_calibration


def test_exact_split_is_not_srm():
    r = check_srm(850_000, 150_000, expected_ratio=0.85)
    assert not r.is_srm and r.p_value > 0.9


def test_clear_mismatch_is_detected():
    r = check_srm(800_000, 200_000, expected_ratio=0.85)
    assert r.is_srm and r.p_value < 1e-10


def test_false_positive_rate_matches_alpha():
    out = simulate_srm_calibration(n=200_000, n_sims=2000, seed=3)
    assert out["rejection_rate_at_0.05"] < 0.08     # ~0.05 plus MC slack
    assert out["rejection_rate_at_0.001"] < 0.01
```

### 5.9 Phase 2 done when

- [ ] `make experiment` produces the full readout
- [ ] You can explain why the SRM test rejects and why that is not a data-quality problem
- [ ] Your CUPAC variance reduction is reported with its ρ² explanation
- [ ] The allocation-efficiency table is in the README
- [ ] All Phase 2 tests pass

---

## 6. Phase 3 (weeks 5–6) — uplift models and Qini/AUUC

**Goal by end of week 6:** five uplift models trained, ranked by Qini **with bootstrap confidence intervals**, plus a CATE calibration plot.

### 6.1 The framing you must get right

Uplift modeling estimates τ(x) = E[Y(1) − Y(0) | X = x], the *incremental* effect. You never observe τ for any individual — you see one potential outcome per unit. So:

- **You cannot compute per-unit error.** No RMSE against ground truth. Anyone reporting one is confused.
- **Evaluation is at the group level.** Rank units by predicted τ̂, then check whether the top-ranked groups actually show a larger treated-vs-control gap. That is what Qini and AUUC do.
- **AUC is the wrong metric.** AUC measures how well you predict *who responds*, not *who responds because of the treatment*. A model that perfectly ranks response probability can have zero uplift value. Report AUC if you like, but label it "response-model AUC (not the objective)" so the reader knows you know.

### 6.2 Feature prep — `src/uplift/models/features.py`

Almost nothing to do, which is itself worth stating in the README.

```python
FEATURES = [f"f{i}" for i in range(12)]

def load_split(con, split: str, outcome: str = "visited"):
    """Load one split. NOTE: is_exposed is deliberately excluded - it is a
    post-treatment variable and using it as a feature would break identification."""
    df = con.execute(
        f"""
        SELECT {', '.join(FEATURES)}, is_treated, {outcome} AS y
        FROM units WHERE split = ?
        """,
        [split],
    ).df()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    w = df["is_treated"].to_numpy(dtype=np.int8)
    y = df["y"].to_numpy(dtype=np.int8)
    return X, w, y
```

- No scaling: LightGBM is scale-invariant.
- No encoding: everything is already numeric.
- No imputation: check for nulls, but there should be none.
- **No `is_exposed`.** Put the reason in a comment, as above. This is the sort of comment interviewers notice.

### 6.3 The models — `src/uplift/models/learners.py`

Train five, in increasing order of sophistication. The comparison is the story.

```python
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor


def _base_clf(seed: int = 0, **kw) -> LGBMClassifier:
    params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=-1,
    )
    params.update(kw)
    return LGBMClassifier(**params)


# ---------- 1. S-learner ----------
class SLearner:
    """One model on [X, W]; tau_hat = f(X,1) - f(X,0).

    Known failure mode: when the treatment signal is weak relative to the
    covariates, the trees may barely split on W and tau_hat collapses toward 0.
    We check for this explicitly rather than hoping.
    """

    def __init__(self, seed: int = 0):
        self.model = _base_clf(seed)

    def fit(self, X, w, y):
        self.model.fit(np.column_stack([X, w]), y)
        # diagnose the pathology: how much gain came from the treatment column?
        gains = self.model.booster_.feature_importance(importance_type="gain")
        self.treatment_gain_share_ = float(gains[-1] / gains.sum())
        return self

    def predict(self, X):
        n = len(X)
        p1 = self.model.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
        return p1 - p0


# ---------- 2. T-learner ----------
class TLearner:
    """Separate models per arm. Simple and strong, but the control model sees
    only 15% of the data here, so it is the noisier of the two."""

    def __init__(self, seed: int = 0):
        self.m1, self.m0 = _base_clf(seed), _base_clf(seed + 1)

    def fit(self, X, w, y):
        self.m1.fit(X[w == 1], y[w == 1])
        self.m0.fit(X[w == 0], y[w == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


# ---------- 3. Class transformation (Athey-Imbens transformed outcome) ----------
class ClassTransformation:
    """Z = Y*W/e - Y*(1-W)/(1-e); E[Z|X] = tau(X). One model, and it targets
    uplift directly. Cheap, and a surprisingly hard baseline to beat."""

    def __init__(self, seed: int = 0):
        self.model = LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=500, verbose=-1, random_state=seed, n_jobs=-1,
        )

    def fit(self, X, w, y, propensity: float | np.ndarray | None = None):
        e = w.mean() if propensity is None else propensity
        z = y * w / e - y * (1 - w) / (1 - e)
        self.model.fit(X, z)
        return self

    def predict(self, X):
        return self.model.predict(X)
```

X-learner, DR-learner, and causal forest come from EconML:

```python
from econml.dr import DRLearner
from econml.dml import CausalForestDML
from econml.metalearners import XLearner
from sklearn.linear_model import LogisticRegression


def fit_x_learner(X, w, y, seed=0):
    est = XLearner(
        models=_base_clf(seed),
        propensity_model=LogisticRegression(max_iter=1000),
        cate_models=LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, verbose=-1, random_state=seed),
    )
    est.fit(y, w, X=X)
    return est


def fit_dr_learner(X, w, y, seed=0):
    """Doubly robust + cross-fitted. Consistent if EITHER the outcome model or
    the propensity model is right - the strongest guarantee of the five."""
    est = DRLearner(
        model_propensity=LogisticRegression(max_iter=1000),
        model_regression=LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                       num_leaves=31, verbose=-1, random_state=seed),
        model_final=LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, verbose=-1, random_state=seed),
        cv=3,
        random_state=seed,
    )
    est.fit(y, w, X=X)
    return est


def fit_causal_forest(X, w, y, seed=0, max_n=2_000_000):
    """Causal forests give honest CONFIDENCE INTERVALS on tau(x), which none of
    the meta-learners above do. Subsample: 14M rows will not finish."""
    if len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, w, y = X[idx], w[idx], y[idx]
    est = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed),
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=200,
        max_samples=0.4,
        cv=3,
        random_state=seed,
    )
    est.fit(y, w, X=X)
    return est
```

EconML's fit signature is `est.fit(Y, T, X=X, W=W)` — outcome first, treatment second. Getting this backwards silently produces garbage, so assert your shapes.

Only the causal forest gives you `est.effect_interval(X, alpha=0.05)`. Use it: showing a **CATE with uncertainty** is a differentiator, and it lets you say honestly "for most units the CI on τ(x) covers zero, so the heterogeneity I can detect is limited."

**Runtime budget.** T-learner on 8.4M training rows: minutes. DR-learner with `cv=3`: tens of minutes. Causal forest on 2M: tens of minutes to hours depending on cores. Develop on a 1M-row subsample (`USING SAMPLE 1000000 ROWS`) and only run the full fit when the code is settled.

### 6.4 Evaluation — `src/uplift/evaluation/qini.py`

**Use `scikit-uplift` for the numbers you report.** Write your own implementation too, so you understand it and can test it — but report theirs.

```python
from sklift.metrics import (
    qini_auc_score, uplift_auc_score, uplift_at_k, weighted_average_uplift,
)
from sklift.viz import plot_qini_curve, plot_uplift_by_percentile

qini  = qini_auc_score(y_true=y_test, uplift=tau_hat, treatment=w_test)
auuc  = uplift_auc_score(y_true=y_test, uplift=tau_hat, treatment=w_test)
top30 = uplift_at_k(y_true=y_test, uplift=tau_hat, treatment=w_test,
                    strategy="overall", k=0.3)

plot_qini_curve(y_test, tau_hat, w_test, perfect=True, name="DR-learner")
plot_uplift_by_percentile(y_test, tau_hat, w_test, strategy="overall", bins=10)
```

Your own reference implementation, verified:

```python
import numpy as np


def qini_curve(y: np.ndarray, w: np.ndarray, score: np.ndarray):
    """Cumulative incremental outcome as we treat units in descending score order.

    At depth k:  Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k)
    The N_t/N_c factor rescales the control response to the treated group size,
    which is what makes the curve interpretable as 'incremental outcomes gained'.
    """
    order = np.argsort(-score, kind="mergesort")     # stable: ties are reproducible
    y, w = y[order], w[order]
    n_t, n_c = np.cumsum(w), np.cumsum(1 - w)
    y_t, y_c = np.cumsum(y * w), np.cumsum(y * (1 - w))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / np.maximum(n_c, 1), 0.0)
    return np.arange(1, len(y) + 1), np.nan_to_num(q)


def qini_coefficient(y, w, score) -> float:
    x, q = qini_curve(y, w, score)
    q_random = q[-1] * x / len(y)
    oracle = np.where((w == 1) & (y == 1), 2.0, np.where((w == 0) & (y == 0), 1.0, 0.0))
    _, q_perfect = qini_curve(y, w, oracle)
    return float(
        (np.trapezoid(q, x) - np.trapezoid(q_random, x))
        / (np.trapezoid(q_perfect, x) - np.trapezoid(q_random, x))
    )
```

Sanity-checked on 200k synthetic units where true uplift was 0.05 for `x1 > 0` and 0 otherwise:

| Ranking used | Qini coefficient |
|---|---|
| Oracle (true τ) | **+0.138** |
| Random scores | **+0.004** ≈ 0 |
| Anti-oracle (−τ) | **−0.137** |

Random lands at zero and the anti-oracle mirrors the oracle, which is what a correct implementation must do. Note that even the *oracle* only scores 0.138 — because the "perfect" normalizing curve is built from realized outcomes, which no model can achieve. **Qini values are not comparable across implementations** with different normalizations. Say which one you used.

### 6.5 Bootstrap confidence intervals on Qini — do not skip this

This is what stops you from claiming a model is better when it is not.

```python
def bootstrap_qini_ci(y, w, score, n_boot=200, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = qini_auc_score(y_true=y[idx], uplift=score[idx], treatment=w[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi)


def paired_bootstrap_difference(y, w, score_a, score_b, n_boot=200, seed=0):
    """Resample ONCE per iteration and score both models on the same resample.
    Paired resampling removes shared sampling noise, which is the only way to
    tell two similar models apart at this signal level."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = (
            qini_auc_score(y[idx], score_a[idx], w[idx])
            - qini_auc_score(y[idx], score_b[idx], w[idx])
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(diffs.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)
```

Your results table should look like this — and if the CIs overlap, **say so** rather than declaring a winner:

| Model | Qini | 95% CI | vs. T-learner | Response AUC |
|---|---|---|---|---|
| Class transformation | 0.0xx | [.., ..] | — | 0.6xx |
| S-learner | 0.0xx | [.., ..] | overlapping | 0.6xx |
| T-learner | 0.0xx | [.., ..] | (reference) | 0.6xx |
| X-learner | 0.0xx | [.., ..] | overlapping | 0.6xx |
| DR-learner | 0.0xx | [.., ..] | +0.00x, CI excludes 0 | 0.6xx |
| Causal forest | 0.0xx | [.., ..] | overlapping | 0.6xx |

> "Four of the six models are statistically indistinguishable on Qini at n = 2.8M held-out units. Reporting a leaderboard without confidence intervals would have manufactured a winner out of noise."

That sentence is worth more than a higher number.

### 6.6 CATE calibration — `src/uplift/evaluation/calibration.py`

Very few portfolio projects do this, and it directly answers "how do you know your uplift model is any good?"

```python
import numpy as np
import pandas as pd


def cate_calibration(y, w, tau_hat, n_bins=10):
    """Bin by PREDICTED uplift, then measure OBSERVED uplift in each bin.
    A calibrated model produces a monotone increasing observed_uplift column
    that tracks predicted_uplift along the diagonal."""
    bins = pd.qcut(tau_hat, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in range(bins.max() + 1):
        m = bins == b
        yt, yc = y[m & (w == 1)], y[m & (w == 0)]
        if len(yt) < 30 or len(yc) < 30:
            continue
        obs = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append({
            "decile": b + 1,
            "n_treatment": len(yt), "n_control": len(yc),
            "predicted_uplift": float(tau_hat[m].mean()),
            "observed_uplift": float(obs),
            "ci_low": float(obs - 1.96 * se), "ci_high": float(obs + 1.96 * se),
        })
    return pd.DataFrame(rows)
```

Two things to check and report:
1. **Monotonicity** — does observed uplift increase across deciles? Compute the rank correlation between predicted and observed.
2. **Calibration** — regress observed on predicted; a slope near 1 means calibrated, a slope near 0 means the model ranks nothing.

Expect deciles at the extremes to have wide CIs. Show them.

### 6.7 The results artifact

Write `evals/results.json` on every eval run. This is what the CI regression gate reads, what the dashboard displays, and what the README quotes.

```json
{
  "run_id": "2026-09-04T14:22:11Z",
  "git_sha": "a1b2c3d",
  "dataset": {"name": "criteo-uplift-v2.1", "rows": 13979592,
              "sha256": "2716e1bf...", "split": "test", "n_test": 2795918},
  "outcome": "visit",
  "ate": {"estimator": "difference-in-means", "value": 0.0, "ci": [0.0, 0.0]},
  "models": {
    "dr_learner": {"qini": 0.0, "qini_ci": [0.0, 0.0], "auuc": 0.0,
                   "uplift_at_30pct": 0.0, "response_auc": 0.0,
                   "calibration_slope": 0.0, "train_seconds": 0}
  },
  "policy": {"threshold": 0.0, "treated_fraction": 0.0,
             "estimated_incremental_visits": 0,
             "policy_value_dr": 0.0, "policy_value_ci": [0.0, 0.0]}
}
```

`git_sha` and the dataset checksum make every number traceable to the exact code and exact data that produced it. That is the reproducibility story in two fields.

### 6.8 Tests for Phase 3

`tests/test_qini.py`:

```python
import numpy as np
from uplift.evaluation.qini import qini_coefficient


def _synthetic(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    w = rng.binomial(1, 0.85, n)
    tau = 0.05 * (x > 0)
    y = rng.binomial(1, 0.05 + w * tau)
    return x, w, y, tau


def test_random_scores_score_near_zero():
    x, w, y, _ = _synthetic()
    rng = np.random.default_rng(99)
    assert abs(qini_coefficient(y, w, rng.normal(size=len(y)))) < 0.02


def test_oracle_beats_random():
    x, w, y, tau = _synthetic()
    rng = np.random.default_rng(99)
    assert qini_coefficient(y, w, tau) > qini_coefficient(y, w, rng.normal(size=len(y))) + 0.05


def test_reversed_ranking_flips_the_sign():
    x, w, y, tau = _synthetic()
    assert np.isclose(qini_coefficient(y, w, tau), -qini_coefficient(y, w, -tau), atol=0.01)
```

These three properties — random ≈ 0, oracle > random, reversal flips sign — catch essentially every implementation bug in a Qini function.

### 6.9 Phase 3 done when

- [ ] Six models trained and evaluated on the held-out test split
- [ ] Qini reported **with** bootstrap CIs, and pairwise comparisons stated honestly
- [ ] CATE calibration table and plot produced
- [ ] Response AUC reported and explicitly labelled as *not* the objective
- [ ] `evals/results.json` written with git SHA and dataset checksum
- [ ] You checked the S-learner's treatment-feature gain share and can say whether it degenerated

---

## 7. Phase 4 (weeks 7–8) — quasi-experimental, IV, sensitivity

**This is the most differentiating phase in the project.** If you only have time for one thing beyond a working pipeline, do §7.1–7.3.

### 7.1 The core idea: use the RCT as an answer key

Every observational causal-inference project has the same weakness — you estimate an effect and have no way to know whether you got it right. You are in the rare position of holding a randomized experiment. So:

1. Estimate the true effect from the RCT. This is ground truth.
2. **Deliberately destroy the randomization** by dropping units in a way that depends on their covariates, creating a confounded observational dataset from the same underlying data.
3. Run the observational toolkit — naive comparison, IPW, matching, doubly-robust, DML — on the corrupted slice.
4. Compare each estimate to the ground truth. Produce a bias table.

You now have something almost no portfolio has: **quantified evidence about when causal methods work and when they fail**, on real data, with a known answer.

### 7.2 Injecting confounding — `src/uplift/causal/confounding.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfoundedSample:
    idx: np.ndarray            # indices kept from the RCT
    selection_score: np.ndarray  # e(X) for every ORIGINAL unit
    keep_prob: np.ndarray        # P(keep | X, W) for every original unit
    ground_truth_ate: float
    ground_truth_se: float
    strength: float


def inject_confounding(
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray | None = None,
    strength: float = 1.0,
    seed: int = 0,
) -> ConfoundedSample:
    """Create a confounded observational slice from randomized data.

    Mechanism: build a selection score e(X) from covariates, then keep a
    TREATED unit with probability e(X) and a CONTROL unit with probability
    1 - e(X). Treatment is now correlated with X in the surviving sample, so
    a naive comparison is confounded - exactly like real observational data.

    Crucially, potential outcomes are untouched. Only WHO WE OBSERVE changes.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if coefs is None:
        coefs = np.zeros(X.shape[1])
        coefs[:4] = [1.2, -0.8, 0.6, 0.9]      # a few covariates drive selection
    e = 1.0 / (1.0 + np.exp(-strength * (Xs @ coefs)))

    keep_prob = np.where(w == 1, e, 1.0 - e)
    keep = rng.random(n) < keep_prob
    idx = np.flatnonzero(keep)

    # ---- ground truth for THIS slice, computed from the full RCT ----
    # The slice's covariate distribution is the s(X)-reweighted population,
    # where s(X) = P(keep|X) = P(W=1)*e + P(W=0)*(1-e).
    p_w = w.mean()
    s = p_w * e + (1 - p_w) * (1 - e)
    truth, truth_se = _weighted_diff_in_means(y, w, s)

    return ConfoundedSample(
        idx=idx, selection_score=e, keep_prob=keep_prob,
        ground_truth_ate=truth, ground_truth_se=truth_se, strength=strength,
    )


def _weighted_diff_in_means(y, w, weights):
    """Weighted difference in means on the RCT + its standard error.
    Valid as the slice's target because W is independent of X in the RCT."""
    def wmean_var(vals, wt):
        wt = wt / wt.sum()
        m = np.sum(wt * vals)
        # variance of a weighted mean with normalized weights
        v = np.sum(wt**2 * (vals - m) ** 2)
        return m, v

    m1, v1 = wmean_var(y[w == 1], weights[w == 1])
    m0, v0 = wmean_var(y[w == 0], weights[w == 0])
    return float(m1 - m0), float(np.sqrt(v1 + v0))
```

**The subtlety that makes this correct** — and it is worth understanding rather than copying, because an interviewer may probe it:

When treatment shares are 50/50, `s(X) = 0.5·e + 0.5·(1−e) = 0.5` is constant, so the slice has the same covariate distribution as the population and the ground truth is just the population ATE. **Criteo is 85/15, so `s(X)` is not constant** — the slice is a reweighted population and its target estimand is the `s(X)`-weighted ATE. Using the plain population ATE as your "truth" would build a bias into your bias table.

I verified the weighted-ground-truth formula on 3M simulated units under an 85/15 design with a non-constant `s(X)` ranging from 0.151 to 0.850. Across 12 seeds the estimator's error had a t-statistic of −0.95 against zero — no detectable bias. The seed-to-seed standard deviation was 6.4e-4 on an effect of 1.1e-2, i.e. **about 6% noise on the "ground truth" itself**, which is why the next section puts a confidence interval on the truth.

### 7.3 The estimators and the bias table — `src/uplift/causal/estimators.py`

```python
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def fit_propensity(X, w, seed=0, cross_fit=True, n_folds=5):
    """Cross-fitted propensity scores. Cross-fitting avoids the overfitting bias
    that makes in-sample propensity scores too extreme."""
    if not cross_fit:
        return LGBMClassifier(n_estimators=300, verbose=-1,
                              random_state=seed).fit(X, w).predict_proba(X)[:, 1]
    ps = np.zeros(len(w))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, w):
        m = LGBMClassifier(n_estimators=300, num_leaves=31, min_child_samples=200,
                           verbose=-1, random_state=seed).fit(X[tr], w[tr])
        ps[te] = m.predict_proba(X[te])[:, 1]
    return ps


def check_overlap(ps, w, lo=0.01, hi=0.99):
    """Positivity/overlap diagnostic. Report this BEFORE any estimate:
    if there is no overlap, no amount of adjustment saves you."""
    return {
        "ps_min": float(ps.min()), "ps_max": float(ps.max()),
        "frac_below": float((ps < lo).mean()), "frac_above": float((ps > hi).mean()),
        "treated_ps_p05": float(np.quantile(ps[w == 1], 0.05)),
        "control_ps_p95": float(np.quantile(ps[w == 0], 0.95)),
    }


def naive_ate(y, w):
    return float(y[w == 1].mean() - y[w == 0].mean())


def ipw_ate(y, w, ps, clip=(0.02, 0.98), stabilized=True):
    """Inverse propensity weighting. Clipping trades a little bias for a large
    variance reduction when some weights explode - always report the clip."""
    ps = np.clip(ps, *clip)
    if stabilized:
        p = w.mean()
        w1 = w * p / ps
        w0 = (1 - w) * (1 - p) / (1 - ps)
        return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))
    return float(np.mean(w * y / ps) - np.mean((1 - w) * y / (1 - ps)))


def aipw_ate(y, w, X, ps, seed=0, clip=(0.02, 0.98), n_folds=5):
    """Augmented IPW / doubly robust. Consistent if EITHER the outcome model
    or the propensity model is correctly specified. Cross-fitted."""
    ps = np.clip(ps, *clip)
    m1 = np.zeros(len(y)); m0 = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, w):
        tr1, tr0 = tr[w[tr] == 1], tr[w[tr] == 0]
        m1[te] = LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed) \
            .fit(X[tr1], y[tr1]).predict_proba(X[te])[:, 1]
        m0[te] = LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed) \
            .fit(X[tr0], y[tr0]).predict_proba(X[te])[:, 1]

    scores = m1 - m0 + w * (y - m1) / ps - (1 - w) * (y - m0) / (1 - ps)
    ate = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(len(scores)))   # influence-function SE
    return ate, se
```

**Empirical validation of the whole design.** I ran this end to end on 400k simulated units (true ATE = 0.040, three covariates, selection driven by two of them):

| Estimator | Estimate | Bias vs. truth | Relative error |
|---|---|---|---|
| Ground truth (from the RCT) | 0.0401 | — | — |
| **Naive difference-in-means** | **0.0726** | **+0.0325** | **+81%** |
| IPW (clipped, stabilized) | 0.0378 | −0.0023 | −5.8% |
| AIPW / doubly robust | 0.0378 | −0.0024 | −5.9% |

The naive comparison is off by 81%; adjustment recovers the answer to within 6%. **That table is the single best artifact in this project.** Reproduce it on Criteo and put it in the README.

Then push further, because this is where it gets genuinely interesting — vary the confounding strength and show where the methods break:

```python
for strength in [0.0, 0.5, 1.0, 2.0, 4.0]:
    cs = inject_confounding(X, w, y, strength=strength, seed=0)
    ...  # record naive / IPW / AIPW bias and the overlap diagnostics
```

At `strength = 0` everything agrees (no confounding). As strength rises, naive bias grows and — the important finding — **the adjusted estimators start to fail too**, because extreme selection destroys overlap. Plot bias against strength with the overlap diagnostic on a second axis. You now have a defensible answer to "when should I *not* trust a propensity-score analysis?", grounded in your own experiment.

Also run the estimators with a **deliberately misspecified** propensity model (logistic regression with linear terms only, when selection is nonlinear) to show the doubly-robust estimator surviving where IPW does not. That is the demonstration of what "doubly robust" actually buys you.

### 7.4 ITT vs CACE — `src/uplift/causal/iv.py`

This module exists because of the `exposure` column, and it is where you show you did not fall into the trap.

The structure: `treatment` (random assignment, Z) → `exposure` (was actually shown an ad, D) → `visit` (outcome, Y). Not every assigned user gets shown an ad, so there is **one-sided non-compliance**: `P(D = 1 | Z = 0) = 0` by design.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class IVResult:
    itt: float
    itt_se: float
    first_stage: float           # P(exposed | assigned)
    cace: float
    cace_se: float
    naive_exposed_vs_control: float
    naive_bias_vs_cace: float


def itt_and_cace(y: np.ndarray, z: np.ndarray, d: np.ndarray) -> IVResult:
    """z = randomized assignment (treatment), d = actual exposure, y = outcome.

    ITT  = E[Y|Z=1] - E[Y|Z=0]                 <- always valid, randomization only
    CACE = ITT / (E[D|Z=1] - E[D|Z=0])         <- Wald estimator
    Under one-sided non-compliance E[D|Z=0]=0, so CACE = ITT / P(D=1|Z=1).

    Assumptions for CACE: relevance (nonzero first stage), exclusion (assignment
    affects Y only through exposure), monotonicity (no defiers), and
    independence (holds by randomization).
    """
    y1, y0 = y[z == 1], y[z == 0]
    itt = y1.mean() - y0.mean()
    itt_se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))

    first_stage = d[z == 1].mean() - d[z == 0].mean()
    cace = itt / first_stage
    cace_se = itt_se / abs(first_stage)          # delta method, first stage ~exact at 14M

    # what you get if you naively compare EXPOSED users to controls
    naive = y[d == 1].mean() - y0.mean()

    return IVResult(
        itt=float(itt), itt_se=float(itt_se),
        first_stage=float(first_stage),
        cace=float(cace), cace_se=float(cace_se),
        naive_exposed_vs_control=float(naive),
        naive_bias_vs_cace=float(naive - cace),
    )
```

**The point of the module.** Report all three numbers side by side:

| Estimand | What it answers | Valid? |
|---|---|---|
| ITT | "What happens if we launch the campaign?" | Always — pure randomization |
| CACE | "What is the effect on users who actually see an ad?" | Under exclusion + monotonicity |
| Exposed-vs-control | *nothing* | **Invalid** — conditions on a post-treatment variable |

Then quantify: the naive exposed-vs-control number will be much larger than the CACE, because exposure is not random — users who get shown ads are those who were browsing, and browsing users visit sites more anyway. Report the size of that bias. It is a self-contained interview story: *"the dataset has a column that will silently break your analysis if you use it as a feature or a filter; here is the number showing how badly."*

For heterogeneous CACE, EconML has the estimator:

```python
from econml.iv.dr import LinearIntentToTreatDRIV

driv = LinearIntentToTreatDRIV(
    model_y_xw=LGBMRegressor(n_estimators=200, verbose=-1),
    model_t_xwz=LGBMClassifier(n_estimators=200, verbose=-1),
    flexible_model_effect=LGBMRegressor(n_estimators=200, verbose=-1),
    cv=3,
)
driv.fit(Y=y, T=d, Z=z, X=X)
tau_cace = driv.effect(X_test)
```

Also state the assumption that is *not* guaranteed: **exclusion**. Being assigned to the campaign could in principle affect a user through channels other than seeing the ad (retargeting bid changes, frequency capping elsewhere). Naming an assumption you cannot test, rather than pretending it holds automatically, is exactly the habit these teams hire for.

### 7.5 Difference-in-differences — `src/uplift/causal/did.py`

Criteo has no time dimension, so a real DiD needs a second dataset. Two options:

**Option A (recommended): Olist.** The Brazilian e-commerce dataset (`olistbr/brazilian-ecommerce` on Kaggle, ~100k orders, 2016–2018) has genuine order timestamps, sellers, categories, and review scores. Define a plausible treatment (e.g. sellers who first appear in a category after some date, or a category-level shock), then:

1. **Plot parallel trends first.** Before any regression, show pre-period trends for treated and control groups. If they diverge, DiD is not identified and the honest move is to say so.
2. **Event study.** Regress the outcome on leads and lags relative to treatment. Pre-period coefficients should be flat and near zero — this is the testable implication of parallel trends.
3. **Two-way fixed effects** for the headline number, via `statsmodels` or `linearmodels.PanelOLS`.
4. **Name the staggered-adoption problem.** If units adopt at different times, TWFE can be badly biased under heterogeneous effects (Goodman-Bacon decomposition). Mention Callaway–Sant'Anna as the modern fix, and use `differences` or `pyfixest` if you want to implement it.

**Option B (cheaper): skip DiD entirely** and say why in the README — "the Criteo data has no time dimension, so rather than synthesize one I demonstrated identification through selection-on-observables with a validated ground truth, which this data actually supports." That is a legitimate, defensible scoping decision. A well-argued omission beats a fabricated panel.

If you are short on time, choose B. §7.1–7.3 already carry the causal-inference signal.

### 7.6 Sensitivity analysis — `src/uplift/causal/sensitivity.py`

Any observational estimate is conditional on "no unmeasured confounding," which is untestable. Sensitivity analysis asks: *how strong would a hidden confounder need to be to overturn my conclusion?*

```python
import numpy as np


def e_value(rr: float, rr_ci_bound: float | None = None) -> dict[str, float]:
    """VanderWeele & Ding E-value: the minimum association strength (on the risk
    ratio scale) that an unmeasured confounder would need with BOTH treatment and
    outcome to explain away the observed effect."""
    def _ev(r):
        r = 1 / r if r < 1 else r
        return r + np.sqrt(r * (r - 1))
    out = {"e_value_point": float(_ev(rr))}
    if rr_ci_bound is not None:
        out["e_value_ci"] = 1.0 if (rr > 1) == (rr_ci_bound > 1) and \
            min(rr, rr_ci_bound) <= 1 <= max(rr, rr_ci_bound) else float(_ev(rr_ci_bound))
    return out


def negative_control_outcome(X, w, feature_idx):
    """A pre-treatment covariate CANNOT be affected by treatment. So regressing
    one on treatment must give zero. In the RCT it does. In the confounded slice
    it will not - and the size of that non-zero effect measures your bias."""
    x = X[:, feature_idx]
    m1, m0 = x[w == 1].mean(), x[w == 0].mean()
    se = np.sqrt(x[w == 1].var(ddof=1) / (w == 1).sum()
                 + x[w == 0].var(ddof=1) / (w == 0).sum())
    return {"effect": float(m1 - m0), "se": float(se), "z": float((m1 - m0) / se)}
```

**The negative-control demonstration is the one to lead with**, because on this data it is unusually clean:

- Run it on the RCT → effects on `f0…f11` are all ≈ 0. The negative control passes, as it must.
- Run the *same* check on your confounded slice → several features show large "effects" of treatment. The negative control fails loudly, correctly flagging confounding you injected yourself and therefore know is there.
- Run it after IPW/AIPW reweighting → the imbalance should shrink back toward zero. This is your *diagnostic that the adjustment worked*, independent of knowing the true ATE.

That is a complete, self-validating story about a technique most candidates can only define. Add the E-value on your adjusted estimate, plus optionally a Rosenbaum bounds analysis on a matched version and a Cinelli–Hazlett robustness value (`PySensemakr`).

### 7.7 Tests for Phase 4 — `tests/test_estimators_recover_truth.py`

**This is the highest-signal test file in the repo.** It also lets CI verify your causal code without touching the 14M-row dataset.

```python
import numpy as np
import pytest

from uplift.causal.confounding import inject_confounding
from uplift.causal.estimators import aipw_ate, fit_propensity, ipw_ate, naive_ate
from uplift.data.synthetic import make_rct


@pytest.mark.stat
def test_aipw_recovers_truth_under_injected_confounding():
    X, w, y, true_ate = make_rct(n=300_000, treatment_share=0.85, seed=0)

    cs = inject_confounding(X, w, y, strength=1.0, seed=1)
    Xo, wo, yo = X[cs.idx], w[cs.idx], y[cs.idx]
    truth = cs.ground_truth_ate

    naive = naive_ate(yo, wo)
    ps = fit_propensity(Xo, wo, seed=2)
    ipw = ipw_ate(yo, wo, ps)
    dr, dr_se = aipw_ate(yo, wo, Xo, ps, seed=2)

    # the confounding must actually bite, or the test proves nothing
    assert abs(naive - truth) > 3 * abs(dr - truth)
    # doubly robust must land within ~4 SEs of the truth
    assert abs(dr - truth) < 4 * max(dr_se, 1e-4)


@pytest.mark.stat
def test_no_confounding_means_no_bias():
    X, w, y, _ = make_rct(n=200_000, treatment_share=0.85, seed=3)
    cs = inject_confounding(X, w, y, strength=0.0, seed=4)
    Xo, wo, yo = X[cs.idx], w[cs.idx], y[cs.idx]
    assert abs(naive_ate(yo, wo) - cs.ground_truth_ate) < 0.005


@pytest.mark.stat
def test_cace_exceeds_itt_under_partial_compliance():
    from uplift.causal.iv import itt_and_cace
    rng = np.random.default_rng(5)
    n = 400_000
    z = rng.binomial(1, 0.85, n)
    d = z * rng.binomial(1, 0.5, n)          # 50% of assigned are exposed
    y = rng.binomial(1, 0.05 + 0.04 * d)     # effect flows ONLY through exposure
    r = itt_and_cace(y, z, d)
    assert abs(r.first_stage - 0.5) < 0.01
    assert abs(r.cace - 0.04) < 0.005        # recovers the true complier effect
    assert r.cace > r.itt                    # CACE is always larger under partial compliance
    assert r.naive_exposed_vs_control > r.cace   # and the naive comparison overstates it
```

`make_rct` in `src/uplift/data/synthetic.py` generates covariates, a randomized assignment, a heterogeneous τ(x), and outcomes with a known ATE. **Keep the outcome probability well away from 0 and 1** — if `p0 + τ` gets clipped at the boundary, the realized effect stops matching the nominal τ and your test will fail for a reason that has nothing to do with your estimator. I lost time to exactly this; use bounded functions like `tanh` for the effect and a baseline around 0.05.

### 7.8 Phase 4 done when

- [ ] Bias table: naive vs IPW vs matching vs AIPW vs DML against RCT ground truth
- [ ] Bias-vs-confounding-strength curve with overlap diagnostics
- [ ] Doubly-robust survival under a misspecified propensity model demonstrated
- [ ] ITT / CACE / naive-exposed table with the bias quantified
- [ ] Negative-control outcomes: pass on RCT, fail on confounded slice, recover after adjustment
- [ ] E-value on the adjusted estimate
- [ ] `test_estimators_recover_truth.py` passing in CI

---

## 8. Phase 5 (weeks 9–10) — policy, API, dashboard, ops

**Goal:** a live URL, a scheduled pipeline, and CI that blocks a regression.

### 8.1 From scores to a decision — `src/uplift/policy/targeting.py`

A Qini curve is not a decision. A policy is.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetingPolicy:
    threshold: float
    treated_fraction: float
    expected_incremental_outcomes: float
    expected_profit: float
    value_per_outcome: float
    cost_per_treatment: float


def optimal_threshold(
    tau_hat: np.ndarray,
    value_per_outcome: float,
    cost_per_treatment: float,
    grid: int = 200,
) -> TargetingPolicy:
    """Treat unit i when tau_hat_i * value > cost.

    The break-even threshold is cost/value; we still sweep a grid so the
    dashboard can show the profit curve and so the decision is visibly a
    business decision, not a statistical one.
    """
    candidates = np.quantile(tau_hat, np.linspace(0, 1, grid))
    best = None
    for c in candidates:
        treat = tau_hat >= c
        if treat.sum() == 0:
            continue
        inc = float(tau_hat[treat].sum())
        profit = inc * value_per_outcome - treat.sum() * cost_per_treatment
        if best is None or profit > best.expected_profit:
            best = TargetingPolicy(
                threshold=float(c),
                treated_fraction=float(treat.mean()),
                expected_incremental_outcomes=inc,
                expected_profit=float(profit),
                value_per_outcome=value_per_outcome,
                cost_per_treatment=cost_per_treatment,
            )
    return best
```

**The trap this walks into on purpose:** `expected_profit` above is computed from the model's *own* predictions. If the model is overconfident, so is the profit number. That is why the next section exists, and saying so in the README is the whole point.

### 8.2 Off-policy evaluation — `src/uplift/policy/ope.py`

Evaluate the learned policy on **held-out randomized data**, where you know the propensity exactly because it was assigned by design. This is the honest number.

```python
import numpy as np


def ipw_policy_value(y, w, policy_treat, propensity):
    """Value of a deterministic policy under randomized logging.

    Only units whose ACTUAL assignment matches what the policy would do are
    informative; each is reweighted by the probability of that match.
    """
    p = np.where(policy_treat, propensity, 1 - propensity)
    match = (w == policy_treat.astype(int))
    vals = np.where(match, y / p, 0.0)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def dr_policy_value(y, w, policy_treat, propensity, mu0, mu1):
    """Doubly robust policy value: lower variance than IPW, unbiased if either
    the outcome model or the propensity is right (propensity IS right here)."""
    mu_pi = np.where(policy_treat, mu1, mu0)
    p = np.where(policy_treat, propensity, 1 - propensity)
    match = (w == policy_treat.astype(int))
    scores = mu_pi + match * (y - mu_pi) / p
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores)))
```

Report your policy against three baselines, all with confidence intervals:

| Policy | Value (visit rate) | 95% CI | Treated fraction | Est. incremental visits per 1M |
|---|---|---|---|---|
| Treat nobody | ... | ... | 0% | 0 |
| Treat everybody | ... | ... | 100% | ... |
| **Model-based targeting** | ... | ... | xx% | ... |
| Random targeting at same fraction | ... | ... | xx% | ... |

That last row is the one that matters and the one most projects omit. If your model's policy does not beat *random targeting at the same budget*, the model adds nothing — and finding that out is a legitimate, publishable result. Given how weak the uplift signal is in Criteo, be prepared for the gap to be small, and report it either way.

Then convert to money, with the assumptions visible:

```
Incremental visits per 1M users targeted:  X
Value per conversion (ASSUMPTION):         $25
Cost per treatment (ASSUMPTION):           $0.01
Estimated incremental profit per 1M:       $Y  [CI: $A – $B]
```

Never state a dollar figure without the assumption line directly above it.

### 8.3 The API — `src/uplift/api/main.py`

```python
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uplift.config import settings

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    art = settings.artifacts_dir
    STATE["model"] = joblib.load(art / "uplift_model.joblib")
    STATE["policy"] = json.loads((art / "policy.json").read_text())
    STATE["deciles"] = np.load(art / "decile_edges.npy")
    STATE["meta"] = json.loads((art / "model_card.json").read_text())
    yield
    STATE.clear()


app = FastAPI(
    title="Uplift Targeting API",
    description="Scores users by estimated incremental effect and returns a targeting decision.",
    version="0.1.0",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    features: list[float] = Field(..., min_length=12, max_length=12,
                                  description="f0..f11, in order")


class ScoreResponse(BaseModel):
    predicted_uplift: float
    decile: int = Field(..., ge=1, le=10)
    treat: bool
    threshold: float
    model_version: str
    caveat: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model-info")
def model_info():
    return STATE["meta"]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    x = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
    tau = float(STATE["model"].predict(x)[0])
    thr = STATE["policy"]["threshold"]
    return ScoreResponse(
        predicted_uplift=tau,
        decile=int(np.searchsorted(STATE["deciles"], tau).clip(1, 10)),
        treat=tau >= thr,
        threshold=thr,
        model_version=STATE["meta"]["version"],
        caveat=("Point estimate of a heterogeneous treatment effect. "
                "Individual-level uplift is not identified; use for ranking "
                "and budget allocation, not for individual claims."),
    )


class BatchRequest(BaseModel):
    rows: list[list[float]] = Field(..., max_length=10_000)


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    X = np.asarray(req.rows, dtype=np.float32)
    if X.shape[1] != 12:
        raise HTTPException(422, f"expected 12 features, got {X.shape[1]}")
    tau = STATE["model"].predict(X)
    thr = STATE["policy"]["threshold"]
    return {"n": len(tau), "treat_count": int((tau >= thr).sum()),
            "predicted_uplift": tau.tolist()}
```

The `caveat` field in the response is deliberate. Shipping the epistemic limit of the model *in the payload* is the kind of detail that signals you have thought about how a model gets misused downstream.

Ship a `model_card.json` alongside: training date, dataset checksum, git SHA, Qini with CI, calibration slope, known limitations, and the intended-use statement.

### 8.4 The dashboard — `src/uplift/dashboard/app.py`

Five tabs, in this order — it mirrors how an analyst would actually reason:

1. **Experiment readout** — arm sizes, SRM verdict with the large-sample caveat, balance table, ATE with CI, the allocation-efficiency table.
2. **Uplift models** — Qini curves with CIs, model comparison table, uplift-by-decile, CATE calibration plot.
3. **Targeting policy** — profit curve with `value_per_conversion` and `cost_per_treatment` as **sliders**, threshold selection, off-policy value with CIs vs. all four baselines.
4. **Causal validation** — the bias table, bias-vs-confounding-strength curve, negative-control results, ITT/CACE/naive comparison.
5. **Method notes** — the assumptions, the honest limitations, links to the code.

Sliders on the business assumptions matter: they make the sensitivity of your dollar figure visible instead of hiding it behind one hardcoded number. Cache with `@st.cache_data` reading from `evals/results.json` — the dashboard should never retrain anything.

### 8.5 Orchestration — `orchestration/definitions.py`

```python
from pathlib import Path

from dagster import (
    AssetCheckResult, AssetExecutionContext, AssetKey, Definitions,
    ScheduleDefinition, asset, asset_check, define_asset_job,
)
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

DBT_PROJECT = DbtProject(project_dir=Path(__file__).parent.parent / "dbt")
DBT_PROJECT.prepare_if_dev()


@asset(compute_kind="python", group_name="ingestion")
def criteo_raw(context: AssetExecutionContext) -> None:
    from uplift.data.download import download_criteo
    path = download_criteo()
    context.add_output_metadata({"path": str(path)})


@asset(deps=[criteo_raw], compute_kind="duckdb", group_name="ingestion")
def criteo_units() -> None:
    from uplift.data.ingest import ingest
    ingest()


@dbt_assets(manifest=DBT_PROJECT.manifest_path)
def dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()


@asset_check(asset=AssetKey(["main", "mart_arm_summary"]), blocking=True)
def check_no_srm() -> AssetCheckResult:
    """Block the pipeline if the observed allocation drifts materially from design."""
    from uplift.experiment.srm import check_srm
    from uplift.data.io import read_arm_summary
    s = read_arm_summary()
    r = check_srm(s["n_treatment"], s["n_control"], expected_ratio=0.85)
    # gate on EFFECT SIZE, not the p-value - see section 5.1
    passed = r.abs_deviation < 0.005
    return AssetCheckResult(
        passed=passed,
        metadata={"observed_ratio": r.observed_ratio, "p_value": r.p_value,
                  "abs_deviation": r.abs_deviation},
    )


@asset_check(asset=AssetKey(["main", "mart_covariate_balance"]), blocking=True)
def check_covariate_balance() -> AssetCheckResult:
    from uplift.data.io import read_balance
    b = read_balance()
    worst = b["smd"].abs().max()
    return AssetCheckResult(passed=bool(worst < 0.02),
                            metadata={"max_abs_smd": float(worst)})


@asset(deps=[dbt_models], compute_kind="python", group_name="modeling")
def uplift_models(context: AssetExecutionContext) -> None:
    from uplift.models.train import train_all
    context.add_output_metadata(train_all())


@asset(deps=[uplift_models], compute_kind="python", group_name="modeling")
def uplift_evaluation(context: AssetExecutionContext) -> None:
    from uplift.evaluation.report import evaluate_and_write
    context.add_output_metadata(evaluate_and_write())


@asset(deps=[uplift_evaluation], compute_kind="python", group_name="serving")
def targeting_policy(context: AssetExecutionContext) -> None:
    from uplift.policy.targeting import build_and_save_policy
    context.add_output_metadata(build_and_save_policy())


full_refresh = define_asset_job("full_refresh", selection="*")

defs = Definitions(
    assets=[criteo_raw, criteo_units, dbt_models, uplift_models,
            uplift_evaluation, targeting_policy],
    asset_checks=[check_no_srm, check_covariate_balance],
    jobs=[full_refresh],
    schedules=[ScheduleDefinition(job=full_refresh, cron_schedule="0 3 * * 1")],
    resources={"dbt": DbtCliResource(project_dir=DBT_PROJECT)},
)
```

**Blocking asset checks on SRM and balance are the best design decision in this file.** They encode a real principle: if the randomization looks broken, no downstream model should be trained, because every downstream number would be meaningless. Screenshot the Dagster asset graph for the README.

Note the check gates on **effect size**, not the p-value — consistent with §5.1. Being internally consistent about that across three different files is exactly what a careful reviewer looks for.

### 8.6 Docker

`Dockerfile` (multi-stage, non-root):

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra causal --extra serve --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --extra causal --extra serve

FROM python:3.12-slim AS runtime
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ src/
COPY --chown=app:app artifacts/ artifacts/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "uplift.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:

```yaml
services:
  api:
    build: {context: ., target: runtime}
    ports: ["8000:8000"]
    volumes: ["./artifacts:/app/artifacts:ro"]
    environment: {UPLIFT_ARTIFACTS_DIR: /app/artifacts}

  dashboard:
    build: {context: ., target: runtime}
    command: streamlit run src/uplift/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    volumes: ["./evals:/app/evals:ro", "./artifacts:/app/artifacts:ro"]
    depends_on: [api]

  dagster:
    build: {context: ., target: runtime}
    command: dagster dev -m orchestration.definitions -h 0.0.0.0 -p 3000
    ports: ["3000:3000"]
    volumes: ["./data:/app/data", "./dbt:/app/dbt", "./evals:/app/evals"]
```

Add this to the README verbatim:

> **Why not Kubernetes?** This is a batch analytics pipeline with a single-user serving surface. Docker Compose plus a weekly scheduled run is the correct amount of infrastructure. A `deploy/k8s/` manifest is included so the scaling path is concrete, but it is not what runs — running a cluster for a portfolio demo would be paying for complexity that buys nothing.

### 8.7 CI — `.github/workflows/`

**`ci.yml`** — every push, must finish in under five minutes:

```yaml
name: ci
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: {enable-cache: true}
      - run: uv sync --frozen --extra causal --extra dev
      - run: uv run ruff check src tests orchestration
      - run: uv run ruff format --check src tests orchestration
      - run: uv run mypy src
      - run: uv run pytest -m "not slow" --cov=src/uplift --cov-report=term-missing
```

**`dbt.yml`** — builds the whole dbt project against the committed 50k seed, in memory:

```yaml
name: dbt
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    defaults: {run: {working-directory: dbt}}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dbt
      - run: uv run dbt deps
      - run: uv run dbt seed --target ci
      - run: uv run dbt build --target ci --vars '{"use_seed": true, "smd_tolerance": 0.05}'
```

**`eval.yml`** — the regression gate:

```yaml
name: eval
on:
  schedule: [{cron: "0 4 * * 1"}]
  workflow_dispatch:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra causal --extra dev
      - run: uv run uplift download && uv run uplift ingest
      - run: uv run uplift train --sample 2000000
      - run: uv run uplift evaluate
      - name: Block on metric regression
        run: uv run python scripts/check_regression.py evals/results.json evals/baseline.json
      - uses: actions/upload-artifact@v4
        with: {name: eval-results, path: evals/}
```

`scripts/check_regression.py` compares the new Qini against `evals/baseline.json` and exits non-zero if it falls more than one bootstrap standard error below the floor. This is the "CI gate on a quality metric" story, and it is directly transferable to LLM eval harnesses — which is exactly the parallel to draw when you discuss this project alongside Project 1.

**The dbt-project seed trick makes CI free and fast.** Never download 311 MB in CI. Your statistical correctness tests run on synthetic data with known ground truth (§7.7), your dbt tests run on the committed sample, and the full-data eval runs weekly on a schedule.

### 8.8 Deploying a live URL

You need one clickable link. Cheapest path that works:

- **Hugging Face Space (Streamlit SDK)** for the dashboard. Free, no credit card, and reads `evals/results.json` committed to the Space. Best effort-to-signal ratio.
- **Fly.io or Cloud Run** for the FastAPI service. Both have free-ish tiers and both scale to zero. Put the `/docs` Swagger URL in the README so a reader can call the endpoint from the browser.

Then record a **2–3 minute Loom**: the business question, the bias table, the Qini curves with CIs, the policy profit curve, and one honest limitation. Embed it at the top of the README.

### 8.9 Phase 5 done when

- [ ] Policy value estimated by IPW **and** DR, with CIs, against all four baselines
- [ ] API running with `/score`, `/score/batch`, `/health`, `/model-info`, plus a model card
- [ ] Dashboard with the five tabs and business-assumption sliders
- [ ] Dagster asset graph runs end to end with blocking SRM and balance checks
- [ ] `docker compose up` works from a clean clone
- [ ] Three CI workflows green; the regression gate demonstrably fails when you lower the baseline
- [ ] Live URL in the README
- [ ] Loom recorded

---

## 9. README, blog post, résumé

### 9.1 README structure

Order matters. Recruiters and hiring managers read the first screen and then decide.

```markdown
# Uplift — Causal Experimentation & Uplift-Modeling Platform

> Which users should we target, and what is the *incremental* effect?

**[Live dashboard](...)** · **[API docs](...)** · **[3-min walkthrough](...)** · **[Blog post](...)**

## Headline results
| | |
|---|---|
| Data | Criteo Uplift v2.1 — 13,979,592 randomized units, 85/15 allocation |
| ATE (visit) | +X.XX% relative [95% CI ...] |
| Best uplift model | DR-learner, Qini 0.0XX [95% CI ...] |
| Naive observational bias | **+81%** vs. RCT truth; doubly-robust recovers to within 6% |
| Design efficiency | 85/15 split is **51%** as efficient as 50/50 |
| Variance reduction (CUPAC) | X.X% (= ρ², see notes) |

![architecture](docs/architecture.png)

## What this demonstrates
- Experiment analysis: SRM, covariate balance, power/MDE, CUPED/CUPAC, Lin estimator
- Heterogeneous effects: S/T/X/DR-learners + causal forest, evaluated by Qini/AUUC **with bootstrap CIs**
- Causal identification validated against randomized ground truth (the bias table)
- ITT vs CACE via instrumental variables — and why `exposure` breaks a naive analysis
- Analytics engineering: dbt staging→marts, 14 data tests, a metrics layer, lineage
- Ops: Dagster assets with blocking data-quality checks, Docker Compose, CI regression gate

## Quickstart
[3 commands, from clone to running]

## Results in detail
[bias table, model comparison, calibration, policy value]

## Method notes and honest limitations
[the section below]

## How this would scale in production
[what changes at 100x, and what you would NOT change]
```

### 9.2 The limitations section — write this, do not skip it

This section will be read more carefully than any other by good interviewers. Include at minimum:

- **Uplift signal in this data is weak.** Several models are statistically indistinguishable; CIs are reported so the reader can see this rather than trusting a leaderboard.
- **Features are anonymized and randomly projected**, so there is no feature interpretability and no domain story about *who* responds.
- **No time dimension**, so no genuine sequential monitoring and no classic DiD. The peeking problem is demonstrated by simulation and labelled as such.
- **CACE relies on the exclusion restriction**, which is untestable and could plausibly fail here (assignment may affect users through channels other than ad exposure).
- **The revenue figures rest on two assumed constants** (value per conversion, cost per treatment), both surfaced as dashboard sliders.
- **The confounding in the observational module is injected by me**, so it is a controlled demonstration of estimator behaviour, not evidence about real-world confounding.
- **MetricFlow on DuckDB is not on dbt's officially supported adapter list**, though it works; the fallback is documented.

Every one of these is a sentence you would otherwise have to improvise under pressure in an interview. Writing them down now means you have already thought them through.

### 9.3 The blog post

One post, on the hardest decision. Suggested title: **"Your uplift model is probably indistinguishable from the next one — here's how I checked."** Structure:

1. The problem: uplift evaluation has no ground truth per unit, so people report point estimates and pick winners out of noise.
2. Bootstrap CIs on Qini, and how many of my six models were actually distinguishable.
3. The bigger idea: I had an RCT, so I could break it deliberately and check which observational methods recover the truth.
4. The bias table, and the point where adjustment stops working (overlap).
5. The `exposure` trap: how one column in a public dataset silently invalidates a naive analysis.

That is a post with a real finding in it, not a tutorial. It will get read.

### 9.4 Résumé bullets

Replace the generic version from the strategy doc with something specific to what you actually built:

> **Causal Experimentation & Uplift Platform** — *Python, EconML, dbt, DuckDB, Dagster, FastAPI* · [live demo] · [repo]
> Built an end-to-end causal platform on 14M randomized-trial records: dbt-modeled warehouse with 14 data tests including an automated randomization-balance gate; CUPAC variance reduction and SRM monitoring; S/T/X/DR-learner and causal-forest uplift models evaluated by Qini/AUUC with bootstrap confidence intervals; and a targeting-policy API whose policy value is estimated by doubly-robust off-policy evaluation.
> Validated observational estimators against randomized ground truth by injecting selection bias into the RCT — naive comparison biased +81%, doubly-robust recovered the true effect within 6% — and quantified the design's 51% allocation efficiency versus a balanced split.

The second bullet is the one that will get you asked about in interviews. It contains a specific finding, not a list of tools.

Also reorder your Skills section as the strategy doc recommends: put **Causal Inference & Experimentation** (A/B testing, CUPED, uplift/CATE, propensity/IPW/DR, IV, power analysis) and **MLOps** above classical ML/DL. Recruiters and résumé-parsing models read top-of-section as priority.

---

## Appendix A — command cheat sheet

**One-time setup**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --extra causal --extra dbt --extra serve --extra orchestrate --extra dev
uv run pre-commit install
```

**The daily loop**
```bash
make download          # once; verifies SHA256 against three mirrors
make ingest            # csv.gz -> parquet -> duckdb, with assertions
make dbt-build         # dbt deps + build (models + tests)
make experiment        # SRM, balance, ATE, CUPED, power readout
make train             # fit the six uplift models
make eval              # Qini + CIs + calibration -> evals/results.json
make causal            # bias table, IV/CACE, sensitivity
make test              # fast tests only
make lint              # ruff + mypy
```

**Serving**
```bash
make api               # http://localhost:8000/docs
make dashboard         # http://localhost:8501
make dagster           # http://localhost:3000
make docker-up         # all three
```

**dbt specifics**
```bash
cd dbt
uv run dbt deps
uv run dbt build                       # run + test everything
uv run dbt test --select mart_covariate_balance
uv run dbt build --select +mart_arm_summary   # the model and its upstream
uv run dbt docs generate && uv run dbt docs serve
uv run mf list metrics
uv run mf query --metrics visit_rate --group-by unit__is_treated
```

**Developing on a subsample** (do this until the code is settled)
```bash
uv run uplift train --sample 1000000
```

**Useful DuckDB one-liners**
```bash
uv run python -c "
import duckdb; con = duckdb.connect('data/uplift.duckdb')
print(con.execute('''
  SELECT split, is_treated, count(*) n, avg(visited::DOUBLE) visit_rate,
         avg(converted::DOUBLE) conv_rate, avg(is_exposed::DOUBLE) exposure_rate
  FROM units GROUP BY 1,2 ORDER BY 1,2
''').df())"
```

---

## Appendix B — pitfalls that make a repo look junior

| Pitfall | Why it is wrong | What to do instead |
|---|---|---|
| Reporting AUC for an uplift model | AUC measures *who responds*, not *who responds because of treatment* | Report Qini/AUUC; include AUC only labelled "not the objective" |
| Using `exposure` as a feature or filter | Post-treatment variable; destroys identification | ITT on `treatment`; CACE via IV; quantify the naive bias |
| Qini leaderboard with no CIs | Manufactures winners from noise | Bootstrap CIs plus paired bootstrap for pairwise comparisons |
| Applying CUPED with no pre-period covariate | Textbook CUPED needs a pre-experiment measurement | CUPAC (cross-fitted control-only prediction) or Lin's estimator; explain why |
| Claiming 30% variance reduction | Variance reduction is capped at ρ²; weak features cannot deliver it | Report the number you got, plus the ρ² explanation |
| SRM test at α = 0.05 | Runs constantly; 1-in-20 false alarms | α = 0.001, gate on effect size, report the observed ratio with a CI |
| Announcing SRM on 14M rows without context | The test detects deviations smaller than the published design precision | Explain the large-sample failure mode explicitly |
| Homoskedastic OLS standard errors on a binary outcome | Error variance differs by arm | `cov_type="HC1"` |
| Dividing the absolute CI by the baseline to get a relative CI | The baseline is estimated too | Delta method on `log(m1/m0)` |
| Modeling uplift on `conversion` | 0.3% base rate; hopeless ranking noise | Use `visit` as primary; `conversion` as a caveated secondary |
| Regenerating `unit_id` each run | Silently reshuffles your train/test split | Assign once, persist to Parquet, treat as immutable |
| Committing the 311 MB CSV | Repo becomes unclonable | Commit the download script + checksum; `check-added-large-files` hook |
| Downloading the full data in CI | Slow, flaky, eventually breaks | Synthetic-data tests plus a committed 50k seed |
| One giant notebook as the deliverable | Not reviewable, not testable, not runnable | Packaged `src/`, notebooks for exploration only |
| Kubernetes for a single-user demo | Complexity with no payoff | Docker Compose; keep a k8s manifest to *discuss* and say so |
| Asserting causality without an identification strategy | The core error in observational work | State the assumption, then run sensitivity analysis |
| Stating dollar figures without assumptions | Unfalsifiable | Assumptions in config, exposed as sliders, printed above the figure |
| Claiming interpretability of `f0`–`f11` | They are randomly projected; nobody knows what they mean | Say so explicitly in the README |
| Silently `WHERE`-ing away rows that fail a structural check | Hides a finding | Log it, investigate it, write it up |

---

## Appendix C — verified dataset facts

**Criteo Uplift Prediction Dataset v2.1**

| Property | Value |
|---|---|
| Rows | 13,979,592 |
| Columns | `f0`–`f11`, `treatment`, `exposure`, `visit`, `conversion` |
| Treatment ratio | ≈ 0.85 |
| Visit rate | ≈ 0.046992 |
| Conversion rate | ≈ 0.0029 |
| File | `criteo-research-uplift-v2.1.csv.gz`, ~311 MB compressed |
| SHA256 | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |
| Design | Assembled from randomized incrementality trials; treatment independent of features **by design** |
| Features | 12 numeric, anonymized and randomly projected — **no interpretability** |

**Mirrors** (try in this order):
1. `https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz`
2. `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`
3. `https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz`

**Or via scikit-uplift**, which handles caching for you:
```python
from sklift.datasets import fetch_criteo
ds = fetch_criteo(target_col="visit", treatment_col="treatment",
                  percent10=False, return_X_y_t=True)
# treatment_col accepts 'treatment', 'exposure', or 'all'
# target_col accepts 'visit', 'conversion', or 'all'
# percent10=True gives a 10% sample - use it while developing
```

Note the older **v2.0** file has 25,309,483 rows. v2.1 (13,979,592) is the one this guide uses and the one the Hugging Face card describes. State which you used.

**Library versions verified current as of September 2026:**

| Package | Version | Notes |
|---|---|---|
| EconML | 0.17.0 (Jul 31, 2026) | Now maintained under `py-why/EconML`, not `microsoft/` |
| CausalML | 0.17.0 (Jul 4, 2026) | Python ≥ 3.11; wheels for cp311/cp312 only; Linux needs glibc ≥ 2.28 |
| dbt-core | 1.12.0 (Aug 2026 compatible track) | |
| dbt-duckdb | targets dbt-core ≥ 1.8, duckdb ≥ 1.0; Python 3.10–3.13 | Let pip resolve dbt-core rather than pinning both |
| scikit-uplift | 0.5.1 | Canonical Qini/AUUC + `fetch_criteo` |
| NumPy | 2.5.x current | 2.5.0 dropped Python 3.11 |

Re-verify before committing your lockfile; these move.

**Optional second datasets**
- **Hillstrom MineThatData** — 64,000 customers, three arms (men's / women's / control). A clean second uplift benchmark, useful for showing your metrics generalize beyond one dataset.
- **Olist Brazilian E-Commerce** (`olistbr/brazilian-ecommerce`, Kaggle) — ~100k orders 2016–2018 with real timestamps. The only realistic route to a genuine DiD in this project.

---

## Appendix D — weekly checklist

| Week | Deliverable | Commit message when done |
|---|---|---|
| 1 | Environment, scaffold, download with checksum, Parquet ingest | `feat: reproducible ingestion with checksum verification` |
| 2 | dbt staging→marts, 14 tests, metrics layer, docs lineage | `feat: dbt models with randomization-balance tests` |
| 3 | SRM (+ calibration sim), balance, difference-in-means, Lin estimator | `feat: experiment analysis module` |
| 4 | CUPAC, power/MDE, allocation efficiency, `make experiment` readout | `feat: variance reduction and power analysis` |
| 5 | S/T/X-learners, class transformation, Qini implementation + tests | `feat: uplift meta-learners with Qini evaluation` |
| 6 | DR-learner, causal forest, bootstrap CIs, CATE calibration | `feat: bootstrap CIs and CATE calibration` |
| 7 | Confounding injection, IPW/AIPW/DML, **the bias table** | `feat: validate observational estimators against RCT truth` |
| 8 | ITT/CACE via IV, negative controls, E-value, sensitivity curve | `feat: instrumental variables and sensitivity analysis` |
| 9 | Targeting policy, off-policy evaluation, FastAPI, model card | `feat: targeting policy API with DR off-policy evaluation` |
| 10 | Dashboard, Dagster, Docker, three CI workflows, README, deploy, Loom | `docs: results, limitations, and live demo` |

**If you fall behind, cut in this order:**
1. Sequential testing (§5.6)
2. Difference-in-differences (§7.5) — say why in the README
3. Causal forest (keep S/T/X/DR)
4. Dagster (keep the Makefile + CI)
5. Hillstrom as a second benchmark

**Never cut:** the bias table (§7.3), the ITT/CACE distinction (§7.4), bootstrap CIs on Qini (§6.5), off-policy evaluation (§8.2), or the limitations section (§9.2). Those five are the project.

---

## One closing note on framing

When you present this, resist describing it as "an uplift modeling project." Describe it as: *"I had a randomized experiment, so I used it as an answer key to test which causal methods actually recover the truth when you take the randomization away — and then I built the targeting system on top of what survived."*

The models are the easy part and everyone has them. The validation is the part almost nobody does, and it is the part that makes the rest of it credible.
