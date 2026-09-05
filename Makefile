.PHONY: help install download ingest dbt-build dbt-test experiment train eval causal api dashboard dagster test test-all lint fmt docker-up clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:        ## sync the environment
	uv sync --extra causal --extra dbt --extra serve --extra orchestrate --extra dev

download:       ## fetch + verify the Criteo file
	uv run uplift download

ingest:         ## csv.gz -> parquet -> duckdb
	uv run uplift ingest

seed:           ## regenerate the committed 50k-row CI seed
	uv run python scripts/make_seed.py

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

policy:         ## targeting policy + off-policy evaluation
	uv run uplift policy-report

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
	uv run ruff check src tests orchestration scripts
	uv run mypy src

fmt:
	uv run ruff format src tests orchestration scripts
	uv run ruff check --fix src tests orchestration scripts

docker-up:
	docker compose up --build

clean:
	rm -rf data/*.parquet data/*.duckdb dbt/target dbt/logs artifacts/*
