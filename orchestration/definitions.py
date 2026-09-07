"""Dagster assets, checks and schedules.

The best design decision in this file is that the SRM and covariate-balance
checks are BLOCKING. They encode a real principle: if the randomization looks
broken, no downstream model should be trained, because every downstream number
would be meaningless.

Two details that keep them honest:

* They gate on EFFECT SIZE, not the p-value. At n = 14M a chi-square test
  resolves deviations of 4e-4, far below the precision to which the design ratio
  is even published, so a p-value gate would fire on nothing. This matches
  `Settings.srm_effect_size_gate` and the reasoning in section 5.1 of the guide.
* They fail CLOSED. `read_arm_summary` raises `MartNotFoundError` when the mart
  is missing or empty rather than returning something falsy, so a check that
  cannot read its input crashes instead of quietly reporting "no SRM".
"""

# NOTE: deliberately no `from __future__ import annotations` here. Dagster
# resolves the `context` parameter's annotation by introspection, and postponed
# (string) annotations make it fail with a confusingly self-contradictory
# "Cannot annotate `context` with AssetExecutionContext ... must be annotated
# with AssetExecutionContext".

from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    Definitions,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

from uplift.config import settings

DBT_PROJECT = DbtProject(project_dir=Path(__file__).parent.parent / "dbt")
DBT_PROJECT.prepare_if_dev()


@asset(compute_kind="python", group_name="ingestion")
def criteo_raw(context: AssetExecutionContext) -> None:
    """Download and checksum-verify the source file."""
    from uplift.data.download import download_criteo

    path = download_criteo()
    context.add_output_metadata({"path": str(path), "sha256": settings.criteo_sha256})


@asset(deps=[criteo_raw], compute_kind="duckdb", group_name="ingestion")
def criteo_units(context: AssetExecutionContext) -> None:
    """csv.gz -> Parquet -> DuckDB, then the deterministic 60/20/20 split."""
    from uplift.data.ingest import connect, ingest
    from uplift.data.splits import build_splits

    metrics = ingest()
    con = connect()
    try:
        splits = build_splits(con)
    finally:
        con.close()
    context.add_output_metadata({**metrics, **{f"n_{k}": v["n"] for k, v in splits.items()}})


@dbt_assets(manifest=DBT_PROJECT.manifest_path)
def dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()


@asset_check(asset=AssetKey(["main_marts", "mart_arm_summary"]), blocking=True)
def check_no_srm() -> AssetCheckResult:
    """Block the pipeline if the observed allocation drifts materially from design.

    Gates on `abs_deviation`, not `p_value` - see the module docstring.
    """
    from uplift.data.io import read_arm_summary
    from uplift.experiment.srm import check_srm

    df = read_arm_summary()
    n_t = int(df["n_treatment"].sum())
    n_c = int(df["n_control"].sum())
    r = check_srm(n_t, n_c, expected_ratio=settings.designed_treatment_ratio)
    return AssetCheckResult(
        passed=bool(r.abs_deviation < settings.srm_effect_size_gate),
        metadata={
            "observed_ratio": r.observed_ratio,
            "abs_deviation": r.abs_deviation,
            "gate": settings.srm_effect_size_gate,
            "p_value": r.p_value,
            "note": "gated on effect size; the p-value is reported, not gated on",
        },
    )


@asset_check(asset=AssetKey(["main_marts", "mart_covariate_balance"]), blocking=True)
def check_covariate_balance() -> AssetCheckResult:
    """Block on gross imbalance only.

    The gate is the matching-literature rule of thumb (|SMD| < 0.10), not the
    tighter 0.02 a clean RCT should satisfy. Criteo v2.1 fails 0.02 on 7 of 12
    features and that failure is a documented FINDING, not a reason to halt the
    pipeline - so it is a dbt WARN while this stays an ERROR.
    """
    from uplift.data.io import read_balance

    b = read_balance()
    worst = float(b["smd"].abs().max())
    over_warn = int((b["smd"].abs() > settings.smd_threshold).sum())
    return AssetCheckResult(
        passed=bool(worst < settings.smd_blocking_threshold),
        metadata={
            "max_abs_smd": worst,
            "blocking_threshold": settings.smd_blocking_threshold,
            "features_over_warn_threshold": over_warn,
            "warn_threshold": settings.smd_threshold,
        },
    )


@asset(deps=[dbt_models], compute_kind="python", group_name="modeling")
def uplift_models(context: AssetExecutionContext) -> None:
    from uplift.models.train import train_all

    meta = train_all()
    context.add_output_metadata(
        {k: v["train_seconds"] for k, v in meta["models"].items()}
        | {"n_train_available": meta["n_train_available"]}
    )


@asset(deps=[uplift_models], compute_kind="python", group_name="modeling")
def uplift_evaluation(context: AssetExecutionContext) -> None:
    from uplift.evaluation.report import evaluate_and_write

    res = evaluate_and_write()
    context.add_output_metadata(
        {"ranking": ", ".join(res["ranking"])} | {k: v["qini"] for k, v in res["models"].items()}
    )


@asset_check(asset=uplift_evaluation, blocking=False)
def check_no_metric_regression() -> AssetCheckResult:
    """The same floor CI enforces, surfaced in the asset graph.

    Non-blocking on purpose: a regression should be loud and visible, but it
    should not stop the policy artifact from being rebuilt for inspection.
    """
    import json

    from uplift.evaluation.regression import gate

    results_p = settings.evals_dir / "results.json"
    baseline_p = settings.evals_dir / "baseline.json"
    if not baseline_p.exists():
        return AssetCheckResult(passed=True, metadata={"note": "no baseline yet"})
    ok, lines = gate(json.loads(results_p.read_text()), json.loads(baseline_p.read_text()))
    return AssetCheckResult(passed=ok, metadata={"detail": "\n".join(lines)})


@asset(deps=[uplift_evaluation], compute_kind="python", group_name="serving")
def targeting_policy(context: AssetExecutionContext) -> None:
    from uplift.policy.report import policy_report

    pol = policy_report()
    context.add_output_metadata(
        {
            "threshold": pol["threshold"],
            "treated_fraction": pol["model_predicted_policy"]["treated_fraction"],
        }
    )


full_refresh = define_asset_job("full_refresh", selection="*")

defs = Definitions(
    assets=[
        criteo_raw,
        criteo_units,
        dbt_models,
        uplift_models,
        uplift_evaluation,
        targeting_policy,
    ],
    asset_checks=[check_no_srm, check_covariate_balance, check_no_metric_regression],
    jobs=[full_refresh],
    schedules=[ScheduleDefinition(job=full_refresh, cron_schedule="0 3 * * 1")],
    resources={"dbt": DbtCliResource(project_dir=DBT_PROJECT)},
)
