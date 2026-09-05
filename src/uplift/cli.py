"""Typer entrypoint. `uv run uplift --help` lists everything."""

from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="Causal experimentation & uplift-modeling platform.")


@app.command()
def download(force: bool = typer.Option(False, help="Re-download even if the checksum matches.")):
    """Fetch and verify the Criteo file."""
    from uplift.data.download import download_criteo

    typer.echo(str(download_criteo(force=force)))


@app.command()
def ingest():
    """csv.gz -> Parquet -> DuckDB, then build the deterministic splits."""
    from uplift.data.ingest import connect
    from uplift.data.ingest import ingest as _ingest
    from uplift.data.splits import build_splits

    metrics = _ingest()
    con = connect()
    splits = build_splits(con)
    con.close()
    typer.echo(json.dumps({"ingest": metrics, "splits": splits}, indent=2))


@app.command("experiment-report")
def experiment_report(
    outcome: str = "visit",
    sample: int = typer.Option(None, help="Subsample N rows (development)."),
    no_calibration: bool = typer.Option(False, help="Skip the SRM calibration simulation."),
):
    """SRM, balance, ATE, CUPAC variance reduction, power/MDE."""
    from uplift.experiment.report import experiment_report as _report

    _report(outcome=outcome, sample=sample, run_calibration=not no_calibration)


@app.command()
def train(
    sample: int = typer.Option(None, help="Train on a subsample (development)."),
    outcome: str = "visit",
    models: str = typer.Option("all", help="Comma-separated model names, or 'all'."),
):
    """Fit the uplift models."""
    from uplift.models.train import train_all

    typer.echo(
        json.dumps(
            train_all(sample=sample, outcome=outcome, models=models), indent=2, default=float
        )
    )


@app.command()
def evaluate(
    outcome: str = "visit",
    n_boot: int = typer.Option(200, help="Bootstrap replicates for Qini CIs."),
    eval_sample: int = typer.Option(None, help="Subsample the test split."),
):
    """Qini/AUUC with bootstrap CIs, calibration -> evals/results.json."""
    from uplift.evaluation.report import evaluate_and_write

    typer.echo(
        json.dumps(
            evaluate_and_write(outcome=outcome, n_boot=n_boot, eval_sample=eval_sample),
            indent=2,
            default=float,
        )
    )


@app.command("causal-report")
def causal_report(
    sample: int = typer.Option(2_000_000, help="Rows to use for the confounding study."),
    outcome: str = "visit",
):
    """Bias table vs RCT ground truth, ITT/CACE, negative controls, sensitivity."""
    from uplift.causal.report import causal_report as _report

    _report(sample=sample, outcome=outcome)


@app.command("policy-report")
def policy_report(outcome: str = "visit"):
    """Targeting policy + doubly-robust off-policy evaluation."""
    from uplift.policy.report import policy_report as _report

    _report(outcome=outcome)


@app.command()
def info():
    """Print the observed dataset facts from DuckDB."""
    from uplift.data.ingest import connect

    con = connect(read_only=True)
    df = con.execute(
        """
        SELECT split, treatment, count(*) AS n,
               avg(visit::DOUBLE) AS visit_rate,
               avg(conversion::DOUBLE) AS conversion_rate,
               avg(exposure::DOUBLE) AS exposure_rate
        FROM units GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()
    con.close()
    typer.echo(df.to_string(index=False))


if __name__ == "__main__":
    app()
