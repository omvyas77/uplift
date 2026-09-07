"""Block a merge when a headline metric regresses.

Compares `evals/results.json` against the committed floor in
`evals/baseline.json`. The CLI shim is `scripts/check_regression.py`; the logic
lives here so it is importable and unit-testable rather than trapped in a
script.

The design decision worth stating: the gate fires on a metric falling below the
baseline by more than one bootstrap standard error, NOT on any decrease at all.
Qini on this data has a bootstrap SE of roughly 0.006, so a strict
"never decrease" rule would fail about half of all reruns on noise alone and the
team would learn to ignore it. A gate that cries wolf is worse than no gate.

Usage:
    python scripts/check_regression.py evals/results.json evals/baseline.json
    python scripts/check_regression.py --update      # rewrite the baseline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOLERANCE_SE = 1.0  # how many bootstrap SEs below baseline is allowed


def _se_from_ci(ci: list[float]) -> float:
    """Bootstrap SE implied by a 95% percentile interval."""
    return (ci[1] - ci[0]) / (2 * 1.96) if ci and len(ci) == 2 else 0.0


def gate(results: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    base_models = baseline.get("models", {})
    cur_models = results.get("models", {})

    missing = sorted(set(base_models) - set(cur_models))
    if missing:
        ok = False
        lines.append(f"FAIL  models present in the baseline but missing from this run: {missing}")

    for name in sorted(base_models):
        if name not in cur_models:
            continue
        floor = base_models[name]["qini"]
        cur = cur_models[name]["qini"]
        se = _se_from_ci(cur_models[name].get("qini_ci", []))
        allowed = floor - TOLERANCE_SE * se
        verdict = "ok  " if cur >= allowed else "FAIL"
        if cur < allowed:
            ok = False
        lines.append(
            f"{verdict}  {name:<22} qini {cur:+.5f}  floor {floor:+.5f}  "
            f"se {se:.5f}  allowed >= {allowed:+.5f}"
        )

    # The dataset must not change underneath the comparison.
    for key in ("rows", "sha256"):
        b, c = baseline.get("dataset", {}).get(key), results.get("dataset", {}).get(key)
        if b is not None and c is not None and b != c:
            ok = False
            lines.append(f"FAIL  dataset {key} changed: baseline {b!r} vs current {c!r}")

    return ok, lines


def build_baseline(results: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the current results as a new regression floor."""
    return {
        "_note": (
            "Regression floor. Update deliberately, with the reason in the commit "
            "message - never to make a red gate green."
        ),
        "updated_from_run_id": results.get("run_id"),
        "git_sha": results.get("git_sha"),
        "dataset": results.get("dataset"),
        "outcome": results.get("outcome"),
        "tolerance_se": TOLERANCE_SE,
        "models": {
            k: {"qini": v["qini"], "qini_ci": v.get("qini_ci")}
            for k, v in results.get("models", {}).items()
        },
    }


def run(results_path: Path, baseline_path: Path, update: bool = False) -> int:
    if not results_path.exists():
        print(f"no results at {results_path}; run `uplift evaluate` first")
        return 2
    results = json.loads(results_path.read_text())

    if update:
        baseline_path.write_text(json.dumps(build_baseline(results), indent=2))
        print(f"baseline written to {baseline_path} from run {results.get('run_id')}")
        return 0

    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}; create one with --update")
        return 2

    ok, lines = gate(results, json.loads(baseline_path.read_text()))
    print(f"REGRESSION GATE  (tolerance: {TOLERANCE_SE} bootstrap SE below the floor)")
    for line in lines:
        print("  " + line)
    print("PASS" if ok else "FAIL - a headline metric regressed below its floor")
    return 0 if ok else 1
