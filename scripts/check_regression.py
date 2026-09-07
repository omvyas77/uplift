"""CLI shim for the CI regression gate.

The logic lives in `uplift.evaluation.regression` so it can be imported and
unit-tested; this file only parses arguments. Called by
.github/workflows/eval.yml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift.evaluation.regression import run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="evals/results.json")
    ap.add_argument("baseline", nargs="?", default="evals/baseline.json")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline from results")
    args = ap.parse_args()
    return run(Path(args.results), Path(args.baseline), update=args.update)


if __name__ == "__main__":
    raise SystemExit(main())
