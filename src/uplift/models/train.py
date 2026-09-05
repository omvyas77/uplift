"""Fit the uplift models and persist their held-out scores.

Design decision worth stating: models are fitted on `train`, and their scores on
`valid` and `test` are written to Parquet. Evaluation then reads scores, never
models. That keeps `make eval` cheap, makes the bootstrap affordable, and means
the test split is touched by exactly one process at exactly one point.

Runtime on the full 8.4M-row train split is hours for the EconML estimators, so
`--sample` exists and every artifact records how many rows it actually saw.
"""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from uplift.config import settings
from uplift.data.io import load_split
from uplift.logging import get_logger
from uplift.models.learners import (
    ClassTransformation,
    EconMLWrapper,
    SLearner,
    TLearner,
    fit_causal_forest,
    fit_dr_learner,
    fit_x_learner,
)

log = get_logger(__name__)

ALL_MODELS = [
    "class_transformation",
    "s_learner",
    "t_learner",
    "x_learner",
    "dr_learner",
    "causal_forest",
]

# EconML's cross-fitting builds dense per-fold copies, so these estimators get a
# hard row cap independent of --sample. The caps are tuned for an 8GB machine
# and are recorded in the artifact rather than hidden.
ECONML_CAPS = {"x_learner": 1_500_000, "dr_learner": 1_500_000, "causal_forest": 400_000}


def _subsample(X, w, y, cap: int, seed: int):
    if len(X) <= cap:
        return X, w, y, len(X)
    idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
    return X[idx], w[idx], y[idx], cap


def train_all(
    sample: int | None = None,
    outcome: str = "visit",
    models: str = "all",
    seed: int | None = None,
) -> dict[str, Any]:
    seed = settings.random_seed if seed is None else seed
    wanted = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]
    settings.ensure_dirs()

    X_tr, w_tr, y_tr = load_split("train", outcome=outcome, sample=sample, seed=seed)
    log.info("train_loaded", rows=len(y_tr), outcome=outcome, treated=int(w_tr.sum()))

    scores: dict[str, dict[str, np.ndarray]] = {}
    meta: dict[str, Any] = {}
    holdouts = {s: load_split(s, outcome=outcome) for s in ("valid", "test")}
    for split_name, (_, _, yh) in holdouts.items():
        log.info("holdout_loaded", split=split_name, rows=len(yh))

    for name in wanted:
        t0 = time.time()
        cap = ECONML_CAPS.get(name, len(X_tr))
        Xs, ws, ys, n_used = _subsample(X_tr, w_tr, y_tr, cap, seed)
        log.info("fitting", model=name, rows=n_used)

        if name == "s_learner":
            est: Any = SLearner(seed).fit(Xs, ws, ys)
        elif name == "t_learner":
            est = TLearner(seed).fit(Xs, ws, ys)
        elif name == "class_transformation":
            est = ClassTransformation(seed).fit(Xs, ws, ys)
        elif name == "x_learner":
            est = EconMLWrapper(fit_x_learner(Xs, ws, ys, seed), name, n_used)
        elif name == "dr_learner":
            est = EconMLWrapper(fit_dr_learner(Xs, ws, ys, seed), name, n_used)
        elif name == "causal_forest":
            est = EconMLWrapper(
                fit_causal_forest(Xs, ws, ys, seed, max_n=cap), name, min(n_used, cap)
            )
        else:
            raise ValueError(f"unknown model: {name}")

        elapsed = time.time() - t0
        scores[name] = {s: est.predict(holdouts[s][0]) for s in holdouts}
        meta[name] = {
            "train_seconds": round(elapsed, 1),
            "n_train_rows": int(n_used),
            "capped": bool(n_used < len(X_tr)),
            "score_mean": float(scores[name]["test"].mean()),
            "score_sd": float(scores[name]["test"].std()),
            **est.diagnostics(),
        }
        log.info("fitted", model=name, seconds=round(elapsed, 1), **est.diagnostics())

        if name in ("s_learner", "t_learner", "class_transformation"):
            joblib.dump(est, settings.artifacts_dir / f"{name}.joblib")

    for split in holdouts:
        df = pd.DataFrame({n: scores[n][split] for n in scores})
        df["y"] = holdouts[split][2]
        df["w"] = holdouts[split][1]
        path = settings.artifacts_dir / f"scores_{split}.parquet"
        df.to_parquet(path, index=False)
        log.info("scores_written", split=split, path=str(path), rows=len(df))

    out = {
        "outcome": outcome,
        "seed": seed,
        "n_train_available": len(y_tr),
        "sample": sample,
        "models": meta,
    }
    (settings.artifacts_dir / "train_meta.json").write_text(
        json.dumps(out, indent=2, default=float)
    )
    return out
