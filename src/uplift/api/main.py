"""Targeting API.

Scores users by estimated incremental effect and returns a decision.

Two deliberate choices worth noticing:

* The `caveat` field ships the model's epistemic limit IN THE PAYLOAD. Individual
  uplift is not identified, and a consumer downstream should not have to read the
  README to learn that.
* The threshold comes from `artifacts/policy.json`, which was chosen by
  off-policy evaluation on held-out randomized data - not from the model's own
  predicted profit, which would be circular.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from uplift.api.schemas import (
    N_FEATURES,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from uplift.config import settings

STATE: dict[str, Any] = {}


def load_state() -> dict[str, Any]:
    """Load the model, policy and deciles. Missing artifacts are not fatal:
    /health reports the truth and scoring returns 503, which is more useful than
    a container that will not boot."""
    art = settings.artifacts_dir
    state: dict[str, Any] = {}
    model_path = art / "uplift_model.joblib"
    if not model_path.exists():  # fall back to whichever learner was persisted
        for name in ("s_learner", "class_transformation", "t_learner"):
            if (art / f"{name}.joblib").exists():
                model_path = art / f"{name}.joblib"
                break
    if model_path.exists():
        state["model"] = joblib.load(model_path)
        state["model_name"] = model_path.stem

    policy_path = art / "policy.json"
    if policy_path.exists():
        pol = json.loads(policy_path.read_text())
        state["threshold"] = pol.get("threshold", 0.0)
        state["policy"] = pol

    card_path = art / "model_card.json"
    state["meta"] = (
        json.loads(card_path.read_text())
        if card_path.exists()
        else {
            "version": "unbuilt",
            "note": "model_card.json not found; run `uplift train` and `uplift policy-report`",
        }
    )

    deciles_path = art / "decile_edges.npy"
    if deciles_path.exists():
        state["deciles"] = np.load(deciles_path)
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_state())
    yield
    STATE.clear()


app = FastAPI(
    title="Uplift Targeting API",
    description=__doc__,
    version="0.1.0",
    lifespan=lifespan,
)


def _require_model():
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded; run `uplift train` first")


def _decile(tau: np.ndarray) -> np.ndarray:
    edges = STATE.get("deciles")
    if edges is None:
        return np.ones(len(tau), dtype=int)
    return np.clip(np.searchsorted(edges, tau) + 1, 1, 10)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in STATE)


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model": STATE.get("model_name"),
        "threshold": STATE.get("threshold"),
        "policy": STATE.get("policy"),
        **STATE.get("meta", {}),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    _require_model()
    x = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
    tau = float(np.asarray(STATE["model"].predict(x)).ravel()[0])
    thr = float(STATE.get("threshold", 0.0))
    return ScoreResponse(
        predicted_uplift=tau,
        decile=int(_decile(np.array([tau]))[0]),
        treat=tau >= thr,
        threshold=thr,
        model_version=str(STATE.get("meta", {}).get("version", "unknown")),
    )


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest) -> BatchResponse:
    _require_model()
    X = np.asarray(req.rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise HTTPException(422, f"expected {N_FEATURES} features per row, got shape {X.shape}")
    tau = np.asarray(STATE["model"].predict(X)).ravel()
    thr = float(STATE.get("threshold", 0.0))
    treat = tau >= thr
    return BatchResponse(
        n=len(tau),
        treat_count=int(treat.sum()),
        treated_fraction=float(treat.mean()),
        predicted_uplift=[float(v) for v in tau],
        threshold=thr,
        model_version=str(STATE.get("meta", {}).get("version", "unknown")),
    )
