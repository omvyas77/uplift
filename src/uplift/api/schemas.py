"""Request and response models for the targeting API."""

from __future__ import annotations

from pydantic import BaseModel, Field

N_FEATURES = 12

CAVEAT = (
    "Point estimate of a heterogeneous treatment effect. Individual-level uplift "
    "is not identified - no unit's true tau is ever observed. Use for ranking and "
    "budget allocation, not for claims about an individual."
)


class ScoreRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=N_FEATURES, max_length=N_FEATURES, description="f0..f11, in order"
    )


class ScoreResponse(BaseModel):
    predicted_uplift: float
    decile: int = Field(..., ge=1, le=10)
    treat: bool
    threshold: float
    model_version: str
    caveat: str = CAVEAT


class BatchRequest(BaseModel):
    rows: list[list[float]] = Field(..., max_length=10_000)


class BatchResponse(BaseModel):
    n: int
    treat_count: int
    treated_fraction: float
    predicted_uplift: list[float]
    threshold: float
    model_version: str
    caveat: str = CAVEAT


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
