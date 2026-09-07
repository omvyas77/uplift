"""API contract tests.

These run against whatever artifacts exist. The service is deliberately built to
boot WITHOUT a model - /health reports the truth and scoring returns 503 - so
these tests are meaningful on a fresh clone as well as after a training run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uplift.api.main import app
from uplift.api.schemas import N_FEATURES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def has_model(client):
    return client.get("/health").json()["model_loaded"]


def test_health_always_answers(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert isinstance(r.json()["model_loaded"], bool)


def test_model_info_is_serialisable(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_wrong_feature_count_is_rejected(client):
    """Pydantic must reject a short vector before it reaches the model."""
    r = client.post("/score", json={"features": [0.0] * (N_FEATURES - 1)})
    assert r.status_code == 422


def test_batch_rejects_wrong_width(client, has_model):
    if not has_model:
        pytest.skip("no trained model present")
    r = client.post("/score/batch", json={"rows": [[0.0] * 5, [0.0] * 5]})
    assert r.status_code == 422


def test_score_returns_a_decision(client, has_model):
    if not has_model:
        pytest.skip("no trained model present")
    r = client.post("/score", json={"features": [1.0] * N_FEATURES})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= body["decile"] <= 10
    assert isinstance(body["treat"], bool)
    # the decision must follow the published threshold, not float around
    assert body["treat"] == (body["predicted_uplift"] >= body["threshold"])


def test_the_caveat_ships_in_the_payload(client, has_model):
    """The model's epistemic limit travels WITH the prediction. A consumer should
    not have to read the README to learn that individual uplift is not identified."""
    if not has_model:
        pytest.skip("no trained model present")
    r = client.post("/score", json={"features": [0.5] * N_FEATURES})
    assert "not identified" in r.json()["caveat"]


def test_batch_agrees_with_single_scoring(client, has_model):
    """Same input, same answer, whichever endpoint you call."""
    if not has_model:
        pytest.skip("no trained model present")
    feats = [0.25] * N_FEATURES
    single = client.post("/score", json={"features": feats}).json()
    batch = client.post("/score/batch", json={"rows": [feats]}).json()
    assert batch["n"] == 1
    assert abs(batch["predicted_uplift"][0] - single["predicted_uplift"]) < 1e-9
    assert batch["treat_count"] == int(single["treat"])


def test_batch_treated_fraction_is_consistent(client, has_model):
    if not has_model:
        pytest.skip("no trained model present")
    rows = [[float(i % 3 - 1)] * N_FEATURES for i in range(30)]
    body = client.post("/score/batch", json={"rows": rows}).json()
    assert body["n"] == 30
    assert body["treat_count"] == sum(1 for t in body["predicted_uplift"] if t >= body["threshold"])
    assert abs(body["treated_fraction"] - body["treat_count"] / 30) < 1e-9
