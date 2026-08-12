# tests/test_api.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_config_returns_expected_fields():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "llm_model",
        "embedding_model",
        "sparse_embedding_model",
        "reranker_model",
        "judge_model",
        "qdrant_url",
        "qdrant_collection",
    }


def test_get_config_never_exposes_secrets():
    response = client.get("/config")
    body = response.text.lower()
    assert "key" not in body
    assert "token" not in body
