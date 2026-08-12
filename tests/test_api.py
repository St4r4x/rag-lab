# tests/test_api.py
import json

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


def test_list_eval_runs_empty_when_no_results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path / "missing")
    response = client.get("/eval/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_eval_runs_returns_summaries(monkeypatch, tmp_path):
    (tmp_path / "20260101T000000Z.json").write_text(
        json.dumps([{"faithfulness": 4, "correctness": 5}]), encoding="utf-8"
    )
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs")
    assert response.status_code == 200
    assert response.json() == [
        {"id": "20260101T000000Z", "count": 1, "avg_faithfulness": 4.0, "avg_correctness": 5.0}
    ]


def test_get_eval_run_rejects_malformed_run_id():
    response = client.get("/eval/runs/not-a-valid-id")
    assert response.status_code == 400


def test_get_eval_run_returns_404_for_missing_well_formed_id(monkeypatch, tmp_path):
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs/20260101T000000Z")
    assert response.status_code == 404


def test_get_eval_run_returns_full_detail(monkeypatch, tmp_path):
    detail = [{"id": "q01", "faithfulness": 4, "correctness": 5}]
    (tmp_path / "20260101T000000Z.json").write_text(json.dumps(detail), encoding="utf-8")
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs/20260101T000000Z")
    assert response.status_code == 200
    assert response.json() == detail
