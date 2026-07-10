"""Unit tests for the human feedback API router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rag_system.api.routers.feedback import router


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.rag_system.api.routers.feedback as fb_module

    monkeypatch.setattr(fb_module, "_FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestFeedbackEndpoint:
    def test_submit_thumbs_up(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "qid_001",
                "query_text": "What was Q3 revenue?",
                "answer_text": "Revenue was $23.35B.",
                "rating": "thumbs_up",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert "feedback_id" in data

    def test_submit_thumbs_down_with_comment(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "qid_002",
                "query_text": "What was EBITDA margin?",
                "answer_text": "EBITDA was 45%.",
                "rating": "thumbs_down",
                "comment": "Answer referenced wrong page",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"

    def test_invalid_rating_rejected(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "qid_003",
                "query_text": "test",
                "answer_text": "test",
                "rating": "invalid_rating",
            },
        )
        assert resp.status_code == 422

    def test_feedback_summary_empty(self, client):
        resp = client.get("/api/v1/feedback/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_feedback" in data
        assert data["total_feedback"] == 0

    def test_feedback_summary_counts_correctly(self, client):
        for rating in ["thumbs_up", "thumbs_up", "thumbs_down"]:
            client.post(
                "/api/v1/feedback",
                json={
                    "query_id": "q",
                    "query_text": "q?",
                    "answer_text": "a",
                    "rating": rating,
                    "tenant_id": "default",
                },
            )
        resp = client.get("/api/v1/feedback/summary")
        data = resp.json()
        assert data["thumbs_up"] == 2
        assert data["thumbs_down"] == 1
        assert abs(data["satisfaction_rate"] - 0.667) < 0.01
