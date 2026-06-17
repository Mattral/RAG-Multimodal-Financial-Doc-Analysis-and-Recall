"""API integration tests using FastAPI TestClient.

Tests the full request→middleware→router→pipeline→response cycle
with in-memory components (no real OpenAI calls).
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import os
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("VECTOR_STORE_CONFIG__PROVIDER", "memory")
os.environ.setdefault("CACHE_CONFIG__BACKEND", "memory")


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.health_check = AsyncMock(return_value={
        "status": "healthy",
        "components": {"vector_store": "ok", "retriever": "ok"},
    })
    pipeline.query = AsyncMock(return_value={
        "status": "success",
        "tenant_id": "test",
        "query": "What was revenue?",
        "answer": "Revenue was $23.35B [Source: tesla.pdf, Page 4].",
        "answer_obj": None,
        "sources": [{"document": "tesla.pdf", "page": 4, "score": 0.92,
                     "text_preview": "Revenue was $23.35B."}],
        "guardrails": {"overall_passed": True},
        "metrics": {"total_latency_ms": 1200, "cost_usd": 0.0001, "num_chunks": 3},
        "analysis": {"intent": "factual", "complexity": "simple", "use_pot": False,
                     "rewritten_query": "What was revenue?", "filters_applied": {}},
        "pot_result": None,
    })
    pipeline.ingest = AsyncMock(return_value={
        "status": "success", "tenant_id": "test",
        "num_files": 1, "num_chunks": 42, "skipped": 0, "latency_s": 3.2,
    })
    pipeline.list_documents = AsyncMock(return_value=[
        {"source_uri": "file://test.pdf", "filename": "test.pdf",
         "version": 1, "content_hash": "abc123", "is_deleted": False},
    ])
    pipeline.delete_document = AsyncMock(return_value={
        "status": "deleted", "source_uri": "file://test.pdf"
    })
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    from src.rag_system.api.app import create_app
    app = create_app()
    app.state.pipeline = mock_pipeline
    return TestClient(app)


class TestHealthEndpoints:
    def test_liveness(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_k8s_liveness(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_readiness_with_healthy_pipeline(self, client, mock_pipeline):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_readiness_without_pipeline(self):
        from src.rag_system.api.app import create_app
        app = create_app()
        app.state.pipeline = None
        c = TestClient(app)
        resp = c.get("/readyz")
        assert resp.status_code == 503


class TestQueryEndpoint:
    def test_query_success(self, client):
        resp = client.post("/api/v1/query", json={
            "query": "What was revenue in Q3 2023?",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["answer"] is not None
        assert len(data["sources"]) > 0

    def test_query_empty_string_rejected(self, client):
        resp = client.post("/api/v1/query", json={"query": "", "top_k": 5})
        assert resp.status_code == 422

    def test_query_too_long_rejected(self, client):
        resp = client.post("/api/v1/query", json={"query": "x" * 3000})
        assert resp.status_code == 422

    def test_query_top_k_too_large_rejected(self, client):
        resp = client.post("/api/v1/query", json={"query": "What was revenue?", "top_k": 100})
        assert resp.status_code == 422

    def test_query_without_pipeline_returns_503(self):
        from src.rag_system.api.app import create_app
        app = create_app()
        app.state.pipeline = None
        c = TestClient(app)
        resp = c.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 503


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert data["total"] == 1

    def test_delete_document(self, client):
        resp = client.delete("/api/v1/documents/file%3A%2F%2Ftest.pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client, tmp_path, monkeypatch):
        import src.rag_system.api.routers.feedback as fb
        monkeypatch.setattr(fb, "_FEEDBACK_PATH", tmp_path / "fb.jsonl")
        resp = client.post("/api/v1/feedback", json={
            "query_id": "q001",
            "query_text": "What was revenue?",
            "answer_text": "Revenue was $23.35B.",
            "rating": "thumbs_up",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"


class TestTenantsEndpoint:
    def test_create_tenant(self, client):
        resp = client.post("/api/v1/tenants", json={"tenant_id": "new_tenant"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["tenant_id"] == "new_tenant"

    def test_get_usage(self, client):
        resp = client.get("/api/v1/tenants/any_tenant/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_cost_usd" in data
