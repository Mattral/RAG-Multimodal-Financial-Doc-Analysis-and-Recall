"""Tests for the X-API-Key authentication middleware."""

from __future__ import annotations

import hmac

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rag_system.api.middleware.auth import APIKeyMiddleware


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RAG_API_MASTER_KEY", "correct-key")
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/v1/thing")
    async def thing():
        return {"ok": True}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return TestClient(app)


def test_valid_key_is_accepted(client):
    assert client.get("/api/v1/thing", headers={"X-API-Key": "correct-key"}).status_code == 200


def test_wrong_key_is_rejected(client):
    assert client.get("/api/v1/thing", headers={"X-API-Key": "wrong"}).status_code == 403


def test_missing_key_is_rejected(client):
    assert client.get("/api/v1/thing").status_code == 401


def test_health_probe_is_exempt(client):
    assert client.get("/healthz").status_code == 200


def test_comparison_uses_constant_time_compare(client, monkeypatch):
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    client.get("/api/v1/thing", headers={"X-API-Key": "wrong"})
    assert len(calls) == 1
    assert len(calls[0][0]) == len(calls[0][1]) == 32  # SHA-256 digests, never the raw keys
