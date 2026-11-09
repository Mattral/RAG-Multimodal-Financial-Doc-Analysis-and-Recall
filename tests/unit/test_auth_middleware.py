"""Tests for the X-API-Key authentication middleware."""

from __future__ import annotations

import hmac

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from filing_qa.api.middleware.auth import APIKeyMiddleware


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


# ── Behaviour without RAG_API_MASTER_KEY ─────────────────────────────────────
# The middleware used to let every request through when the key was unset, in any
# environment. It must fail closed unless development or AUTH_DISABLED is explicit.


def _unkeyed_client(monkeypatch, **env):
    from filing_qa.config import reset_config

    monkeypatch.delenv("RAG_API_MASTER_KEY", raising=False)
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_config()
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/v1/thing")
    async def thing():
        return {"ok": True}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return TestClient(app)


@pytest.mark.parametrize(
    "env",
    [
        {"ENVIRONMENT": "production"},
        {"ENVIRONMENT": "staging"},
        {"ENVIRONMENT": "testing"},
        {},  # ENVIRONMENT unset: the config default must not open the API
        {"ENVIRONMENT": "production", "AUTH_DISABLED": "false"},
    ],
)
def test_unset_key_rejects_requests(monkeypatch, env):
    client = _unkeyed_client(monkeypatch, **env)
    resp = client.get("/api/v1/thing")
    assert resp.status_code == 401
    assert resp.json() == {"error": "API authentication is not configured"}
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize(
    "env, reason",
    [
        ({"ENVIRONMENT": "development"}, "ENVIRONMENT=development"),
        ({"ENVIRONMENT": "production", "AUTH_DISABLED": "true"}, "AUTH_DISABLED=true"),
    ],
)
def test_unset_key_allowed_when_explicit_and_warns(monkeypatch, env, reason):
    from structlog.testing import capture_logs

    from filing_qa.api.middleware.auth import log_auth_startup_status

    client = _unkeyed_client(monkeypatch, **env)
    assert client.get("/api/v1/thing").status_code == 200
    with capture_logs() as logs:
        log_auth_startup_status()
    assert [e["log_level"] for e in logs] == ["warning"]
    assert logs[0]["event"] == "api_auth_disabled"
    assert logs[0]["reason"] == reason


def test_unset_key_in_production_logs_error_at_startup(monkeypatch):
    from structlog.testing import capture_logs

    from filing_qa.api.middleware.auth import log_auth_startup_status

    _unkeyed_client(monkeypatch, ENVIRONMENT="production")
    with capture_logs() as logs:
        log_auth_startup_status()
    assert [(e["log_level"], e["event"]) for e in logs] == [("error", "api_auth_unconfigured")]


def test_key_is_required_even_when_auth_disabled_is_set(monkeypatch):
    client = _unkeyed_client(monkeypatch, AUTH_DISABLED="true", RAG_API_MASTER_KEY="k")
    assert client.get("/api/v1/thing").status_code == 401
    assert client.get("/api/v1/thing", headers={"X-API-Key": "k"}).status_code == 200
