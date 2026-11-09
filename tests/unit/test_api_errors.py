"""API error responses must not echo exception text to clients.

Before the fix, 500s returned ``str(exc)`` (router handlers) or its first 200
characters (global handler), which can carry connection strings, file paths or
provider error bodies. Clients now get ``{code, message, request_id}``; the
exception is logged server-side under the same request id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

PROBE = "connection to postgres://admin:hunter2@db.internal:5432/filings failed"


def _client(pipeline):
    from filing_qa.api.app import create_app

    app = create_app()
    app.state.pipeline = pipeline

    @app.get("/api/v1/_boom")
    async def boom():
        raise RuntimeError(PROBE)

    return TestClient(app, raise_server_exceptions=False)


def _failing_pipeline():
    pipeline = MagicMock()
    pipeline.health_check = AsyncMock(side_effect=RuntimeError(PROBE))
    pipeline.query = AsyncMock(side_effect=RuntimeError(PROBE))
    pipeline.ingest = AsyncMock(side_effect=RuntimeError(PROBE))
    pipeline.list_documents = AsyncMock(side_effect=RuntimeError(PROBE))
    pipeline.delete_document = AsyncMock(side_effect=RuntimeError(PROBE))

    async def query_stream(**kwargs):
        raise RuntimeError(PROBE)
        yield  # pragma: no cover

    pipeline.query_stream = query_stream
    return pipeline


@pytest.mark.parametrize(
    "method, path, kwargs, status, code",
    [
        ("post", "/api/v1/query", {"json": {"query": "What was revenue?"}}, 500, "query_failed"),
        (
            "post",
            "/api/v1/ingest",
            {"files": {"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}},
            500,
            "ingest_failed",
        ),
        ("get", "/api/v1/documents", {}, 500, "document_list_failed"),
        ("delete", "/api/v1/documents/file://a.pdf", {}, 500, "document_delete_failed"),
        ("get", "/readyz", {}, 503, "health_check_failed"),
        ("get", "/api/v1/_boom", {}, 500, "internal_error"),
    ],
)
def test_error_responses_hide_exception_text(method, path, kwargs, status, code):
    with _client(_failing_pipeline()) as client, capture_logs() as logs:
        resp = getattr(client, method)(path, **kwargs)

    assert resp.status_code == status
    assert PROBE not in resp.text
    assert "hunter2" not in resp.text
    body = resp.json()
    assert body["code"] == code
    assert body["message"]
    assert body["request_id"]
    # The detail is kept server-side, under the same request id.
    logged = [e for e in logs if e.get("request_id") == body["request_id"]]
    assert logged and PROBE in logged[0]["error"]


def test_stream_error_event_hides_exception_text():
    with _client(_failing_pipeline()) as client:
        resp = client.post("/api/v1/query/stream", json={"query": "q"})
    assert PROBE not in resp.text
    assert '"code": "query_failed"' in resp.text


def test_well_formed_request_id_is_echoed():
    with _client(_failing_pipeline()) as client:
        resp = client.post(
            "/api/v1/query", json={"query": "q"}, headers={"X-Request-ID": "abc-123"}
        )
    assert resp.json()["request_id"] == "abc-123"
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_malformed_request_id_is_replaced():
    bad = "x" * 200 + "\r\nSet-Cookie: a=b"
    with _client(_failing_pipeline()) as client:
        resp = client.post(
            "/api/v1/query", json={"query": "q"}, headers={"X-Request-ID": bad[:200]}
        )
    rid = resp.json()["request_id"]
    assert rid != bad[:200]
    assert len(rid) == 32
