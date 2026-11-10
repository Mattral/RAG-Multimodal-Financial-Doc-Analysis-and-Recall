"""Vector store adapters against the real backends (no mocks).

Each adapter upserts chunks, is queried, then a fresh adapter instance (as a
restarted process would create) queries the same store again. Results must carry
the exact text, source, chunk id and page number that were written.

- deeplake runs when the package is installed (local dataset in a temp dir).
- qdrant runs when TEST_QDRANT_URL points at a Qdrant server.
- pgvector runs when TEST_PGVECTOR_DSN points at Postgres with the vector extension.

CI provides the servers as service containers; locally these cases are skipped
with the reason shown.
"""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from filing_qa.components.base import DocumentElement, RetrievedChunk
from filing_qa.components.vector_store import build_vector_store
from filing_qa.config import reset_config

DIM = 8
SOURCE = "tesla_10q_q3_2023.pdf"
CHUNKS = [
    ("Total revenues were $23,350 million for the quarter.", 4),
    ("Operating cash flow was $3,308 million.", 7),
    ("Gross margin decreased to 17.9%.", 5),
]


def _vec(i: int) -> list[float]:
    """One-hot vectors, so each query has exactly one best match."""
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _elements(tenant: str) -> list[DocumentElement]:
    return [
        DocumentElement(
            type="text",
            text=text,
            source_document=SOURCE,
            page_number=page,
            content_hash=f"hash-{i}",
            tenant_id=tenant,
        )
        for i, (text, page) in enumerate(CHUNKS)
    ]


def _assert_chunk(chunk: RetrievedChunk, index: int) -> None:
    text, page = CHUNKS[index]
    assert type(chunk.text) is str and chunk.text == text
    assert type(chunk.source_document) is str and chunk.source_document == SOURCE
    assert type(chunk.page_number) is int and chunk.page_number == page
    assert type(chunk.chunk_id) is str and chunk.chunk_id == f"hash-{index}"
    assert type(chunk.score) is float


def _skip_unless(condition: bool, reason: str):
    return pytest.mark.skipif(not condition, reason=reason)


BACKENDS = [
    pytest.param(
        "deeplake",
        marks=_skip_unless(
            importlib.util.find_spec("deeplake") is not None, "deeplake not installed"
        ),
    ),
    pytest.param(
        "qdrant",
        marks=_skip_unless(bool(os.environ.get("TEST_QDRANT_URL")), "TEST_QDRANT_URL not set"),
    ),
    pytest.param(
        "pgvector",
        marks=_skip_unless(bool(os.environ.get("TEST_PGVECTOR_DSN")), "TEST_PGVECTOR_DSN not set"),
    ),
]


@pytest.fixture
def backend_env(request, tmp_path, monkeypatch):
    provider = request.param
    name = f"roundtrip_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("VECTOR_STORE_CONFIG__PROVIDER", provider)
    monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("VECTOR_STORE_CONFIG__COLLECTION_NAME", name)
    if provider == "deeplake":
        monkeypatch.setenv("VECTOR_STORE_CONFIG__DATASET_PATH", str(tmp_path / name))
    elif provider == "qdrant":
        monkeypatch.setenv("VECTOR_STORE_CONFIG__QDRANT_URL", os.environ["TEST_QDRANT_URL"])
    elif provider == "pgvector":
        monkeypatch.setenv(
            "VECTOR_STORE_CONFIG__CONNECTION_STRING", os.environ["TEST_PGVECTOR_DSN"]
        )
    reset_config()
    yield provider
    reset_config()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_env", BACKENDS, indirect=True)
async def test_upsert_query_reopen_query(backend_env):
    provider = backend_env
    tenant = "acme"

    store = build_vector_store(provider)
    await store.initialize(tenant_id=tenant)
    await store.upsert(_elements(tenant), [_vec(i) for i in range(len(CHUNKS))], tenant_id=tenant)

    first = await store.search(_vec(1), top_k=2, tenant_id=tenant)
    assert len(first) == 2
    _assert_chunk(first[0], 1)
    assert first[0].score == pytest.approx(1.0, abs=1e-4)

    # A new instance holds no in-process state: everything must come from the store.
    reopened = build_vector_store(provider)
    await reopened.initialize(tenant_id=tenant)
    second = await reopened.search(_vec(2), top_k=3, tenant_id=tenant)
    assert len(second) == 3
    _assert_chunk(second[0], 2)
    assert sorted(c.chunk_id for c in second) == ["hash-0", "hash-1", "hash-2"]
    for chunk in second:
        _assert_chunk(chunk, int(chunk.chunk_id.split("-")[1]))
