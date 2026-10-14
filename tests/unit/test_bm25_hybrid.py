"""Unit tests for the production BM25 path: ingest populates the keyword index,
hybrid retrieval returns BM25 hits, and tenants never see each other's chunks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag_system.components.base import DocumentElement, RetrievedChunk
from src.rag_system.components.retriever import BM25Index, HybridRetriever
from src.rag_system.components.vector_store import InMemoryVectorStore
from src.rag_system.components.version_manager import DocumentVersionManager
from src.rag_system.pipeline import RAGPipeline

# ── Fixtures ──────────────────────────────────────────────────────────────


def _elem(text: str, source: str, page: int, h: str, tenant: str, **metadata) -> DocumentElement:
    return DocumentElement(
        type="text",
        text=text,
        source_document=source,
        page_number=page,
        content_hash=h,
        tenant_id=tenant,
        metadata=metadata,
    )


ACME_ELEMENTS = [
    _elem("Acme Q3 revenue was $42.3M, driven by widget sales.", "acme_10q.pdf", 4, "a1", "acme"),
    _elem("Acme gross margin contracted to 31% in Q3.", "acme_10q.pdf", 5, "a2", "acme"),
]
GLOBEX_ELEMENTS = [
    _elem("Globex Q3 revenue was $99.9M on hammock exports.", "globex_10q.pdf", 2, "g1", "globex"),
    _elem(
        "Globex disclosed a confidential merger with Initech.", "globex_10q.pdf", 7, "g2", "globex"
    ),
]


@pytest.fixture
def embedder():
    emb = MagicMock()
    emb.name = "fake_embedder"
    emb.dimension = 3
    emb.embed = AsyncMock(side_effect=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    emb.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return emb


@pytest.fixture
def empty_dense_store():
    """Vector store whose search always misses, so any result must come from BM25."""
    store = MagicMock()
    store.name = "empty_dense"
    store.upsert = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


def _pipeline(embedder, vector_store, retriever, tmp_path, elements_by_call) -> RAGPipeline:
    parser = MagicMock()
    parser.name = "fake_parser"
    parser.parse_batch = AsyncMock(side_effect=elements_by_call)
    return RAGPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        audit_logger=MagicMock(),
        version_manager=DocumentVersionManager(registry_path=str(tmp_path / "versions.json")),
        semantic_cache=False,
    )


# ── BM25Index ─────────────────────────────────────────────────────────────


def test_build_without_tenant_is_backward_compatible():
    idx = BM25Index()
    idx.build([RetrievedChunk(text="Tesla Q3 revenue", score=0, source_document="a.pdf")])
    assert [r.source_document for r in idx.search("revenue")] == ["a.pdf"]
    # Legacy calls land in the "default" partition, the same one tenant_id=None uses.
    assert idx.count() == 1
    assert idx.search("revenue", tenant_id="other") == []


def test_add_is_incremental_and_upserts_by_chunk_id():
    idx = BM25Index()
    idx.add([RetrievedChunk(text="alpha", score=0, source_document="a.pdf", chunk_id="1")])
    idx.add([RetrievedChunk(text="beta", score=0, source_document="b.pdf", chunk_id="2")])
    assert idx.count() == 2

    idx.add([RetrievedChunk(text="gamma", score=0, source_document="a.pdf", chunk_id="1")])
    assert idx.count() == 2
    assert idx.search("alpha") == []
    assert [r.chunk_id for r in idx.search("gamma")] == ["1"]


def test_search_excludes_chunks_with_no_query_terms():
    idx = BM25Index()
    idx.build(
        [
            RetrievedChunk(text="revenue grew", score=0, source_document="a.pdf"),
            RetrievedChunk(text="unrelated text", score=0, source_document="b.pdf"),
        ]
    )
    assert [r.source_document for r in idx.search("revenue")] == ["a.pdf"]


def test_search_is_scoped_to_tenant_partition():
    idx = BM25Index()
    idx.add([RetrievedChunk(text="acme revenue", score=0, source_document="a.pdf")], "acme")
    idx.add([RetrievedChunk(text="globex revenue", score=0, source_document="g.pdf")], "globex")

    assert [r.source_document for r in idx.search("revenue", tenant_id="acme")] == ["a.pdf"]
    assert [r.source_document for r in idx.search("revenue", tenant_id="globex")] == ["g.pdf"]
    assert idx.search("globex", tenant_id="acme") == []
    assert idx.search("revenue") == []  # default partition is its own tenant


def test_idf_is_not_influenced_by_other_tenants():
    """Term statistics are per tenant, so scores can't leak another tenant's corpus."""
    chunk = RetrievedChunk(text="revenue margin", score=0, source_document="a.pdf")

    alone = BM25Index()
    alone.add([chunk], "acme")

    shared = BM25Index()
    shared.add([chunk], "acme")
    shared.add(
        [RetrievedChunk(text="revenue", score=0, source_document=f"g{i}.pdf") for i in range(50)],
        "globex",
    )

    assert alone.search("revenue", tenant_id="acme")[0].score == pytest.approx(
        shared.search("revenue", tenant_id="acme")[0].score
    )


def test_search_applies_filters_like_dense_path():
    idx = BM25Index()
    idx.add(
        [
            RetrievedChunk(
                text="revenue", score=0, source_document="a.pdf", metadata={"doc_type": "10-Q"}
            ),
            RetrievedChunk(
                text="revenue", score=0, source_document="b.pdf", metadata={"doc_type": "10-K"}
            ),
            RetrievedChunk(text="revenue", score=0, source_document="c.pdf"),
        ]
    )
    assert [r.source_document for r in idx.search("revenue", filters={"doc_type": "10-K"})] == [
        "b.pdf"
    ]
    # Top-level fields are filterable too.
    assert [
        r.source_document for r in idx.search("revenue", filters={"source_document": "c.pdf"})
    ] == ["c.pdf"]


# ── Pipeline ingest → BM25 ────────────────────────────────────────────────


async def test_ingest_populates_bm25(embedder, empty_dense_store, tmp_path):
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store=empty_dense_store, embedder=embedder, bm25_index=bm25)
    pipeline = _pipeline(embedder, empty_dense_store, retriever, tmp_path, [ACME_ELEMENTS])

    result = await pipeline.ingest(["acme_10q.pdf"], tenant_id="acme", process_vision=False)

    assert result["num_chunks"] == 2
    assert bm25.count("acme") == 2
    hit = bm25.search("gross margin", tenant_id="acme")[0]
    assert hit.source_document == "acme_10q.pdf"
    assert hit.page_number == 5
    assert hit.chunk_id == "a2"
    assert hit.metadata["tenant_id"] == "acme"
    assert hit.metadata["chunk_type"] == "text"


async def test_reingest_does_not_duplicate_bm25_chunks(embedder, empty_dense_store, tmp_path):
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store=empty_dense_store, embedder=embedder, bm25_index=bm25)
    pipeline = _pipeline(
        embedder, empty_dense_store, retriever, tmp_path, [ACME_ELEMENTS, ACME_ELEMENTS]
    )

    await pipeline.ingest(["acme_10q.pdf"], tenant_id="acme", process_vision=False)
    await pipeline.ingest(
        ["acme_10q.pdf"], tenant_id="acme", process_vision=False, skip_unchanged=False
    )

    assert bm25.count("acme") == 2


async def test_hybrid_retrieval_returns_bm25_hits(embedder, empty_dense_store, tmp_path):
    retriever = HybridRetriever(
        vector_store=empty_dense_store, embedder=embedder, bm25_index=BM25Index()
    )
    pipeline = _pipeline(embedder, empty_dense_store, retriever, tmp_path, [ACME_ELEMENTS])
    await pipeline.ingest(["acme_10q.pdf"], tenant_id="acme", process_vision=False)

    results = await retriever.retrieve("widget sales revenue", tenant_id="acme")

    # Dense search returned nothing, so these can only be BM25 hits.
    assert results
    assert results[0].chunk_id == "a1"
    assert results[0].source_document == "acme_10q.pdf"


async def test_tenant_never_sees_other_tenants_chunks(embedder, tmp_path):
    store = InMemoryVectorStore()
    retriever = HybridRetriever(vector_store=store, embedder=embedder, bm25_index=BM25Index())
    pipeline = _pipeline(embedder, store, retriever, tmp_path, [ACME_ELEMENTS, GLOBEX_ELEMENTS])
    await pipeline.ingest(["acme_10q.pdf"], tenant_id="acme", process_vision=False)
    await pipeline.ingest(["globex_10q.pdf"], tenant_id="globex", process_vision=False)

    # Query with terms that only appear in Globex's documents.
    acme_results = await retriever.retrieve("Globex confidential merger Initech", tenant_id="acme")
    assert acme_results  # dense still returns Acme's own chunks
    assert {c.source_document for c in acme_results} == {"acme_10q.pdf"}
    assert not any("Globex" in c.text for c in acme_results)

    globex_results = await retriever.retrieve("Acme widget sales", tenant_id="globex")
    assert {c.source_document for c in globex_results} == {"globex_10q.pdf"}


async def test_create_wires_one_shared_bm25_index(embedder, empty_dense_store):
    empty_dense_store.initialize = AsyncMock()
    pipeline = await RAGPipeline.create(
        embedder=embedder,
        vector_store=empty_dense_store,
        reranker=MagicMock(),
        parser=MagicMock(),
        vision_describer=MagicMock(),
        generator=MagicMock(),
    )
    assert isinstance(pipeline._bm25_index, BM25Index)
    assert pipeline._bm25_index is pipeline._retriever.bm25_index
