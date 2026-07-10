"""Tests for ColPali late-interaction visual retriever — MaxSim and index."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import pytest

from src.rag_system.components.colpali_retriever import (
    ColPaliRetriever,
    PageEmbedding,
    _maxsim,
)


class TestMaxSim:
    def test_identical_vectors(self):
        v = [[1.0, 0.0, 0.0]]
        assert _maxsim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        assert _maxsim([[1.0, 0.0]], [[0.0, 1.0]]) == pytest.approx(0.0, abs=1e-5)

    def test_multi_patch_sums_maxima(self):
        # q[0]=[1,0] max over d = 1.0; q[1]=[0,1] max over d = 1.0 => total 2.0
        q = [[1.0, 0.0], [0.0, 1.0]]
        d = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        assert _maxsim(q, d) == pytest.approx(2.0, abs=1e-4)

    def test_empty_doc_vecs_returns_zero(self):
        assert _maxsim([[1.0, 0.0]], []) == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        assert isinstance(_maxsim([[0.5, 0.5]], [[0.5, 0.5]]), float)

    def test_more_similar_page_scores_higher(self):
        q = [[1.0, 0.0]]
        assert _maxsim(q, [[0.99, 0.01]]) > _maxsim(q, [[0.1, 0.99]])

    def test_high_dimensional(self):
        dim = 128
        v = [[1.0 / math.sqrt(dim)] * dim]
        assert _maxsim(v, v) == pytest.approx(1.0, abs=1e-4)


class TestPageEmbedding:
    def test_creation(self):
        pe = PageEmbedding("tesla.pdf", 5, [[0.1, 0.2]])
        assert pe.source_document == "tesla.pdf"
        assert pe.page_number == 5

    def test_default_metadata_empty(self):
        assert PageEmbedding("d.pdf", 1, [[0.1]]).metadata == {}


class TestColPaliRetriever:
    def test_name_property(self):
        r = ColPaliRetriever(model_name="vidore/colqwen2-v1.0")
        assert "colpali" in r.name and "colqwen2" in r.name

    def test_check_deps_false_without_library(self):
        r = ColPaliRetriever()
        with patch.dict("sys.modules", {"colpali_engine": None}):
            assert r._has_deps() is False

    @pytest.mark.asyncio
    async def test_retrieve_empty_index_returns_empty(self):
        r = ColPaliRetriever(index_path=None)
        assert await r.retrieve("revenue?") == []

    @pytest.mark.asyncio
    async def test_retrieve_empty_when_model_unavailable(self):
        r = ColPaliRetriever(index_path=None)
        r._index = [PageEmbedding("d.pdf", 1, [[0.5, 0.5]])]
        with patch.object(r, "_load_model", return_value=False):
            assert await r.retrieve("query") == []

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        r = ColPaliRetriever(index_path=None)
        r._index = [PageEmbedding("doc.pdf", i + 1, [[float(i + 1), 0.0]]) for i in range(10)]
        with (
            patch.object(r, "_load_model", return_value=True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=[[1.0, 0.0]])),
        ):
            results = await r.retrieve("revenue", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_sorted_descending(self):
        r = ColPaliRetriever(index_path=None)
        r._index = [
            PageEmbedding("d.pdf", 1, [[1.0, 0.0]]),
            PageEmbedding("d.pdf", 2, [[0.0, 1.0]]),
            PageEmbedding("d.pdf", 3, [[0.7, 0.3]]),
        ]
        with (
            patch.object(r, "_load_model", return_value=True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=[[1.0, 0.0]])),
        ):
            results = await r.retrieve("query", top_k=3)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieve_applies_source_document_filter(self):
        r = ColPaliRetriever(index_path=None)
        r._index = [
            PageEmbedding("tesla.pdf", 1, [[1.0, 0.0]]),
            PageEmbedding("apple.pdf", 2, [[0.9, 0.1]]),
            PageEmbedding("tesla.pdf", 3, [[0.8, 0.2]]),
        ]
        with (
            patch.object(r, "_load_model", return_value=True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=[[1.0, 0.0]])),
        ):
            results = await r.retrieve("q", filters={"source_document": "tesla.pdf"})
        assert all(res.source_document == "tesla.pdf" for res in results)
        assert len(results) == 2

    def test_load_index_missing_file_returns_false(self):
        r = ColPaliRetriever(index_path="/nonexistent/path.json")
        assert r.load_index() is False

    def test_save_and_load_roundtrip(self, tmp_path):
        idx = str(tmp_path / "idx.json")
        r = ColPaliRetriever(index_path=idx)
        r._index = [PageEmbedding("d.pdf", 1, [[0.1, 0.9]], thumbnail_path="t.png")]
        r._save_index()
        r2 = ColPaliRetriever(index_path=idx)
        assert r2.load_index() is True
        assert r2._index[0].source_document == "d.pdf"
        assert r2._index[0].patch_embeddings == [[0.1, 0.9]]
        assert r2._index[0].thumbnail_path == "t.png"

    @pytest.mark.asyncio
    async def test_retrieve_result_metadata(self):
        r = ColPaliRetriever(index_path=None)
        r._index = [PageEmbedding("doc.pdf", 7, [[1.0, 0.0]])]
        with (
            patch.object(r, "_load_model", return_value=True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=[[1.0, 0.0]])),
        ):
            results = await r.retrieve("q", top_k=1)
        assert results[0].page_number == 7
        assert results[0].metadata["method"] == "colpali_maxsim"
