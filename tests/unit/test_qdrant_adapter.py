"""Tests for the Qdrant vector store adapter (mocked HTTP, no live instance)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_system.components.base import DocumentElement
from src.rag_system.components.vector_store.qdrant_adapter import QdrantAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_CONFIG__QDRANT_URL", "http://localhost:6333")
    from src.rag_system.config import reset_config

    reset_config()
    a = QdrantAdapter()
    yield a
    reset_config()


class TestQdrantAdapterUnit:
    def test_name_property(self, adapter):
        assert adapter.name == "qdrant"

    def test_collection_name_default_tenant(self, adapter):
        assert adapter._collection_name("default") == adapter._collection_name(None)

    def test_collection_name_custom_tenant(self, adapter):
        assert "acme" in adapter._collection_name("acme")

    def test_custom_tenant_differs_from_default(self, adapter):
        assert adapter._collection_name("acme") != adapter._collection_name("default")

    def test_check_deps_false_when_not_installed(self, adapter):
        with patch.dict("sys.modules", {"qdrant_client": None}):
            assert adapter._check_deps() is False

    def test_get_url_from_env(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__QDRANT_URL", "http://custom:6333")
        from src.rag_system.config import reset_config

        reset_config()
        a = QdrantAdapter()
        assert a._get_url() == "http://custom:6333"
        reset_config()

    def test_get_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__QDRANT_API_KEY", "key-123")
        a = QdrantAdapter()
        assert a._get_api_key() == "key-123"

    def test_get_api_key_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("VECTOR_STORE_CONFIG__QDRANT_API_KEY", raising=False)
        a = QdrantAdapter()
        assert a._get_api_key() is None

    @pytest.mark.asyncio
    async def test_upsert_empty_is_noop(self, adapter):
        with patch.object(adapter, "_check_deps", return_value=True):
            await adapter.upsert([], [], tenant_id="test")

    @pytest.mark.asyncio
    async def test_search_returns_empty_without_deps(self, adapter):
        with patch.object(adapter, "_check_deps", return_value=False):
            assert await adapter.search([0.1] * 384, top_k=5) == []

    @pytest.mark.asyncio
    async def test_delete_does_not_raise_without_deps(self, adapter):
        with patch.object(adapter, "_check_deps", return_value=False):
            await adapter.delete(["id1"])

    @pytest.mark.asyncio
    async def test_upsert_calls_qdrant_client(self, adapter):
        mock_client = AsyncMock()
        elements = [
            DocumentElement(
                type="text",
                text="Revenue $23B",
                source_document="t.pdf",
                page_number=1,
                content_hash="abc",
            )
        ]
        with (
            patch.object(adapter, "_check_deps", return_value=True),
            patch.object(adapter, "_get_async_client", return_value=mock_client),
            patch.object(adapter, "_ensure_collection", AsyncMock()),
        ):
            await adapter.upsert(elements, [[0.1] * 384], tenant_id="acme")
            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_maps_qdrant_results(self, adapter):
        hit = MagicMock()
        hit.score = 0.92
        hit.payload = {
            "text": "Revenue $23B",
            "source_document": "t.pdf",
            "page_number": 4,
            "content_hash": "abc",
            "metadata": {},
        }
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[hit])
        with (
            patch.object(adapter, "_check_deps", return_value=True),
            patch.object(adapter, "_get_async_client", return_value=mock_client),
        ):
            results = await adapter.search([0.1] * 384, top_k=5, tenant_id="acme")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.92)
        assert results[0].source_document == "t.pdf"
        assert results[0].page_number == 4

    @pytest.mark.asyncio
    async def test_search_passes_filter_to_qdrant(self, adapter):
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        with (
            patch.object(adapter, "_check_deps", return_value=True),
            patch.object(adapter, "_get_async_client", return_value=mock_client),
        ):
            await adapter.search([0.1] * 384, filters={"doc_type": "10-K"}, tenant_id="t1")
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_no_filter_when_empty_dict(self, adapter):
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        with (
            patch.object(adapter, "_check_deps", return_value=True),
            patch.object(adapter, "_get_async_client", return_value=mock_client),
        ):
            await adapter.search([0.1] * 384, filters={}, tenant_id="t1")
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs.get("query_filter") is None

    @pytest.mark.asyncio
    async def test_search_handles_exception_gracefully(self, adapter):
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=Exception("connection refused"))
        with (
            patch.object(adapter, "_check_deps", return_value=True),
            patch.object(adapter, "_get_async_client", return_value=mock_client),
        ):
            results = await adapter.search([0.1] * 384)
        assert results == []

    def test_factory_builds_qdrant(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__PROVIDER", "qdrant")
        monkeypatch.setenv("VECTOR_STORE_CONFIG__QDRANT_URL", "http://localhost:6333")
        from src.rag_system.config import reset_config

        reset_config()
        from src.rag_system.components.vector_store import build_vector_store

        store = build_vector_store("qdrant")
        assert isinstance(store, QdrantAdapter)
        reset_config()
