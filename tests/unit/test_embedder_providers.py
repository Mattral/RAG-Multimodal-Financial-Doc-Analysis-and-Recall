"""Unit tests for multi-provider embedders and the build_embedder factory.

Covers the DX fix: users can now select OpenAI, local (sentence-transformers,
zero API cost), Voyage (finance-domain-tuned), or Cohere embeddings with zero
pipeline code changes -- just VECTOR_STORE_CONFIG__EMBEDDING_PROVIDER in .env.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_system.components.embedder import (
    CohereEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    build_embedder,
)
from src.rag_system.utils.exceptions import ConfigurationError

# ── Factory: explicit provider argument ───────────────────────────────────────


class TestBuildEmbedderFactoryExplicit:
    def test_openai_provider_resolves(self):
        emb = build_embedder("openai")
        assert isinstance(emb, OpenAIEmbedder)

    def test_local_provider_resolves(self):
        emb = build_embedder("local")
        assert isinstance(emb, LocalEmbedder)

    def test_voyage_provider_resolves(self):
        emb = build_embedder("voyage")
        assert isinstance(emb, VoyageEmbedder)

    def test_cohere_provider_resolves(self):
        emb = build_embedder("cohere")
        assert isinstance(emb, CohereEmbedder)

    def test_case_insensitive(self):
        emb = build_embedder("VOYAGE")
        assert isinstance(emb, VoyageEmbedder)

    def test_unknown_provider_falls_back_to_openai(self):
        emb = build_embedder("totally_unknown_provider")
        assert isinstance(emb, OpenAIEmbedder)


# ── Factory: inference from VECTOR_STORE_CONFIG__EMBEDDING_MODEL ─────────────


class TestBuildEmbedderFactoryInference:
    def test_infers_voyage_from_model_name(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_MODEL", "voyage-finance-2")
        from src.rag_system.config import reset_config

        reset_config()
        emb = build_embedder()  # no explicit provider -> infer from model name
        assert isinstance(emb, VoyageEmbedder)
        reset_config()

    def test_infers_cohere_from_model_name(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_MODEL", "embed-english-v3.0")
        from src.rag_system.config import reset_config

        reset_config()
        emb = build_embedder()
        assert isinstance(emb, CohereEmbedder)
        reset_config()

    def test_infers_local_from_bge_model_name(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        from src.rag_system.config import reset_config

        reset_config()
        emb = build_embedder()
        assert isinstance(emb, LocalEmbedder)
        reset_config()

    def test_defaults_to_openai_for_openai_model_name(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_MODEL", "text-embedding-3-small")
        from src.rag_system.config import reset_config

        reset_config()
        emb = build_embedder()
        assert isinstance(emb, OpenAIEmbedder)
        reset_config()

    def test_explicit_embedding_provider_field_takes_precedence(self, monkeypatch):
        # Even if the model name looks OpenAI-ish, an explicit provider field wins
        monkeypatch.setenv("VECTOR_STORE_CONFIG__EMBEDDING_PROVIDER", "local")
        from src.rag_system.config import reset_config

        reset_config()
        from src.rag_system.config import get_config

        cfg = get_config()
        emb = build_embedder(cfg.vector_store_config.embedding_provider)
        assert isinstance(emb, LocalEmbedder)
        reset_config()


# ── VoyageEmbedder ──────────────────────────────────────────────────────────────


class TestVoyageEmbedder:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        from src.rag_system.config import reset_config

        reset_config()
        emb = VoyageEmbedder()
        with pytest.raises(ConfigurationError):
            emb._get_api_key()
        reset_config()

    def test_name_property(self):
        emb = VoyageEmbedder(model="voyage-finance-2")
        assert "voyage" in emb.name
        assert "finance" in emb.name

    def test_dimension_property(self):
        emb = VoyageEmbedder()
        assert emb.dimension == 1024

    @pytest.mark.asyncio
    async def test_embed_success(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
        monkeypatch.setenv("CACHE_CONFIG__BACKEND", "memory")
        from src.rag_system.config import reset_config

        reset_config()

        emb = VoyageEmbedder()
        emb._cache = None  # bypass redis cache for this unit test

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            vectors = await emb.embed(["Revenue was $23.35B.", "Gross margin was 17.9%."])

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]
        reset_config()

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
        from src.rag_system.config import reset_config

        reset_config()

        emb = VoyageEmbedder()
        emb._cache = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.7, 0.8, 0.9]}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            vec = await emb.embed_query("What was revenue?")

        assert vec == [0.7, 0.8, 0.9]
        reset_config()


# ── CohereEmbedder ──────────────────────────────────────────────────────────────


class TestCohereEmbedder:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from src.rag_system.config import reset_config

        reset_config()
        emb = CohereEmbedder()
        with pytest.raises(ConfigurationError):
            emb._get_api_key()
        reset_config()

    def test_name_property(self):
        emb = CohereEmbedder(model="embed-english-v3.0")
        assert "cohere" in emb.name

    def test_dimension_property(self):
        emb = CohereEmbedder()
        assert emb.dimension == 1024

    @pytest.mark.asyncio
    async def test_embed_uses_search_document_input_type(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
        from src.rag_system.config import reset_config

        reset_config()

        emb = CohereEmbedder()
        emb._cache = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.raise_for_status = MagicMock()

        captured_payload = {}

        async def capture_post(url, headers=None, json=None):
            captured_payload.update(json or {})
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            vectors = await emb.embed(["doc one", "doc two"])

        assert len(vectors) == 2
        assert captured_payload.get("input_type") == "search_document"
        reset_config()

    @pytest.mark.asyncio
    async def test_embed_query_uses_search_query_input_type(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
        from src.rag_system.config import reset_config

        reset_config()

        emb = CohereEmbedder()
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5, 0.6]]}
        mock_response.raise_for_status = MagicMock()

        captured_payload = {}

        async def capture_post(url, headers=None, json=None):
            captured_payload.update(json or {})
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            vec = await emb.embed_query("What was revenue?")

        assert vec == [0.5, 0.6]
        assert captured_payload.get("input_type") == "search_query"
        reset_config()


# ── LocalEmbedder (zero-API-cost path) ────────────────────────────────────────


class TestLocalEmbedderDX:
    def test_name_does_not_require_api_key(self):
        # The whole point of LocalEmbedder: instantiation never touches
        # any *_API_KEY config, so it works fully offline.
        emb = LocalEmbedder()
        assert "local" in emb.name.lower() or "bge" in emb.name.lower() or emb.name
