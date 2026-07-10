"""Unit tests for multi-provider LLM generators and the build_generator factory.

Covers the DX fix: users can now select OpenAI, Gemini, Anthropic, or a fully
local/open-source model (via vLLM) for text generation with zero pipeline
code changes -- just LLM_CONFIG__PROVIDER in .env.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_system.components.base import RetrievedChunk
from src.rag_system.components.generator import (
    AnthropicGenerator,
    GeminiGenerator,
    LocalVLLMGenerator,
    OpenAIGenerator,
    _build_context_block,
    _is_complex_query,
    build_generator,
)
from src.rag_system.utils.exceptions import ConfigurationError


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            text="Q3 2023 revenue was $23.35B.",
            score=0.9,
            source_document="tesla.pdf",
            page_number=4,
        ),
    ]


# ── Factory tests ──────────────────────────────────────────────────────────────


class TestBuildGeneratorFactory:
    def test_openai_provider_resolves(self):
        gen = build_generator("openai")
        assert isinstance(gen, OpenAIGenerator)

    def test_gemini_provider_resolves(self):
        gen = build_generator("gemini")
        assert isinstance(gen, GeminiGenerator)

    def test_google_alias_resolves_to_gemini(self):
        gen = build_generator("google")
        assert isinstance(gen, GeminiGenerator)

    def test_anthropic_provider_resolves(self):
        gen = build_generator("anthropic")
        assert isinstance(gen, AnthropicGenerator)

    def test_claude_alias_resolves_to_anthropic(self):
        gen = build_generator("claude")
        assert isinstance(gen, AnthropicGenerator)

    def test_local_vllm_provider_resolves(self):
        gen = build_generator("local_vllm")
        assert isinstance(gen, LocalVLLMGenerator)

    def test_local_alias_resolves_to_local_vllm(self):
        gen = build_generator("local")
        assert isinstance(gen, LocalVLLMGenerator)

    def test_together_alias_resolves_to_local_vllm_protocol(self):
        # Together.ai exposes an OpenAI-compatible endpoint, same wire protocol
        gen = build_generator("together")
        assert isinstance(gen, LocalVLLMGenerator)

    def test_unknown_provider_falls_back_to_openai(self):
        gen = build_generator("some_unknown_provider_xyz")
        assert isinstance(gen, OpenAIGenerator)

    def test_case_insensitive(self):
        gen = build_generator("GEMINI")
        assert isinstance(gen, GeminiGenerator)

    def test_no_provider_arg_reads_from_config(self):
        # Falls through to cfg.provider (default "openai")
        gen = build_generator()
        assert isinstance(gen, OpenAIGenerator)


# ── Shared helper function tests ──────────────────────────────────────────────


class TestComplexQueryHeuristic:
    @pytest.mark.parametrize(
        "query",
        [
            "What was the CAGR from 2020 to 2023?",
            "Calculate the gross margin percentage",
            "Compare revenue versus last year",
            "What is the EBITDA growth rate?",
        ],
    )
    def test_numerical_queries_detected(self, query):
        assert _is_complex_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "Who is the CEO?",
            "When was the company founded?",
            "What is mentioned in the risk factors section?",
        ],
    )
    def test_simple_queries_not_flagged(self, query):
        assert _is_complex_query(query) is False


class TestBuildContextBlock:
    def test_includes_source_and_page(self, sample_chunks):
        block = _build_context_block(sample_chunks)
        assert "tesla.pdf" in block
        assert "Page 4" in block
        assert "$23.35B" in block

    def test_empty_chunks_returns_empty_string(self):
        assert _build_context_block([]) == ""

    def test_multiple_chunks_separated(self):
        chunks = [
            RetrievedChunk(text="A", score=0.9, source_document="a.pdf", page_number=1),
            RetrievedChunk(text="B", score=0.8, source_document="b.pdf", page_number=2),
        ]
        block = _build_context_block(chunks)
        assert "Source 1" in block
        assert "Source 2" in block


# ── OpenAIGenerator tests (mocked HTTP) ────────────────────────────────────────


class TestOpenAIGenerator:
    @pytest.mark.asyncio
    async def test_generate_success(self, sample_chunks, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config

        reset_config()

        gen = OpenAIGenerator()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Revenue was $23.35B [Source: tesla.pdf, Page 4]."}}
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await gen.generate("What was revenue?", sample_chunks, tenant_id="test")

        assert "23.35B" in result.answer
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 40
        assert result.estimated_cost_usd >= 0
        reset_config()

    def test_name_property(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config

        reset_config()
        gen = OpenAIGenerator()
        assert "openai" in gen.name
        reset_config()


# ── GeminiGenerator tests ──────────────────────────────────────────────────────


class TestGeminiGenerator:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        gen = GeminiGenerator()
        with pytest.raises(ConfigurationError):
            gen._get_api_key()

    @pytest.mark.asyncio
    async def test_generate_success(self, sample_chunks, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        gen = GeminiGenerator()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Revenue was $23.35B."}]}}],
            "usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 30},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await gen.generate("What was revenue?", sample_chunks, tenant_id="test")

        assert "23.35B" in result.answer
        assert result.prompt_tokens == 150

    def test_name_property(self):
        gen = GeminiGenerator()
        assert "gemini" in gen.name

    def test_defaults_to_flash_when_model_not_gemini(self, monkeypatch):
        # If LLM_CONFIG__MODEL is left as the OpenAI default, Gemini generator
        # should still pick a sane Gemini default rather than sending "gpt-4o-mini"
        gen = GeminiGenerator()
        assert "gemini" in gen._default_model


# ── AnthropicGenerator tests ───────────────────────────────────────────────────


class TestAnthropicGenerator:
    def test_missing_api_key_raises(self):
        from src.rag_system.config import reset_config

        reset_config()
        gen = AnthropicGenerator()
        with pytest.raises(ConfigurationError):
            gen._get_api_key()
        reset_config()

    @pytest.mark.asyncio
    async def test_generate_success(self, sample_chunks, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from src.rag_system.config import reset_config

        reset_config()

        gen = AnthropicGenerator()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Revenue was $23.35B."}],
            "usage": {"input_tokens": 180, "output_tokens": 25},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await gen.generate("What was revenue?", sample_chunks, tenant_id="test")

        assert "23.35B" in result.answer
        assert result.prompt_tokens == 180
        reset_config()

    def test_name_property(self):
        gen = AnthropicGenerator()
        assert "anthropic" in gen.name


# ── LocalVLLMGenerator tests ───────────────────────────────────────────────────


class TestLocalVLLMGenerator:
    def test_name_property(self):
        gen = LocalVLLMGenerator(base_url="http://localhost:8090/v1")
        assert "local_vllm" in gen.name

    def test_default_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_VLLM_GENERATOR_BASE_URL", "http://gpu-box:9000/v1")
        gen = LocalVLLMGenerator()
        assert gen._base_url == "http://gpu-box:9000/v1"

    @pytest.mark.asyncio
    async def test_generate_success(self, sample_chunks):
        gen = LocalVLLMGenerator(base_url="http://localhost:8090/v1")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Revenue was $23.35B."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await gen.generate("What was revenue?", sample_chunks, tenant_id="test")

        assert "23.35B" in result.answer
        # Local inference is tracked at zero marginal API cost
        assert result.estimated_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_connection_error_raises_with_helpful_log(self, sample_chunks):
        import httpx

        gen = LocalVLLMGenerator(base_url="http://localhost:19999/v1")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await gen.generate("test", sample_chunks)
