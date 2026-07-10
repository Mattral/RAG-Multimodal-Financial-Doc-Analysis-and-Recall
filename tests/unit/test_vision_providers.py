"""Unit tests for vision provider adapters and fallback chain."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag_system.components.base import DocumentElement
from src.rag_system.components.vision.fallback_chain import (
    FallbackVisionDescriber,
)


class TestFallbackVisionDescriber:
    def _make_provider(self, name: str, returns: object = None) -> MagicMock:
        p = MagicMock()
        p.name = name
        p.describe = AsyncMock(return_value=returns)
        return p

    def _make_element(self) -> DocumentElement:
        return DocumentElement(
            type="graph",
            text="Revenue bar chart description.",
            source_document="tesla.pdf",
            page_number=3,
        )

    @pytest.mark.asyncio
    async def test_returns_primary_on_success(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        elem = self._make_element()
        primary = self._make_provider("primary", returns=elem)
        fallback = self._make_provider("fallback", returns=None)
        describer = FallbackVisionDescriber([primary, fallback])
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is not None
        primary.describe.assert_called_once()
        fallback.describe.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_returns_none(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        elem = self._make_element()
        primary = self._make_provider("primary", returns=None)
        fallback = self._make_provider("fallback", returns=elem)
        describer = FallbackVisionDescriber([primary, fallback])
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is not None
        assert result.text == elem.text
        fallback.describe.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_exception(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        elem = self._make_element()
        primary = self._make_provider("primary")
        primary.describe = AsyncMock(side_effect=Exception("API timeout"))
        fallback = self._make_provider("fallback", returns=elem)
        describer = FallbackVisionDescriber([primary, fallback])
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_fail(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        p1 = self._make_provider("p1", returns=None)
        p2 = self._make_provider("p2", returns=None)
        describer = FallbackVisionDescriber([p1, p2])
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is None

    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            FallbackVisionDescriber([])

    def test_name_shows_chain(self):
        p1 = self._make_provider("openai/gpt-4o")
        p2 = self._make_provider("gemini/gemini-2.5-flash")
        describer = FallbackVisionDescriber([p1, p2])
        assert "openai" in describer.name
        assert "gemini" in describer.name
        assert "→" in describer.name

    @pytest.mark.asyncio
    async def test_describe_batch_filters_none(self, tmp_path):
        imgs = [str(tmp_path / f"img{i}.png") for i in range(3)]
        for img in imgs:
            with open(img, "wb") as f:
                f.write(b"PNG")

        elem = self._make_element()
        provider = self._make_provider("p1")
        # Return element for first, None for rest
        provider.describe = AsyncMock(side_effect=[elem, None, elem])
        describer = FallbackVisionDescriber([provider])
        results = await describer.describe_batch(imgs, "tesla.pdf")
        assert len(results) == 2  # None filtered out


class TestLocalVLLMDescriber:
    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        from src.rag_system.components.vision.local_vllm_adapter import LocalVLLMDescriber

        describer = LocalVLLMDescriber(base_url="http://localhost:9999/v1")
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is None

    def test_name_property(self):
        from src.rag_system.components.vision.local_vllm_adapter import LocalVLLMDescriber

        d = LocalVLLMDescriber(model="Qwen/Qwen2-VL-7B-Instruct")
        assert "local_vllm" in d.name
        assert "Qwen2-VL-7B" in d.name


class TestGeminiVisionDescriber:
    @pytest.mark.asyncio
    async def test_returns_none_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        img = tmp_path / "chart.png"
        img.write_bytes(b"PNG")
        from src.rag_system.components.vision.gemini_adapter import GeminiVisionDescriber

        describer = GeminiVisionDescriber()
        result = await describer.describe(str(img), "tesla.pdf")
        assert result is None

    def test_name_property(self):
        from src.rag_system.components.vision.gemini_adapter import GeminiVisionDescriber

        d = GeminiVisionDescriber(model="gemini-2.5-flash")
        assert "gemini" in d.name
        assert "flash" in d.name
