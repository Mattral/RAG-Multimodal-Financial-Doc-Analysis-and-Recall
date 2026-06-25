"""Vision/multimodal describer implementations.

Providers:
  openai      → OpenAIVisionDescriber    (GPT-4o, highest accuracy)
  gemini      → GeminiVisionDescriber    (Gemini 2.5 Flash, cheapest cloud)
  qwen2-vl    → Qwen2VLDescriber         (open-source, via Together.ai)
  local_vllm  → LocalVLLMDescriber       (any HF model via vLLM, fully private)

Fallback chain:
  FallbackVisionDescriber wraps multiple providers in priority order.

All implement BaseVisionDescriber — switch via config with zero code changes:
    VISION_CONFIG__PROVIDER=gemini
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import structlog

from src.rag_system.components.base import BaseVisionDescriber, DocumentElement
from src.rag_system.config import get_config
from src.rag_system.utils.cost_tracker import get_cost_tracker
from src.rag_system.utils.telemetry import async_trace_span

logger = structlog.get_logger(__name__)

FINANCIAL_CHART_PROMPT = (
    "You are analyzing a chart or figure from a financial report (e.g. 10-K, earnings release). "
    "Extract ALL of the following with precision:\n"
    "1. Chart type (bar, line, pie, table, etc.)\n"
    "2. Title and subtitle\n"
    "3. All axis labels and units\n"
    "4. All data values (exact numbers, percentages, dates)\n"
    "5. Legend entries\n"
    "6. Key trends, peaks, troughs\n"
    "7. Any footnotes or source attributions\n"
    "Be exhaustive — every number matters for financial accuracy."
)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OpenAIVisionDescriber(BaseVisionDescriber):
    """GPT-4o vision — highest accuracy for complex financial documents."""

    def __init__(self) -> None:
        self._cfg = get_config().vision_config
        self._cost_tracker = get_cost_tracker()

    @property
    def name(self) -> str:
        return f"openai/{self._cfg.model}"

    async def describe(
        self,
        image_path: str,
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[DocumentElement]:
        async with async_trace_span("vision_describe", {"model": self._cfg.model}):
            try:
                cfg = get_config()
                api_key = cfg.get_openai_key()
                image_b64 = await asyncio.to_thread(_encode_image, image_path)
                payload = {
                    "model": self._cfg.model,
                    "max_tokens": self._cfg.max_tokens,
                    "temperature": self._cfg.temperature,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FINANCIAL_CHART_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": self._cfg.detail_level,
                            }},
                        ],
                    }],
                }
                for attempt in range(self._cfg.retry_max_attempts):
                    try:
                        async with httpx.AsyncClient(
                            timeout=self._cfg.timeout_seconds
                        ) as client:
                            resp = await client.post(
                                "https://api.openai.com/v1/chat/completions",
                                headers={"Authorization": f"Bearer {api_key}"},
                                json=payload,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                        break
                    except httpx.HTTPStatusError as exc:
                        if (
                            exc.response.status_code == 429
                            and attempt < self._cfg.retry_max_attempts - 1
                        ):
                            await asyncio.sleep(self._cfg.retry_backoff_factor ** attempt)
                        else:
                            raise

                description = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                self._cost_tracker.record(
                    tenant_id=tenant_id or "default",
                    model=self._cfg.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                return DocumentElement(
                    type="graph",
                    text=description,
                    source_document=source_document,
                    ingest_timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash=hashlib.sha256(description.encode()).hexdigest()[:12],
                    tenant_id=tenant_id,
                    metadata={"vision_model": self._cfg.model, "image_path": image_path},
                )
            except Exception as exc:
                logger.error("vision_describe_failed", image=image_path, error=str(exc))
                return None

    async def describe_batch(
        self,
        image_paths: List[str],
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentElement]:
        sem = asyncio.Semaphore(4)

        async def _describe_with_sem(path: str) -> Optional[DocumentElement]:
            async with sem:
                return await self.describe(path, source_document, tenant_id)

        results = await asyncio.gather(*[_describe_with_sem(p) for p in image_paths])
        return [r for r in results if r is not None]


class Qwen2VLDescriber(BaseVisionDescriber):
    """Qwen2-VL via Together.ai — open-source, 5–20× cheaper, private inference."""

    def __init__(
        self,
        base_url: str = "https://api.together.xyz/v1",
        model: str = "Qwen/Qwen2-VL-72B-Instruct",
    ) -> None:
        self._base_url = base_url
        self._model = model

    @property
    def name(self) -> str:
        return f"qwen2vl/{self._model}"

    async def describe(
        self,
        image_path: str,
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[DocumentElement]:
        import os
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            logger.warning("TOGETHER_API_KEY not set — skipping Qwen2-VL")
            return None
        try:
            image_b64 = await asyncio.to_thread(_encode_image, image_path)
            payload = {
                "model": self._model,
                "max_tokens": 1500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FINANCIAL_CHART_PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }],
            }
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                description = resp.json()["choices"][0]["message"]["content"]
            return DocumentElement(
                type="graph", text=description, source_document=source_document,
                ingest_timestamp=datetime.now(timezone.utc).isoformat(),
                content_hash=hashlib.sha256(description.encode()).hexdigest()[:12],
                tenant_id=tenant_id,
                metadata={"vision_model": self._model, "image_path": image_path},
            )
        except Exception as exc:
            logger.error("qwen2vl_describe_failed", image=image_path, error=str(exc))
            return None

    async def describe_batch(
        self,
        image_paths: List[str],
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentElement]:
        results = await asyncio.gather(
            *[self.describe(p, source_document, tenant_id) for p in image_paths]
        )
        return [r for r in results if r is not None]


def build_vision_describer(
    provider: Optional[str] = None,
    use_fallback_chain: bool = True,
) -> BaseVisionDescriber:
    """Factory: build vision describer by provider name.

    With use_fallback_chain=True (default), wraps in FallbackVisionDescriber
    so failures automatically try the next provider.
    """
    from src.rag_system.components.vision.gemini_adapter import GeminiVisionDescriber
    from src.rag_system.components.vision.local_vllm_adapter import LocalVLLMDescriber
    from src.rag_system.components.vision.fallback_chain import FallbackVisionDescriber

    cfg = get_config().vision_config
    name = provider or cfg.provider

    _PROVIDERS = {
        "openai": OpenAIVisionDescriber,
        "gemini": GeminiVisionDescriber,
        "qwen2-vl": Qwen2VLDescriber,
        "qwen2vl": Qwen2VLDescriber,
        "local_vllm": LocalVLLMDescriber,
    }

    primary_cls = _PROVIDERS.get(name, OpenAIVisionDescriber)
    primary = primary_cls()

    if use_fallback_chain and cfg.fallback_model:
        # Build a simple 2-provider chain: primary → openai fallback
        fallback = OpenAIVisionDescriber()
        if name != "openai" and fallback.name != primary.name:
            return FallbackVisionDescriber([primary, fallback])

    return primary
