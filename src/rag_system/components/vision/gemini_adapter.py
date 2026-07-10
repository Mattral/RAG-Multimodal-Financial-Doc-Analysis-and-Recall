"""Google Gemini vision adapter (2.5 Flash/Pro) — generous free tier, strong document understanding.

Guideline §7: 'Google Gemini Flash and Pro — excellent free or cheap alternatives.'
(Guideline originally named the 2.0 series; 2.0 was retired by Google in favor of
2.5/3.x — this adapter tracks whichever Gemini generation is currently GA.)

Usage: set VISION_CONFIG__PROVIDER=gemini and GOOGLE_API_KEY in .env
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import httpx
import structlog

from src.rag_system.components.base import BaseVisionDescriber, DocumentElement
from src.rag_system.components.vision import FINANCIAL_CHART_PROMPT
from src.rag_system.utils.cost_tracker import get_cost_tracker

logger = structlog.get_logger(__name__)

# Gemini pricing (per 1M tokens, mid-2025 approximate)
_GEMINI_PRICING = {
    "gemini-2.5-flash": {"prompt": 0.15, "completion": 0.60},
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-3.5-flash": {"prompt": 0.15, "completion": 0.60},
    "gemini-3.1-flash-lite": {"prompt": 0.05, "completion": 0.20},
}


class GeminiVisionDescriber(BaseVisionDescriber):
    """Google Gemini vision adapter for financial chart extraction.

    Models supported: gemini-2.5-flash (recommended default), gemini-2.5-pro.
    Note: gemini-2.0-flash and gemini-1.5-* are retired — do not use.
    Uses Google AI Studio API (not Vertex) for simpler auth.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1500,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._cost_tracker = get_cost_tracker()

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def _get_api_key(self) -> str:
        import os

        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        return key

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def describe(
        self, image_path: str, source_document: str, tenant_id: Optional[str] = None
    ) -> Optional[DocumentElement]:
        try:
            api_key = self._get_api_key()
            image_b64 = await asyncio.to_thread(self._encode_image, image_path)

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self._model}:generateContent?key={api_key}"
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": FINANCIAL_CHART_PROMPT},
                            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self._temperature,
                    "maxOutputTokens": self._max_tokens,
                },
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            description = data["candidates"][0]["content"]["parts"][0]["text"]

            # Approximate cost tracking (Gemini doesn't always return usage)
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 500)
            completion_tokens = usage.get("candidatesTokenCount", 200)
            pricing = _GEMINI_PRICING.get(self._model, {"prompt": 0.10, "completion": 0.40})
            cost_usd = (
                prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
            ) / 1_000_000

            self._cost_tracker.record(
                tenant_id=tenant_id or "default",
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            logger.info(
                "gemini_vision_complete",
                model=self._model,
                image=Path(image_path).name,
                cost_usd=round(cost_usd, 6),
            )

            return DocumentElement(
                type="graph",
                text=description,
                source_document=source_document,
                ingest_timestamp=datetime.now(UTC).isoformat(),
                content_hash=hashlib.sha256(description.encode()).hexdigest()[:12],
                tenant_id=tenant_id,
                metadata={
                    "vision_model": self._model,
                    "image_path": image_path,
                    "cost_usd": cost_usd,
                },
            )

        except Exception as exc:
            logger.error("gemini_vision_failed", image=image_path, error=str(exc))
            return None

    async def describe_batch(
        self, image_paths: List[str], source_document: str, tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        sem = asyncio.Semaphore(4)  # Gemini allows higher concurrency

        async def _describe(path: str) -> Optional[DocumentElement]:
            async with sem:
                return await self.describe(path, source_document, tenant_id)

        results = await asyncio.gather(*[_describe(p) for p in image_paths])
        return [r for r in results if r is not None]
