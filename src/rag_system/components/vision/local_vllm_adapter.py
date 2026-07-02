"""Local vLLM vision adapter — any HuggingFace vision-language model served via vLLM.

Guideline §7: 'LocalVLLMDescriber (generic for any HF model served with vLLM)'
+ 'Private inference keeps data inside VPC or air-gapped environment.'

Supported models (examples):
- Qwen/Qwen2-VL-7B-Instruct
- Qwen/Qwen2-VL-72B-Instruct
- mistralai/Pixtral-12B-2409
- InternVL2-8B
- meta-llama/Llama-3.2-11B-Vision-Instruct

Startup:
    pip install vllm
    vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8080 --host 0.0.0.0

Config:
    VISION_CONFIG__PROVIDER=local_vllm
    LOCAL_VLLM_BASE_URL=http://localhost:8080/v1
    VISION_CONFIG__MODEL=Qwen/Qwen2-VL-7B-Instruct
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


class LocalVLLMDescriber(BaseVisionDescriber):
    """Generic vision adapter for any model served via vLLM OpenAI-compatible API.

    Zero data leaves your infrastructure — fully private inference.
    Drop-in replacement for OpenAIVisionDescriber via config change.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_tokens: int = 1500,
        temperature: float = 0.1,
        timeout: int = 180,
        api_key: str = "local",  # vLLM accepts any key
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._api_key = api_key
        self._cost_tracker = get_cost_tracker()
        logger.info(
            "local_vllm_describer_created",
            model=model,
            base_url=base_url,
            note="Data stays on-premise — no external API calls",
        )

    @property
    def name(self) -> str:
        return f"local_vllm/{self._model}"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def describe(
        self,
        image_path: str,
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[DocumentElement]:
        try:
            image_b64 = await asyncio.to_thread(self._encode_image, image_path)
            payload = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FINANCIAL_CHART_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }],
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            description = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 300)
            completion_tokens = usage.get("completion_tokens", 200)

            # Local models have near-zero marginal cost — track as $0 unless pricing configured
            self._cost_tracker.record(
                tenant_id=tenant_id or "default",
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            logger.info(
                "local_vllm_vision_complete",
                model=self._model,
                image=Path(image_path).name,
                tokens=prompt_tokens + completion_tokens,
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
                    "vision_backend": "local_vllm",
                    "image_path": image_path,
                    "private_inference": True,
                },
            )

        except httpx.ConnectError:
            logger.error(
                "local_vllm_connection_failed",
                base_url=self._base_url,
                hint=f"Start vLLM: vllm serve {self._model} --host 0.0.0.0 --port 8080",
            )
            return None
        except Exception as exc:
            logger.error("local_vllm_describe_failed", image=image_path, error=str(exc))
            return None

    async def describe_batch(
        self,
        image_paths: List[str],
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentElement]:
        sem = asyncio.Semaphore(2)  # conservative for local GPU

        async def _describe(path: str) -> Optional[DocumentElement]:
            async with sem:
                return await self.describe(path, source_document, tenant_id)

        results = await asyncio.gather(*[_describe(p) for p in image_paths])
        return [r for r in results if r is not None]
