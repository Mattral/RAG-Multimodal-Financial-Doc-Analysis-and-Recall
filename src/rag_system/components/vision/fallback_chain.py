"""Vision provider fallback chain.

Guideline §7: 'Fallback chain: primary_provider → fallback_providers → None (skip image)'

Usage (config-driven):
    VISION_CONFIG__PROVIDER=qwen2-vl
    # Falls back to gemini, then openai, then skips image on all failures

Provides FallbackVisionDescriber that tries providers in order and returns
the first successful description.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import structlog

from src.rag_system.components.base import BaseVisionDescriber, DocumentElement

logger = structlog.get_logger(__name__)


class FallbackVisionDescriber(BaseVisionDescriber):
    """Try vision providers in order; return first success or None.

    This is the production-recommended configuration for robustness:
        primary: qwen2-vl (cheap, private)
        fallback[0]: gemini-flash (cheap, cloud)
        fallback[1]: openai (expensive, highest quality)

    If all providers fail, returns None (image is skipped gracefully).
    """

    def __init__(self, providers: List[BaseVisionDescriber]) -> None:
        if not providers:
            raise ValueError("FallbackVisionDescriber requires at least one provider")
        self._providers = providers

    @property
    def name(self) -> str:
        return "fallback_chain[" + " → ".join(p.name for p in self._providers) + "]"

    async def describe(
        self,
        image_path: str,
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[DocumentElement]:
        last_error: Optional[str] = None
        for provider in self._providers:
            try:
                result = await provider.describe(image_path, source_document, tenant_id)
                if result is not None:
                    if provider.name != self._providers[0].name:
                        logger.info(
                            "vision_fallback_used",
                            primary=self._providers[0].name,
                            used=provider.name,
                        )
                    return result
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "vision_provider_failed",
                    provider=provider.name,
                    error=str(exc)[:120],
                    trying_next=True,
                )
                continue

        logger.error(
            "all_vision_providers_failed",
            image=image_path,
            providers=[p.name for p in self._providers],
            last_error=last_error,
        )
        return None  # Skip gracefully

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


def build_vision_fallback_chain(
    primary: str = "openai",
    fallbacks: Optional[List[str]] = None,
    **kwargs,
) -> FallbackVisionDescriber:
    """Build a fallback chain from provider names.

    Args:
        primary: Primary provider name
        fallbacks: List of fallback provider names (in order)
        **kwargs: Passed to provider constructors

    Returns:
        FallbackVisionDescriber wrapping all providers in order
    """
    from src.rag_system.components.vision import (
        OpenAIVisionDescriber,
        Qwen2VLDescriber,
    )
    from src.rag_system.components.vision.gemini_adapter import GeminiVisionDescriber
    from src.rag_system.components.vision.local_vllm_adapter import LocalVLLMDescriber

    provider_map = {
        "openai": lambda: OpenAIVisionDescriber(),
        "gemini": lambda: GeminiVisionDescriber(),
        "qwen2-vl": lambda: Qwen2VLDescriber(),
        "local_vllm": lambda: LocalVLLMDescriber(
            base_url=kwargs.get("local_vllm_base_url", "http://localhost:8080/v1"),
            model=kwargs.get("model", "Qwen/Qwen2-VL-7B-Instruct"),
        ),
    }

    all_providers = [primary] + (fallbacks or [])
    resolved_providers = []
    for name in all_providers:
        factory = provider_map.get(name)
        if factory:
            try:
                resolved_providers.append(factory())
            except Exception as exc:
                logger.warning("vision_provider_init_failed", provider=name, error=str(exc))
        else:
            logger.warning("unknown_vision_provider", provider=name)

    if not resolved_providers:
        resolved_providers = [OpenAIVisionDescriber()]  # Safe default

    return FallbackVisionDescriber(resolved_providers)
