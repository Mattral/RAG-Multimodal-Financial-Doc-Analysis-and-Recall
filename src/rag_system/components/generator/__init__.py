"""Multi-provider LLM generator with cost-based routing and fallback.

Supports: OpenAI, Anthropic, Azure OpenAI, Together.ai (open models).
Routes complex/numerical queries to high-capability models automatically.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

import httpx
import structlog

from src.rag_system.components.base import BaseGenerator, GeneratedAnswer, RetrievedChunk
from src.rag_system.config import get_config
from src.rag_system.utils.cost_tracker import get_cost_tracker
from src.rag_system.utils.telemetry import async_trace_span

logger = structlog.get_logger(__name__)

# Prompt template
FINANCIAL_RAG_SYSTEM_PROMPT = """\
You are an expert financial analyst AI assistant. Answer the user's question
STRICTLY based on the provided source passages. Follow these rules:

1. ONLY use information explicitly present in the source passages.
2. If the answer is not in the sources, say "This information is not available in the provided documents."
3. For numerical claims, ALWAYS include the exact figure from the source.
4. ALWAYS cite your sources with [Source: <document_name>, Page <N>] after each claim.
5. NEVER extrapolate, estimate, or invent numbers.
6. Format monetary values consistently (e.g., $42.3M, not 42.3 million).
"""

_NUMERICAL_QUERY_RE = re.compile(
    r"\b(calculat|cagr|growth rate|revenue|eps|ebitda|margin|return|ratio|"
    r"percent|increase|decrease|compare|versus|yoy|qoq)\b",
    re.I,
)


def _is_complex_query(query: str) -> bool:
    """Heuristic: route to complex model for numerical/analytical queries."""
    return bool(_NUMERICAL_QUERY_RE.search(query))


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = f", Page {chunk.page_number}" if chunk.page_number else ""
        parts.append(
            f"[Source {i}: {chunk.source_document}{page}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


class OpenAIGenerator(BaseGenerator):
    """OpenAI-based generator with gpt-4o / gpt-4o-mini routing."""

    def __init__(self) -> None:
        self._cfg = get_config().llm_config
        self._cost_tracker = get_cost_tracker()

    @property
    def name(self) -> str:
        return f"openai/{self._cfg.model}"

    async def generate(
        self,
        query: str,
        context: List[RetrievedChunk],
        tenant_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        from src.rag_system.config import get_config

        cfg = get_config()
        api_key = cfg.get_openai_key()

        # Model routing
        if self._cfg.enable_model_routing and _is_complex_query(query):
            model = self._cfg.complex_query_model
        else:
            model = self._cfg.model

        context_block = _build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or FINANCIAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
        }

        start = time.perf_counter()
        async with async_trace_span("llm_generation", {"model": model, "tenant_id": tenant_id or ""}):
            try:
                async with httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as exc:
                # Try fallback model
                if self._cfg.fallback_model and exc.response.status_code in (429, 503):
                    logger.warning("llm_fallback_triggered", model=model, status=exc.response.status_code)
                    payload["model"] = self._cfg.fallback_model
                    async with httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client:
                        response = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()
                        model = self._cfg.fallback_model
                else:
                    raise

        latency_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        answer_text = data["choices"][0]["message"]["content"]

        cost_record = self._cost_tracker.record(
            tenant_id=tenant_id or "default",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=context,
            model_used=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost_record.cost_usd,
            latency_ms=latency_ms,
            tenant_id=tenant_id,
        )
