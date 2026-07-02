"""Multi-provider LLM generator with cost-based routing and fallback.

Providers (all implement BaseGenerator, switch via config — zero code changes):
  openai      -> OpenAIGenerator      (GPT-4o-mini / GPT-4o, default)
  gemini      -> GeminiGenerator      (Gemini 2.5 Flash / Pro, 5-30x cheaper)
  anthropic   -> AnthropicGenerator   (Claude 3.5 Sonnet / Haiku)
  local_vllm  -> LocalVLLMGenerator   (any HF model via vLLM, fully private)

    LLM_CONFIG__PROVIDER=gemini
    LLM_CONFIG__MODEL=gemini-2.5-flash

Every provider:
  - Routes simple queries to a cheap model and numerical/analytical queries
    to a stronger model (LLM_CONFIG__ENABLE_MODEL_ROUTING).
  - Retries once on a transient/fallback-eligible failure.
  - Records token usage and cost to the shared CostTracker.
  - Returns the same GeneratedAnswer shape regardless of provider.
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

# ── Shared prompt + heuristics ────────────────────────────────────────────────

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
        parts.append(f"[Source {i}: {chunk.source_document}{page}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


# ── OpenAI ─────────────────────────────────────────────────────────────────────

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
        cfg = get_config()
        api_key = cfg.get_openai_key()

        model = self._cfg.complex_query_model if (
            self._cfg.enable_model_routing and _is_complex_query(query)
        ) else self._cfg.model

        context_block = _build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                        headers=headers, json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as exc:
                if self._cfg.fallback_model and exc.response.status_code in (429, 503):
                    logger.warning("llm_fallback_triggered", model=model, status=exc.response.status_code)
                    payload["model"] = self._cfg.fallback_model
                    async with httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client:
                        response = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers, json=payload,
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
            tenant_id=tenant_id or "default", model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text, citations=context, model_used=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated_cost_usd=cost_record.cost_usd, latency_ms=latency_ms,
            tenant_id=tenant_id,
        )


# ── Google Gemini ────────────────────────────────────────────────────────────

class GeminiGenerator(BaseGenerator):
    """Google Gemini generator — 10-40x cheaper than GPT-4o, generous free tier.

    Set GOOGLE_API_KEY in .env. Model defaults to gemini-2.5-flash (stable GA);
    set LLM_CONFIG__COMPLEX_QUERY_MODEL=gemini-2.5-pro for harder queries.

    Note: gemini-2.0-flash and the gemini-1.5-* family were retired by Google
    in favor of the 2.5 and 3.x generations. Pricing below reflects 2.5-series
    rates — verify current pricing at ai.google.dev/gemini-api/docs/pricing
    before relying on these numbers for budgeting, as Google updates pricing
    independently of this codebase.
    """

    _PRICING = {
        "gemini-2.5-flash":      {"prompt": 0.15, "completion": 0.60},
        "gemini-2.5-pro":        {"prompt": 1.25, "completion": 5.00},
        "gemini-3.5-flash":      {"prompt": 0.15, "completion": 0.60},
        "gemini-3.1-flash-lite": {"prompt": 0.05, "completion": 0.20},
    }

    def __init__(self) -> None:
        self._cfg = get_config().llm_config
        self._cost_tracker = get_cost_tracker()
        # If the user hasn't overridden the model away from the OpenAI default,
        # use a sensible Gemini default instead.
        self._default_model = (
            self._cfg.model if "gemini" in self._cfg.model else "gemini-2.5-flash"
        )
        self._complex_model = (
            self._cfg.complex_query_model
            if "gemini" in self._cfg.complex_query_model
            else "gemini-2.5-pro"
        )

    @property
    def name(self) -> str:
        return f"gemini/{self._default_model}"

    def _get_api_key(self) -> str:
        import os
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            from src.rag_system.utils.exceptions import ConfigurationError
            raise ConfigurationError(
                "GOOGLE_API_KEY not set — required for LLM_CONFIG__PROVIDER=gemini",
                config_key="GOOGLE_API_KEY",
            )
        return key

    async def generate(
        self,
        query: str,
        context: List[RetrievedChunk],
        tenant_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        api_key = self._get_api_key()
        model = self._complex_model if (
            self._cfg.enable_model_routing and _is_complex_query(query)
        ) else self._default_model

        context_block = _build_context_block(context)
        prompt = (
            f"{system_prompt or FINANCIAL_RAG_SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}"
        )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self._cfg.temperature,
                "maxOutputTokens": self._cfg.max_tokens,
            },
        }

        start = time.perf_counter()
        async with (
            async_trace_span("llm_generation", {"model": model, "tenant_id": tenant_id or ""}),
            httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client,
        ):
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        # Cost tracker uses a fixed internal pricing table; record raw tokens
        # and also compute Gemini-specific cost directly since the shared
        # tracker may not know Gemini pricing.
        pricing = self._PRICING.get(model, {"prompt": 0.10, "completion": 0.40})
        cost_usd = (
            prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
        ) / 1_000_000

        self._cost_tracker.record(
            tenant_id=tenant_id or "default", model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text, citations=context, model_used=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated_cost_usd=cost_usd, latency_ms=latency_ms,
            tenant_id=tenant_id,
        )


# ── Anthropic Claude ──────────────────────────────────────────────────────────

class AnthropicGenerator(BaseGenerator):
    """Anthropic Claude generator — strong reasoning, large context window."""

    _PRICING = {
        "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
        "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.00},
        "claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
    }

    def __init__(self) -> None:
        self._cfg = get_config().llm_config
        self._cost_tracker = get_cost_tracker()
        self._default_model = (
            self._cfg.model if "claude" in self._cfg.model
            else "claude-3-5-haiku-20241022"
        )
        self._complex_model = (
            self._cfg.complex_query_model if "claude" in self._cfg.complex_query_model
            else "claude-3-5-sonnet-20241022"
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self._default_model}"

    def _get_api_key(self) -> str:
        cfg = get_config()
        if not cfg.anthropic_api_key:
            from src.rag_system.utils.exceptions import ConfigurationError
            raise ConfigurationError(
                "ANTHROPIC_API_KEY not set — required for LLM_CONFIG__PROVIDER=anthropic",
                config_key="ANTHROPIC_API_KEY",
            )
        return cfg.anthropic_api_key.get_secret_value()

    async def generate(
        self,
        query: str,
        context: List[RetrievedChunk],
        tenant_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        api_key = self._get_api_key()
        model = self._complex_model if (
            self._cfg.enable_model_routing and _is_complex_query(query)
        ) else self._default_model

        context_block = _build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "system": system_prompt or FINANCIAL_RAG_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

        start = time.perf_counter()
        async with (
            async_trace_span("llm_generation", {"model": model, "tenant_id": tenant_id or ""}),
            httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client,
        ):
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload,
            )
            response.raise_for_status()
            data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        answer_text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        pricing = self._PRICING.get(model, {"prompt": 3.00, "completion": 15.00})
        cost_usd = (
            prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
        ) / 1_000_000

        self._cost_tracker.record(
            tenant_id=tenant_id or "default", model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text, citations=context, model_used=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated_cost_usd=cost_usd, latency_ms=latency_ms,
            tenant_id=tenant_id,
        )


# ── Local vLLM (open-source, fully private) ───────────────────────────────────

class LocalVLLMGenerator(BaseGenerator):
    """Generic generator for any open-source LLM served via vLLM's OpenAI-compatible API.

    Zero data leaves your infrastructure — best for regulated environments
    that cannot send financial documents to any external API.

    Supported models (examples):
        meta-llama/Llama-3.1-8B-Instruct
        meta-llama/Llama-3.1-70B-Instruct
        Qwen/Qwen2.5-72B-Instruct
        mistralai/Mistral-Large-Instruct-2407

    Startup:
        pip install vllm
        vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8090 --host 0.0.0.0

    Config:
        LLM_CONFIG__PROVIDER=local_vllm
        LOCAL_VLLM_GENERATOR_BASE_URL=http://localhost:8090/v1
        LLM_CONFIG__MODEL=meta-llama/Llama-3.1-8B-Instruct
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "local",
    ) -> None:
        import os
        self._cfg = get_config().llm_config
        self._cost_tracker = get_cost_tracker()
        self._base_url = (
            base_url
            or os.environ.get("LOCAL_VLLM_GENERATOR_BASE_URL", "http://localhost:8090/v1")
        ).rstrip("/")
        self._api_key = api_key

    @property
    def name(self) -> str:
        return f"local_vllm/{self._cfg.model}"

    async def generate(
        self,
        query: str,
        context: List[RetrievedChunk],
        tenant_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        context_block = _build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt or FINANCIAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
        }

        start = time.perf_counter()
        try:
            async with async_trace_span(
                "llm_generation", {"model": self._cfg.model, "tenant_id": tenant_id or ""}
            ), httpx.AsyncClient(timeout=self._cfg.timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            logger.error(
                "local_vllm_generator_connection_failed",
                base_url=self._base_url,
                hint=f"Start vLLM: vllm serve {self._cfg.model} --host 0.0.0.0 --port 8090",
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        answer_text = data["choices"][0]["message"]["content"]

        # Local inference: near-zero marginal cost (infra cost only).
        self._cost_tracker.record(
            tenant_id=tenant_id or "default", model=self._cfg.model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text, citations=context, model_used=self._cfg.model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated_cost_usd=0.0, latency_ms=latency_ms,
            tenant_id=tenant_id,
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def build_generator(provider: Optional[str] = None) -> BaseGenerator:
    """Factory: build a text generator by provider name.

    Reads LLM_CONFIG__PROVIDER from config if provider is not passed explicitly.
    This is the single switch that lets users choose OpenAI, Gemini, Anthropic,
    or a fully local/open-source model with zero pipeline code changes.
    """
    cfg = get_config().llm_config
    name = (provider or cfg.provider).lower()

    providers = {
        "openai": OpenAIGenerator,
        "azure_openai": OpenAIGenerator,  # same wire protocol; point base URL via env if needed
        "gemini": GeminiGenerator,
        "google": GeminiGenerator,
        "anthropic": AnthropicGenerator,
        "claude": AnthropicGenerator,
        "local": LocalVLLMGenerator,
        "local_vllm": LocalVLLMGenerator,
        "together": LocalVLLMGenerator,  # Together exposes an OpenAI-compatible endpoint too
    }

    generator_cls = providers.get(name)
    if generator_cls is None:
        logger.warning(
            "unknown_llm_provider", provider=name, fallback="openai",
            available=sorted(set(providers.keys())),
        )
        generator_cls = OpenAIGenerator

    logger.info("generator_provider_selected", provider=name, resolved_class=generator_cls.__name__)
    return generator_cls()
