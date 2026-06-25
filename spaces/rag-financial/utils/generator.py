"""utils/generator.py — Multi-provider LLM generation with transparent cost tracking.

Supports: OpenAI (GPT-4o, GPT-4o-mini) and Google Gemini (2.5 Flash, 2.5 Pro).

Model currency note (v2.0): Google retired gemini-2.0-flash and the
gemini-1.5-* family in favor of the 2.5 and 3.x generations. This module
defaults to gemini-2.5-flash (stable GA as documented in the official
google-genai SDK). If Google ships a newer stable default before this
code is next updated, override via the model dropdown in app.py or pass
a different `model` argument directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from utils.retriever import RetrievedChunk

FINANCIAL_RAG_SYSTEM_PROMPT = """\
You are an expert financial analyst AI assistant. Your task is to answer questions
about financial documents with precision and integrity.

STRICT RULES:
1. Answer ONLY using information explicitly present in the provided source passages.
2. For every numeric claim (revenue, margin, EPS, ratio, percentage), cite the exact
   figure from the source and include [Source: <filename>, Page <N>].
3. If the information is NOT in the provided context, say exactly:
   "This information is not available in the provided document excerpts."
4. NEVER invent, estimate, or extrapolate numbers not directly stated in context.
5. If a chart or visual description mentions data, treat it as authoritative.
6. Format monetary values consistently ($42.3M, not 42.3 million dollars).
7. End your answer with a one-sentence summary of confidence level.
"""


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    provider: str
    steps: List[str]


# ── Cost tables (USD per 1M tokens) — verify current pricing at provider docs ─
# OpenAI: https://openai.com/api/pricing
# Google: https://ai.google.dev/gemini-api/docs/pricing

_OPENAI_PRICING = {
    "gpt-4o":      {"prompt": 5.00,  "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15,  "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
}

_GEMINI_PRICING = {
    # Current stable (2.5 series) as of this codebase's last verification.
    "gemini-2.5-flash":      {"prompt": 0.15, "completion": 0.60},
    "gemini-2.5-pro":        {"prompt": 1.25, "completion": 5.00},
    # Newer generation, offered as an option in the model dropdown.
    "gemini-3.5-flash":      {"prompt": 0.15, "completion": 0.60},
    "gemini-3.1-flash-lite": {"prompt": 0.05, "completion": 0.20},
}


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int, pricing: dict) -> float:
    rates = pricing.get(model, {"prompt": 0.0, "completion": 0.0})
    return (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1_000_000


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, r in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {r.source}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def generate_openai(
    query: str, chunks: List[RetrievedChunk], api_key: str, model: str = "gpt-4o-mini",
) -> GenerationResult:
    steps = [f"Generating answer with OpenAI {model}..."]
    start = time.perf_counter()
    context = _build_context(chunks)

    try:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": FINANCIAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cost = _compute_cost(model, pt, ct, _OPENAI_PRICING)

        steps.append(f"Generated {ct} tokens in {latency_ms:.0f}ms | Est. cost: ${cost:.5f}")
        return GenerationResult(
            answer=answer, model=model, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost, latency_ms=latency_ms, provider="openai", steps=steps,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        err = str(exc)
        if "401" in err or "Unauthorized" in err.lower():
            msg = "Invalid OpenAI API key. Please check your key and try again."
        elif "429" in err:
            msg = "OpenAI rate limit hit. Please wait a moment and retry."
        elif "insufficient_quota" in err.lower():
            msg = "OpenAI quota exceeded. Please check your billing at platform.openai.com."
        else:
            msg = f"OpenAI generation failed: {err[:120]}"
        steps.append(msg)
        return GenerationResult(
            answer=msg, model=model, prompt_tokens=0, completion_tokens=0,
            cost_usd=0.0, latency_ms=latency_ms, provider="openai", steps=steps,
        )


# ── Google Gemini ─────────────────────────────────────────────────────────────

def generate_gemini(
    query: str, chunks: List[RetrievedChunk], api_key: str, model: str = "gemini-2.5-flash",
) -> GenerationResult:
    steps = [f"Generating answer with Google {model}..."]
    start = time.perf_counter()
    context = _build_context(chunks)
    full_prompt = f"{FINANCIAL_RAG_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"

    try:
        import httpx
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1500},
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        pt = usage.get("promptTokenCount", 0)
        ct = usage.get("candidatesTokenCount", 0)
        cost = _compute_cost(model, pt, ct, _GEMINI_PRICING)

        steps.append(f"Generated {ct} tokens in {latency_ms:.0f}ms | Est. cost: ${cost:.5f}")
        return GenerationResult(
            answer=answer, model=model, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost, latency_ms=latency_ms, provider="gemini", steps=steps,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        err = str(exc)
        if "400" in err or "API_KEY_INVALID" in err:
            msg = "Invalid Google API key. Get one free at aistudio.google.com."
        elif "404" in err or "NOT_FOUND" in err:
            msg = (
                f"Model '{model}' not found — it may have been retired or renamed. "
                "Try gemini-2.5-flash or check aistudio.google.com for current model names."
            )
        elif "429" in err or "RESOURCE_EXHAUSTED" in err:
            msg = "Gemini rate limit hit. Please wait a moment and retry."
        else:
            msg = f"Gemini generation failed: {err[:120]}"
        steps.append(msg)
        return GenerationResult(
            answer=msg, model=model, prompt_tokens=0, completion_tokens=0,
            cost_usd=0.0, latency_ms=latency_ms, provider="gemini", steps=steps,
        )


# ── Router ────────────────────────────────────────────────────────────────────

def generate(
    query: str, chunks: List[RetrievedChunk], provider: str, model: str, api_key: str,
) -> GenerationResult:
    if not api_key or not api_key.strip():
        return GenerationResult(
            answer=(
                "**No API key provided.**\n\n"
                "Please enter your API key in the sidebar:\n"
                "- **OpenAI**: Get a key at [platform.openai.com](https://platform.openai.com)\n"
                "- **Google Gemini**: Get a free key at [aistudio.google.com](https://aistudio.google.com)\n\n"
                "Gemini has a generous free tier and works well for financial document analysis."
            ),
            model=model, prompt_tokens=0, completion_tokens=0,
            cost_usd=0.0, latency_ms=0.0, provider=provider,
            steps=["Generation skipped - no API key"],
        )

    if provider.lower() in ("openai", "gpt"):
        return generate_openai(query, chunks, api_key, model)
    elif provider.lower() in ("gemini", "google"):
        return generate_gemini(query, chunks, api_key, model)
    else:
        return generate_openai(query, chunks, api_key, "gpt-4o-mini")
