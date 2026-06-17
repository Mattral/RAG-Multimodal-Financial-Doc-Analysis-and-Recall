# ADR 006: Multi-Provider Vision Strategy — Gemini as Primary Alternative

> **Update**: the guideline and this ADR originally named the Gemini 2.0 series. Google retired 2.0-flash/2.0-pro in favor of the 2.5 and 3.x generations; the codebase now defaults to `gemini-2.5-flash`/`gemini-2.5-pro`. The adapter's `model` parameter and pricing table are updated accordingly — see `src/rag_system/components/vision/gemini_adapter.py`.


**Status:** Accepted  
**Date:** 2024-07  
**Deciders:** Core team

## Context

GPT-4o vision is excellent for complex financial documents but expensive (~$0.015–0.05 per image) and sends data to OpenAI. Regulated financial institutions often cannot send client data outside their VPC or to specific third parties.

Guideline §7 explicitly names "Google Gemini 2.0 Flash and Gemini 2.0 Pro — excellent free or cheap alternatives" and requires Qwen2-VL support.

## Decision

Add `GeminiVisionDescriber` as a production-quality alternative. Config-driven selection:

```
VISION_CONFIG__PROVIDER=gemini   # switch with zero code changes
GOOGLE_API_KEY=...
```

Fallback chain: `primary_provider → fallback_providers → None (skip image)`.

## Rationale

- **Gemini 2.0 Flash:** 10-40× cheaper than GPT-4o, comparable quality on financial charts. Free tier for development.
- **Qwen2-VL-72B:** Open-source, can run on Together.ai or private vLLM. No data leaves org.
- **Pixtral-12B:** Strong open alternative from Mistral, good license.
- **Single interface:** `BaseVisionDescriber` ABC means pipeline code never changes when switching providers.
- **Cost tracking:** All providers record tokens/cost to `CostTracker` using provider-specific pricing.

## Consequences

- **Positive:** Cost reduction of 10-40× for chart extraction at scale. Privacy compliance path via local/VPC inference.
- **Negative:** Gemini quality on very dense tables occasionally lower than GPT-4o. Mitigated by the fallback chain and the detailed prompt.
- **Future:** Add `LocalVLLMDescriber` for any Hugging Face model served with vLLM.
