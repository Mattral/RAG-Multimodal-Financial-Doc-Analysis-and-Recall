# ADR 007: LangGraph Agentic Flow for Complex Queries

**Status:** Accepted (feature-flagged)  
**Date:** 2024-07  
**Deciders:** Core team

## Context

Simple RAG (retrieve → generate) fails on complex multi-step queries like:
- "Identify all numeric anomalies across the last 4 quarterly filings"
- "Compare risk factors across 5 different 10-Ks and rank by severity"
- "Calculate CAGR for each revenue segment and flag which are declining"

These require iterative retrieval, intermediate reasoning, self-correction when guardrails fail, and tool use (PoT calculator).

## Decision

Implement a LangGraph `StateGraph` with nodes:

```
[Analyze] → [Retrieve] → [Verify] → [Calculate] → [Synthesize]
                 ↑_______________|
                (retry if numeric discrepancy detected, max 3 iterations)
```

Enable via `ENABLE_LANGGRAPH_AGENTIC=true` in config. Disabled by default.

## Rationale

- **LangGraph over bare asyncio loops:** Explicit state machine makes the flow testable, observable (each node is a traced span), and easy to extend with new nodes.
- **Feature flag:** Agentic flow is more expensive (multiple LLM calls) and slower. Reserve for queries where `QueryAnalyzer` detects `AGENTIC` or `COMPARATIVE` intent.
- **Self-correction loop:** If the Verify node finds ungrounded numbers in the draft answer, it signals a refined retrieval query and loops back. Max 3 iterations prevents runaway cost.
- **Graceful degradation:** If `langgraph` is not installed or the graph fails, falls back to the standard single-pass pipeline transparently.

## Consequences

- **Positive:** 20-40% better answer quality on complex multi-document queries. Hallucination rate drops because the loop forces numeric grounding.
- **Negative:** 2-5× more expensive per query (multiple retrieval + generation calls). Add cost alerts in Grafana when agentic mode is active.
- **Future:** Add tool-use nodes (web search, screener API calls) for real-time data augmentation.
