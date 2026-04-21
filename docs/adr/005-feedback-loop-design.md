# ADR 005: Human Feedback Loop Design

**Status:** Accepted  
**Date:** 2024-07  
**Deciders:** Core team

## Context

Without user feedback, we have no signal on whether generated answers are actually useful to the analyst — only proxy metrics (faithfulness, latency). Guideline §3 explicitly requires a thumbs-up/down + free-text feedback mechanism stored with `query_id` for later use in reranker fine-tuning and reward modelling.

## Decision

Implement a lightweight feedback endpoint (`POST /api/v1/feedback`) that:
1. Accepts `rating` (thumbs_up / thumbs_down / neutral) + optional free-text `comment`
2. Writes to an append-only `data/feedback.jsonl` (same pattern as audit log)
3. Stores `query_id`, `tenant_id`, answer preview, model used, latency, number of sources
4. Exposes a `GET /api/v1/feedback/summary` for monitoring dashboards

## Rationale

- **JSONL over database:** Simplest possible durable store. Zero new dependencies. Easy to ship to S3 or Postgres later by changing the writer.
- **Stored with query_id:** Enables joining feedback to the audit log for per-query analysis.
- **Thumbs + free-text:** Thumbs gives a quantitative signal; free text gives qualitative. Both are needed for reranker preference pairs (thumbs_down + explanation = negative training example).
- **Tenant-scoped:** Different tenants have different quality expectations; per-tenant satisfaction rates are meaningful.

## Usage

Feedback data at `data/feedback.jsonl` can be used to:
1. Create preference pairs for cross-encoder reranker fine-tuning
2. Build a reward model for RLHF on the generator
3. Expand the golden evaluation dataset with high-confidence rated answers
4. Power a Grafana satisfaction rate dashboard

## Consequences

- **Positive:** Zero-friction quality signal from real users. Actionable for model improvement.
- **Negative:** Requires frontend integration to surface the rating UI. Without UI, feedback volume will be low.
- **Future:** Replace JSONL with a vector store of (query, answer, rating) triplets to enable semantic search for similar past feedback.
