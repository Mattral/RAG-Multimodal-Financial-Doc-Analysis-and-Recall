# REST API Reference

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ingest` | Ingest a PDF document |
| `POST` | `/api/v1/query` | Query ingested documents (one JSON response) |
| `POST` | `/api/v1/query/stream` | Query with the answer streamed as server-sent events |
| `POST` | `/api/v1/tenants` | Register a new tenant |
| `GET` | `/api/v1/tenants/{id}/usage` | Get tenant usage stats |
| `GET` | `/healthz` | K8s liveness probe |
| `GET` | `/readyz` | K8s readiness probe |

## Authentication
All endpoints (except health probes) require: `X-API-Key: your-key`
Set tenant with: `X-Tenant-ID: your-tenant`

## Streaming answers: `POST /api/v1/query/stream`

The request body is the same as `POST /api/v1/query`. The response is `text/event-stream`:

| Event | `data` | When |
|---|---|---|
| `sources` | `{"sources": [...], "analysis": {...}}` | Retrieval (and Program-of-Thought, if used) finished |
| `token` | `{"text": "..."}` | Each piece of the answer as the model generates it |
| `done` | The same object `POST /query` returns | Generation, guardrails and audit finished |
| `error` | `{"error": "..."}` | The pipeline failed mid-stream |

```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" -H "X-API-Key: $RAG_API_KEY" \
  -d '{"query": "What was Tesla gross margin in Q3 2023?", "tenant_id": "demo"}'
```

```js
// Browser: EventSource only supports GET, so read the POST response body as a stream.
const res = await fetch("/api/v1/query/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": key },
  body: JSON.stringify({ query, tenant_id: "demo" }),
});
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
// Split on blank lines; each block has an "event:" and a "data:" line.
```

From Python, the SDK exposes the same stream without HTTP:

```python
async for event in rag.query_stream("What was Tesla gross margin in Q3 2023?"):
    if event["event"] == "token":
        print(event["text"], end="", flush=True)
```

Things to know:

* **Guardrails run on the complete answer.** The numeric-grounding and other checks need the
  whole text, so they run after the last token. Treat streamed text as provisional until the
  `done` event's `guardrails` field confirms it, and replace or flag the answer in your UI if
  `guardrails.overall_passed` is `false`.
* **Token-level streaming** is native for the OpenAI and Gemini generators. Other providers
  (Anthropic, local vLLM) currently send the whole answer as a single `token` event, using
  the default `BaseGenerator.generate_stream`.
* **Single-event responses.** Semantic-cache hits and early exits (quota exceeded, blocked
  query) arrive as one `done` event.
* **Proxies.** The response sets `X-Accel-Buffering: no` and `Cache-Control: no-cache` so
  nginx-style proxies pass tokens through immediately.
* **OpenAI rate limits.** A 429/503 before the first token falls back to the non-streaming
  call on `LLM_CONFIG__FALLBACK_MODEL`, so you still get an answer, as one `token` event.

See interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)
