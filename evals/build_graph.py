"""Run LLM entity extraction over the benchmark chunks and cache the raw output.

The cache feeds ``python -m evals.retrieval_benchmark --graph-cache <file>``,
which builds an ``InMemoryGraphStore`` from it and adds a hybrid+graph mode.
Chunking matches the benchmark (same --chunk-size/--overlap), so chunk ids line
up. Uses the extractor's configured provider (LLM_CONFIG__PROVIDER/MODEL).

This makes one LLM call per chunk: ~600 at --chunk-size 4000, ~3,200 at 1000.
Requests are limited to --rpm and --concurrency, transient failures (timeouts,
5xx, 429) are retried, and every result is appended to the cache, so an
interrupted run resumes without repeating calls.

Usage:
    python -m evals.build_graph --chunk-size 4000 --limit 5        # smoke test
    python -m evals.build_graph --chunk-size 4000 --rpm 60 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from evals.retrieval_benchmark import chunk_text
from evals.sources import DEFAULT_CACHE, extract_text, fetch, load_provenance, source_urls
from src.rag_system.components.knowledge_graph import EntityExtractor


def benchmark_chunks(chunk_size: int, overlap: int, cache_dir: Path):
    chunks = []
    for doc, url in sorted(source_urls(load_provenance()).items()):
        chunks.extend(chunk_text(extract_text(fetch(url, cache_dir)), doc, chunk_size, overlap))
    return chunks


def default_cache_path(chunk_size: int) -> Path:
    return Path(f"evals/results/graph_extractions_chunk{chunk_size}.jsonl")


async def run(args: argparse.Namespace) -> None:
    out_path = Path(args.output) if args.output else default_cache_path(args.chunk_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["chunk_id"])

    chunks = benchmark_chunks(args.chunk_size, args.overlap, Path(args.cache_dir))
    todo = [c for c in chunks if c.chunk_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(chunks)} chunks, {len(done)} cached, {len(todo)} to extract")

    semaphore = asyncio.Semaphore(args.concurrency)
    spacing = 60.0 / args.rpm
    next_slot = [time.monotonic()]
    totals = {"ok": 0, "failed": 0, "input": 0, "output": 0}
    out = open(out_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in finally

    async def extract_one(chunk) -> None:
        extractor = EntityExtractor()  # one per task: last_usage is per instance
        prompt = extractor.build_prompt(chunk.text)
        for attempt in range(1, 4):
            async with semaphore:
                wait = next_slot[0] - time.monotonic()
                next_slot[0] = max(next_slot[0], time.monotonic()) + spacing
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    raw = await extractor._complete(prompt)
                    extractor.parse(raw, chunk.source_document)  # validate before caching
                    break
                except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    retryable = status is None or status >= 500 or status == 429
                    if attempt == 3 or not retryable:
                        totals["failed"] += 1
                        print(f"  failed {chunk.chunk_id}: {type(exc).__name__} {str(exc)[:80]}")
                        return
            await asyncio.sleep(10 * attempt)
        totals["ok"] += 1
        totals["input"] += extractor.last_usage.get("input", 0)
        totals["output"] += extractor.last_usage.get("output", 0)
        out.write(json.dumps({"chunk_id": chunk.chunk_id, "raw": raw}, ensure_ascii=False) + "\n")
        out.flush()
        if totals["ok"] % 25 == 0:
            print(f"  {totals['ok']}/{len(todo)} extracted")

    try:
        await asyncio.gather(*(extract_one(c) for c in todo))
    finally:
        out.close()

    n = max(totals["ok"], 1)
    print(
        f"done: {totals['ok']} extracted, {totals['failed']} failed; provider-reported tokens "
        f"{totals['input']:,} input / {totals['output']:,} output incl. thinking "
        f"(avg {totals['input'] / n:,.0f} / {totals['output'] / n:,.0f} per chunk)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache LLM entity extraction for the benchmark")
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--rpm", type=float, default=30.0, help="max requests per minute")
    parser.add_argument("--concurrency", type=int, default=4, help="max requests in flight")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
