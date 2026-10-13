"""Answer-quality eval: hybrid retrieval over the real sources + the configured generator.

For each golden question this retrieves context from an in-memory index of the
16 primary documents (same setup as evals/retrieval_benchmark.py: BM25 + local
dense embeddings fused with RRF, no reranker), asks the generator selected by
LLM_CONFIG__PROVIDER to answer, and scores the answer without an LLM judge:

    expected_value_recall  share of expected_numeric_values stated in the answer
    cites_expected_doc     answer cites one of the expected source documents
    abstained              answer says the information is not available
    evidence_in_context    an answer-bearing chunk was in the retrieved context

This is not the full pipeline: there is no query analyzer, reranker, guardrails
or unstructured.io parsing. It measures retrieval + generation only.

Built for rate-limited free tiers: requests are spaced to --rpm, 429s back off
exponentially, and every answer is appended to --answers so an interrupted run
resumes without repeating calls.

Usage:
    LLM_CONFIG__ENABLE_MODEL_ROUTING=false python -m evals.answer_eval --limit 5
    LLM_CONFIG__ENABLE_MODEL_ROUTING=false python -m evals.answer_eval --rpm 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import httpx

from evals.retrieval_benchmark import DATASET, DenseIndex, chunk_text
from evals.sources import (
    DEFAULT_CACHE,
    extract_text,
    fetch,
    load_provenance,
    normalize,
    source_urls,
)
from src.rag_system.components.embedder import LocalEmbedder
from src.rag_system.components.evaluator import numeric_recall
from src.rag_system.components.generator import build_generator
from src.rag_system.components.retriever import BM25Index, _reciprocal_rank_fusion
from src.rag_system.config import RetrieverConfig, get_config

_CITED_DOC_RE = re.compile(r"Source(?: \d+)?:\s*([\w.\-]+\.pdf)", re.I)
_ABSTAIN_RE = re.compile(r"not available in the provided documents", re.I)


def score_text(answer: str, row: Dict) -> Dict:
    """Metrics computed from the answer text alone (recomputed at summary time)."""
    cited = {m.group(1).lower() for m in _CITED_DOC_RE.finditer(answer)}
    expected_docs = {d.lower() for d in row["source_documents"]}
    values = row["expected_numeric_values"]
    headline = numeric_recall(answer, values[:1])
    return {
        "expected_value_recall": numeric_recall(answer, values),
        # The first expected value is the figure the question asks for; the rest
        # are supporting context (prior period, growth rate) it may not request.
        "headline_value_stated": None if headline is None else headline == 1.0,
        "cites_expected_doc": bool(cited & expected_docs),
        "abstained": bool(_ABSTAIN_RE.search(answer)),
    }


def score_answer(answer: str, row: Dict, evidence: List[str], context) -> Dict:
    return {
        **score_text(answer, row),
        "evidence_in_context": any(e in c.text for c in context for e in evidence),
    }


_RETRYABLE = {429, 500, 502, 503, 504}


async def generate_with_backoff(generator, question, context, max_attempts: int = 5):
    """Call the generator, backing off on 429/5xx.

    Raises RuntimeError (which stops the run) on daily-quota exhaustion, on any
    other 4xx (e.g. a retired model name), or when retries run out.
    """
    delay = 30.0
    for attempt in range(1, max_attempts + 1):
        try:
            return await generator.generate(question, context, tenant_id="eval")
        except httpx.HTTPStatusError as exc:
            status, body = exc.response.status_code, exc.response.text[:500]
            if status not in _RETRYABLE:
                raise RuntimeError(f"HTTP {status}: {body[:200]}") from exc
            if status == 429 and ("PerDay" in body or "per day" in body.lower()):
                raise RuntimeError("daily quota exhausted") from exc
            print(f"  HTTP {status} (attempt {attempt}/{max_attempts}); sleeping {delay:.0f}s")
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("still failing after retries")


async def run(args: argparse.Namespace) -> None:
    provenance = {p["question"]: p for p in load_provenance()}
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    answers_path = Path(args.answers)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Dict] = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["question"]] = rec
    todo = [] if args.summarize_only else [r for r in rows if r["question"] not in done]
    print(f"{len(rows)} questions, {len(done)} already answered, {len(todo)} to go")

    generator = build_generator()
    print(f"generator: {generator.name}; routing={get_config().llm_config.enable_model_routing}")

    if todo:
        chunks = []
        for doc, url in sorted(source_urls(list(provenance.values())).items()):
            chunks.extend(
                chunk_text(
                    extract_text(fetch(url, Path(args.cache_dir))),
                    doc,
                    args.chunk_size,
                    args.overlap,
                )
            )
        cfg = RetrieverConfig()
        bm25 = BM25Index()
        bm25.build(chunks)
        dense = DenseIndex(LocalEmbedder(), chunks)
        await dense.build(cache_dir=Path(args.cache_dir))
        print(f"indexed {len(chunks):,} chunks")

        spacing = 60.0 / args.rpm
        last_call = 0.0
        with open(answers_path, "a", encoding="utf-8") as out:
            for i, row in enumerate(todo, 1):
                q = row["question"]
                fused = _reciprocal_rank_fusion(
                    [await dense.search(q, cfg.top_k_dense), bm25.search(q, cfg.top_k_bm25)],
                    k=cfg.rrf_k,
                )
                context = fused[: args.context_k]
                wait = spacing - (time.monotonic() - last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                last_call = time.monotonic()
                try:
                    answer = await generate_with_backoff(generator, q, context)
                except RuntimeError as exc:
                    print(f"stopping early: {exc}. Re-run later to resume.")
                    break
                except Exception as exc:  # record the failure, keep going
                    print(f"  [{i}/{len(todo)}] error: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
                evidence = [normalize(e) for s in provenance[q]["sources"] for e in s["evidence"]]
                rec = {
                    "question": q,
                    "type": row["tags"][0],
                    "answer": answer.answer,
                    "model": answer.model_used,
                    "latency_ms": round(answer.latency_ms),
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "context_documents": [c.source_document for c in context],
                    **score_answer(answer.answer, row, evidence, context),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done[q] = rec
                print(
                    f"  [{i}/{len(todo)}] recall={rec['expected_value_recall']} "
                    f"ctx_evidence={rec['evidence_in_context']} {q[:60]}"
                )

    # Re-score from the saved answer text so metric changes never need new API calls.
    records = [
        {**done[r["question"]], **score_text(done[r["question"]]["answer"], r)}
        for r in rows
        if r["question"] in done
    ]
    summarize(records, args.output)


def summarize(records: List[Dict], output: str | None) -> None:
    if not records:
        print("no answers yet")
        return

    def block(recs: List[Dict]) -> Dict:
        scored = [
            r["expected_value_recall"] for r in recs if r["expected_value_recall"] is not None
        ]
        headline = [
            r["headline_value_stated"] for r in recs if r["headline_value_stated"] is not None
        ]
        return {
            "n": len(recs),
            "headline_value_stated": round(sum(headline) / len(headline), 3) if headline else None,
            "expected_value_recall": round(sum(scored) / len(scored), 3) if scored else None,
            "all_values_stated": (
                round(sum(v == 1.0 for v in scored) / len(scored), 3) if scored else None
            ),
            "cites_expected_doc": round(sum(r["cites_expected_doc"] for r in recs) / len(recs), 3),
            "abstained": round(sum(r["abstained"] for r in recs) / len(recs), 3),
            "evidence_in_context": round(
                sum(r["evidence_in_context"] for r in recs) / len(recs), 3
            ),
        }

    with_ctx = [r for r in records if r["evidence_in_context"]]
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    lat = sorted(r["latency_ms"] for r in records)
    report = {
        "model": sorted({r["model"] for r in records}),
        "overall": block(records),
        "when_evidence_retrieved": block(with_ctx) if with_ctx else None,
        "by_type": {t: block(rs) for t, rs in sorted(by_type.items())},
        "latency_ms": {
            "p50": lat[len(lat) // 2],
            "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        },
        "tokens": {
            "prompt": sum(r["prompt_tokens"] for r in records),
            "completion": sum(r["completion_tokens"] for r in records),
        },
    }
    print(json.dumps(report, indent=2))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval + generation eval (no LLM judge)")
    parser.add_argument("--rpm", type=float, default=5.0, help="max generator requests per minute")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--context-k", type=int, default=5, help="chunks passed to the generator")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--answers", default="evals/results/answers.jsonl")
    parser.add_argument("--output", default="evals/results/answer_eval.json")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="score saved answers without calling the generator",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
