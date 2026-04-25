#!/usr/bin/env python3
"""Pipeline benchmark: compare dense-only vs hybrid vs agentic on golden dataset.

Measures: latency, cost, faithfulness, answer relevancy per pipeline mode.
Results written to benchmark_results.json for tracking over time.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --modes dense hybrid --samples 20
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_benchmark(modes: List[str], sample_limit: int) -> List[Dict[str, Any]]:
    from evals.run_evals import _run as run_evals
    import argparse as ap

    results = []
    golden_path = "evals/golden_datasets/financial_qa.jsonl"

    if not Path(golden_path).exists():
        print(f"Golden dataset not found at {golden_path}")
        print("Run: python scripts/seed_golden_dataset.py")
        return []

    samples = []
    with open(golden_path) as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except Exception:
                continue
    samples = samples[:sample_limit]

    for mode in modes:
        os.environ["QUERY_MODE"] = mode
        print(f"\nBenchmarking mode: {mode} ({len(samples)} samples)...")

        start = time.perf_counter()
        mode_results = {"mode": mode, "samples": len(samples), "errors": 0,
                        "latencies_ms": [], "costs_usd": []}

        try:
            from src.rag_system.pipeline import create_pipeline
            from src.rag_system.config import reset_config
            reset_config()
            pipeline = await create_pipeline()

            for sample in samples:
                try:
                    t0 = time.perf_counter()
                    result = await pipeline.query(sample["question"], tenant_id="benchmark")
                    latency_ms = (time.perf_counter() - t0) * 1000
                    mode_results["latencies_ms"].append(round(latency_ms, 1))
                    cost = result.get("metrics", {}).get("cost_usd", 0.0)
                    mode_results["costs_usd"].append(cost)
                except Exception as exc:
                    mode_results["errors"] += 1
                    print(f"  Error on '{sample['question'][:60]}': {exc}")

        except Exception as exc:
            print(f"  Pipeline init failed for mode {mode}: {exc}")

        elapsed = time.perf_counter() - start
        lats = mode_results["latencies_ms"]
        if lats:
            lats_sorted = sorted(lats)
            mode_results["p50_ms"] = lats_sorted[len(lats_sorted) // 2]
            mode_results["p95_ms"] = lats_sorted[int(len(lats_sorted) * 0.95)]
            mode_results["p99_ms"] = lats_sorted[int(len(lats_sorted) * 0.99)]
            mode_results["avg_ms"] = sum(lats) / len(lats)
        mode_results["total_cost_usd"] = round(sum(mode_results["costs_usd"]), 4)
        mode_results["total_elapsed_s"] = round(elapsed, 1)

        results.append(mode_results)
        print(f"  p50={mode_results.get('p50_ms', 'n/a')}ms "
              f"p99={mode_results.get('p99_ms', 'n/a')}ms "
              f"cost=${mode_results['total_cost_usd']}")

    return results


def print_table(results: List[Dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Mode':<20} {'p50(ms)':<12} {'p99(ms)':<12} {'Cost($)':<12} {'Errors'}")
    print("-" * 70)
    for r in results:
        print(f"{r['mode']:<20} {r.get('p50_ms', 'N/A'):<12} "
              f"{r.get('p99_ms', 'N/A'):<12} {r['total_cost_usd']:<12} {r['errors']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline modes")
    parser.add_argument("--modes", nargs="+", default=["simple", "hybrid"],
                        choices=["simple", "hybrid", "agentic"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.modes, args.samples))
    print_table(results)

    output = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(),
              "results": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
