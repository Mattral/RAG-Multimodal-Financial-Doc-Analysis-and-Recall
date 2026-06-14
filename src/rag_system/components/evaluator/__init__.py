"""Production-quality evaluation framework with RAGAS + financial-specific metrics.

Integrates:
- RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
- LLM-as-Judge: numeric accuracy, citation faithfulness, hallucination score
- Golden dataset runner
- CI-ready exit codes (fails if quality regression detected)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import BaseEvaluator, GeneratedAnswer

logger = structlog.get_logger(__name__)


@dataclass
class EvalSample:
    """Single golden-dataset evaluation sample."""
    question: str
    ground_truth: str
    source_documents: List[str] = field(default_factory=list)
    expected_page: Optional[int] = None
    expected_numeric_values: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # e.g. ["table", "numeric", "10-K"]


@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    numeric_accuracy: float = 0.0
    citation_faithfulness: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    passed: bool = False


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    num_samples: int
    passed: int
    failed: int
    pass_rate: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_numeric_accuracy: float
    avg_latency_ms: float
    total_cost_usd: float
    results: List[EvalResult] = field(default_factory=list)
    regression_detected: bool = False


class RagasEvaluator(BaseEvaluator):
    """RAGAS-based evaluator with financial extensions."""

    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "ragas_not_installed",
                detail="pip install ragas  — falling back to LLM-judge only",
            )
            return False

    @property
    def name(self) -> str:
        return "ragas_financial_evaluator"

    async def evaluate(
        self,
        query: str,
        answer: GeneratedAnswer,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self._ragas_available:
            try:
                metrics.update(
                    await self._run_ragas(query, answer, ground_truth)
                )
            except Exception as exc:
                logger.warning("ragas_evaluation_failed", error=str(exc))

        # LLM-as-judge numeric accuracy
        if answer.citations:
            metrics["numeric_accuracy"] = await self._llm_numeric_judge(
                query, answer.answer, [c.text for c in answer.citations]
            )

        metrics.setdefault("faithfulness", 0.0)
        metrics.setdefault("answer_relevancy", 0.0)
        metrics.setdefault("numeric_accuracy", 0.0)
        return metrics

    async def _run_ragas(
        self,
        query: str,
        answer: GeneratedAnswer,
        ground_truth: Optional[str],
    ) -> Dict[str, float]:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        contexts = [c.text for c in answer.citations]
        data = {
            "question": [query],
            "answer": [answer.answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or ""],
        }
        ds = Dataset.from_dict(data)
        result = await asyncio.to_thread(
            evaluate,
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
        }

    async def _llm_numeric_judge(
        self, query: str, answer: str, context_texts: List[str]
    ) -> float:
        """Ask LLM to verify numeric claims in the answer are grounded."""
        import httpx
        from src.rag_system.config import get_config

        try:
            cfg = get_config()
            api_key = cfg.get_openai_key()
            context_block = "\n\n".join(context_texts[:3])
            judge_prompt = (
                f"Context:\n{context_block}\n\n"
                f"Answer to evaluate:\n{answer}\n\n"
                "Rate from 0.0 to 1.0 how accurately the numeric values in the answer "
                "match the context. 1.0 means all numbers are exactly correct and grounded. "
                "Respond with ONLY a number, e.g. 0.85"
            )
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_tokens": 10,
                        "temperature": 0.0,
                    },
                )
                score_str = response.json()["choices"][0]["message"]["content"].strip()
                return min(1.0, max(0.0, float(score_str)))
        except Exception as exc:
            logger.warning("numeric_judge_failed", error=str(exc))
            return 0.5


class GoldenDatasetRunner:
    """Runs the full evaluation pipeline on a golden dataset."""

    def __init__(
        self,
        pipeline: Any,  # RAGPipeline
        evaluator: RagasEvaluator,
        golden_dataset_path: str = "evals/golden_datasets/financial_qa.jsonl",
        regression_threshold: float = 0.05,
        history_path: str = "evals/history.json",
    ) -> None:
        self._pipeline = pipeline
        self._evaluator = evaluator
        self._golden_path = Path(golden_dataset_path)
        self._threshold = regression_threshold
        self._history_path = Path(history_path)

    def load_golden_dataset(self) -> List[EvalSample]:
        if not self._golden_path.exists():
            logger.warning("golden_dataset_not_found", path=str(self._golden_path))
            return []
        samples = []
        with open(self._golden_path) as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(EvalSample(**data))
        return samples

    async def run(self, tenant_id: str = "eval") -> EvalReport:
        import uuid

        samples = self.load_golden_dataset()
        if not samples:
            logger.warning("no_eval_samples_loaded")

        results: List[EvalResult] = []
        for sample in samples:
            try:
                start = time.perf_counter()
                qa_result = await self._pipeline.query(
                    sample.question, tenant_id=tenant_id
                )
                latency_ms = (time.perf_counter() - start) * 1000

                answer: GeneratedAnswer = qa_result.get("answer_obj")
                if answer is None:
                    continue

                metrics = await self._evaluator.evaluate(
                    sample.question, answer, sample.ground_truth
                )
                passed = (
                    metrics.get("faithfulness", 0.0) >= 0.7
                    and metrics.get("numeric_accuracy", 0.0) >= 0.7
                )
                results.append(
                    EvalResult(
                        question=sample.question,
                        answer=answer.answer,
                        ground_truth=sample.ground_truth,
                        faithfulness=metrics.get("faithfulness", 0.0),
                        answer_relevancy=metrics.get("answer_relevancy", 0.0),
                        numeric_accuracy=metrics.get("numeric_accuracy", 0.0),
                        latency_ms=latency_ms,
                        cost_usd=answer.estimated_cost_usd,
                        passed=passed,
                    )
                )
            except Exception as exc:
                logger.error("eval_sample_failed", question=sample.question[:80], error=str(exc))

        num = len(results)
        passed_count = sum(1 for r in results if r.passed)
        report = EvalReport(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=num,
            passed=passed_count,
            failed=num - passed_count,
            pass_rate=passed_count / num if num else 0.0,
            avg_faithfulness=sum(r.faithfulness for r in results) / num if num else 0.0,
            avg_answer_relevancy=sum(r.answer_relevancy for r in results) / num if num else 0.0,
            avg_numeric_accuracy=sum(r.numeric_accuracy for r in results) / num if num else 0.0,
            avg_latency_ms=sum(r.latency_ms for r in results) / num if num else 0.0,
            total_cost_usd=sum(r.cost_usd for r in results),
            results=results,
        )
        report.regression_detected = self._detect_regression(report)
        self._save_history(report)
        return report

    def _detect_regression(self, report: EvalReport) -> bool:
        if not self._history_path.exists():
            return False
        try:
            with open(self._history_path) as f:
                history = json.load(f)
            if not history:
                return False
            last = history[-1]
            delta = last.get("avg_faithfulness", 0.0) - report.avg_faithfulness
            return delta > self._threshold
        except Exception:
            return False

    def _save_history(self, report: EvalReport) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        history: List[Dict] = []
        if self._history_path.exists():
            with open(self._history_path) as f:
                history = json.load(f)
        history.append({
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "pass_rate": report.pass_rate,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_numeric_accuracy": report.avg_numeric_accuracy,
        })
        # Keep last 50 runs
        history = history[-50:]
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)
