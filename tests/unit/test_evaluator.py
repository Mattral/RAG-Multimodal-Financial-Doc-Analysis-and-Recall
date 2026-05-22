"""Unit tests for src/rag_system/components/evaluator/__init__.py.

Primary focus: _llm_numeric_judge(), which had a real bug fixed in this
session — it called float() directly on raw LLM output with no
raise_for_status() check, silently masking API errors (4xx/5xx, malformed
JSON, instruction-disobeying responses) as a fake neutral 0.5 score. These
tests pin down the corrected behaviour: genuine infra failures still return
0.5 (by design — neutral, not a false negative on quality), but they must
go through the explicit except branches, not happen to survive by luck.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.rag_system.components.base import GeneratedAnswer, RetrievedChunk
from src.rag_system.components.evaluator import (
    EvalSample,
    EvalResult,
    EvalReport,
    RagasEvaluator,
)


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            text="Q3 2023 revenue was $23.35 billion, up 9% year-over-year.",
            score=0.9, source_document="tesla.pdf", page_number=4,
        ),
    ]


@pytest.fixture
def sample_answer(sample_chunks):
    return GeneratedAnswer(
        answer="Revenue was $23.35 billion [Source: tesla.pdf, Page 4].",
        citations=sample_chunks, model_used="gpt-4o-mini",
        prompt_tokens=200, completion_tokens=40,
        estimated_cost_usd=0.0001, latency_ms=1200.0,
    )


def _mock_response(content: str, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response that mimics the OpenAI chat completions shape."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
        resp.text = '{"error": "rate limited"}'
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ── Data model sanity ───────────────────────────────────────────────────────────

class TestEvalDataModels:
    def test_eval_sample_defaults(self):
        s = EvalSample(question="What was revenue?", ground_truth="$23.35B")
        assert s.source_documents == []
        assert s.tags == []

    def test_eval_result_defaults_unpassed(self):
        r = EvalResult(question="q", answer="a", ground_truth="gt")
        assert r.passed is False
        assert r.faithfulness == 0.0

    def test_eval_report_results_default_empty_list(self):
        report = EvalReport(
            run_id="r1", timestamp="2024-01-01", num_samples=0,
            passed=0, failed=0, pass_rate=0.0,
            avg_faithfulness=0.0, avg_answer_relevancy=0.0,
            avg_numeric_accuracy=0.0, avg_latency_ms=0.0, total_cost_usd=0.0,
        )
        assert report.results == []


# ── RagasEvaluator basic wiring ────────────────────────────────────────────────

class TestRagasEvaluatorWiring:
    def test_name_property(self):
        ev = RagasEvaluator(openai_api_key="sk-test")
        assert ev.name == "ragas_financial_evaluator"

    def test_ragas_not_installed_falls_back_gracefully(self):
        with patch.dict("sys.modules", {"ragas": None}):
            ev = RagasEvaluator(openai_api_key="sk-test")
            assert ev._ragas_available is False

    @pytest.mark.asyncio
    async def test_evaluate_sets_defaults_when_ragas_unavailable(self, sample_answer):
        ev = RagasEvaluator(openai_api_key="sk-test")
        ev._ragas_available = False
        with patch.object(ev, "_llm_numeric_judge", AsyncMock(return_value=0.9)):
            metrics = await ev.evaluate("What was revenue?", sample_answer)
        assert metrics["faithfulness"] == 0.0  # never ran, but key still present
        assert metrics["numeric_accuracy"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_skips_numeric_judge_without_citations(self):
        ev = RagasEvaluator(openai_api_key="sk-test")
        ev._ragas_available = False
        answer_no_citations = GeneratedAnswer(
            answer="No sources found.", citations=[], model_used="gpt-4o-mini",
            prompt_tokens=50, completion_tokens=10,
            estimated_cost_usd=0.0, latency_ms=500.0,
        )
        metrics = await ev.evaluate("test query", answer_no_citations)
        assert metrics["numeric_accuracy"] == 0.0  # default, judge never called


# ── _llm_numeric_judge: the bug-fix coverage ──────────────────────────────────

class TestLLMNumericJudgeHappyPath:
    @pytest.mark.asyncio
    async def test_clean_numeric_response(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("0.85")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == pytest.approx(0.85)
        reset_config()

    @pytest.mark.asyncio
    async def test_response_with_trailing_punctuation(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("0.85.")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == pytest.approx(0.85)
        reset_config()

    @pytest.mark.asyncio
    async def test_response_wrapped_in_prose(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("Score: 0.92")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == pytest.approx(0.92)
        reset_config()

    @pytest.mark.asyncio
    async def test_score_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("1.5")  # out of range
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 1.0
        reset_config()

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("-0.3")  # out of range
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.0
        reset_config()


class TestLLMNumericJudgeFailureModes:
    """These pin down the exact bug fixed this session: previously a 4xx/5xx
    or malformed response would either raise inside response.json() and get
    masked by a bare `except Exception`, or (for empty/garbage content)
    raise ValueError from float(). Both used to collapse into the same
    return 0.5 — now we verify they STILL return 0.5, but via the correct,
    specific except branch rather than accidentally."""

    @pytest.mark.asyncio
    async def test_api_4xx_returns_neutral_score_not_exception(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response('{"error": "invalid_api_key"}', status_code=401)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()

    @pytest.mark.asyncio
    async def test_api_5xx_returns_neutral_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response('{"error": "internal"}', status_code=503)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()

    @pytest.mark.asyncio
    async def test_connection_timeout_returns_neutral_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_neutral_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = _mock_response("I cannot determine a score for this answer.")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()

    @pytest.mark.asyncio
    async def test_malformed_json_shape_returns_neutral_score(self, monkeypatch):
        """choices[0] missing entirely -> IndexError, must be caught explicitly."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key="sk-test")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": []}  # empty -> IndexError on [0]
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_neutral_score(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.rag_system.config import reset_config
        reset_config()
        ev = RagasEvaluator(openai_api_key=None)
        ev._api_key = ""

        score = await ev._llm_numeric_judge("q", "a", ["context"])
        assert score == 0.5
        reset_config()
