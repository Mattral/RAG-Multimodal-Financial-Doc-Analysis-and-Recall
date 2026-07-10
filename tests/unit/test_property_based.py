"""Property-based tests using Hypothesis.

Tests edge cases that unit tests miss: empty inputs, adversarial strings,
boundary numeric values, unicode, very long texts, malformed metadata.

Guideline §5: 'Property-based testing with Hypothesis for edge cases
(empty docs, malformed tables, adversarial queries).'
"""

from __future__ import annotations

import math
import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.rag_system.components.base import DocumentElement, RetrievedChunk
from src.rag_system.components.guardrails import FinancialGuardrails
from src.rag_system.components.layout_parser import LayoutAwareParser
from src.rag_system.components.pot_executor import ASTSandboxValidator
from src.rag_system.components.query_analyzer import QueryAnalyzer
from src.rag_system.components.retriever import BM25Index
from src.rag_system.utils.cost_tracker import CostRecord, CostTracker

# ── Strategies ────────────────────────────────────────────────────────────────

financial_text = st.text(
    alphabet=string.printable,
    min_size=0,
    max_size=2000,
)

safe_query = st.text(
    alphabet=string.ascii_letters + string.digits + " .,?%$-",
    min_size=1,
    max_size=500,
)

positive_float = st.floats(
    min_value=0.01,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

small_positive_int = st.integers(min_value=0, max_value=10_000_000)


# ── QueryAnalyzer property tests ──────────────────────────────────────────────


class TestQueryAnalyzerProperties:
    """QueryAnalyzer must never crash on arbitrary input."""

    @given(query=financial_text)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_analyze_never_raises(self, query):
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(query)
        assert result is not None
        assert result.original_query == query
        assert result.intent is not None
        assert result.complexity is not None

    @given(query=safe_query)
    @settings(max_examples=100)
    def test_rewritten_query_never_empty(self, query):
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(query)
        assert len(result.rewritten_query) > 0

    @given(queries=st.lists(safe_query, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_batch_analyze_length_matches(self, queries):
        analyzer = QueryAnalyzer()
        results = analyzer.batch_analyze(queries)
        assert len(results) == len(queries)

    @given(query=financial_text)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_injection_detection_is_bool(self, query):
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(query)
        assert isinstance(result.is_injection, bool)

    @given(query=safe_query)
    @settings(max_examples=100)
    def test_metadata_filters_is_dict(self, query):
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(query)
        assert isinstance(result.metadata_filters, dict)

    @given(query=safe_query)
    @settings(max_examples=100)
    def test_suggested_top_k_positive(self, query):
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(query)
        assert result.suggested_top_k > 0


# ── FinancialGuardrails property tests ───────────────────────────────────────


class TestGuardrailsProperties:
    """Guardrails must handle arbitrary text without crashing."""

    @given(answer=financial_text, context=st.lists(financial_text, min_size=0, max_size=5))
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_numeric_grounding_never_raises(self, answer, context):
        g = FinancialGuardrails()
        passed, ungrounded = g.check_numeric_grounding(answer, context)
        assert isinstance(passed, bool)
        assert isinstance(ungrounded, list)

    @given(query=financial_text)
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_injection_check_never_raises(self, query):
        g = FinancialGuardrails()
        result = g.check_prompt_injection(query)
        assert isinstance(result, bool)

    @given(
        query=safe_query,
        answer=financial_text,
        context=st.lists(financial_text, min_size=0, max_size=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_run_all_checks_returns_dict(self, query, answer, context):
        g = FinancialGuardrails()
        result = g.run_all_checks(query=query, answer=answer, context_chunks=context)
        assert "overall_passed" in result
        assert isinstance(result["overall_passed"], bool)

    @given(answer=st.just(""))
    @settings(max_examples=10)
    def test_empty_answer_passes_grounding(self, answer):
        """Empty answer has no numbers to check — should pass."""
        g = FinancialGuardrails()
        passed, ungrounded = g.check_numeric_grounding(answer, ["some context"])
        assert passed is True
        assert ungrounded == []


# ── ASTSandboxValidator property tests ───────────────────────────────────────


class TestASTSandboxProperties:
    """Validator must never crash, must always return str or None."""

    @given(code=financial_text)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_never_raises(self, code):
        v = ASTSandboxValidator()
        result = v.validate(code)
        assert result is None or isinstance(result, str)

    @given(
        a=positive_float,
        b=positive_float,
    )
    @settings(max_examples=100)
    def test_safe_arithmetic_passes_validation(self, a, b):
        """Arithmetic code should always pass AST validation."""
        v = ASTSandboxValidator()
        code = f"x = {a}\ny = {b}\nresult = x + y"
        result = v.validate(code)
        assert result is None, f"Expected safe code to pass, got: {result}"

    @given(
        dangerous=st.sampled_from(
            [
                "import os",
                "import sys",
                "exec('1')",
                "eval('1')",
                "open('/etc/passwd')",
                "__builtins__",
            ]
        )
    )
    @settings(max_examples=30)
    def test_dangerous_patterns_blocked(self, dangerous):
        v = ASTSandboxValidator()
        result = v.validate(dangerous)
        assert result is not None, f"Expected '{dangerous}' to be blocked"


# ── BM25Index property tests ──────────────────────────────────────────────────


class TestBM25IndexProperties:
    """BM25 must handle empty queries, empty index, unicode, duplicates."""

    @given(
        texts=st.lists(
            st.text(alphabet=string.ascii_lowercase + " ", min_size=1, max_size=200),
            min_size=0,
            max_size=30,
        ),
        query=st.text(alphabet=string.ascii_lowercase + " ", min_size=0, max_size=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_search_never_raises(self, texts, query):
        chunks = [
            RetrievedChunk(text=t, score=0.0, source_document="d.pdf") for t in texts if t.strip()
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search(query, top_k=5)
        assert isinstance(results, list)

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_results_not_exceed_top_k(self, texts):
        chunks = [RetrievedChunk(text=t, score=0.0, source_document="d.pdf") for t in texts]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("revenue profit margin", top_k=3)
        assert len(results) <= 3

    @given(query=safe_query)
    @settings(max_examples=50)
    def test_empty_index_returns_empty(self, query):
        idx = BM25Index()
        idx.build([])
        assert idx.search(query) == []


# ── CostTracker property tests ────────────────────────────────────────────────


class TestCostTrackerProperties:
    """Cost computations must be non-negative, monotonic, and never crash."""

    @given(
        prompt_tokens=small_positive_int,
        completion_tokens=small_positive_int,
        model=st.sampled_from(["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "unknown-model"]),
    )
    @settings(max_examples=200)
    def test_cost_record_non_negative(self, prompt_tokens, completion_tokens, model):
        rec = CostRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
        assert rec.cost_usd >= 0.0
        assert not math.isnan(rec.cost_usd)
        assert not math.isinf(rec.cost_usd)

    @given(
        n_queries=st.integers(min_value=1, max_value=50),
        prompt_tokens=small_positive_int,
    )
    @settings(max_examples=100)
    def test_total_cost_monotonically_increasing(self, n_queries, prompt_tokens):
        tracker = CostTracker()
        prev_cost = 0.0
        for _ in range(n_queries):
            tracker.record("tenant", "gpt-4o-mini", prompt_tokens=prompt_tokens)
            summary = tracker.get_tenant_summary("tenant")
            assert summary is not None
            assert summary.total_cost_usd >= prev_cost
            prev_cost = summary.total_cost_usd

    @given(
        limit=small_positive_int,
        used=small_positive_int,
    )
    @settings(max_examples=100)
    def test_quota_check_consistent(self, limit, used):
        tracker = CostTracker()
        tracker.record("t", "gpt-4o-mini", prompt_tokens=used)
        result = tracker.check_quota("t", monthly_token_limit=limit)
        summary = tracker.get_tenant_summary("t")
        assert summary is not None
        expected = summary.total_tokens <= limit
        assert result == expected


# ── DocumentElement property tests ───────────────────────────────────────────


class TestDocumentElementProperties:
    """DocumentElement must accept and preserve arbitrary text content."""

    @given(
        text=financial_text,
        source=st.text(min_size=1, max_size=200, alphabet=string.printable),
        page=st.one_of(st.none(), st.integers(min_value=1, max_value=9999)),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_creation_never_raises(self, text, source, page):
        elem = DocumentElement(
            type="text",
            text=text,
            source_document=source,
            page_number=page,
        )
        assert elem.text == text
        assert elem.page_number == page

    @given(text=financial_text)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_model_dump_roundtrip(self, text):
        elem = DocumentElement(type="text", text=text, source_document="d.pdf")
        d = elem.model_dump()
        restored = DocumentElement(**d)
        assert restored.text == text


# ── LayoutAwareParser property tests ─────────────────────────────────────────


class TestLayoutParserProperties:
    """Layout parser must handle any list of valid DocumentElements without crashing."""

    @given(
        element_types=st.lists(
            st.sampled_from(["text", "table", "graph", "image"]),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_parse_never_raises(self, element_types):
        parser = LayoutAwareParser()
        elements = [
            DocumentElement(
                type=et,
                text=f"Sample {et} content with some financial data $42.3M",
                source_document="test.pdf",
                page_number=i + 1,
            )
            for i, et in enumerate(element_types)
        ]
        chunks = parser.parse(elements)
        assert isinstance(chunks, list)

    @given(
        n_elements=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_output_chunks_lte_input_elements(self, n_elements):
        """Chunking should merge, not create more chunks than elements."""
        parser = LayoutAwareParser()
        elements = [
            DocumentElement(
                type="text",
                text=f"Paragraph {i}: Revenue grew 9% to $23.35B in Q3 2023.",
                source_document="doc.pdf",
                page_number=i + 1,
            )
            for i in range(n_elements)
        ]
        chunks = parser.parse(elements)
        # Chunks should be <= elements (merger combines consecutive text)
        assert len(chunks) <= max(n_elements, 1)
