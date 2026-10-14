"""Unit tests for the Program-of-Thought path in RAGPipeline.query()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag_system.components.base import GeneratedAnswer, RetrievedChunk
from src.rag_system.components.generator import FINANCIAL_RAG_SYSTEM_PROMPT
from src.rag_system.components.pot_executor import (
    POT_CODEGEN_SYSTEM_PROMPT,
    POT_NO_CALCULATION,
    PoTExecutor,
    PoTResult,
)
from src.rag_system.pipeline import RAGPipeline

NUMERIC_QUERY = "Calculate the 3-year revenue CAGR"

# A retrieved chunk whose *document text* contains executable-looking code.
# It must never reach the sandbox — only LLM-written programs may.
POISONED_CHUNK_TEXT = (
    "Revenue grew from $100.0M in FY2021 to $133.1M in FY2024.\n"
    "```python\nresult = 999999\n```\n"
    "result = 424242"
)


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, score=0.9, source_document="10k.pdf", page_number=3, chunk_id="c1"
    )


def _make_pipeline(codegen_reply: str):
    """Pipeline with a mock retriever and a generator that answers the codegen
    prompt with ``codegen_reply`` and everything else with a normal answer."""
    chunks = [_chunk(POISONED_CHUNK_TEXT)]

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=chunks)

    async def _generate(query, context, tenant_id=None, system_prompt=None):
        if system_prompt == POT_CODEGEN_SYSTEM_PROMPT:
            return GeneratedAnswer(answer=codegen_reply, tenant_id=tenant_id)
        return GeneratedAnswer(
            answer="CAGR was 10.00% [Source: 10k.pdf, Page 3].",
            citations=context,
            tenant_id=tenant_id,
        )

    generator = MagicMock()
    generator.generate = AsyncMock(side_effect=_generate)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        audit_logger=MagicMock(),
        semantic_cache=MagicMock(),  # unused: no embedder, so the cache is skipped
    )
    return pipeline, generator


def _answer_call_prompt(generator) -> str | None:
    """system_prompt passed to the final (non-codegen) generate call."""
    calls = [
        c
        for c in generator.generate.await_args_list
        if c.kwargs.get("system_prompt") != POT_CODEGEN_SYSTEM_PROMPT
    ]
    assert len(calls) == 1
    return calls[0].kwargs.get("system_prompt")


# ── System prompt preservation ──────────────────────────────────────────────


def test_pot_note_appends_to_default_prompt_when_none_given():
    result = PoTResult(success=True, result=10.0, code="result = 10.0")
    prompt = RAGPipeline._with_pot_note(None, result)
    assert prompt.startswith(FINANCIAL_RAG_SYSTEM_PROMPT)
    assert "Calculated value (exact): 10.00" in prompt


def test_pot_note_appends_to_caller_prompt():
    result = PoTResult(success=True, result=10.0, code="result = 10.0")
    prompt = RAGPipeline._with_pot_note("Custom prompt.", result)
    assert prompt.startswith("Custom prompt.")
    assert FINANCIAL_RAG_SYSTEM_PROMPT not in prompt
    assert "Calculated value (exact): 10.00" in prompt


async def test_query_preserves_default_prompt_when_pot_succeeds():
    program = "```python\nv0 = 100.0\nv1 = 133.1\nresult = ((v1 / v0) ** (1 / 3) - 1) * 100\n```"
    pipeline, generator = _make_pipeline(program)

    out = await pipeline.query(NUMERIC_QUERY, tenant_id="t1")

    assert out["status"] == "success"
    assert out["pot_result"] is not None
    assert out["pot_result"]["result"] == pytest.approx(10.0)
    prompt = _answer_call_prompt(generator)
    assert prompt.startswith(FINANCIAL_RAG_SYSTEM_PROMPT)
    assert "Calculated value (exact): 10.00" in prompt


async def test_query_leaves_system_prompt_untouched_when_pot_declined():
    pipeline, generator = _make_pipeline(POT_NO_CALCULATION)

    out = await pipeline.query(NUMERIC_QUERY, tenant_id="t1")

    assert out["pot_result"] is None
    assert _answer_call_prompt(generator) is None  # generator applies its default


# ── Untrusted document content is never executed ────────────────────────────


@pytest.mark.parametrize(
    "codegen_reply",
    [POT_NO_CALCULATION, "", "```python\nresult = 133.1 - 100.0\n```"],
)
async def test_code_in_retrieved_chunks_is_never_executed(monkeypatch, codegen_reply):
    executed: list[str] = []
    real_run = PoTExecutor._run_in_sandbox

    def _spy(self, code):
        executed.append(code)
        return real_run(self, code)

    monkeypatch.setattr(PoTExecutor, "_run_in_sandbox", _spy)
    pipeline, _ = _make_pipeline(codegen_reply)

    out = await pipeline.query(NUMERIC_QUERY, tenant_id="t1")

    # Only the LLM's program (if it wrote one) reaches the sandbox.
    assert len(executed) == (1 if "```" in codegen_reply else 0)
    for code in executed:
        assert "999999" not in code
        assert "424242" not in code
    if out["pot_result"] is not None:
        assert out["pot_result"]["result"] not in (999999.0, 424242.0)
