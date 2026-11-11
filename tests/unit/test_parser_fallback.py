"""The unstructured parser's fallback is explicit, logged and recorded per document.

The hi_res path used to fail on every PDF (pillow 10.4 + unstructured-inference 1.6:
"Operation on closed image") and the parser silently fell back to pypdf text, so nothing
showed that tables and OCR were missing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from filing_qa.components.parser import UnstructuredParser
from filing_qa.config import reset_config
from filing_qa.utils.exceptions import PDFParsingError

FIXTURE = str(Path(__file__).parents[1] / "fixtures" / "smoke_filing.pdf")


class _Chunk:
    def __init__(self, text: str, page: int) -> None:
        self._text = text
        self.metadata = types.SimpleNamespace(page_number=page)

    def __str__(self) -> str:
        return self._text


@pytest.fixture
def fake_unstructured(monkeypatch):
    """Stand-in for unstructured's partition/chunk functions (an external library)."""
    partition = MagicMock(return_value=["raw"])
    chunk = MagicMock(return_value=[_Chunk("Operating cash flow was 3,308 million.", 2)])
    pdf_mod = types.ModuleType("unstructured.partition.pdf")
    pdf_mod.partition_pdf = partition
    title_mod = types.ModuleType("unstructured.chunking.title")
    title_mod.chunk_by_title = chunk
    for name in ("unstructured", "unstructured.partition", "unstructured.chunking"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf_mod)
    monkeypatch.setitem(sys.modules, "unstructured.chunking.title", title_mod)
    return partition


@pytest.fixture
def parser_env(monkeypatch):
    def make(**env):
        for key in (
            "PDF_PARSING_CONFIG__FALLBACK_PARSER",
            "PDF_PARSING_CONFIG__UNSTRUCTURED_STRATEGY",
        ):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        reset_config()
        return UnstructuredParser()

    yield make
    reset_config()


@pytest.mark.asyncio
async def test_hi_res_success_is_recorded_per_element(fake_unstructured, parser_env):
    parser = parser_env()
    with capture_logs() as logs:
        elements = await parser.parse(FIXTURE, tenant_id="t")

    assert fake_unstructured.call_args.kwargs["strategy"] == "hi_res"
    chunk_by_title = sys.modules["unstructured.chunking.title"].chunk_by_title
    assert chunk_by_title.call_args.kwargs["multipage_sections"] is False
    assert [e.metadata["parser"] for e in elements] == ["unstructured/hi_res"]
    assert elements[0].page_number == 2
    assert not [e for e in logs if e["event"] == "pdf_parser_fallback"]


@pytest.mark.asyncio
async def test_failure_falls_back_to_pypdf_with_a_warning(fake_unstructured, parser_env):
    fake_unstructured.side_effect = ValueError("Operation on closed image")
    parser = parser_env()
    with capture_logs() as logs:
        elements = await parser.parse(FIXTURE, tenant_id="t")

    events = [e for e in logs if e["event"] == "pdf_parser_fallback"]
    assert len(events) == 1
    assert events[0]["log_level"] == "warning"
    assert events[0]["primary"] == "unstructured/hi_res"
    assert events[0]["fallback"] == "pypdf"
    assert events[0]["error_type"] == "ValueError"
    assert events[0]["error"] == "Operation on closed image"
    # pypdf read the real fixture: one element per page, each marked as a fallback.
    assert [e.page_number for e in elements] == [1, 2]
    assert {e.metadata["parser"] for e in elements} == {"pypdf"}
    assert {e.metadata["parser_fallback_from"] for e in elements} == {"unstructured/hi_res"}
    assert "Operating cash flow was 3,308 million dollars" in elements[1].text


@pytest.mark.asyncio
async def test_no_fallback_configured_fails_the_document(fake_unstructured, parser_env):
    fake_unstructured.side_effect = ValueError("Operation on closed image")
    parser = parser_env(PDF_PARSING_CONFIG__FALLBACK_PARSER="none")
    with pytest.raises(PDFParsingError, match="unstructured/hi_res failed on smoke_filing.pdf"):
        await parser.parse(FIXTURE, tenant_id="t")


@pytest.mark.asyncio
async def test_batch_does_not_drop_a_failed_document(fake_unstructured, parser_env):
    fake_unstructured.side_effect = ValueError("broken")
    parser = parser_env(PDF_PARSING_CONFIG__FALLBACK_PARSER="none")
    with pytest.raises(PDFParsingError, match="Could not parse: smoke_filing.pdf"):
        await parser.parse_batch([FIXTURE], tenant_id="t")


def test_strategy_comes_from_config(fake_unstructured, parser_env):
    parser = parser_env(PDF_PARSING_CONFIG__UNSTRUCTURED_STRATEGY="fast")
    assert parser.variant == "unstructured/fast"


@pytest.mark.asyncio
async def test_ingest_result_names_the_parser_per_document(fake_unstructured, parser_env):
    from unittest.mock import AsyncMock

    from filing_qa.pipeline import RAGPipeline

    fake_unstructured.side_effect = ValueError("Operation on closed image")
    parser = parser_env()
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
    store = MagicMock()
    store.upsert = AsyncMock()
    version_manager = MagicMock()
    version_manager.needs_reindex = MagicMock(return_value=True)
    pipeline = RAGPipeline(
        parser=parser, embedder=embedder, vector_store=store, version_manager=version_manager
    )
    result = await pipeline.ingest([FIXTURE], tenant_id="t", process_vision=False)
    assert result["parsers"] == {"smoke_filing.pdf": "pypdf"}


def _el(kind: str, page):
    return type(kind, (), {"metadata": types.SimpleNamespace(page_number=page)})()


def test_group_by_page_splits_on_page_change_and_drops_page_breaks():
    from filing_qa.components.parser import _group_by_page

    a, b, brk, c = _el("Title", 1), _el("NarrativeText", 1), _el("PageBreak", None), _el("Title", 2)
    groups = _group_by_page([a, b, brk, c, _el("PageBreak", None)])
    assert groups == [[a, b], [c]]


@pytest.mark.asyncio
async def test_each_page_is_chunked_separately(fake_unstructured, parser_env):
    # chunk_by_title merges small sections across pages even with
    # multipage_sections=False, so a chunk could be cited to the wrong page.
    page1, page2 = _el("NarrativeText", 1), _el("NarrativeText", 2)
    fake_unstructured.return_value = [page1, _el("PageBreak", None), page2]
    chunk_by_title = sys.modules["unstructured.chunking.title"].chunk_by_title
    parser = parser_env()
    await parser.parse(FIXTURE, tenant_id="t")
    assert [c.args[0] for c in chunk_by_title.call_args_list] == [[page1], [page2]]
