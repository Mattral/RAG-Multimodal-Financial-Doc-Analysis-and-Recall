"""Unit tests for the layout-aware semantic chunker."""
from src.rag_system.components.base import DocumentElement
from src.rag_system.components.layout_parser import (
    LayoutAwareParser,
    LayoutChunk,
    _wrap_figure,
    _wrap_section,
    _wrap_table,
)


def _make_elem(text, etype="text", page=1, source="doc.pdf"):
    return DocumentElement(
        type=etype, text=text, source_document=source,
        page_number=page, content_hash=str(hash(text))[:8],
    )


class TestLayoutChunk:
    def test_page_range_single_page(self):
        c = LayoutChunk(text="x", html="x", element_types=["text"],
                        source_document="d.pdf", page_start=5, page_end=5)
        assert c.page_range == "5"

    def test_page_range_multi_page(self):
        c = LayoutChunk(text="x", html="x", element_types=["table"],
                        source_document="d.pdf", page_start=3, page_end=7, is_continuation=True)
        assert c.page_range == "3-7"

    def test_page_range_no_page(self):
        c = LayoutChunk(text="x", html="x", element_types=["text"], source_document="d.pdf")
        assert c.page_range == "?"


class TestHTMLWrappers:
    def test_wrap_table_no_caption(self):
        html = _wrap_table("| A | B |\n| 1 | 2 |")
        assert "<table" in html
        assert "financial-table" in html
        assert "<caption>" not in html

    def test_wrap_table_with_caption(self):
        html = _wrap_table("data", caption="Table 1: Revenue Summary")
        assert "<caption>Table 1: Revenue Summary</caption>" in html

    def test_wrap_figure_with_caption(self):
        html = _wrap_figure("Chart description", caption="Figure 2: Revenue Trend")
        assert "<figcaption>Figure 2: Revenue Trend</figcaption>" in html
        assert "<figure" in html

    def test_wrap_section_with_heading(self):
        html = _wrap_section("Some text", heading="Item 1A. Risk Factors")
        assert "<h2>Item 1A. Risk Factors</h2>" in html
        assert "<section>" in html


class TestLayoutAwareParser:
    def setup_method(self):
        self.parser = LayoutAwareParser()

    def test_parse_empty_returns_empty(self):
        assert self.parser.parse([]) == []

    def test_parse_single_text(self):
        elems = [_make_elem("Revenue was $23.35B in Q3 2023.")]
        chunks = self.parser.parse(elems)
        assert len(chunks) >= 1
        assert "Revenue was $23.35B" in chunks[0].text

    def test_parse_table_gets_wrapped(self):
        elems = [_make_elem("| Revenue | $23.35B |", etype="table", page=5)]
        chunks = self.parser.parse(elems)
        assert len(chunks) == 1
        assert chunks[0].element_types == ["table"]
        assert "<table" in chunks[0].html

    def test_parse_graph_gets_wrapped(self):
        elems = [_make_elem("Bar chart showing revenue growth.", etype="graph", page=3)]
        chunks = self.parser.parse(elems)
        assert len(chunks) == 1
        assert chunks[0].element_types == ["graph"]
        assert "<figure" in chunks[0].html

    def test_parse_caption_paired_with_table(self):
        elems = [
            _make_elem("Table 1: Revenue by Quarter", etype="text", page=4),
            _make_elem("| Q1 | Q2 | Q3 |\n| 20 | 21 | 23 |", etype="table", page=4),
        ]
        chunks = self.parser.parse(elems)
        table_chunks = [c for c in chunks if "table" in c.element_types]
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata.get("caption") is not None

    def test_parse_multi_page_table_merged(self):
        table_p1 = _make_elem("| Header A | Header B |\n| data1 | data2 |", etype="table", page=5)
        table_p2 = _make_elem("| data3 | data4 |\n| data5 | data6 |", etype="table", page=6)
        chunks = self.parser.parse([table_p1, table_p2])
        table_chunks = [c for c in chunks if "table" in c.element_types]
        assert len(table_chunks) == 1  # merged
        assert table_chunks[0].is_continuation is True
        assert table_chunks[0].page_start == 5
        assert table_chunks[0].page_end == 6

    def test_parse_heading_sets_section_heading(self):
        elems = [
            _make_elem("RISK FACTORS", etype="text", page=10),
            _make_elem("We face intense competition from legacy automakers.", etype="text", page=10),
        ]
        chunks = self.parser.parse(elems)
        text_chunks = [c for c in chunks if "text" in c.element_types]
        assert any(c.heading is not None for c in text_chunks)

    def test_to_document_elements_preserves_html(self):
        elems = [_make_elem("| Revenue | $23B |", etype="table")]
        chunks = self.parser.parse(elems)
        doc_elems = self.parser.to_document_elements(chunks, tenant_id="acme")
        assert len(doc_elems) == 1
        assert "<table" in doc_elems[0].text  # HTML is embedded in text
        assert doc_elems[0].tenant_id == "acme"

    def test_large_text_chunk_split_respects_limit(self):
        long_text = "word " * 1000  # 5000 chars
        elems = [_make_elem(long_text)]
        chunks = self.parser.parse(elems)
        # Each chunk should respect max_chunk_chars
        for chunk in chunks:
            assert len(chunk.text) <= 4200  # allow some headroom

    def test_mixed_document(self):
        elems = [
            _make_elem("MANAGEMENT'S DISCUSSION AND ANALYSIS", page=1),
            _make_elem("Revenue grew 9% year-over-year.", page=1),
            _make_elem("Figure 1: Revenue Trend", page=2),
            _make_elem("Bar chart showing quarterly revenue.", etype="graph", page=2),
            _make_elem("Table 2: Key Metrics", page=3),
            _make_elem("| EPS | $0.53 |\n| Revenue | $23.35B |", etype="table", page=3),
        ]
        chunks = self.parser.parse(elems)
        types = {tuple(c.element_types) for c in chunks}
        assert ("text",) in types
        assert ("graph",) in types
        assert ("table",) in types
