"""Unit tests for structured table extraction."""

import json

import pytest

from src.rag_system.components.base import DocumentElement
from src.rag_system.components.table_extractor import (
    ExtractedTable,
    TableExtractor,
    _parse_html_table,
    _parse_markdown_table,
)

# ── Markdown parser tests ─────────────────────────────────────────────────────


class TestMarkdownTableParser:
    def test_basic_table(self):
        text = "| Metric | Q3 2023 | Q3 2022 |\n|---|---|---|\n| Revenue | $23.35B | $21.45B |"
        headers, rows = _parse_markdown_table(text)
        assert headers == ["Metric", "Q3 2023", "Q3 2022"]
        assert len(rows) == 1
        assert rows[0] == ["Revenue", "$23.35B", "$21.45B"]

    def test_multi_row_table(self):
        text = (
            "| Metric | Value |\n|---|---|\n"
            "| Revenue | $23.35B |\n"
            "| Gross Margin | 17.9% |\n"
            "| EPS | $0.53 |"
        )
        headers, rows = _parse_markdown_table(text)
        assert headers == ["Metric", "Value"]
        assert len(rows) == 3

    def test_empty_text(self):
        headers, rows = _parse_markdown_table("")
        assert headers == []
        assert rows == []

    def test_no_pipe_chars(self):
        headers, rows = _parse_markdown_table("Plain text without tables")
        assert headers == []
        assert rows == []

    def test_separator_row_skipped(self):
        text = "| A | B |\n|:---|---:|\n| x | y |"
        headers, rows = _parse_markdown_table(text)
        assert headers == ["A", "B"]
        assert rows == [["x", "y"]]

    def test_whitespace_trimmed(self):
        text = "|  Revenue  |  $23.35B  |\n|---|---|\n|  EPS  |  $0.53  |"
        headers, rows = _parse_markdown_table(text)
        assert headers == ["Revenue", "$23.35B"]
        assert rows[0] == ["EPS", "$0.53"]


# ── HTML parser tests ─────────────────────────────────────────────────────────


class TestHTMLTableParser:
    def test_basic_html_table(self):
        html = (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            "<tr><td>Revenue</td><td>$23.35B</td></tr>"
            "</table>"
        )
        headers, rows = _parse_html_table(html)
        assert headers == ["Metric", "Value"]
        assert rows == [["Revenue", "$23.35B"]]

    def test_html_table_multiple_rows(self):
        html = (
            "<table>"
            "<tr><th>Q</th><th>Rev</th></tr>"
            "<tr><td>Q1</td><td>$20B</td></tr>"
            "<tr><td>Q2</td><td>$21B</td></tr>"
            "<tr><td>Q3</td><td>$23B</td></tr>"
            "</table>"
        )
        headers, rows = _parse_html_table(html)
        assert len(rows) == 3

    def test_malformed_html_returns_empty(self):
        headers, rows = _parse_html_table("<not a table>")
        assert headers == []
        assert rows == []

    def test_html_with_caption(self):
        html = (
            "<table class='financial-table'>"
            "<caption>Table 1: Revenue Summary</caption>"
            "<tr><th>Period</th><th>Amount</th></tr>"
            "<tr><td>Q3 2023</td><td>$23.35B</td></tr>"
            "</table>"
        )
        headers, rows = _parse_html_table(html)
        assert headers == ["Period", "Amount"]
        assert len(rows) == 1


# ── ExtractedTable tests ──────────────────────────────────────────────────────


class TestExtractedTable:
    @pytest.fixture
    def table(self):
        return ExtractedTable(
            source_document="tesla_10q.pdf",
            page_number=5,
            caption="Table 1: Key Metrics",
            headers=["Metric", "Q3 2023", "Q3 2022"],
            rows=[
                ["Revenue", "$23.35B", "$21.45B"],
                ["Gross Margin", "17.9%", "25.1%"],
                ["EPS", "$0.53", "$1.05"],
            ],
            raw_text="| Metric | Q3 2023 | Q3 2022 |",
        )

    def test_num_rows(self, table):
        assert table.num_rows == 3

    def test_num_cols(self, table):
        assert table.num_cols == 3

    def test_to_json(self, table):
        j = json.loads(table.to_json())
        assert j["source_document"] == "tesla_10q.pdf"
        assert j["page_number"] == 5
        assert len(j["data"]) == 3
        assert j["data"][0]["Metric"] == "Revenue"

    def test_to_csv(self, table):
        csv = table.to_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "Revenue" in lines[1]

    def test_to_dataframe_returns_df_or_none(self, table):
        result = table.to_dataframe()
        # Either a DataFrame or None (if pandas not installed)
        assert result is None or hasattr(result, "shape")

    def test_to_dataframe_shape(self, table):
        import importlib.util

        if importlib.util.find_spec("pandas") is None:
            pytest.skip("pandas not installed")
        df = table.to_dataframe()
        assert df is not None
        assert df.shape == (3, 3)


# ── TableExtractor tests ──────────────────────────────────────────────────────


class TestTableExtractor:
    def setup_method(self):
        self.extractor = TableExtractor()

    def test_extract_markdown_table(self):
        text = "| Revenue | $23.35B |\n|---|---|\n| EPS | $0.53 |"
        result = self.extractor.extract_from_text(text, source_document="tesla.pdf", page_number=5)
        assert result is not None
        assert result.num_rows == 1
        assert result.source_document == "tesla.pdf"

    def test_extract_html_table(self):
        html = (
            "<table><tr><th>Metric</th><th>Value</th></tr>"
            "<tr><td>Revenue</td><td>$23B</td></tr></table>"
        )
        result = self.extractor.extract_from_text(html, source_document="doc.pdf")
        assert result is not None
        assert result.headers == ["Metric", "Value"]

    def test_extract_plain_text_returns_none(self):
        result = self.extractor.extract_from_text("No table here, just prose.")
        assert result is None

    def test_extract_empty_table_returns_none(self):
        result = self.extractor.extract_from_text("|---|---|")
        assert result is None

    def test_extract_from_elements(self):
        elements = [
            DocumentElement(
                type="table",
                text="| Q | Rev |\n|---|---|\n| Q3 | $23B |",
                source_document="tesla.pdf",
                page_number=5,
            ),
            DocumentElement(
                type="text",
                text="Some narrative text.",
                source_document="tesla.pdf",
                page_number=4,
            ),
        ]
        tables = self.extractor.extract_from_elements(elements)
        assert len(tables) == 1
        assert tables[0].page_number == 5

    def test_extract_from_empty_elements(self):
        assert self.extractor.extract_from_elements([]) == []

    def test_to_pot_context_format(self):
        table = ExtractedTable(
            source_document="tesla.pdf",
            page_number=5,
            caption="Revenue Table",
            headers=["Quarter", "Revenue"],
            rows=[["Q3 2023", "$23.35B"], ["Q2 2023", "$21.45B"]],
            raw_text="",
        )
        code = self.extractor.to_pot_context(table)
        assert "# Table: Revenue Table" in code
        assert "# Columns:" in code
        assert "data = [" in code
        assert "Q3 2023" in code

    def test_to_pot_context_max_rows(self):
        table = ExtractedTable(
            source_document="d.pdf",
            page_number=1,
            caption=None,
            headers=["A", "B"],
            rows=[[str(i), str(i * 2)] for i in range(50)],
            raw_text="",
        )
        code = self.extractor.to_pot_context(table, max_rows=5)
        assert "truncated" in code

    def test_caption_propagated(self):
        elem = DocumentElement(
            type="table",
            text="| Metric | Value |\n|---|---|\n| EPS | $0.53 |",
            source_document="doc.pdf",
            page_number=3,
            metadata={"caption": "Table 3: EPS Summary"},
        )
        tables = self.extractor.extract_from_elements([elem])
        assert len(tables) == 1
        assert tables[0].caption == "Table 3: EPS Summary"
