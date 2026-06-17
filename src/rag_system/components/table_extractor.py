"""Structured table extraction to pandas DataFrames / JSON.

Guideline §4: 'Post-processing to extract structured tables (pandas DataFrames
or JSON). Optional integration with code interpreter for calculations.'

Converts LayoutChunk table HTML or markdown pipe tables into:
  - pandas DataFrame (for downstream calculations / PoT injection)
  - JSON (structured storage alongside vector chunks)
  - CSV string (display / export)

Handles: markdown pipe tables, HTML <table> elements, whitespace-delimited.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedTable:
    """A structured financial table extracted from a document."""

    source_document: str
    page_number: Optional[int]
    caption: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_document": self.source_document,
                "page_number": self.page_number,
                "caption": self.caption,
                "headers": self.headers,
                "data": [dict(zip(self.headers, row)) for row in self.rows],
            },
            indent=2,
        )

    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame. Requires pandas."""
        try:
            import pandas as pd
            return pd.DataFrame(
                self.rows, columns=self.headers if self.headers else None
            )
        except ImportError:
            logger.warning("pandas_not_installed", detail="pip install pandas")
            return None

    def to_csv(self) -> str:
        lines = []
        if self.headers:
            lines.append(",".join(f'"{h}"' for h in self.headers))
        for row in self.rows:
            lines.append(",".join(f'"{cell}"' for cell in row))
        return "\n".join(lines)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        if self.headers:
            return len(self.headers)
        return len(self.rows[0]) if self.rows else 0


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_markdown_table(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a markdown pipe table into headers + rows."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return [], []

    def _parse_row(line: str) -> List[str]:
        return [c.strip() for c in line.strip("|").split("|") if c.strip()]

    headers: List[str] = []
    rows: List[List[str]] = []

    for line in lines:
        if "|" not in line:
            continue
        # Skip separator rows (e.g., |---|---|)
        if re.match(r"^\|[\s\-:]+\|", line):
            continue
        parsed = _parse_row(line)
        if not parsed:
            continue
        if not headers:
            headers = parsed
        else:
            rows.append(parsed)

    return headers, rows


def _parse_html_table(html: str) -> Tuple[List[str], List[List[str]]]:
    """Parse an HTML <table> element into headers + rows."""
    try:
        from html.parser import HTMLParser

        class _TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.headers: List[str] = []
                self.rows: List[List[str]] = []
                self._current_row: List[str] = []
                self._in_header = False
                self._in_data = False
                self._current_text = ""

            def handle_starttag(self, tag, attrs):
                if tag == "th":
                    self._in_header = True
                elif tag == "td":
                    self._in_data = True
                elif tag == "tr":
                    self._current_row = []

            def handle_endtag(self, tag):
                if tag == "th":
                    self.headers.append(self._current_text.strip())
                    self._current_text = ""
                    self._in_header = False
                elif tag == "td":
                    self._current_row.append(self._current_text.strip())
                    self._current_text = ""
                    self._in_data = False
                elif tag == "tr" and self._current_row:
                    self.rows.append(self._current_row)

            def handle_data(self, data):
                if self._in_header or self._in_data:
                    self._current_text += data

        p = _TableParser()
        p.feed(html)
        return p.headers, p.rows
    except Exception as exc:
        logger.warning("html_table_parse_failed", error=str(exc))
        return [], []


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class TableExtractor:
    """Extract structured tables from document text chunks.

    Handles:
    - Markdown pipe tables  (| Header | Value |)
    - HTML <table> elements  (from layout parser HTML wrapping)
    """

    def extract_from_text(
        self,
        text: str,
        source_document: str = "unknown",
        page_number: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> Optional[ExtractedTable]:
        """Extract a table from raw text or HTML."""
        if "<table" in text.lower():
            headers, rows = _parse_html_table(text)
        elif "|" in text:
            headers, rows = _parse_markdown_table(text)
        else:
            return None

        if not rows:
            return None

        return ExtractedTable(
            source_document=source_document,
            page_number=page_number,
            caption=caption,
            headers=headers,
            rows=rows,
            raw_text=text,
        )

    def extract_from_elements(self, elements: List[Any]) -> List[ExtractedTable]:
        """Extract all tables from a list of DocumentElements."""
        tables: List[ExtractedTable] = []
        for elem in elements:
            if getattr(elem, "type", "") == "table":
                caption = (
                    elem.metadata.get("caption")
                    if hasattr(elem, "metadata") and elem.metadata
                    else None
                )
                table = self.extract_from_text(
                    text=elem.text,
                    source_document=elem.source_document,
                    page_number=elem.page_number,
                    caption=caption,
                )
                if table:
                    tables.append(table)
        logger.info("table_extraction_complete", num_tables=len(tables))
        return tables

    def to_pot_context(self, table: ExtractedTable, max_rows: int = 20) -> str:
        """Format a table as PoT-ready Python code context.

        Returns a string injected into PoT code so the LLM can reference
        structured data in its calculations:

            # Table: Revenue by Quarter (Page 5)
            # Columns: ['Quarter', 'Revenue', 'YoY']
            data = [
                {'Quarter': 'Q3 2023', 'Revenue': '$23.35B', 'YoY': '9%'},
                ...
            ]
        """
        label = table.caption or table.source_document
        lines = [
            f"# Table: {label} (Page {table.page_number})",
            f"# Columns: {table.headers}",
            "data = [",
        ]
        for row in table.rows[:max_rows]:
            lines.append(f"    {dict(zip(table.headers, row))},")
        if len(table.rows) > max_rows:
            lines.append(f"    # ... {len(table.rows) - max_rows} more rows truncated")
        lines.append("]")
        return "\n".join(lines)
