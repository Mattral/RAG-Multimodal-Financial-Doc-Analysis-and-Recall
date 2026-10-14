"""Fetch and extract the primary source documents behind the golden dataset.

Shared by scripts/verify_golden_dataset.py (evidence check) and
evals/retrieval_benchmark.py (offline retrieval benchmark), so both see
exactly the same text.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List

PROVENANCE = Path(__file__).parent / "golden_datasets" / "provenance.json"
DEFAULT_CACHE = Path(".cache/golden_sources")
DEFAULT_UA = "rag-financial-golden-verifier (set RAG_EVAL_USER_AGENT)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"


def normalize(text: str) -> str:
    """Collapse table separators, quotes and whitespace so evidence matching is stable."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("/dollarsign", "$ ").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)</(tr|p|div|br|h\d)>", "\n", raw)
    raw = re.sub(r"(?i)</t[dh]>", " | ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return html.unescape(raw).replace("\xa0", " ")


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding noise on Apple's PDF
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


def fetch(url: str, cache_dir: Path = DEFAULT_CACHE) -> bytes:
    """Download ``url`` once and cache it. SEC requests use RAG_EVAL_USER_AGENT."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / hashlib.sha256(url.encode()).hexdigest()[:24]
    if cached.exists():
        return cached.read_bytes()
    ua = os.environ.get("RAG_EVAL_USER_AGENT", DEFAULT_UA) if "sec.gov" in url else BROWSER_UA
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    cached.write_bytes(data)
    return data


def extract_text(data: bytes) -> str:
    """Normalized plain text of a PDF or HTML document."""
    if data[:5] == b"%PDF-":
        return normalize(pdf_to_text(data))
    return normalize(html_to_text(data.decode("utf-8", "ignore")))


def load_provenance(path: Path = PROVENANCE) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["samples"]


def source_urls(samples: List[dict]) -> Dict[str, str]:
    """Map each canonical document name to its URL."""
    return {src["document"]: src["url"] for s in samples for src in s["sources"]}
