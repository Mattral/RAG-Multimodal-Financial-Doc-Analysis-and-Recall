"""Offline integrity checks for the golden evaluation dataset.

These tests do not call any model or network. They check that every sample
loads into ``EvalSample``, that the ground truth actually states each expected
value, that every sample has a provenance record, and that every recorded
calculation reproduces. ``scripts/verify_golden_dataset.py`` performs the
online check of evidence strings against the primary filings.
"""

from __future__ import annotations

import ast
import json
import operator
from collections import Counter
from pathlib import Path

import pytest

from src.rag_system.components.evaluator import EvalSample, numeric_recall

DATA_DIR = Path(__file__).resolve().parents[2] / "evals" / "golden_datasets"
DATASET = DATA_DIR / "financial_qa.jsonl"
PROVENANCE = DATA_DIR / "provenance.json"

QUESTION_TYPES = {"factual", "calculation", "table", "chart", "comparative"}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a pure arithmetic expression (numbers and + - * / only)."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression: {expr}")

    return walk(ast.parse(expr, mode="eval"))


def _rows() -> list[dict]:
    with open(DATASET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


ROWS = _rows()
PROV = json.loads(PROVENANCE.read_text(encoding="utf-8"))["samples"]


class TestDatasetShape:
    def test_has_at_least_100_samples(self):
        assert len(ROWS) >= 100

    def test_every_row_loads_into_eval_sample(self):
        for row in ROWS:
            EvalSample(**row)  # raises TypeError on unknown or missing keys

    def test_questions_are_unique(self):
        dupes = [q for q, n in Counter(r["question"] for r in ROWS).items() if n > 1]
        assert dupes == []

    def test_required_fields_are_populated(self):
        for row in ROWS:
            assert row["question"].strip() and row["ground_truth"].strip()
            assert row["source_documents"], row["question"]
            page = row["expected_page"]
            assert page is None or (isinstance(page, int) and page > 0)

    def test_first_tag_is_a_question_type(self):
        for row in ROWS:
            assert row["tags"] and row["tags"][0] in QUESTION_TYPES, row["question"]

    def test_question_type_mix(self):
        tags = Counter(t for r in ROWS for t in r["tags"])
        assert tags["calculation"] >= 20
        assert tags["comparative"] >= 20
        assert tags["table"] >= 30
        assert tags["chart"] >= 8

    def test_covers_multiple_companies(self):
        companies = {r["source_documents"][0].split("_")[0] for r in ROWS}
        assert {"tesla", "apple", "msft", "nvidia", "jpmorgan"} <= companies


class TestGroundTruthConsistency:
    @pytest.mark.parametrize("row", ROWS, ids=lambda r: r["question"][:50])
    def test_ground_truth_states_every_expected_value(self, row):
        recall = numeric_recall(row["ground_truth"], row["expected_numeric_values"])
        assert recall in (None, 1.0), (row["expected_numeric_values"], row["ground_truth"])


class TestProvenance:
    def test_every_sample_has_exactly_one_provenance_record(self):
        prov_questions = Counter(p["question"] for p in PROV)
        assert set(prov_questions) == {r["question"] for r in ROWS}
        assert max(prov_questions.values()) == 1

    def test_provenance_sources_match_dataset(self):
        by_q = {p["question"]: p for p in PROV}
        for row in ROWS:
            docs = [s["document"] for s in by_q[row["question"]]["sources"]]
            assert docs == row["source_documents"]

    def test_every_source_has_url_and_evidence(self):
        for record in PROV:
            for src in record["sources"]:
                assert src["url"].startswith("https://")
                assert src["evidence"] and all(e.strip() for e in src["evidence"])

    @pytest.mark.parametrize(
        "calc",
        [c for p in PROV for c in p["calculations"]],
        ids=lambda c: c["expression"][:40],
    )
    def test_calculations_reproduce(self, calc):
        got = _safe_eval(calc["expression"])
        assert abs(got - calc["expected"]) <= calc["tolerance"]
