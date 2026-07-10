"""Regression tests for demo/app.py — Streamlit widget key uniqueness.

Background (PR #9, github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall):
Example-question buttons originally used `key=f"ex_{ex[:10]}"` — the first 10
characters of the question text. Streamlit requires every widget key to be
unique within a render; any two example questions sharing the same first 10
characters produced a duplicate key, and Streamlit raised
`StreamlitDuplicateElementKey`, crashing the demo on load.

Fix: switch to the loop index (`key=f"ex_{i}"`), which is unique by
construction regardless of question text content or length.

Streamlit itself is not installed in this environment (and the demo needs a
running Streamlit server to fully exercise), so these tests verify the fix
two ways that don't require the framework:
  1. Static source inspection — assert the buggy pattern is gone and the
     fixed pattern is present in demo/app.py.
  2. Direct simulation of Streamlit's key-uniqueness constraint against the
     actual `examples` list in the file, including the literal collision
     scenario from the bug report.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_DEMO_APP_PATH = Path(__file__).resolve().parents[2] / "demo" / "app.py"


@pytest.fixture(scope="module")
def demo_app_source() -> str:
    assert _DEMO_APP_PATH.exists(), f"demo/app.py not found at {_DEMO_APP_PATH}"
    return _DEMO_APP_PATH.read_text()


@pytest.fixture(scope="module")
def examples_list(demo_app_source: str) -> list[str]:
    """Extract the literal `examples = [...]` list from demo/app.py via AST,
    so this test tracks the real production data rather than a hand-copied
    duplicate that could drift out of sync."""
    tree = ast.parse(demo_app_source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "examples"
            and isinstance(node.value, ast.List)
        ):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    pytest.fail("Could not locate `examples = [...]` list in demo/app.py")


# ── Static source inspection ──────────────────────────────────────────────────


class TestExampleButtonKeySource:
    def test_buggy_truncated_text_key_pattern_is_absent(self, demo_app_source: str):
        """The original bug: key derived from ex[:10] (question text prefix)."""
        assert 'key=f"ex_{ex[' not in demo_app_source, (
            "Found the PR #9 regression pattern — example-question button keys "
            "must not be derived from a truncated slice of the question text, "
            "since two questions can share the same prefix and collide."
        )

    def test_enumerate_index_key_pattern_is_present(self, demo_app_source: str):
        """The fix: key derived from the loop index, unique by construction."""
        assert re.search(r'key=f"ex_\{i\}"', demo_app_source), (
            "Expected example-question buttons to use the enumerate index "
            '(key=f"ex_{i}") as the Streamlit widget key.'
        )

    def test_example_buttons_loop_uses_enumerate(self, demo_app_source: str):
        assert "enumerate(zip(cols, examples, strict=True))" in demo_app_source, (
            "Expected the example-button loop to enumerate over "
            "zip(cols, examples) so an index is available for the widget key."
        )


# ── Behavioural simulation of Streamlit's key-uniqueness constraint ──────────


def _simulate_streamlit_render(examples: list[str], key_fn) -> None:
    """Reproduce Streamlit's own duplicate-key check well enough to catch
    this bug class without requiring the framework installed.

    Streamlit raises StreamlitDuplicateElementKey the moment two widgets in
    the same render produce the same key. We simulate that exact check.
    """
    seen_keys: set[str] = set()
    for i, ex in enumerate(examples):
        key = key_fn(i, ex)
        if key in seen_keys:
            raise DuplicateElementKeyError(
                f"Duplicate widget key {key!r} for example {i} ({ex!r}); "
                f"already used by an earlier button in this render."
            )
        seen_keys.add(key)


class DuplicateElementKeyError(Exception):
    """Stand-in for streamlit.errors.StreamlitDuplicateElementKey."""


class TestExampleButtonKeyUniqueness:
    def test_current_examples_list_produces_unique_keys(self, examples_list: list[str]):
        """The actual production examples list must never collide under the
        fixed (index-based) key scheme."""

        def fixed_key_fn(i, ex):
            return f"ex_{i}"

        _simulate_streamlit_render(examples_list, fixed_key_fn)  # must not raise

    def test_index_based_keys_never_collide_regardless_of_content(self):
        """Property-style check: index-based keys are unique by construction
        for ANY list of questions, including pathological cases — identical
        questions, empty strings, or many questions sharing a long prefix."""
        pathological_cases = [
            ["What was Q3 revenue?", "What was Q3 revenue?"],  # exact duplicates
            ["", "", ""],  # empty strings
            ["What was Q3 2023 revenue?", "What was Q3 2023 margin?"],  # PR #9's actual collision
            [f"Question variant number {i} about quarterly results" for i in range(50)],
        ]

        def fixed_key_fn(i, ex):
            return f"ex_{i}"

        for case in pathological_cases:
            _simulate_streamlit_render(case, fixed_key_fn)  # must not raise for any case

    def test_truncated_text_key_scheme_reproduces_the_original_bug(self):
        """Sanity check on the test harness itself: confirm the OLD (buggy)
        key scheme genuinely does collide on a realistic pair of questions
        sharing a 10-character prefix, proving this regression test would
        have caught PR #9's bug if it had existed beforehand."""
        colliding_questions = [
            "What was Q3 2023 revenue?",
            "What was Q3 2023 margin?",  # same first 10 chars: "What was Q"
        ]
        assert colliding_questions[0][:10] == colliding_questions[1][:10]

        def buggy_key_fn(i, ex):
            return f"ex_{ex[:10]}"

        with pytest.raises(DuplicateElementKeyError):
            _simulate_streamlit_render(colliding_questions, buggy_key_fn)

    def test_truncated_text_key_scheme_fails_on_real_examples_if_extended(self):
        """Guard against a future edit to the examples list reintroducing a
        collision risk: if someone reverts to slice-based keys, two
        questions starting with "What was the " would collide. This test
        documents that risk explicitly using realistic financial questions,
        independent of whatever the current `examples` list happens to be."""
        realistic_extension = [
            "What was the 3-year revenue CAGR?",
            "What was the gross margin in Q3?",  # same first 10 chars as above
        ]
        assert realistic_extension[0][:10] == realistic_extension[1][:10]

        def buggy_key_fn(i, ex):
            return f"ex_{ex[:10]}"

        with pytest.raises(DuplicateElementKeyError):
            _simulate_streamlit_render(realistic_extension, buggy_key_fn)
