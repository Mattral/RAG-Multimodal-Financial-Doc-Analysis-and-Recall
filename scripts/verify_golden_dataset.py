#!/usr/bin/env python3
"""Verify the golden dataset against its primary sources.

Downloads every source listed in evals/golden_datasets/provenance.json (SEC
EDGAR filings, company press releases and investor decks), extracts the text
and checks that each recorded evidence string appears in it. Calculations are
checked offline by tests/unit/test_golden_dataset.py.

Usage:
    python scripts/verify_golden_dataset.py
    python scripts/verify_golden_dataset.py --cache-dir .cache/golden_sources

SEC asks automated clients to identify themselves; set RAG_EVAL_USER_AGENT to
something like "Your Name you@example.com" before running.

Exit codes: 0 all evidence found, 1 evidence missing, 2 download failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.sources import (  # noqa: E402
    DEFAULT_CACHE,
    PROVENANCE,
    extract_text,
    fetch,
    load_provenance,
    normalize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify golden dataset evidence")
    parser.add_argument("--provenance", default=str(PROVENANCE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    args = parser.parse_args()

    samples = load_provenance(Path(args.provenance))
    urls = sorted({src["url"] for s in samples for src in s["sources"]})

    texts = {}
    for url in urls:
        try:
            texts[url] = extract_text(fetch(url, Path(args.cache_dir)))
        except Exception as exc:  # network errors, 403s, timeouts
            print(f"DOWNLOAD FAILED {url}: {exc}")
            return 2
        print(f"fetched {url} ({len(texts[url]):,} chars)")

    missing = 0
    checked = 0
    for sample in samples:
        for src in sample["sources"]:
            for evidence in src["evidence"]:
                checked += 1
                if normalize(evidence) not in texts[src["url"]]:
                    missing += 1
                    print(f"MISSING [{src['document']}] {evidence!r}\n   for: {sample['question']}")

    print(f"\n{checked - missing}/{checked} evidence strings found across {len(samples)} samples")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
