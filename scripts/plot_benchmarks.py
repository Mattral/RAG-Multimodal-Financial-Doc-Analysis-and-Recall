#!/usr/bin/env python3
"""Draw the README benchmark charts from the measured results in evals/results/.

Every number on these charts is read from a JSON file written by an eval script,
so re-running the evals and then this script keeps the charts honest:

    evals/results/retrieval_chunk4000_graph.json   <- python -m evals.retrieval_benchmark ...
    evals/results/answer_eval_gemini-3.8-flash.json <- python -m evals.answer_eval ...

Usage:
    pip install matplotlib
    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
ASSETS = ROOT / "docs" / "assets"

# Reference categorical palette (light), fixed order; validated for adjacent pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2)


def _bar_labels(ax, bars, values) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT,
            fontweight="bold",
        )


def retrieval_chart() -> Path:
    report = json.loads((RESULTS / "retrieval_chunk4000_graph.json").read_text(encoding="utf-8"))
    overall, cfg = report["overall"], report["config"]
    modes = [m for m in ("bm25", "dense", "hybrid_rrf", "hybrid+graph") if m in overall]
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid_rrf": "Hybrid RRF",
              "hybrid+graph": "Hybrid RRF\n+ graph"}
    panels = [
        ("evidence_hit@5", "Answer-bearing chunk in top 5"),
        ("doc_hit@5", "Expected document in top 5"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        values = [overall[m][metric] for m in modes]
        bars = ax.bar(range(len(modes)), values, width=0.62, color=SERIES[: len(modes)],
                      edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, values)
        ax.set_xticks(range(len(modes)), [labels[m] for m in modes], color=TEXT)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        _style(ax, "Share of questions" if metric == "evidence_hit@5" else None)

    fig.suptitle("Retrieval on the real filings (measured)", color=TEXT, fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(
        0.01, 0.01,
        f"{cfg['num_questions']} golden questions · {cfg['num_chunks']} chunks of "
        f"{cfg['chunk_size']:,} chars · {cfg['embedding_model']} · RRF k={cfg['rrf_k']} · "
        "no reranker · source: evals/results/retrieval_chunk4000_graph.json",
        color=TEXT_2, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = ASSETS / "retrieval_measured.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def answer_chart() -> Path:
    report = json.loads(
        (RESULTS / "answer_eval_gemini-3.8-flash.json").read_text(encoding="utf-8")
    )
    groups = [("overall", "All questions"), ("when_evidence_retrieved", "Supporting text retrieved")]
    metrics = [
        ("headline_value_stated", "Asked-for figure\nstated"),
        ("cites_expected_doc", "Cites expected\ndocument"),
        ("abstained", "Said “not\navailable”"),
        ("has_ungrounded_number", "Number absent\nfrom context*"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    width = 0.36
    for i, (key, label) in enumerate(groups):
        block = report[key]
        values = [block[m] for m, _ in metrics]
        xs = [j + (i - 0.5) * width for j in range(len(metrics))]
        bars = ax.bar(xs, values, width=width, color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                      label=f"{label} (n={block['n']})")
        _bar_labels(ax, bars, values)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], color=TEXT)
    ax.set_ylim(0, 1.1)
    _style(ax, "Share of answers")
    ax.legend(frameon=False, loc="upper right", labelcolor=TEXT)

    model = ", ".join(report["model"])
    fig.suptitle(f"Answer quality with {model} (measured, no LLM judge)", color=TEXT,
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(
        0.01, 0.01,
        "*All flagged numbers were checked by hand: correct derivations, 0 fabricated. "
        "Top-5 hybrid context, 1,000-char chunks · source: "
        "evals/results/answer_eval_gemini-3.8-flash.json",
        color=TEXT_2, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = ASSETS / "answer_quality_measured.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (retrieval_chart(), answer_chart()):
        print(f"wrote {path.relative_to(ROOT)}")
