"""app.py — Multimodal Financial RAG · Hugging Face Space (v2.0)

Production-grade RAG for financial documents: chart understanding, hybrid RRF
retrieval, numeric guardrails, source citations.

Model currency (v2.0): defaults to gemini-2.5-flash — Google retired
gemini-2.0-flash and the gemini-1.5-* family in favor of the 2.5/3.x
generations. See utils/generator.py for the full pricing/model table.

GitHub: https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall
Full source of truth: src/rag_system/ in the same repository.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from utils.generator import GenerationResult, generate
from utils.guardrails import GuardrailResult, run_guardrails
from utils.pdf_processor import IngestResult, ingest_pdf
from utils.retriever import EmbeddingModel, VectorIndex

# ── Singletons ────────────────────────────────────────────────────────────────
_embedding_model: Optional[EmbeddingModel] = None
_vector_index: Optional[VectorIndex] = None
_ingested_filename: Optional[str] = None


def _get_embedder() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel("BAAI/bge-small-en-v1.5")
    return _embedding_model


# ── Config — current stable models (v2.0) ─────────────────────────────────────
PROVIDER_MODELS = {
    "Google Gemini (Free tier available)": [
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash", "gemini-3.1-flash-lite",
    ],
    "OpenAI": [
        "gpt-4o-mini", "gpt-4o",
    ],
}

EXAMPLE_QUESTIONS = [
    "What was total revenue in the most recent quarter?",
    "How did gross margin change year-over-year?",
    "What are the key risk factors related to competition?",
    "What guidance did management provide for next quarter?",
    "Describe any charts or tables showing revenue trends.",
    "What was earnings per share (EPS)?",
]

CUSTOM_CSS = """
:root { --color-accent: #6366f1; }
.source-card { border-left: 3px solid var(--color-accent); padding-left: 12px; margin: 10px 0; }
.pipeline-log { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; line-height: 1.7; }
#component-0 { max-width: 1400px; margin: 0 auto; }
.tab-nav { font-weight: 600; }
"""

GITHUB_URL = "https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall"


# ── Vision helper ─────────────────────────────────────────────────────────────

def _make_vision_fn(provider: str, api_key: str):
    prompt = (
        "You are analyzing a page from a financial report. "
        "Extract ALL numeric data: axis values, data points, table cells, "
        "chart titles, legend entries, footnotes. Be exhaustive."
    )
    use_openai = "openai" in provider.lower() or "gpt" in provider.lower()

    def vision_fn(image) -> str:
        import base64
        import io

        import httpx
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        if use_openai:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                ]}],
                "max_tokens": 600,
            }
            with httpx.Client(timeout=60) as c:
                r = c.post("https://api.openai.com/v1/chat/completions",
                           headers={"Authorization": f"Bearer {api_key}"}, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        else:
            # Current stable Gemini vision model (v2.0 — see utils/generator.py note)
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            payload = {"contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": b64}},
            ]}], "generationConfig": {"maxOutputTokens": 600}}
            with httpx.Client(timeout=60) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    return vision_fn


# ── Ingest ────────────────────────────────────────────────────────────────────

def do_ingest(pdf_path: str, enable_vision: bool, provider: str, api_key: str):
    global _vector_index, _ingested_filename

    if not pdf_path:
        return "### No file selected\nPlease upload a PDF above.", gr.update(visible=False)

    vision_fn = None
    if enable_vision and api_key and api_key.strip():
        with contextlib.suppress(Exception):
            vision_fn = _make_vision_fn(provider, api_key)

    try:
        result: IngestResult = ingest_pdf(pdf_path, process_vision=enable_vision, vision_fn=vision_fn)
    except Exception as exc:
        return f"### Ingestion failed\n\n```\n{str(exc)[:400]}\n```", gr.update(visible=False)

    _ingested_filename = result.filename

    try:
        _vector_index = VectorIndex(embedding_model=_get_embedder())
        index_steps = _vector_index.build(result.chunks)
    except Exception as exc:
        return f"### Indexing failed\n\n```\n{str(exc)[:400]}\n```", gr.update(visible=False)

    all_steps = result.processing_steps + [""] + index_steps
    steps_block = "\n".join(f"  {s}" for s in all_steps)

    summary = (
        f"### `{result.filename}` ready for querying\n\n"
        f"| | |\n|---|---|\n"
        f"| **Pages processed** | {result.num_pages} |\n"
        f"| **Text chunks** | {result.num_chunks - result.num_tables - result.num_charts} |\n"
        f"| **Tables extracted** | {result.num_tables} |\n"
        f"| **Visual descriptions** | {result.num_charts} |\n"
        f"| **Total indexed** | {result.num_chunks} chunks |\n\n"
        f"<details><summary>Full pipeline log (click to expand)</summary>\n\n"
        f"```\n{steps_block}\n```\n\n</details>\n\n"
        f"---\n*Enter a question below and click **Analyze** ↓*"
    )
    return summary, gr.update(visible=True)


# ── Query ─────────────────────────────────────────────────────────────────────

def do_query(
    question: str, provider: str, model: str, api_key: str, top_k: int, enable_guardrails: bool,
) -> Tuple[str, str, str, str, str]:

    if not question or not question.strip():
        msg = "### Please enter a question."
        return msg, msg, msg, msg, msg

    if _vector_index is None:
        msg = "### No document indexed — please process a PDF first."
        return msg, msg, msg, msg, msg

    chunks, retrieval_steps = _vector_index.search(question.strip(), top_k=int(top_k))
    if not chunks:
        msg = "### No relevant chunks found. Try rephrasing your question."
        return msg, msg, msg, msg, msg

    prov_key = "openai" if "openai" in provider.lower() else "gemini"
    gen: GenerationResult = generate(question, chunks, prov_key, model, api_key or "")

    if enable_guardrails:
        guard: GuardrailResult = run_guardrails(question, gen.answer, [c.text for c in chunks])
    else:
        guard = GuardrailResult(
            overall_passed=True, numeric_grounding_passed=True,
            pii_detected=False, injection_detected=False,
            ungrounded_numbers=[], pii_entities=[], redacted_query=None,
            details=["Guardrails disabled by user setting."], warnings=[],
        )

    return (
        _fmt_answer(gen, guard), _fmt_sources(chunks),
        _fmt_pipeline(retrieval_steps, gen), _fmt_guardrails(guard),
        _fmt_metrics(gen, chunks),
    )


def _fmt_answer(gen: GenerationResult, guard: GuardrailResult) -> str:
    badge = f"**Guardrails: {'All passed' if guard.overall_passed else 'Warnings — see Guardrails tab'}**"
    md = f"## Answer\n\n{gen.answer}\n\n---\n{badge}"
    for w in guard.warnings:
        md += f"\n\n> {w}"
    if guard.redacted_query:
        md += "\n\n> Your query contained PII — it was redacted before processing."
    return md


def _fmt_sources(chunks) -> str:
    if not chunks:
        return "No sources retrieved."
    type_icons = {"text": "[text]", "table": "[table]", "chart_description": "[chart]"}
    md = f"## Retrieved Sources ({len(chunks)} chunks)\n\n"
    for r in chunks:
        icon = type_icons.get(r.chunk.chunk_type, "[text]")
        md += (
            f"### {icon} {r.source}\n\n"
            f"*Rank #{r.rank} · RRF score: `{r.rrf_score:.5f}` "
            f"(dense: `{r.dense_score:.3f}`, BM25: `{r.bm25_score:.3f}`)*\n\n"
            f"> {r.text[:450]}{'...' if len(r.text) > 450 else ''}\n\n---\n\n"
        )
    return md


def _fmt_pipeline(retrieval_steps, gen: GenerationResult) -> str:
    md = "## Pipeline Transparency\n\n"
    md += f"*Document: `{_ingested_filename or 'unknown'}`*\n\n"
    md += "### Stage 1 — Document Ingestion\n"
    md += "_Text extraction, table detection, semantic chunking, embedding → FAISS index._\n\n"
    md += "### Stage 2 — Hybrid Retrieval\n"
    for s in retrieval_steps:
        md += f"- {s}\n"
    md += "\n### Stage 3 — LLM Generation\n"
    for s in gen.steps:
        md += f"- {s}\n"
    md += "\n### Stage 4 — Guardrails\n_See Guardrails tab for detailed results._\n\n"
    md += "---\n### RRF Algorithm\n"
    md += (
        "```\nscore(d) = 0.7 / (60 + dense_rank(d) + 1)\n"
        "         + 0.3 / (60 + bm25_rank(d)  + 1)\n```\n\n"
        "k=60 per Cormack et al. (2009). "
        f"[See production implementation →]({GITHUB_URL}/blob/main/src/rag_system/components/retriever/__init__.py)"
    )
    return md


def _fmt_guardrails(g: GuardrailResult) -> str:
    md = "## Guardrail Results\n\n"
    md += "\n".join(g.details)
    if g.warnings:
        md += "\n\n### Warnings\n"
        for w in g.warnings:
            md += f"\n- {w}"
    md += (
        "\n\n---\n### Guardrail Chain\n\n"
        "| Check | Method | What it catches |\n|---|---|---|\n"
        "| Injection detection | Regex pattern matching | Jailbreak attempts |\n"
        "| PII redaction | Regex (SSN, IBAN, CUSIP, ISIN, card, email, phone) | Accidental PII |\n"
        "| Numeric grounding | Extract all numbers → verify each appears in context | Hallucinated figures |\n\n"
        f"[Production implementation →]({GITHUB_URL}/blob/main/src/rag_system/components/guardrails/__init__.py)"
    )
    return md


def _fmt_metrics(gen: GenerationResult, chunks) -> str:
    chunk_types: dict = {}
    for r in chunks:
        chunk_types[r.chunk.chunk_type] = chunk_types.get(r.chunk.chunk_type, 0) + 1
    md = "## Performance Metrics\n\n"
    md += "| Metric | Value |\n|---|---|\n"
    md += f"| **Provider** | {gen.provider} · `{gen.model}` |\n"
    md += f"| **Latency** | {gen.latency_ms:.0f} ms |\n"
    md += f"| **Prompt tokens** | {gen.prompt_tokens:,} |\n"
    md += f"| **Completion tokens** | {gen.completion_tokens:,} |\n"
    md += f"| **Estimated cost** | ${gen.cost_usd:.5f} USD |\n"
    md += f"| **Chunks retrieved** | {len(chunks)} |\n"
    for ct, cnt in chunk_types.items():
        md += f"| ↳ {ct} | {cnt} |\n"
    return md


def update_models(provider: str):
    models = PROVIDER_MODELS.get(provider, ["gemini-2.5-flash"])
    return gr.update(choices=models, value=models[0])


# ── UI ────────────────────────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Multimodal Financial RAG",
        theme=gr.themes.Soft(
            primary_hue="indigo", secondary_hue="blue", neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ),
        css=CUSTOM_CSS,
    ) as demo:

        gr.Markdown(f"""
# Multimodal Financial RAG

**Production-grade document intelligence** — charts · tables · hybrid retrieval · numeric guardrails · source citations

[![GitHub](https://img.shields.io/badge/GitHub-Codebase-181717?logo=github&style=flat-square)]({GITHUB_URL})
[![Stars](https://img.shields.io/github/stars/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall?style=social)]({GITHUB_URL})

> Upload a 10-K, 10-Q, or earnings release PDF. Ask a question. Get a grounded, cited answer.
> Every number in the answer is cross-checked against the source.
""")

        with gr.Accordion("Why This System Stands Out", open=False):
            gr.Markdown(f"""
| Feature | What it means for financial analysis |
|---|---|
| **Vision-Language chart extraction** | Describes charts, graphs, and complex tables visually — not just OCR |
| **Hybrid RRF retrieval** | Dense semantic search + BM25 keyword matching fused with RRF |
| **Numeric grounding guardrail** | Every number verified against source text — hallucinations flagged |
| **PII + injection protection** | SSNs, IBANs, CUSIPs redacted; prompt injection blocked |
| **Page-level citations** | Every claim maps to a specific document and page number |
| **Enterprise codebase** | This demo mirrors [`src/rag_system/`]({GITHUB_URL}/tree/main/src/rag_system) |
""")

        gr.Markdown("---")

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=360):

                gr.Markdown("### API Provider")
                with gr.Group():
                    provider_radio = gr.Radio(
                        choices=list(PROVIDER_MODELS.keys()),
                        value="Google Gemini (Free tier available)",
                        label="Choose your LLM provider",
                        info="Gemini has a generous free tier at aistudio.google.com",
                    )
                    model_dd = gr.Dropdown(
                        choices=PROVIDER_MODELS["Google Gemini (Free tier available)"],
                        value="gemini-2.5-flash",
                        label="Model",
                    )
                    api_key_box = gr.Textbox(
                        label="API Key", type="password",
                        placeholder="AIza... (Gemini) or sk-... (OpenAI)",
                        info="Never stored. Sent directly to the provider API from your session.",
                    )

                provider_radio.change(update_models, provider_radio, model_dd)

                gr.Markdown("---")
                gr.Markdown("### Document")
                with gr.Group():
                    pdf_input = gr.File(
                        label="Upload Financial PDF", file_types=[".pdf"], type="filepath",
                    )
                    enable_vision_chk = gr.Checkbox(
                        value=True, label="Visual chart/table extraction (requires API key)",
                        info="Vision LLM describes each page image → richer retrieval",
                    )
                    ingest_btn = gr.Button("Process Document", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Query")

                question_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. What was total revenue in Q3 2023?",
                    lines=2,
                )

                gr.Markdown("**Quick examples:**")
                for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
                    with gr.Row():
                        for ex in EXAMPLE_QUESTIONS[row_start:row_start + 2]:
                            btn = gr.Button(ex[:38] + ("…" if len(ex) > 38 else ""),
                                           size="sm", variant="secondary")
                            btn.click(fn=lambda q=ex: q, outputs=question_box)

                with gr.Accordion("Advanced Settings", open=False):
                    top_k_slider = gr.Slider(
                        minimum=3, maximum=15, value=5, step=1,
                        label="Top-K chunks to retrieve",
                    )
                    guardrails_chk = gr.Checkbox(
                        value=True, label="Enable guardrails (numeric grounding + PII check)",
                    )

                analyze_btn = gr.Button("Analyze & Answer", variant="primary", size="lg")

            with gr.Column(scale=2):

                ingest_status = gr.Markdown(
                    "### Upload a PDF and click **Process Document** to begin."
                )

                with gr.Group(visible=False) as results_group, gr.Tabs():
                    with gr.TabItem("Answer"):
                        answer_out = gr.Markdown(label="")
                    with gr.TabItem("Sources"):
                        sources_out = gr.Markdown(label="")
                    with gr.TabItem("Pipeline"):
                        pipeline_out = gr.Markdown(label="")
                    with gr.TabItem("Guardrails"):
                        guardrail_out = gr.Markdown(label="")
                    with gr.TabItem("Metrics"):
                        metrics_out = gr.Markdown(label="")

        gr.Markdown("---")
        with gr.Accordion("Technical Architecture", open=False):
            gr.Markdown(f"""
```
PDF
 ├── pdfplumber ─────────────────→ text + table extraction (per page)
 └── pdf2image + Vision LLM ─────→ chart/graph descriptions

         ▼
 Semantic chunker (≤800 chars, 100-char overlap, paragraph boundaries)
         ▼
 BAAI/bge-small-en-v1.5 ─────────→ 384-dim embeddings
                          ┌──────→ FAISS IndexFlatIP (cosine similarity)
                          └──────→ BM25Okapi (keyword index)
         ▼
 Query → RRF fusion: 0.7/(60+dense_rank+1) + 0.3/(60+bm25_rank+1)
                    ▼
                top-k chunks
                    ▼
         System prompt + context + question
                    ▼
    OpenAI GPT-4o-mini  OR  Gemini 2.5 Flash
                    ▼
             Generated answer
                    ▼
         Guardrails: injection → PII → numeric grounding
                    ▼
         Grounded answer + citations
```

**Model currency (v2.0)**: defaults to `gemini-2.5-flash`. Google retired
`gemini-2.0-flash` and the `gemini-1.5-*` family — this demo tracks
whichever Gemini generation is currently GA. See `utils/generator.py`
for the full model/pricing table.

**This demo mirrors [`src/rag_system/`]({GITHUB_URL}/tree/main/src/rag_system).**
The full enterprise system additionally includes: multi-tenancy · pgvector/Qdrant ·
Redis semantic cache · LangGraph agentic flow · Program-of-Thought calculator ·
OpenTelemetry · Prometheus/Grafana · Kubernetes · Terraform · RAGAS evaluation · 520+ tests.

[**View the full codebase →**]({GITHUB_URL})
""")

        gr.Markdown("---")
        gr.Markdown(f"""
<div align="center">

### Useful? Star the repo.

[**View Full Codebase**]({GITHUB_URL}) · [**Report a Bug**]({GITHUB_URL}/issues)

*This Space lives in `spaces/rag-financial/` in the main repository and is kept
in sync with the production model lineup.*

MIT License · Built with Gradio · Embeddings: BAAI/bge-small-en-v1.5

</div>
""")

        ingest_btn.click(
            fn=do_ingest,
            inputs=[pdf_input, enable_vision_chk, provider_radio, api_key_box],
            outputs=[ingest_status, results_group],
            show_progress="full",
        )

        output_list = [answer_out, sources_out, pipeline_out, guardrail_out, metrics_out]
        analyze_btn.click(
            fn=do_query,
            inputs=[question_box, provider_radio, model_dd, api_key_box, top_k_slider, guardrails_chk],
            outputs=output_list, show_progress="full",
        )
        question_box.submit(
            fn=do_query,
            inputs=[question_box, provider_radio, model_dd, api_key_box, top_k_slider, guardrails_chk],
            outputs=output_list, show_progress="full",
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
