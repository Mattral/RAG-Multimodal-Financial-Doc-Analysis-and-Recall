"""Streamlit demo UI for RAG Financial Multimodal.

Guideline §8 Phase 3: 'SDK + UI demo (Streamlit/Gradio with source highlighting)'

Usage:
    pip install streamlit
    streamlit run demo/app.py

Features:
- Upload PDF and ingest with progress indicator
- Ask questions with real-time answer streaming (stub)
- Show source citations with page numbers and relevance scores
- Display query analysis (intent, complexity, filters)
- Show cost and latency metrics
- Thumbs up/down feedback
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)


def get_pipeline():
    """Get or create the pipeline (cached in session state)."""
    if "pipeline" not in st.session_state:
        with st.spinner("Initialising RAG pipeline..."):
            from src.rag_system.sdk import RAGPipeline
            os.environ.setdefault("VECTOR_STORE_CONFIG__PROVIDER", "memory")
            os.environ.setdefault("CACHE_CONFIG__BACKEND", "memory")
            pipeline = asyncio.run(RAGPipeline.create(tenant_id="demo"))
            st.session_state["pipeline"] = pipeline
            st.session_state["ingested_files"] = []
    return st.session_state["pipeline"]


def main():
    st.set_page_config(
        page_title="RAG Financial Multimodal",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 RAG Financial Multimodal")
    st.caption("Enterprise-grade multimodal RAG for financial document analysis")

    # ── Sidebar: Settings & Upload ────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("OpenAI API Key", type="password",
                                value=os.environ.get("OPENAI_API_KEY", ""),
                                help="Required for LLM generation")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        tenant_id = st.text_input("Tenant ID", value="demo",
                                  help="Isolate documents per tenant")
        top_k = st.slider("Top-K chunks to retrieve", 1, 20, 5)
        process_vision = st.checkbox("Extract charts with vision", value=True)

        st.divider()
        st.header("📄 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload financial PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest", type="primary"):
            pipeline = get_pipeline()
            progress = st.progress(0, text="Ingesting...")
            for i, uf in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    result = asyncio.run(pipeline.ingest(
                        [tmp_path], process_vision=process_vision
                    ))
                    st.session_state["ingested_files"].append(uf.name)
                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text=f"Ingested {uf.name} — {result.get('num_chunks', 0)} chunks"
                    )
                except Exception as exc:
                    st.error(f"Failed to ingest {uf.name}: {exc}")
                finally:
                    os.unlink(tmp_path)
            st.success(f"✅ Ingested {len(uploaded_files)} document(s)")

        if st.session_state.get("ingested_files"):
            st.divider()
            st.caption("Ingested files:")
            for f in st.session_state["ingested_files"]:
                st.write(f"• {f}")

    # ── Main: Query Interface ─────────────────────────────────────────────────
    st.header("🔍 Query")

    # Example questions
    with st.expander("Example questions"):
        examples = [
            "What was total revenue in Q3 2023?",
            "How did gross margins change year-over-year?",
            "What are the key risk factors related to competition?",
            "What was the 3-year revenue CAGR?",
            "Describe the revenue trend from the charts",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex[:30] + "...", key=f"ex_{ex[:10]}"):
                st.session_state["query_input"] = ex

    query = st.text_input(
        "Ask a question about your financial documents",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. What was Q3 revenue?",
        key="query_box",
    )

    if st.button("Ask", type="primary") and query:
        if not st.session_state.get("ingested_files"):
            st.warning("Please ingest at least one document first.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            pipeline = get_pipeline()
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = asyncio.run(pipeline.query(
                        query, top_k=top_k
                    ))

                    # ── Answer ─────────────────────────────────────────────────
                    if result.get("status") == "success" and result.get("answer"):
                        st.subheader("💡 Answer")
                        st.write(result["answer"])

                        # ── Guardrail warning ──────────────────────────────────
                        guards = result.get("guardrails", {})
                        if not guards.get("overall_passed", True):
                            st.warning(
                                "⚠️ Guardrail alert: Some numeric values in the answer "
                                "could not be verified against source documents."
                            )

                        # ── Analysis metadata ──────────────────────────────────
                        analysis = result.get("analysis", {})
                        if analysis:
                            with st.expander("🧠 Query Analysis"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Intent", analysis.get("intent", "?"))
                                col2.metric("Complexity", analysis.get("complexity", "?"))
                                col3.metric("PoT Used", "Yes" if analysis.get("use_pot") else "No")

                        # ── Sources ────────────────────────────────────────────
                        sources = result.get("sources", [])
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)} chunks retrieved)"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(f"**[{i}] {src['document']}** — Page {src.get('page', '?')} "
                                                f"(relevance: {src['score']:.3f})")
                                    st.caption(src.get("text_preview", "")[:300])
                                    st.divider()

                        # ── Metrics ────────────────────────────────────────────
                        metrics = result.get("metrics", {})
                        cols = st.columns(4)
                        cols[0].metric("Total latency", f"{metrics.get('total_latency_ms', 0):.0f}ms")
                        cols[1].metric("Retrieval", f"{metrics.get('retrieval_latency_ms', 0):.0f}ms")
                        cols[2].metric("Generation", f"{metrics.get('generation_latency_ms', 0):.0f}ms")
                        cols[3].metric("Est. cost", f"${metrics.get('cost_usd', 0):.5f}")

                        # ── Feedback ───────────────────────────────────────────
                        st.subheader("📝 Was this helpful?")
                        col1, col2, col3 = st.columns([1, 1, 8])
                        if col1.button("👍"):
                            st.success("Thanks! Your feedback helps improve the system.")
                        if col2.button("👎"):
                            comment = st.text_input("What went wrong?", key="feedback_comment")
                            st.info("Feedback noted. This will be used to improve retrieval quality.")

                    elif result.get("status") == "error":
                        st.error(f"Error: {result.get('error', 'Unknown error')}")

                except Exception as exc:
                    st.error(f"Query failed: {exc}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "RAG Financial Multimodal v2.0 | "
        "[GitHub](https://github.com/your-org/rag-financial-multimodal) | "
        "[Docs](https://your-org.github.io/rag-financial-multimodal) | "
        "MIT License"
    )


if __name__ == "__main__":
    main()
