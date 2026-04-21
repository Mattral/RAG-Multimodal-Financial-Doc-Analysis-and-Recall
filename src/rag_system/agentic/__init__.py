"""LangGraph agentic flow for complex multi-step financial analysis.

Guideline §8: 'LangGraph agentic RAG flow for complex multi-step financial
queries (identify anomalies, summarise risk factors across multiple 10-Ks).'

Enable via: ENABLE_LANGGRAPH_AGENTIC=true

Flow:
  [Analyze] → [Retrieve] → [Verify] → [Calculate] → [Synthesize] → [Guard]
       ↑__________________________|

The agent can self-correct: if Verify finds numeric discrepancies, it
loops back to Retrieve with refined filters.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import structlog

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """Mutable state threaded through all LangGraph nodes."""
    query: str
    tenant_id: str
    intent: str
    iterations: int
    retrieved_chunks: List[Any]
    pot_result: Optional[Dict[str, Any]]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    guardrail_passed: bool
    error: Optional[str]
    metadata: Dict[str, Any]


MAX_ITERATIONS = 3


def _build_graph(pipeline: Any) -> Any:
    """Build the LangGraph StateGraph for agentic RAG.

    Returns None if langgraph is not installed — falls back to standard pipeline.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.warning(
            "langgraph_not_installed",
            detail="pip install langgraph  — agentic mode unavailable",
        )
        return None

    async def node_analyze(state: AgentState) -> AgentState:
        """Classify query intent and extract metadata filters."""
        from src.rag_system.components.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(state["query"], tenant_id=state["tenant_id"])
        if analysis.is_injection:
            state["error"] = f"Blocked: {analysis.injection_reason}"
            state["final_answer"] = None
            return state
        state["intent"] = analysis.intent.value
        state["metadata"] = {
            "filters": analysis.metadata_filters,
            "use_pot": analysis.use_pot,
            "top_k": analysis.suggested_top_k,
            "rewritten_query": analysis.rewritten_query,
        }
        return state

    async def node_retrieve(state: AgentState) -> AgentState:
        """Retrieve relevant chunks using hybrid RRF retrieval."""
        if state.get("error"):
            return state
        try:
            meta = state.get("metadata", {})
            result = await pipeline.query(
                query_text=meta.get("rewritten_query", state["query"]),
                tenant_id=state["tenant_id"],
                top_k=meta.get("top_k", 10),
                filters=meta.get("filters"),
            )
            state["retrieved_chunks"] = result.get("sources", [])
            state["draft_answer"] = result.get("answer")
        except Exception as exc:
            logger.error("agentic_retrieve_failed", error=str(exc))
            state["error"] = str(exc)
        return state

    async def node_verify(state: AgentState) -> AgentState:
        """Verify numeric claims in draft answer against retrieved context."""
        if state.get("error") or not state.get("draft_answer"):
            return state
        from src.rag_system.components.guardrails import FinancialGuardrails
        g = FinancialGuardrails()
        context_texts = [c.get("text_preview", "") for c in state["retrieved_chunks"]]
        passed, ungrounded = g.check_numeric_grounding(
            state["draft_answer"], context_texts
        )
        state["guardrail_passed"] = passed
        if not passed:
            logger.warning(
                "agentic_numeric_discrepancy",
                ungrounded=ungrounded[:5],
                iteration=state["iterations"],
            )
            # Signal retrieval loop to retry with refined query
            state["metadata"]["refine_query"] = (
                f"{state['query']} Focus specifically on: {', '.join(ungrounded[:3])}"
            )
        return state

    async def node_calculate(state: AgentState) -> AgentState:
        """Run PoT calculator if query requires numeric computation."""
        if state.get("error") or not state["metadata"].get("use_pot"):
            return state
        try:
            from src.rag_system.components.pot_executor import PoTExecutor
            executor = PoTExecutor()
            if state.get("draft_answer"):
                pot_result = await executor.execute_from_llm_response(state["draft_answer"])
                if pot_result.success:
                    state["pot_result"] = {
                        "result": pot_result.result,
                        "formatted": pot_result.formatted(2),
                        "template": pot_result.template_used,
                    }
        except Exception as exc:
            logger.warning("agentic_pot_failed", error=str(exc))
        return state

    async def node_synthesize(state: AgentState) -> AgentState:
        """Produce final grounded answer, injecting PoT result if available."""
        if state.get("error"):
            state["final_answer"] = None
            return state
        answer = state.get("draft_answer", "")
        if state.get("pot_result"):
            pot = state["pot_result"]
            answer += f"\n\n[Calculated value: {pot['formatted']}]"
        state["final_answer"] = answer
        return state

    def should_retry(state: AgentState) -> str:
        """Routing function: retry retrieval or proceed to synthesize."""
        if state.get("error"):
            return "synthesize"
        if not state["guardrail_passed"] and state["iterations"] < MAX_ITERATIONS:
            state["iterations"] += 1
            # Swap refined query for next retrieval
            if state["metadata"].get("refine_query"):
                state["metadata"]["rewritten_query"] = state["metadata"]["refine_query"]
            return "retrieve"
        return "calculate"

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("analyze", node_analyze)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("verify", node_verify)
    graph.add_node("calculate", node_calculate)
    graph.add_node("synthesize", node_synthesize)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges("verify", should_retry, {
        "retrieve": "retrieve",
        "calculate": "calculate",
        "synthesize": "synthesize",
    })
    graph.add_edge("calculate", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


class AgenticRAGPipeline:
    """LangGraph-powered agentic RAG pipeline for complex multi-step queries.

    Falls back to standard pipeline if langgraph is not installed or
    ENABLE_LANGGRAPH_AGENTIC=false.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._graph: Optional[Any] = None

    async def _get_graph(self) -> Optional[Any]:
        if self._graph is None:
            self._graph = _build_graph(self._pipeline)
        return self._graph

    async def query(
        self,
        query_text: str,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute agentic query with self-correction loop."""
        graph = await self._get_graph()
        if graph is None:
            # LangGraph not available — fall back to standard pipeline
            logger.info("agentic_fallback_to_standard_pipeline")
            return await self._pipeline.query(query_text, tenant_id=tenant_id, **kwargs)

        initial_state: AgentState = {
            "query": query_text,
            "tenant_id": tenant_id,
            "intent": "unknown",
            "iterations": 0,
            "retrieved_chunks": [],
            "pot_result": None,
            "draft_answer": None,
            "final_answer": None,
            "guardrail_passed": True,
            "error": None,
            "metadata": {},
        }

        try:
            final_state = await graph.ainvoke(initial_state)
            return {
                "status": "success" if not final_state.get("error") else "error",
                "tenant_id": tenant_id,
                "query": query_text,
                "answer": final_state.get("final_answer"),
                "pot_result": final_state.get("pot_result"),
                "sources": final_state.get("retrieved_chunks", []),
                "guardrails": {"overall_passed": final_state.get("guardrail_passed", True)},
                "agentic": {
                    "iterations": final_state["iterations"],
                    "intent": final_state["intent"],
                },
                "error": final_state.get("error"),
            }
        except Exception as exc:
            logger.error("agentic_graph_failed", error=str(exc))
            return await self._pipeline.query(query_text, tenant_id=tenant_id, **kwargs)
