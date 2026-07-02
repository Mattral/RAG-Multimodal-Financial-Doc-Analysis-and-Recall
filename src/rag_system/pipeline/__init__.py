"""Enterprise RAG Pipeline Orchestrator v2.0.

Orchestrates pluggable components end-to-end:
  Parser → LayoutParser → PIIRedactor → Vision → Embedder → VectorStore
  Query → QueryAnalyzer → HybridRetriever → Reranker → Generator → Guardrails

Integrates: multi-tenancy, versioning, cost tracking, OTel tracing, audit log.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag_system.components.guardrails import FinancialGuardrails, PIIRedactor
from src.rag_system.components.query_analyzer import QueryAnalyzer
from src.rag_system.components.query_analyzer import QueryIntent as QueryIntent
from src.rag_system.components.version_manager import DocumentVersionManager
from src.rag_system.config import get_config
from src.rag_system.utils.audit import AuditLogger
from src.rag_system.utils.cost_tracker import get_cost_tracker
from src.rag_system.utils.logger import get_logger, setup_logging
from src.rag_system.utils.semantic_cache import build_semantic_cache
from src.rag_system.utils.telemetry import (
    async_trace_span,
    record_cache_hit,
    record_ingest,
    record_query,
    setup_telemetry,
)

logger = get_logger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class RAGPipeline:
    """Enterprise-grade multimodal RAG pipeline with pluggable components.

    All components are injected (dependency injection pattern), so any
    parser, vector store, LLM, etc. can be swapped by config with zero
    changes to this class.

    Usage::

        pipeline = await RAGPipeline.create()
        await pipeline.ingest(["report.pdf"], tenant_id="acme")
        result = await pipeline.query("What was Q3 revenue?", tenant_id="acme")
    """

    def __init__(
        self,
        parser=None,
        vision_describer=None,
        embedder=None,
        vector_store=None,
        retriever=None,
        reranker=None,
        generator=None,
        pii_redactor=None,
        guardrails=None,
        audit_logger=None,
        query_analyzer=None,
        version_manager=None,
        semantic_cache=None,
        config=None,
    ) -> None:
        self._config = config or get_config()
        self._parser = parser
        self._vision = vision_describer
        self._embedder = embedder
        self._vector_store = vector_store
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._pii_redactor = pii_redactor or PIIRedactor(
            pii_entities=self._config.security_config.pii_entities,
            enable_financial_patterns=self._config.security_config.enable_financial_redaction,
        )
        self._guardrails = guardrails or FinancialGuardrails()
        self._audit = audit_logger or AuditLogger(
            backend=self._config.security_config.audit_log_backend,
            log_path=self._config.security_config.audit_log_path,
        )
        self._query_analyzer = query_analyzer or QueryAnalyzer()
        self._version_manager = version_manager or DocumentVersionManager()
        self._cost_tracker = get_cost_tracker()
        self._semantic_cache = (
            semantic_cache if semantic_cache is not None
            else build_semantic_cache(self._config.cache_config)
        )

        obs = self._config.observability_config
        setup_telemetry(
            service_name=obs.service_name,
            service_version=obs.service_version,
            otlp_endpoint=obs.otlp_endpoint,
            prometheus_port=obs.prometheus_port,
            sampling_rate=obs.trace_sampling_rate,
        )
        setup_logging(
            level=self._config.logging_config.level,
            format_type=self._config.logging_config.format,
        )
        logger.info("RAGPipeline initialised", environment=self._config.environment)

    @classmethod
    async def create(cls, config=None, **kwargs) -> RAGPipeline:
        """Factory that wires default components from config."""
        cfg = config or get_config()
        try:
            from src.rag_system.components.embedder import build_embedder
            from src.rag_system.components.generator import build_generator
            from src.rag_system.components.parser import build_parser
            from src.rag_system.components.reranker import build_reranker
            from src.rag_system.components.retriever import BM25Index, HybridRetriever
            from src.rag_system.components.vector_store import build_vector_store
            from src.rag_system.components.vision import build_vision_describer

            embedder = kwargs.get("embedder") or build_embedder(cfg.vector_store_config.embedding_provider)
            vector_store = kwargs.get("vector_store") or build_vector_store(cfg.vector_store_config.provider)
            reranker = kwargs.get("reranker") or build_reranker(cfg.reranker_config.provider)
            retriever = kwargs.get("retriever") or HybridRetriever(
                vector_store=vector_store, embedder=embedder,
                reranker=reranker, bm25_index=BM25Index(),
            )
            pipeline = cls(
                parser=kwargs.get("parser") or build_parser(),
                vision_describer=kwargs.get("vision_describer") or build_vision_describer(),
                embedder=embedder,
                vector_store=vector_store,
                retriever=retriever,
                reranker=reranker,
                generator=kwargs.get("generator") or build_generator(cfg.llm_config.provider),
                config=cfg,
            )
        except ImportError as exc:
            logger.warning("component_import_failed", error=str(exc), detail="Using minimal pipeline")
            pipeline = cls(config=cfg)

        if pipeline._vector_store:
            await pipeline._vector_store.initialize()
        return pipeline

    # ── INGEST ────────────────────────────────────────────────────────────────

    async def ingest(
        self,
        file_paths: List[str],
        tenant_id: Optional[str] = None,
        process_vision: bool = True,
        batch_size: Optional[int] = None,
        skip_unchanged: bool = True,
    ) -> Dict[str, Any]:
        """Ingest PDF documents with delta detection, versioning, and full observability."""
        tenant_id = tenant_id or self._config.multi_tenancy_config.default_tenant
        batch_size = batch_size or self._config.batch_size
        start = time.perf_counter()

        async with async_trace_span("ingest_pipeline", {"tenant_id": tenant_id, "num_files": len(file_paths)}):
            # Delta detection — skip unchanged documents
            to_process, skipped = [], []
            for fp in file_paths:
                try:
                    content_preview = Path(fp).read_bytes()[:4096].decode(errors="ignore")
                    if skip_unchanged and not self._version_manager.needs_reindex(fp, content_preview, tenant_id):
                        skipped.append(fp)
                        logger.info("ingest_skipped_unchanged", file=Path(fp).name, tenant_id=tenant_id)
                        continue
                    to_process.append(fp)
                except Exception:
                    to_process.append(fp)

            if not to_process:
                return {"status": "success", "tenant_id": tenant_id,
                        "num_files": 0, "num_chunks": 0, "skipped": len(skipped),
                        "latency_s": round(time.perf_counter() - start, 2)}

            all_elements: List[Any] = []

            # Parse
            if self._parser:
                elements = await self._parser.parse_batch(to_process, tenant_id=tenant_id)
                all_elements.extend(elements)
                logger.info("parsing_complete", num_elements=len(elements))

            # PII redaction
            if self._config.security_config.enable_pii_redaction:
                all_elements = await self._redact_elements(all_elements)

            # Vision processing
            if process_vision and self._vision:
                image_paths = await self._collect_images(to_process)
                if image_paths:
                    vision_elements = await self._process_vision_batched(image_paths, batch_size, tenant_id)
                    all_elements.extend(vision_elements)
                    logger.info("vision_complete", num_graph_elements=len(vision_elements))

            # Embed + store
            if self._embedder and self._vector_store and all_elements:
                texts = [e.text for e in all_elements]
                embeddings = await self._embedder.embed(texts)
                await self._vector_store.upsert(all_elements, embeddings, tenant_id=tenant_id)

            # Version registration + audit
            for fp in to_process:
                try:
                    content_preview = Path(fp).read_bytes()[:4096].decode(errors="ignore")
                    self._version_manager.register(
                        fp, content_preview, tenant_id,
                        page_count=sum(1 for e in all_elements if e.source_document == Path(fp).name),
                    )
                except Exception:
                    pass
                self._audit.log_ingest(
                    tenant_id=tenant_id, source_doc=Path(fp).name,
                    num_chunks=len(all_elements), doc_hash=_sha256(fp),
                    parser=getattr(self._parser, "name", "unknown"),
                )

            latency_s = time.perf_counter() - start
            record_ingest(
                tenant_id=tenant_id,
                parser=getattr(self._parser, "name", "unknown"),
                status="success", num_docs=len(to_process),
                num_chunks=len(all_elements), latency_s=latency_s,
            )
            return {
                "status": "success", "tenant_id": tenant_id,
                "num_files": len(to_process), "num_chunks": len(all_elements),
                "skipped": len(skipped), "latency_s": round(latency_s, 2),
            }

    async def _redact_elements(self, elements: List[Any]) -> List[Any]:
        texts = [e.text for e in elements]
        redacted_pairs = self._pii_redactor.redact_batch(texts)
        result = []
        for elem, (redacted_text, found) in zip(elements, redacted_pairs, strict=True):
            d = elem.model_dump()
            d["text"] = redacted_text
            d["metadata"] = {**d.get("metadata", {}), "pii_redacted": found}
            result.append(elem.__class__(**d))
        return result

    async def _collect_images(self, pdf_paths: List[str]) -> List[str]:
        images: List[str] = []
        for pdf_path in pdf_paths:
            img_dir = Path(pdf_path).parent / (Path(pdf_path).stem + "_images")
            if img_dir.exists():
                images.extend([str(p) for p in img_dir.glob("*.png")])
        return images

    async def _process_vision_batched(self, image_paths: List[str], batch_size: int, tenant_id: str) -> List[Any]:
        if not self._vision:
            return []
        results: List[Any] = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            elements = await self._vision.describe_batch(batch, "vision_batch", tenant_id)
            results.extend(e for e in elements if e is not None)
            if i + batch_size < len(image_paths):
                await asyncio.sleep(0.5)
        return results

    # ── QUERY ─────────────────────────────────────────────────────────────────

    async def query(
        self,
        query_text: str,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a full RAG query with QueryAnalyzer routing, guardrails, and audit."""
        tenant_id = tenant_id or self._config.multi_tenancy_config.default_tenant
        start = time.perf_counter()

        # Quota check
        if not self._cost_tracker.check_quota(
            tenant_id, self._config.multi_tenancy_config.default_tokens_per_month
        ):
            return {"status": "error", "error": "Monthly token quota exceeded", "tenant_id": tenant_id}

        # Query analysis — intent, filters, injection, rewrite
        analysis = self._query_analyzer.analyze(query_text, tenant_id=tenant_id)
        if analysis.is_injection:
            return {"status": "error", "error": f"Query blocked: {analysis.injection_reason}",
                    "tenant_id": tenant_id}

        # Merge analysis-derived filters with caller-supplied filters
        effective_filters = {**(filters or {}), **analysis.metadata_filters}
        effective_top_k = max(top_k, analysis.suggested_top_k)
        effective_query = analysis.rewritten_query

        # Semantic cache check — skip retrieval + generation entirely on a hit.
        # Only attempted for simple/non-PoT queries: numeric/agentic queries
        # have answers that depend on the exact retrieved figures, which is
        # already guaranteed fresh by retrieval, so we don't risk serving a
        # stale calculated value from a semantically-similar but distinct query.
        query_embedding: Optional[List[float]] = None
        if self._semantic_cache and self._embedder and not analysis.use_pot:
            try:
                query_embedding = await self._embedder.embed_query(effective_query)
                cache_result = await self._semantic_cache.get(query_embedding, tenant_id=tenant_id)
                if cache_result.hit and cache_result.answer_payload:
                    record_cache_hit("semantic", tenant_id=tenant_id)
                    logger.info(
                        "semantic_cache_served",
                        tenant_id=tenant_id,
                        similarity=round(cache_result.similarity, 4),
                    )
                    payload = dict(cache_result.answer_payload)
                    payload["query"] = query_text
                    payload["cache"] = {
                        "hit": True,
                        "matched_query": cache_result.matched_query,
                        "similarity": round(cache_result.similarity, 4),
                    }
                    return payload
            except Exception as exc:
                logger.warning("semantic_cache_check_failed", error=str(exc))
                query_embedding = None

        async with async_trace_span("query_pipeline", {
            "tenant_id": tenant_id, "intent": analysis.intent.value,
            "complexity": analysis.complexity.value,
        }):
            # Retrieve
            t0 = time.perf_counter()
            chunks: List[Any] = []
            if self._retriever:
                chunks = await self._retriever.retrieve(
                    query=effective_query, top_k=effective_top_k,
                    filters=effective_filters, tenant_id=tenant_id,
                )
            retrieval_latency_s = time.perf_counter() - t0

            # PoT for numeric queries
            pot_result = None
            if analysis.use_pot and chunks:
                try:
                    from src.rag_system.components.pot_executor import PoTExecutor
                    pot_executor = PoTExecutor()
                    context_text = " ".join(c.text for c in chunks[:3])
                    # Attempt to extract and run any code from context
                    pot_result = await pot_executor.execute_from_llm_response(context_text)
                except Exception:
                    pass

            # Generate
            t1 = time.perf_counter()
            answer = None
            if self._generator and chunks:
                # Inject PoT result into system prompt if available
                effective_system = system_prompt
                if pot_result and pot_result.success:
                    pot_note = f"\n\nCalculated value (exact): {pot_result.formatted(2)}"
                    effective_system = (system_prompt or "") + pot_note
                answer = await self._generator.generate(
                    query=effective_query, context=chunks,
                    tenant_id=tenant_id, system_prompt=effective_system,
                )
            generation_latency_s = time.perf_counter() - t1
            total_latency_s = time.perf_counter() - start

            # Guardrails
            guardrail_results: Dict[str, Any] = {}
            if answer and self._config.security_config.enable_guardrails:
                guardrail_results = self._guardrails.run_all_checks(
                    query=query_text, answer=answer.answer,
                    context_chunks=[c.text for c in chunks],
                )

            # Audit
            if answer:
                self._audit.log_query(
                    tenant_id=tenant_id, query_hash=_sha256(query_text),
                    answer_hash=_sha256(answer.answer),
                    sources_cited=[c.source_document for c in answer.citations],
                    model=answer.model_used, latency_ms=total_latency_s * 1000,
                    cost_usd=answer.estimated_cost_usd,
                    guardrail_passed=guardrail_results.get("overall_passed", True),
                )

            record_query(
                tenant_id=tenant_id, query_mode=self._config.query_mode,
                status="success", total_latency_s=total_latency_s,
                retrieval_latency_s=retrieval_latency_s,
                generation_latency_s=generation_latency_s,
                cost_usd=answer.estimated_cost_usd if answer else 0.0,
                model=answer.model_used if answer else "none",
                prompt_tokens=answer.prompt_tokens if answer else 0,
                completion_tokens=answer.completion_tokens if answer else 0,
                num_chunks=len(chunks),
            )

            response_payload = {
                "status": "success", "tenant_id": tenant_id,
                "query": query_text,
                "analysis": {
                    "intent": analysis.intent.value,
                    "complexity": analysis.complexity.value,
                    "rewritten_query": analysis.rewritten_query,
                    "filters_applied": effective_filters,
                    "use_pot": analysis.use_pot,
                },
                "answer": answer.answer if answer else None,
                "answer_obj": answer,
                "pot_result": {"result": pot_result.result, "code": pot_result.code} if pot_result and pot_result.success else None,
                "sources": [
                    {"document": c.source_document, "page": c.page_number,
                     "score": c.score, "text_preview": c.text[:200]}
                    for c in chunks
                ],
                "guardrails": guardrail_results,
                "metrics": {
                    "total_latency_ms": round(total_latency_s * 1000, 1),
                    "retrieval_latency_ms": round(retrieval_latency_s * 1000, 1),
                    "generation_latency_ms": round(generation_latency_s * 1000, 1),
                    "cost_usd": answer.estimated_cost_usd if answer else 0.0,
                    "num_chunks": len(chunks),
                },
            }

            # Populate semantic cache for future similar queries. Only cache
            # clean successful answers that passed guardrails — never cache
            # a guardrail-flagged or empty answer, since a future "similar"
            # query would then silently inherit the same problem.
            if (
                self._semantic_cache and query_embedding and answer
                and guardrail_results.get("overall_passed", True)
            ):
                try:
                    cacheable_payload = {k: v for k, v in response_payload.items() if k != "answer_obj"}
                    await self._semantic_cache.set(
                        query_text=effective_query,
                        query_embedding=query_embedding,
                        answer_payload=cacheable_payload,
                        tenant_id=tenant_id,
                    )
                except Exception as exc:
                    logger.warning("semantic_cache_store_failed", error=str(exc))

            return response_payload

    # ── DOCUMENT MANAGEMENT ───────────────────────────────────────────────────

    async def list_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all ingested documents and their version info for a tenant."""
        tenant_id = tenant_id or self._config.multi_tenancy_config.default_tenant
        docs = self._version_manager.get_all_docs(tenant_id)
        return [{"source_uri": d.source_uri, "filename": Path(d.source_uri).name,
                 "version": d.version, "content_hash": d.content_hash,
                 "ingest_timestamp": d.ingest_timestamp, "page_count": d.page_count,
                 "is_deleted": d.is_deleted} for d in docs]

    async def delete_document(self, source_uri: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """GDPR/CCPA soft-delete a document."""
        tenant_id = tenant_id or self._config.multi_tenancy_config.default_tenant
        found = self._version_manager.soft_delete(source_uri, tenant_id)
        if found:
            self._audit.log_deletion(tenant_id, source_uri, reason="user_request")
        return {"status": "deleted" if found else "not_found", "source_uri": source_uri}

    # ── HEALTH ────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Return component health for K8s liveness/readiness probes."""
        checks: Dict[str, str] = {}
        for name, component in [
            ("parser", self._parser), ("vision", self._vision),
            ("vector_store", self._vector_store), ("retriever", self._retriever),
            ("generator", self._generator),
        ]:
            if component is None:
                checks[name] = "not_configured"
            elif hasattr(component, "health_check"):
                try:
                    await component.health_check()
                    checks[name] = "ok"
                except Exception as exc:
                    checks[name] = f"error: {str(exc)[:80]}"
            else:
                checks[name] = "ok"
        all_ok = all(v in ("ok", "not_configured") for v in checks.values())
        return {"status": "healthy" if all_ok else "degraded", "components": checks}


async def create_pipeline(**kwargs) -> RAGPipeline:
    return await RAGPipeline.create(**kwargs)
