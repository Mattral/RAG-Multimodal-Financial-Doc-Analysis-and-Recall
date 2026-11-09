"""Immutable audit logging for compliance and data lineage.

Every ingest and query event is written to an append-only audit log
with a SHA-256 content hash, enabling tamper detection.

Backends: file (default), postgres (stub), S3 (stub).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """Append-only audit trail for RAG operations."""

    def __init__(
        self,
        backend: str = "file",
        log_path: str = "./audit_logs",
        service_name: str = "rag-financial-multimodal",
    ) -> None:
        self.backend = backend
        self.service_name = service_name
        self._log_path = Path(log_path)

        if backend == "file":
            self._log_path.mkdir(parents=True, exist_ok=True)

    def _write_file(self, event: Dict[str, Any]) -> None:
        """Append event to a daily JSONL file."""
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        log_file = self._log_path / f"audit_{day}.jsonl"
        line = json.dumps(event, default=str) + "\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def _build_event(
        self,
        event_type: str,
        tenant_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        event = {
            "event_type": event_type,
            "service": self.service_name,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tenant_id": tenant_id,
            **payload,
        }
        # Append content hash for tamper detection
        event["content_hash"] = _sha256(json.dumps(event, sort_keys=True, default=str))
        return event

    def log_ingest(
        self,
        tenant_id: str,
        source_doc: str,
        num_chunks: int,
        doc_hash: Optional[str] = None,
        parser: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = self._build_event(
            "INGEST",
            tenant_id,
            {
                "source_document": source_doc,
                "num_chunks": num_chunks,
                "doc_hash": doc_hash,
                "parser": parser,
                "metadata": metadata or {},
            },
        )
        if self.backend == "file":
            self._write_file(event)
        logger.info("audit_ingest_logged", tenant_id=tenant_id, source=source_doc)

    def log_query(
        self,
        tenant_id: str,
        query_hash: str,
        answer_hash: str,
        sources_cited: List[str],
        model: str,
        latency_ms: float,
        cost_usd: float,
        guardrail_passed: bool,
    ) -> None:
        event = self._build_event(
            "QUERY",
            tenant_id,
            {
                "query_hash": query_hash,  # hash not raw text for privacy
                "answer_hash": answer_hash,
                "sources_cited": sources_cited,
                "model": model,
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(cost_usd, 6),
                "guardrail_passed": guardrail_passed,
            },
        )
        if self.backend == "file":
            self._write_file(event)

    def log_deletion(self, tenant_id: str, doc_id: str, reason: str = "user_request") -> None:
        """GDPR/CCPA data deletion event."""
        event = self._build_event(
            "DELETION",
            tenant_id,
            {"doc_id": doc_id, "reason": reason},
        )
        if self.backend == "file":
            self._write_file(event)
        logger.info("audit_deletion_logged", tenant_id=tenant_id, doc_id=doc_id)
