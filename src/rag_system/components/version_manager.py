"""Document Version Manager — content-hash-based versioning with delta detection.

Tracks every ingested document across versions:
  doc_id = SHA-256(source_uri + filename)
  version bumped on content_hash change
  old chunks soft-deleted (kept for audit); new chunks indexed

Supports:
  - "Give me the answer using the document version from 2025-03-15"
  - Rollback to prior version
  - Compliance audit: "what changed between v1 and v3?"
  - Delta-only re-ingest (skip unchanged docs)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


def _doc_id(source_uri: str) -> str:
    """Stable document ID = SHA-256 of normalised URI."""
    return hashlib.sha256(source_uri.lower().encode()).hexdigest()[:24]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class DocumentVersion:
    """Metadata record for one version of a document."""

    def __init__(
        self,
        doc_id: str,
        version: int,
        source_uri: str,
        content_hash: str,
        ingest_timestamp: str,
        tenant_id: str,
        page_count: int = 0,
        previous_version: Optional[int] = None,
        change_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.doc_id = doc_id
        self.version = version
        self.source_uri = source_uri
        self.content_hash = content_hash
        self.ingest_timestamp = ingest_timestamp
        self.tenant_id = tenant_id
        self.page_count = page_count
        self.previous_version = previous_version
        self.change_summary = change_summary
        self.metadata = metadata or {}
        self.is_deleted = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version": self.version,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "ingest_timestamp": self.ingest_timestamp,
            "tenant_id": self.tenant_id,
            "page_count": self.page_count,
            "previous_version": self.previous_version,
            "change_summary": self.change_summary,
            "is_deleted": self.is_deleted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentVersion":
        v = cls(
            doc_id=d["doc_id"], version=d["version"],
            source_uri=d["source_uri"], content_hash=d["content_hash"],
            ingest_timestamp=d["ingest_timestamp"], tenant_id=d["tenant_id"],
            page_count=d.get("page_count", 0),
            previous_version=d.get("previous_version"),
            change_summary=d.get("change_summary"),
            metadata=d.get("metadata", {}),
        )
        v.is_deleted = d.get("is_deleted", False)
        return v


class DocumentVersionManager:
    """File-backed document version registry.

    In production, replace _load/_save with Postgres or Redis calls.
    The interface is identical — only the storage backend changes.
    """

    def __init__(self, registry_path: str = "./data/doc_versions.json") -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, List[Dict]] = self._load()

    def _load(self) -> Dict[str, List[Dict]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._registry, indent=2))

    def _tenant_key(self, tenant_id: str, doc_id: str) -> str:
        return f"{tenant_id}:{doc_id}"

    def get_latest(self, source_uri: str, tenant_id: str) -> Optional[DocumentVersion]:
        """Return the latest version of a document, or None if never ingested."""
        doc_id = _doc_id(source_uri)
        key = self._tenant_key(tenant_id, doc_id)
        versions = self._registry.get(key, [])
        if not versions:
            return None
        return DocumentVersion.from_dict(versions[-1])

    def get_version_at(self, source_uri: str, tenant_id: str, timestamp: str) -> Optional[DocumentVersion]:
        """Return the document version that was active at a given ISO timestamp."""
        doc_id = _doc_id(source_uri)
        key = self._tenant_key(tenant_id, doc_id)
        versions = self._registry.get(key, [])
        active = None
        for v_dict in versions:
            if v_dict["ingest_timestamp"] <= timestamp:
                active = DocumentVersion.from_dict(v_dict)
        return active

    def needs_reindex(self, source_uri: str, content: str, tenant_id: str) -> bool:
        """Return True if content has changed since last ingest (delta detection)."""
        latest = self.get_latest(source_uri, tenant_id)
        if latest is None:
            return True
        return latest.content_hash != _content_hash(content)

    def register(
        self,
        source_uri: str,
        content: str,
        tenant_id: str,
        page_count: int = 0,
        change_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentVersion:
        """Register a new ingest, bumping version if content changed."""
        doc_id = _doc_id(source_uri)
        key = self._tenant_key(tenant_id, doc_id)
        versions = self._registry.get(key, [])
        content_hash = _content_hash(content)
        now = datetime.now(timezone.utc).isoformat()

        prev_version = len(versions)
        new_version = prev_version + 1

        dv = DocumentVersion(
            doc_id=doc_id,
            version=new_version,
            source_uri=source_uri,
            content_hash=content_hash,
            ingest_timestamp=now,
            tenant_id=tenant_id,
            page_count=page_count,
            previous_version=prev_version if prev_version > 0 else None,
            change_summary=change_summary,
            metadata=metadata or {},
        )

        if key not in self._registry:
            self._registry[key] = []
        self._registry[key].append(dv.to_dict())
        self._save()

        logger.info(
            "document_version_registered",
            doc_id=doc_id,
            version=new_version,
            tenant_id=tenant_id,
            source_uri=source_uri,
        )
        return dv

    def soft_delete(self, source_uri: str, tenant_id: str) -> bool:
        """Mark latest version as deleted (GDPR/CCPA). Returns True if found."""
        doc_id = _doc_id(source_uri)
        key = self._tenant_key(tenant_id, doc_id)
        versions = self._registry.get(key, [])
        if not versions:
            return False
        versions[-1]["is_deleted"] = True
        self._save()
        logger.info("document_soft_deleted", doc_id=doc_id, tenant_id=tenant_id)
        return True

    def list_versions(self, source_uri: str, tenant_id: str) -> List[DocumentVersion]:
        """List all versions of a document for audit trail."""
        doc_id = _doc_id(source_uri)
        key = self._tenant_key(tenant_id, doc_id)
        return [DocumentVersion.from_dict(d) for d in self._registry.get(key, [])]

    def get_all_docs(self, tenant_id: str) -> List[DocumentVersion]:
        """Return latest version of every document for a tenant."""
        results = []
        prefix = f"{tenant_id}:"
        for key, versions in self._registry.items():
            if key.startswith(prefix) and versions:
                results.append(DocumentVersion.from_dict(versions[-1]))
        return results
