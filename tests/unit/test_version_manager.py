"""Unit tests for DocumentVersionManager — versioning, delta detection, soft-delete."""
from datetime import UTC

import pytest

from src.rag_system.components.version_manager import (
    DocumentVersion,
    DocumentVersionManager,
    _content_hash,
    _doc_id,
)


@pytest.fixture
def manager(tmp_path):
    return DocumentVersionManager(registry_path=str(tmp_path / "versions.json"))


def test_doc_id_is_deterministic():
    assert _doc_id("s3://bucket/tesla_10k.pdf") == _doc_id("s3://bucket/tesla_10k.pdf")

def test_doc_id_normalises_case():
    assert _doc_id("S3://Bucket/File.pdf") == _doc_id("s3://bucket/file.pdf")

def test_doc_id_different_uris():
    assert _doc_id("s3://a/file.pdf") != _doc_id("s3://b/file.pdf")

def test_content_hash_deterministic():
    text = "Revenue was $23.35B in Q3 2023."
    assert _content_hash(text) == _content_hash(text)

def test_content_hash_sensitive_to_change():
    assert _content_hash("v1 content") != _content_hash("v2 content")


class TestDocumentVersionManager:

    def test_get_latest_returns_none_for_new_doc(self, manager):
        assert manager.get_latest("s3://bucket/new.pdf", "tenant_a") is None

    def test_register_creates_version_1(self, manager):
        dv = manager.register("s3://bucket/tesla.pdf", "content v1", "acme")
        assert dv.version == 1
        assert dv.previous_version is None
        assert dv.tenant_id == "acme"

    def test_register_second_version_bumps(self, manager):
        manager.register("s3://bucket/tesla.pdf", "content v1", "acme")
        dv2 = manager.register("s3://bucket/tesla.pdf", "content v2", "acme")
        assert dv2.version == 2
        assert dv2.previous_version == 1

    def test_get_latest_returns_most_recent(self, manager):
        manager.register("s3://b/f.pdf", "v1", "t1")
        manager.register("s3://b/f.pdf", "v2", "t1")
        latest = manager.get_latest("s3://b/f.pdf", "t1")
        assert latest is not None
        assert latest.version == 2

    def test_needs_reindex_true_for_new_doc(self, manager):
        assert manager.needs_reindex("s3://b/new.pdf", "any content", "t1") is True

    def test_needs_reindex_false_for_unchanged_content(self, manager):
        manager.register("s3://b/f.pdf", "same content", "t1")
        assert manager.needs_reindex("s3://b/f.pdf", "same content", "t1") is False

    def test_needs_reindex_true_for_changed_content(self, manager):
        manager.register("s3://b/f.pdf", "original content", "t1")
        assert manager.needs_reindex("s3://b/f.pdf", "updated content", "t1") is True

    def test_tenant_isolation(self, manager):
        manager.register("s3://b/f.pdf", "tenant a content", "tenant_a")
        manager.register("s3://b/f.pdf", "tenant b content", "tenant_b")
        a = manager.get_latest("s3://b/f.pdf", "tenant_a")
        b = manager.get_latest("s3://b/f.pdf", "tenant_b")
        assert a is not None and b is not None
        assert a.content_hash != b.content_hash

    def test_list_versions_returns_all(self, manager):
        manager.register("s3://b/f.pdf", "v1", "t1")
        manager.register("s3://b/f.pdf", "v2", "t1")
        manager.register("s3://b/f.pdf", "v3", "t1")
        versions = manager.list_versions("s3://b/f.pdf", "t1")
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    def test_soft_delete_marks_latest_deleted(self, manager):
        manager.register("s3://b/f.pdf", "content", "t1")
        result = manager.soft_delete("s3://b/f.pdf", "t1")
        assert result is True
        latest = manager.get_latest("s3://b/f.pdf", "t1")
        assert latest is not None
        assert latest.is_deleted is True

    def test_soft_delete_nonexistent_returns_false(self, manager):
        assert manager.soft_delete("s3://b/nonexistent.pdf", "t1") is False

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "versions.json")
        m1 = DocumentVersionManager(registry_path=path)
        m1.register("s3://b/f.pdf", "content", "t1")
        m2 = DocumentVersionManager(registry_path=path)
        latest = m2.get_latest("s3://b/f.pdf", "t1")
        assert latest is not None
        assert latest.version == 1

    def test_get_all_docs_for_tenant(self, manager):
        manager.register("s3://b/f1.pdf", "c1", "acme")
        manager.register("s3://b/f2.pdf", "c2", "acme")
        manager.register("s3://b/f3.pdf", "c3", "other_tenant")
        docs = manager.get_all_docs("acme")
        assert len(docs) == 2
        assert all(d.tenant_id == "acme" for d in docs)

    def test_change_summary_stored(self, manager):
        manager.register("s3://b/f.pdf", "v1", "t1")
        dv2 = manager.register("s3://b/f.pdf", "v2", "t1", change_summary="Q4 update added")
        assert dv2.change_summary == "Q4 update added"

    def test_page_count_stored(self, manager):
        dv = manager.register("s3://b/f.pdf", "content", "t1", page_count=142)
        assert dv.page_count == 142

    def test_get_version_at_timestamp(self, manager):
        import time
        from datetime import datetime

        manager.register("s3://b/f.pdf", "v1", "t1")
        time.sleep(0.01)
        mid = datetime.now(UTC).isoformat()
        time.sleep(0.01)
        manager.register("s3://b/f.pdf", "v2", "t1")
        v_at = manager.get_version_at("s3://b/f.pdf", "t1", mid)
        assert v_at is not None
        assert v_at.version == 1

    def test_document_version_to_dict_roundtrip(self, manager):
        dv = manager.register("s3://b/f.pdf", "content", "t1", page_count=10)
        d = dv.to_dict()
        restored = DocumentVersion.from_dict(d)
        assert restored.version == dv.version
        assert restored.content_hash == dv.content_hash
        assert restored.page_count == 10
