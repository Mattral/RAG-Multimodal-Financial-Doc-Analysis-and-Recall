"""Unit tests for enterprise document connectors."""

from pathlib import Path

import pytest

from src.rag_system.components.connectors import (
    DiscoveredDocument,
    LocalFilesystemConnector,
)


@pytest.fixture
def populated_dir(tmp_path):
    """Create a temp directory with PDFs and non-PDFs."""
    (tmp_path / "report_q3.pdf").write_bytes(b"%PDF-1.4 sample")
    (tmp_path / "10k_2023.pdf").write_bytes(b"%PDF-1.4 annual")
    (tmp_path / "notes.txt").write_text("not a pdf")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested_report.pdf").write_bytes(b"%PDF-1.4 nested")
    return tmp_path


class TestLocalFilesystemConnector:

    @pytest.mark.asyncio
    async def test_lists_all_pdfs(self, populated_dir):
        # Explicit extensions=[".pdf"]: the connector's default extension
        # list also includes .docx/.txt (by design, for non-PDF ingestion),
        # and populated_dir includes a notes.txt fixture file — so testing
        # "all PDFs" specifically requires scoping to just .pdf here.
        connector = LocalFilesystemConnector(str(populated_dir), extensions=[".pdf"])
        uris = await connector.list_uris()
        assert len(uris) == 3  # 2 top-level + 1 nested
        assert all(u.endswith(".pdf") for u in uris)

    @pytest.mark.asyncio
    async def test_lists_only_specified_extensions(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir), extensions=[".txt"])
        uris = await connector.list_uris()
        assert len(uris) == 1
        assert uris[0].endswith(".txt")

    @pytest.mark.asyncio
    async def test_stream_yields_discovered_documents(self, populated_dir):
        # Scoped to .pdf only — see test_lists_all_pdfs for why.
        connector = LocalFilesystemConnector(str(populated_dir), extensions=[".pdf"])
        docs = []
        async for doc in connector.stream(tenant_id="test"):
            docs.append(doc)
        assert len(docs) == 3
        assert all(isinstance(d, DiscoveredDocument) for d in docs)

    @pytest.mark.asyncio
    async def test_stream_sets_tenant_id(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        async for doc in connector.stream(tenant_id="acme"):
            assert doc.tenant_id == "acme"

    @pytest.mark.asyncio
    async def test_stream_sets_filename(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        filenames = set()
        async for doc in connector.stream():
            filenames.add(doc.filename)
        assert "report_q3.pdf" in filenames
        assert "10k_2023.pdf" in filenames

    @pytest.mark.asyncio
    async def test_stream_sets_size_bytes(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        async for doc in connector.stream():
            assert doc.size_bytes > 0

    @pytest.mark.asyncio
    async def test_stream_sets_source_uri(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        async for doc in connector.stream():
            assert doc.source_uri.startswith("file://")
            assert doc.filename in doc.source_uri

    @pytest.mark.asyncio
    async def test_stream_sets_local_path(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        async for doc in connector.stream():
            assert Path(doc.local_path).exists()

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        connector = LocalFilesystemConnector(str(tmp_path))
        uris = await connector.list_uris()
        assert uris == []

    @pytest.mark.asyncio
    async def test_connector_metadata_tag(self, populated_dir):
        connector = LocalFilesystemConnector(str(populated_dir))
        async for doc in connector.stream():
            assert doc.metadata.get("connector") == "local_filesystem"


class TestDiscoveredDocument:
    def test_creation(self):
        doc = DiscoveredDocument(
            source_uri="s3://bucket/file.pdf",
            local_path="/tmp/file.pdf",
            filename="file.pdf",
            size_bytes=1024,
            tenant_id="test",
        )
        assert doc.source_uri == "s3://bucket/file.pdf"
        assert doc.metadata == {}

    def test_metadata_default_empty(self):
        doc = DiscoveredDocument(
            source_uri="file:///a.pdf",
            local_path="/a.pdf",
            filename="a.pdf",
            size_bytes=100,
        )
        assert isinstance(doc.metadata, dict)
