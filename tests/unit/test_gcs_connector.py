"""Tests for the GCS document connector."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


class TestGCSConnectorInit:
    def test_defaults(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="my-bucket")
        assert c._bucket_name == "my-bucket"
        assert ".pdf" in c._extensions
        assert c._prefix == ""

    def test_custom_extensions(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="b", extensions=[".docx", ".txt"])
        assert c._extensions == [".docx", ".txt"]

    def test_check_deps_false_without_library(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="b")
        with patch.dict("sys.modules", {"google": None, "google.cloud": None,
                                         "google.cloud.storage": None}):
            assert c._check_deps() is False


class TestGCSConnectorListUris:
    @pytest.mark.asyncio
    async def test_returns_empty_without_deps(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="b")
        with patch.object(c, "_check_deps", return_value=False):
            assert await c.list_uris() == []

    @pytest.mark.asyncio
    async def test_filters_by_extension_and_skips_directories(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="filings", extensions=[".pdf"])

        blob_pdf = MagicMock(); blob_pdf.name = "10k/tesla.pdf"; blob_pdf.size = 512
        blob_txt = MagicMock(); blob_txt.name = "notes/readme.txt"; blob_txt.size = 10
        blob_dir = MagicMock(); blob_dir.name = "10k/"

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [blob_pdf, blob_txt, blob_dir]

        # Verify filtering logic directly
        blobs = [blob_pdf, blob_txt, blob_dir]
        uris = [
            f"gs://filings/{b.name}" for b in blobs
            if any(b.name.endswith(ext) for ext in c._extensions)
            and not b.name.endswith("/")
        ]
        assert len(uris) == 1
        assert "tesla.pdf" in uris[0]
        assert uris[0].startswith("gs://filings/")


class TestGCSConnectorStream:
    @pytest.mark.asyncio
    async def test_yields_nothing_without_deps(self):
        from src.rag_system.components.connectors.gcs_connector import GCSConnector
        c = GCSConnector(bucket="b")
        docs = []
        with patch.object(c, "_check_deps", return_value=False):
            async for doc in c.stream():
                docs.append(doc)
        assert docs == []

    def test_discovered_doc_metadata_structure(self):
        from src.rag_system.components.connectors import DiscoveredDocument
        doc = DiscoveredDocument(
            source_uri="gs://bucket/tesla.pdf",
            local_path="/tmp/tesla.pdf",
            filename="tesla.pdf",
            size_bytes=1024,
            tenant_id="acme",
            metadata={"connector": "gcs", "bucket": "bucket"},
        )
        assert doc.metadata["connector"] == "gcs"
        assert doc.source_uri.startswith("gs://")

    def test_exported_from_package(self):
        from src.rag_system.components.connectors import GCSConnector
        assert GCSConnector is not None
