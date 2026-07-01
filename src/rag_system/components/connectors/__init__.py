"""Enterprise document connectors — S3, Azure Blob, local filesystem."""
from __future__ import annotations
import asyncio, os, tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional
import structlog
logger = structlog.get_logger(__name__)

@dataclass
class DiscoveredDocument:
    source_uri: str
    local_path: str
    filename: str
    size_bytes: int
    last_modified: Optional[str] = None
    tenant_id: str = "default"
    metadata: dict = field(default_factory=dict)

class BaseConnector(ABC):
    @abstractmethod
    async def stream(self, tenant_id: str = "default") -> AsyncIterator[DiscoveredDocument]: ...
    @abstractmethod
    async def list_uris(self) -> List[str]: ...

class LocalFilesystemConnector(BaseConnector):
    def __init__(self, directory: str, extensions: Optional[List[str]] = None):
        self._dir = Path(directory)
        self._extensions = extensions or [".pdf", ".docx", ".txt"]
    async def list_uris(self) -> List[str]:
        return sorted(str(p) for ext in self._extensions for p in self._dir.rglob(f"*{ext}"))
    async def stream(self, tenant_id: str = "default") -> AsyncIterator[DiscoveredDocument]:
        for path_str in await self.list_uris():
            p = Path(path_str)
            yield DiscoveredDocument(source_uri=f"file://{p.absolute()}", local_path=path_str,
                filename=p.name, size_bytes=p.stat().st_size, tenant_id=tenant_id,
                metadata={"connector": "local_filesystem"})

class S3Connector(BaseConnector):
    def __init__(self, bucket: str, prefix: str = "", extensions: Optional[List[str]] = None, region: str = "us-east-1"):
        self._bucket, self._prefix, self._extensions, self._region = bucket, prefix, extensions or [".pdf"], region
    async def list_uris(self) -> List[str]:
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self._region)
            paginator = s3.get_paginator("list_objects_v2")
            return [f"s3://{self._bucket}/{obj['Key']}" for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix)
                    for obj in page.get("Contents", []) if any(obj["Key"].endswith(e) for e in self._extensions)]
        except Exception as exc:
            logger.error("s3_list_failed", error=str(exc)); return []
    async def stream(self, tenant_id: str = "default") -> AsyncIterator[DiscoveredDocument]:
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self._region)
            for uri in await self.list_uris():
                key = uri.replace(f"s3://{self._bucket}/", "")
                with tempfile.NamedTemporaryFile(suffix=Path(key).suffix, delete=False) as tmp:
                    await asyncio.to_thread(s3.download_fileobj, self._bucket, key, tmp)
                    local = tmp.name
                head = await asyncio.to_thread(s3.head_object, Bucket=self._bucket, Key=key)
                yield DiscoveredDocument(source_uri=uri, local_path=local, filename=Path(key).name,
                    size_bytes=head.get("ContentLength", 0), last_modified=str(head.get("LastModified", "")),
                    tenant_id=tenant_id, metadata={"connector": "s3", "bucket": self._bucket})
                os.unlink(local)
        except ImportError:
            logger.warning("boto3_not_installed")

class AzureBlobConnector(BaseConnector):
    def __init__(self, connection_string: str, container: str, prefix: str = "", extensions: Optional[List[str]] = None):
        self._conn_str, self._container, self._prefix = connection_string, container, prefix
        self._extensions = extensions or [".pdf"]
    async def list_uris(self) -> List[str]:
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(self._conn_str)
            return [f"az://{self._container}/{b.name}" for b in client.get_container_client(self._container).list_blobs(name_starts_with=self._prefix)
                    if any(b.name.endswith(e) for e in self._extensions)]
        except ImportError:
            logger.warning("azure_storage_not_installed"); return []
    async def stream(self, tenant_id: str = "default") -> AsyncIterator[DiscoveredDocument]:
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(self._conn_str)
            cc = client.get_container_client(self._container)
            for blob in cc.list_blobs(name_starts_with=self._prefix):
                if not any(blob.name.endswith(e) for e in self._extensions): continue
                with tempfile.NamedTemporaryFile(suffix=Path(blob.name).suffix, delete=False) as tmp:
                    tmp.write(await asyncio.to_thread(cc.get_blob_client(blob.name).download_blob().readall))
                    local = tmp.name
                yield DiscoveredDocument(source_uri=f"az://{self._container}/{blob.name}", local_path=local,
                    filename=Path(blob.name).name, size_bytes=blob.size or 0, tenant_id=tenant_id,
                    metadata={"connector": "azure_blob"})
                os.unlink(local)
        except ImportError:
            logger.warning("azure_storage_not_installed")


# GCS connector — lazy import so google-cloud-storage is optional
try:
    from src.rag_system.components.connectors.gcs_connector import GCSConnector
except ImportError:
    GCSConnector = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseConnector", "DiscoveredDocument",
    "LocalFilesystemConnector", "S3Connector",
    "AzureBlobConnector", "GCSConnector",
]
