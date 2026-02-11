"""Vector indexing component using DeepLake."""

import asyncio
from typing import List, Optional, Dict, Any

from src.rag_system.config import get_config
from src.rag_system.utils.logger import get_logger
from src.rag_system.utils.exceptions import VectorStorageError
from src.rag_system.components.pdf_parser import DocumentElement

logger = get_logger(__name__)


class VectorIndexer:
    """Vector indexing component for DeepLake."""

    def __init__(self):
        """Initialize vector indexer."""
        self.config = get_config()
        self.vector_config = self.config.vector_store_config
        self.vector_store = None
        self.index = None
        logger.info("Vector indexer initialized", config=self.vector_config.dict())

    async def initialize(self) -> None:
        """
        Initialize DeepLake vector store and index.

        Raises:
            VectorStorageError: If initialization fails.
        """
        try:
            logger.info(f"Initializing DeepLake dataset: {self.vector_config.dataset_path}")

            # Run in thread pool to avoid blocking
            await asyncio.to_thread(self._initialize_sync)
            logger.info("DeepLake dataset initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DeepLake: {str(e)}", exc_info=True)
            raise VectorStorageError(
                f"DeepLake initialization failed: {str(e)}",
                dataset_path=self.vector_config.dataset_path,
                details={"error": str(e)},
            )

    def _initialize_sync(self) -> None:
        """Initialize DeepLake synchronously (runs in thread pool)."""
        try:
            from llama_index.vector_stores import DeepLakeVectorStore
            from llama_index.storage.storage_context import StorageContext
            from llama_index import VectorStoreIndex

            self.vector_store = DeepLakeVectorStore(
                dataset_path=self.vector_config.dataset_path,
                runtime={"tensor_db": True},
                read_only=self.vector_config.read_only,
                overwrite=self.vector_config.overwrite,
            )

            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Load existing index or create placeholder
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store, storage_context=storage_context
                )
            except Exception as e:
                logger.warning(
                    "Could not load existing index, will create on first index call",
                    error=str(e),
                )

        except Exception as e:
            logger.error(f"DeepLake sync initialization failed: {str(e)}", exc_info=True)
            raise

    async def index_documents(self, documents: List[DocumentElement]) -> None:
        """
        Index documents into DeepLake.

        Args:
            documents: List of document elements to index.

        Raises:
            VectorStorageError: If indexing fails.
        """
        if not documents:
            logger.warning("No documents to index")
            return

        try:
            logger.info(f"Indexing {len(documents)} documents")

            # Run indexing in thread pool
            await asyncio.to_thread(self._index_documents_sync, documents)
            logger.info(f"Successfully indexed {len(documents)} documents")

        except Exception as e:
            logger.error(f"Document indexing failed: {str(e)}", exc_info=True)
            raise VectorStorageError(
                f"Document indexing failed: {str(e)}",
                dataset_path=self.vector_config.dataset_path,
                details={"num_documents": len(documents), "error": str(e)},
            )

    def _index_documents_sync(self, documents: List[DocumentElement]) -> None:
        """Index documents synchronously (runs in thread pool)."""
        try:
            from llama_index import Document, VectorStoreIndex
            from llama_index.storage.storage_context import StorageContext

            # Convert to LlamaIndex documents
            llama_docs = [
                Document(
                    text=doc.text,
                    metadata={
                        "category": doc.type,
                        "source_document": doc.source_document,
                        **doc.metadata,
                    },
                )
                for doc in documents
            ]

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    llama_docs, storage_context=storage_context
                )
            else:
                # Insert new documents into existing index
                for doc in llama_docs:
                    self.index.insert(doc)

            logger.debug(f"Indexed {len(llama_docs)} documents into vector store")

        except Exception as e:
            logger.error(f"DeepLake sync indexing failed: {str(e)}", exc_info=True)
            raise

    async def query(
        self, query_text: str, top_k: int = 5, use_deep_memory: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the indexed documents.

        Args:
            query_text: Query text.
            top_k: Number of top results.
            use_deep_memory: Whether to use DeepMemory feature.

        Returns:
            List[Dict[str, Any]]: Query results.

        Raises:
            VectorStorageError: If query fails.
        """
        if self.index is None:
            raise VectorStorageError(
                "No documents indexed yet",
                dataset_path=self.vector_config.dataset_path,
            )

        try:
            logger.info(f"Querying: {query_text[:100]}", use_deep_memory=use_deep_memory)

            # Run query in thread pool
            results = await asyncio.to_thread(
                self._query_sync, query_text, top_k, use_deep_memory
            )
            return results

        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            raise VectorStorageError(
                f"Query failed: {str(e)}",
                dataset_path=self.vector_config.dataset_path,
                details={"query": query_text, "error": str(e)},
            )

    def _query_sync(
        self, query_text: str, top_k: int, use_deep_memory: bool
    ) -> List[Dict[str, Any]]:
        """Execute query synchronously (runs in thread pool)."""
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                vector_store_kwargs={"deep_memory": use_deep_memory},
            )

            response = query_engine.query(query_text)

            # Extract and format results
            results = [
                {
                    "text": node.get_content(),
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, "score") else None,
                }
                for node in response.source_nodes
            ]

            return results

        except Exception as e:
            logger.error(f"DeepLake sync query failed: {str(e)}", exc_info=True)
            raise

    async def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict[str, Any]: Dataset statistics.
        """
        try:
            stats = await asyncio.to_thread(self._get_dataset_stats_sync)
            return stats
        except Exception as e:
            logger.warning(f"Failed to get dataset stats: {str(e)}")
            return {"error": str(e)}

    def _get_dataset_stats_sync(self) -> Dict[str, Any]:
        """Get dataset stats synchronously."""
        try:
            if self.vector_store is None or not hasattr(self.vector_store, "vectorstore"):
                return {"status": "not_initialized"}

            ds = self.vector_store.vectorstore.dataset
            return {
                "dataset_path": self.vector_config.dataset_path,
                "num_samples": len(ds),
                "tensors": list(ds.keys()) if hasattr(ds, "keys") else [],
            }
        except Exception as e:
            logger.error(f"Error getting dataset stats: {str(e)}")
            return {"error": str(e)}
