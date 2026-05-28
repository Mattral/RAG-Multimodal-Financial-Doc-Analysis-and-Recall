"""Asynchronous pipeline orchestrator for multimodal RAG."""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.rag_system.config import get_config
from src.rag_system.utils.logger import get_logger, setup_logging
from src.rag_system.utils.exceptions import RAGException
from src.rag_system.components.pdf_parser import PDFParser, DocumentElement
from src.rag_system.components.vision_processor import VisionProcessor
from src.rag_system.components.vector_indexer import VectorIndexer

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates multimodal RAG pipeline."""

    def __init__(self):
        """Initialize RAG pipeline."""
        self.config = get_config()
        setup_logging(
            level=self.config.logging_config.level,
            format_type=self.config.logging_config.format,
        )

        self.pdf_parser = PDFParser()
        self.vision_processor = VisionProcessor()
        self.vector_indexer = VectorIndexer()

        logger.info("RAG Pipeline initialized", environment=self.config.environment)

    async def ingest_documents(
        self,
        pdf_paths: List[str],
        process_vision: bool = True,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ingest and process PDF documents end-to-end.

        Args:
            pdf_paths: List of PDF file paths.
            process_vision: Whether to process images with GPT-4V.
            batch_size: Batch size for vision processing (default: config.batch_size).

        Returns:
            Dict[str, Any]: Ingestion results with statistics.

        Raises:
            RAGException: If ingestion fails.
        """
        batch_size = batch_size or self.config.batch_size

        try:
            logger.info(
                f"Starting ingestion pipeline for {len(pdf_paths)} PDFs",
                process_vision=process_vision,
            )

            # Phase 1: Parse PDFs
            logger.info("Phase 1: Parsing PDFs")
            pdf_elements = await self.pdf_parser.parse_multiple_pdfs(pdf_paths)
            logger.info(f"Extracted {len(pdf_elements)} elements from PDFs")

            all_elements = pdf_elements.copy()

            # Phase 2: Process images (optional)
            if process_vision:
                logger.info("Phase 2: Processing images with GPT-4V")
                image_paths = await asyncio.to_thread(
                    self._generate_image_paths, pdf_paths
                )

                if image_paths:
                    vision_elements = await self._process_images_batched(
                        image_paths, batch_size
                    )
                    all_elements.extend(vision_elements)
                    logger.info(f"Processed {len(image_paths)} images, found {len(vision_elements)} graphs")
                else:
                    logger.warning("No images generated from PDFs")
            else:
                logger.info("Phase 2: Skipping vision processing")

            # Phase 3: Initialize vector store
            logger.info("Phase 3: Initializing vector store")
            await self.vector_indexer.initialize()

            # Phase 4: Index documents
            logger.info("Phase 4: Indexing documents")
            await self.vector_indexer.index_documents(all_elements)

            # Get final statistics
            stats = await self.vector_indexer.get_dataset_stats()

            results = {
                "status": "success",
                "total_elements_processed": len(all_elements),
                "pdf_elements": len(pdf_elements),
                "vision_elements": len(all_elements) - len(pdf_elements),
                "dataset_stats": stats,
            }

            logger.info("Ingestion pipeline completed successfully", results=results)
            return results

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {str(e)}", exc_info=True)
            raise RAGException(f"Ingestion failed: {str(e)}", details={"error": str(e)})

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_deep_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the indexed documents.

        Args:
            query_text: Query text.
            top_k: Number of top results.
            use_deep_memory: Whether to use DeepMemory feature.

        Returns:
            Dict[str, Any]: Query results.
        """
        try:
            logger.info(f"Executing query", query_length=len(query_text), top_k=top_k)

            results = await self.vector_indexer.query(
                query_text, top_k=top_k, use_deep_memory=use_deep_memory
            )

            return {
                "status": "success",
                "query": query_text,
                "num_results": len(results),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "query": query_text,
                "error": str(e),
            }

    async def _process_images_batched(
        self, image_paths: List[str], batch_size: int
    ) -> List[DocumentElement]:
        """Process images in batches with concurrency control."""
        all_elements = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            logger.debug(f"Processing image batch {i // batch_size + 1}/{len(image_paths) // batch_size + 1}")

            # Extract source document names from image paths
            batch_elements = []
            for image_path in batch:
                # Infer source document name from image directory
                source = Path(image_path).parent.name or "default"
                try:
                    element = await self.vision_processor.process_image(image_path, source)
                    if element:
                        batch_elements.append(element)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {str(e)}")

            all_elements.extend(batch_elements)

            # Respect rate limiting between batches
            if i + batch_size < len(image_paths):
                await asyncio.sleep(1)  # Small delay between batches

        return all_elements

    @staticmethod
    def _generate_image_paths(pdf_paths: List[str]) -> List[str]:
        """
        Generate image paths from PDFs (placeholder for PDF-to-image conversion).

        Args:
            pdf_paths: List of PDF paths.

        Returns:
            List[str]: List of generated image paths.
        """
        # In a real implementation, this would convert PDFs to images
        # For now, return empty list as images should be pre-generated
        logger.debug("Image path generation called (would need pdf2image)")
        return []


async def create_pipeline() -> RAGPipeline:
    """
    Factory function to create and initialize RAG pipeline.

    Returns:
        RAGPipeline: Initialized pipeline ready for use.
    """
    pipeline = RAGPipeline()
    logger.info("RAG Pipeline factory created successfully")
    return pipeline


async def main_example():
    """Example usage of the RAG pipeline."""
    # This would be used in actual applications
    try:
        pipeline = await create_pipeline()

        # Example ingestion
        pdf_paths = ["example.pdf"]  # Replace with actual paths
        results = await pipeline.ingest_documents(pdf_paths, process_vision=True)
        logger.info("Ingestion completed", results=results)

        # Example query
        query_results = await pipeline.query("What are the key findings?")
        logger.info("Query completed", results=query_results)

    except Exception as e:
        logger.error(f"Pipeline example failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main_example())
