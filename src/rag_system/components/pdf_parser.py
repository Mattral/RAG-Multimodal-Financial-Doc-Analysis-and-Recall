"""Asynchronous PDF parsing component."""

import asyncio
from typing import List, Any
from pathlib import Path

from pydantic import BaseModel

from src.rag_system.config import get_config
from src.rag_system.utils.logger import get_logger
from src.rag_system.utils.exceptions import PDFParsingError
from src.rag_system.utils.retry_policy import RetryPolicy

logger = get_logger(__name__)


class DocumentElement(BaseModel):
    """Structured document element."""

    type: str  # "text", "table", "graph"
    text: str
    source_document: str
    metadata: dict = {}

    class Config:
        frozen = True


class PDFParser:
    """Asynchronous PDF parser using unstructured.io."""

    def __init__(self):
        """Initialize PDF parser."""
        self.config = get_config()
        self.pdf_config = self.config.pdf_parsing_config
        self.retry_policy = RetryPolicy(
            max_attempts=self.config.rate_limit_config.retry_max_attempts,
            base_delay_seconds=1.0,
            backoff_factor=self.config.rate_limit_config.retry_backoff_factor,
        )
        logger.info("PDF parser initialized", config=self.pdf_config.dict())

    async def parse_pdf(self, file_path: str) -> List[DocumentElement]:
        """
        Parse PDF and extract elements.

        Args:
            file_path: Path to PDF file.

        Returns:
            List[DocumentElement]: Extracted document elements.

        Raises:
            PDFParsingError: If parsing fails.
        """
        try:
            logger.info(f"Parsing PDF: {file_path}")
            path = Path(file_path)

            if not path.exists():
                raise PDFParsingError(f"PDF file not found: {file_path}", file_path=file_path)

            # Run synchronous PDF parsing in thread pool
            elements = await asyncio.to_thread(self._parse_pdf_sync, str(path))
            logger.info(
                f"Successfully parsed PDF: {file_path}",
                num_elements=len(elements),
            )
            return elements

        except PDFParsingError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to parse PDF: {file_path}",
                error=str(e),
                exc_info=True,
            )
            raise PDFParsingError(
                f"PDF parsing failed: {str(e)}",
                file_path=file_path,
                details={"error": str(e)},
            )

    def _parse_pdf_sync(self, file_path: str) -> List[DocumentElement]:
        """
        Synchronous PDF parsing (runs in thread pool).

        Args:
            file_path: Path to PDF file.

        Returns:
            List[DocumentElement]: Extracted elements.
        """
        try:
            from unstructured.partition.pdf import partition_pdf

            raw_elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=self.pdf_config.extract_images,
                infer_table_structure=self.pdf_config.infer_table_structure,
                chunking_strategy="by_title",
                max_characters=self.pdf_config.max_characters,
                new_after_n_chars=self.pdf_config.new_after_n_chars,
                combine_text_under_n_chars=self.pdf_config.combine_text_under_n_chars,
            )

            document_elements = []
            source_name = Path(file_path).stem

            for element in raw_elements:
                element_type = str(type(element))

                if "Table" in element_type:
                    doc_elem = DocumentElement(
                        type="table",
                        text=str(element),
                        source_document=source_name,
                        metadata={"element_type": element_type},
                    )
                elif "CompositeElement" in element_type:
                    doc_elem = DocumentElement(
                        type="text",
                        text=str(element),
                        source_document=source_name,
                        metadata={"element_type": element_type},
                    )
                else:
                    continue

                document_elements.append(doc_elem)

            return document_elements

        except Exception as e:
            logger.error(f"Sync PDF parsing failed: {str(e)}", exc_info=True)
            raise

    async def parse_multiple_pdfs(self, file_paths: List[str]) -> List[DocumentElement]:
        """
        Parse multiple PDFs concurrently with rate limiting.

        Args:
            file_paths: List of PDF file paths.

        Returns:
            List[DocumentElement]: All extracted elements.
        """
        logger.info(f"Parsing {len(file_paths)} PDFs concurrently")
        tasks = [self.parse_pdf(fp) for fp in file_paths]

        all_elements = []
        for coro in asyncio.as_completed(tasks):
            try:
                elements = await coro
                all_elements.extend(elements)
            except PDFParsingError as e:
                logger.error(f"Failed to parse PDF", details=e.details)
                # Continue with other PDFs on error

        logger.info(f"Parsed all PDFs, total elements: {len(all_elements)}")
        return all_elements
