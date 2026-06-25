"""utils/ — modular utilities for the HF Space RAG pipeline.

Modules:
  pdf_processor  — PDF ingestion, text extraction, semantic chunking
  retriever      — EmbeddingModel, FAISS index, BM25, RRF hybrid retrieval
  generator      — OpenAI + Gemini generation with cost tracking (current models: v2.0)
  guardrails     — Numeric grounding, PII detection, injection blocking
"""
from utils.generator import GenerationResult, generate
from utils.guardrails import GuardrailResult, run_guardrails
from utils.pdf_processor import DocumentChunk, IngestResult, ingest_pdf
from utils.retriever import EmbeddingModel, RetrievedChunk, VectorIndex

__all__ = [
    "ingest_pdf", "IngestResult", "DocumentChunk",
    "VectorIndex", "RetrievedChunk", "EmbeddingModel",
    "generate", "GenerationResult",
    "run_guardrails", "GuardrailResult",
]
