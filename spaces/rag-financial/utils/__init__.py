"""utils/ — modular utilities for the HF Space RAG pipeline.

Modules:
  pdf_processor  — PDF ingestion, text extraction, semantic chunking
  retriever      — EmbeddingModel, FAISS index, BM25, RRF hybrid retrieval
  generator      — OpenAI + Gemini generation with cost tracking (current models: v2.0)
  guardrails     — Numeric grounding, PII detection, injection blocking
"""
from utils.pdf_processor import ingest_pdf, IngestResult, DocumentChunk
from utils.retriever import VectorIndex, RetrievedChunk, EmbeddingModel
from utils.generator import generate, GenerationResult
from utils.guardrails import run_guardrails, GuardrailResult

__all__ = [
    "ingest_pdf", "IngestResult", "DocumentChunk",
    "VectorIndex", "RetrievedChunk", "EmbeddingModel",
    "generate", "GenerationResult",
    "run_guardrails", "GuardrailResult",
]
