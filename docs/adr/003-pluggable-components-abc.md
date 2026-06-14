# ADR 003: Pluggable Components via Abstract Base Classes

**Status:** Accepted  
**Date:** 2024-06  
**Deciders:** Core team

## Context

The original codebase had hardcoded calls to specific LLM providers, a specific vector store, and a specific parser. Swapping any component required modifying the pipeline code — violating Open/Closed principle and making testing hard.

## Decision

Define ABCs (`BaseParser`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`, `BaseReranker`, `BaseGenerator`, `BaseEvaluator`) in `components/base.py`. The `RAGPipeline` orchestrator accepts these ABCs via constructor injection. Concrete implementations live in their own subpackages. A factory function builds the default wiring from config.

## Rationale

- **Dependency injection over service locators:** Makes unit testing trivial — inject mock components, test pipeline logic in isolation.
- **ABCs over Protocols for this use case:** ABCs provide `@abstractmethod` enforcement at class definition time, not just at call time. Better error messages when a new implementation forgets a method.
- **Factory pattern (`create_pipeline()`):** Provides a convenient default wiring for production use while keeping DI available for tests and custom deployments.

## Consequences

- **Positive:** Adding a new vector store backend (e.g., Qdrant) requires zero changes to pipeline code. Integration tests run entirely with in-memory mocks — no real API calls.
- **Negative:** More files and indirection than a simple script. Acceptable for a production system; would be over-engineering for a research prototype.

## Alternatives Considered

- **Plugin system with entry points (future):** Would allow third-party components without forking. Tracked for v3.0.
- **Single monolithic class (rejected):** Untestable, unextensible.
