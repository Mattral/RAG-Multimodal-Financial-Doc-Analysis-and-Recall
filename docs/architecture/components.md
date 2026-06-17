# Component Architecture

All components implement Abstract Base Classes from `components/base.py`.
Switch any component by changing one config value — zero code changes required.

## Component Map

```
BaseParser
├── UnstructuredParser   (default, unstructured.io)
└── DoclingParser        (IBM Docling, better tables)

BaseVisionDescriber
├── OpenAIVisionDescriber   (GPT-4o, highest accuracy)
├── GeminiVisionDescriber   (Gemini 2.5 Flash, cheapest)
└── Qwen2VLDescriber        (open-source, private inference)

BaseEmbedder
├── OpenAIEmbedder       (text-embedding-3-small, cached)
└── LocalEmbedder        (BAAI/bge-small-en-v1.5, no API)

BaseVectorStore
├── DeepLakeVectorStoreAdapter   (default)
├── InMemoryVectorStore          (testing/dev)
└── PGVectorAdapter              (Postgres + pgvector)

BaseRetriever
├── HybridRetriever      (dense + BM25 + RRF, default)
└── ColPaliRetriever     (late-interaction, MaxSim — real implementation)

BaseReranker
├── CrossEncoderReranker   (ms-marco-MiniLM, local)
├── CohereReranker         (Cohere Rerank v3, cloud)
└── NoOpReranker           (disabled/testing)

BaseGenerator
└── OpenAIGenerator        (GPT-4o-mini → GPT-4o routing)

BaseEvaluator
└── RagasEvaluator         (RAGAS + LLM-as-judge)
```

## Adding a New Component

1. Implement the relevant ABC in the appropriate subpackage
2. Add to the factory function (`build_parser()`, `build_vector_store()`, etc.)
3. Add config option to the corresponding `*Config` class
4. Write unit tests
5. Update this doc and `.env.example`

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for a full walkthrough.
