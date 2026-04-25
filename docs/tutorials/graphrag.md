# Tutorial: GraphRAG for Relationship-Heavy Financial Analysis

> **Status**: Planned for v3.0. Track progress in [roadmap.md](../../roadmap.md).

GraphRAG augments vector retrieval with an entity-relationship graph:
- "Which companies in our 10-K filings are also Tesla suppliers?"
- "Map ownership relationships between entities in this credit agreement."
- "How does each subsidiary revenue relate to the parent?"

## Architecture (Planned)

```
Ingestion:
  PDF -> Entity Extraction (LLM) -> Knowledge Graph (Neo4j / in-memory)
                                         |
Vector Search <-> Graph Traversal <-> Answer Generation

Entities: Company, Person, MetricValue, Date, Product, Location
Relations: SUBSIDIARY_OF, SUPPLIES_TO, REPORTED_REVENUE, CITES, MANAGED_BY
```

## Current workaround (available now)
```bash
rag-financial query \
  "List all subsidiaries, joint ventures, and related entities mentioned in the filing." \
  --top-k 20 --show-sources
```

See [roadmap.md](../../roadmap.md) for the v3.0 timeline.
