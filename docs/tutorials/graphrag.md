# Tutorial: Knowledge Graph for Relationship-Heavy Financial Analysis

> **Status v2.0**: Entity extraction is **fully implemented** — the system
> makes a real LLM call to extract COMPANY, PERSON, METRIC, DATE entities
> and SUBSIDIARY_OF, REPORTED_REVENUE, SUPPLIES_TO relations from each chunk.
>
> Graph-traversal augmented retrieval (using extracted entities to expand
> the retrieved context) is planned for v3.0. See [ADR-008](../adr/008-knowledge-graph-graphrag.md).

The knowledge graph layer enriches retrieval for relationship-heavy queries:
- "Which companies in our 10-K filings are Tesla suppliers?"
- "Map the ownership hierarchy between entities in this credit agreement."
- "How does each subsidiary revenue relate to the parent?"

## What is implemented now (v2.0)

### Entity and relation extraction
```python
from src.rag_system.components.knowledge_graph import EntityExtractor, InMemoryGraphStore

extractor = EntityExtractor(model="gpt-4o-mini")
entities, relations = await extractor.extract(
    text="Tesla reported revenue of $23.35B in Q3 2023.",
    source_document="tesla_10q.pdf",
    tenant_id="acme",
)
# Returns:
# entities = [Entity(type="COMPANY", name="Tesla"), Entity(type="METRIC", name="Revenue", value="$23.35B")]
# relations = [Relation(predicate="REPORTED_REVENUE", ...)]
```

Enable during ingestion:
```bash
ENABLE_KNOWLEDGE_GRAPH=true rag-financial ingest tesla_10k.pdf --tenant acme
```

### Entity types extracted
`COMPANY` · `PERSON` · `METRIC` · `DATE` · `PRODUCT` · `LOCATION` · `REGULATION`

### Relation types extracted
`SUBSIDIARY_OF` · `REPORTED_REVENUE` · `SUPPLIES_TO` · `CITES` · `MANAGED_BY` · `ISSUED_BY` · `REGULATED_BY` · `COMPETES_WITH`

### InMemoryGraphStore
Query extracted entities after ingest:
```python
from src.rag_system.components.knowledge_graph import InMemoryGraphStore
store = InMemoryGraphStore()
tesla_entity = store.search_by_name("Tesla", tenant_id="acme")
neighbors = store.get_neighbors(tesla_entity[0].id, predicate="SUBSIDIARY_OF", depth=2)
```

## Current workaround for relationship queries (while traversal is v3.0)

Use high `top_k` with hybrid retrieval — exact entity mentions are well-captured by BM25:
```bash
rag-financial query \
  "List all subsidiaries, joint ventures, and related entities mentioned in the filing." \
  --top-k 20 --show-sources
```

## v3.0 roadmap: graph-traversal augmented retrieval

When v3.0 ships, `GraphAugmentedRetriever` will:
1. Extract entity mentions from the query ("Tesla" → COMPANY node)
2. Traverse the extracted graph for related entities (SUBSIDIARY_OF, SUPPLIES_TO)
3. Fetch chunks associated with discovered entities
4. Merge + rerank with the standard hybrid retrieval results

See [ADR-008](../adr/008-knowledge-graph-graphrag.md) for the full design.
