"""Unit tests for the knowledge graph stub (v2.0 interface, v3.0 full impl)."""
import pytest
from src.rag_system.components.knowledge_graph import (
    Entity, Relation, InMemoryGraphStore, EntityExtractor, GraphAugmentedRetriever,
)


class TestEntity:
    def test_creation(self):
        e = Entity(id="e1", type="COMPANY", name="Tesla", source_document="tesla.pdf")
        assert e.type == "COMPANY"
        assert e.tenant_id == "default"

    def test_with_value(self):
        e = Entity(id="m1", type="METRIC", name="Revenue", source_document="tesla.pdf",
                   value="$23.35B")
        assert e.value == "$23.35B"


class TestInMemoryGraphStore:
    def test_add_and_retrieve_entity(self):
        store = InMemoryGraphStore()
        e = Entity(id="e1", type="COMPANY", name="Tesla", source_document="tesla.pdf",
                   tenant_id="acme")
        store.add_entity(e)
        assert "e1" in store._tenant_index["acme"]

    def test_get_neighbors_one_hop(self):
        store = InMemoryGraphStore()
        tesla = Entity(id="tesla", type="COMPANY", name="Tesla", source_document="d.pdf")
        panasonic = Entity(id="panasonic", type="COMPANY", name="Panasonic", source_document="d.pdf")
        store.add_entity(tesla)
        store.add_entity(panasonic)
        store.add_relation(Relation(
            subject_id="panasonic", predicate="SUPPLIES_TO",
            object_id="tesla", source_document="d.pdf",
        ))
        neighbors = store.get_neighbors("tesla", depth=1)
        assert len(neighbors) == 1
        assert neighbors[0].id == "panasonic"

    def test_get_neighbors_two_hop(self):
        store = InMemoryGraphStore()
        for eid, name in [("a", "A"), ("b", "B"), ("c", "C")]:
            store.add_entity(Entity(id=eid, type="COMPANY", name=name, source_document="d.pdf"))
        store.add_relation(Relation(subject_id="a", predicate="SUBSIDIARY_OF",
                                    object_id="b", source_document="d.pdf"))
        store.add_relation(Relation(subject_id="b", predicate="SUBSIDIARY_OF",
                                    object_id="c", source_document="d.pdf"))
        neighbors_1hop = store.get_neighbors("a", depth=1)
        neighbors_2hop = store.get_neighbors("a", depth=2)
        assert len(neighbors_1hop) == 1
        assert len(neighbors_2hop) == 2

    def test_get_neighbors_with_predicate_filter(self):
        store = InMemoryGraphStore()
        for eid in ["a", "b", "c"]:
            store.add_entity(Entity(id=eid, type="COMPANY", name=eid, source_document="d.pdf"))
        store.add_relation(Relation(subject_id="a", predicate="SUBSIDIARY_OF",
                                    object_id="b", source_document="d.pdf"))
        store.add_relation(Relation(subject_id="a", predicate="SUPPLIES_TO",
                                    object_id="c", source_document="d.pdf"))
        filtered = store.get_neighbors("a", predicate="SUBSIDIARY_OF", depth=1)
        assert len(filtered) == 1
        assert filtered[0].id == "b"

    def test_search_by_name(self):
        store = InMemoryGraphStore()
        store.add_entity(Entity(id="e1", type="COMPANY", name="Tesla Inc",
                               source_document="d.pdf", tenant_id="acme"))
        store.add_entity(Entity(id="e2", type="COMPANY", name="Apple Inc",
                               source_document="d.pdf", tenant_id="acme"))
        results = store.search_by_name("tesla", tenant_id="acme")
        assert len(results) == 1
        assert results[0].name == "Tesla Inc"

    def test_search_case_insensitive(self):
        store = InMemoryGraphStore()
        store.add_entity(Entity(id="e1", type="COMPANY", name="TESLA",
                               source_document="d.pdf", tenant_id="t1"))
        results = store.search_by_name("tesla", tenant_id="t1")
        assert len(results) == 1

    def test_stats_empty(self):
        store = InMemoryGraphStore()
        stats = store.stats("nonexistent_tenant")
        assert stats["entities"] == 0

    def test_stats_with_data(self):
        store = InMemoryGraphStore()
        store.add_entity(Entity(id="e1", type="COMPANY", name="Tesla",
                               source_document="d.pdf", tenant_id="acme"))
        store.add_entity(Entity(id="e2", type="COMPANY", name="Apple",
                               source_document="d.pdf", tenant_id="acme"))
        stats = store.stats("acme")
        assert stats["entities"] == 2

    def test_tenant_isolation(self):
        store = InMemoryGraphStore()
        store.add_entity(Entity(id="e1", type="COMPANY", name="A", source_document="d.pdf",
                               tenant_id="tenant_a"))
        store.add_entity(Entity(id="e2", type="COMPANY", name="B", source_document="d.pdf",
                               tenant_id="tenant_b"))
        assert store.stats("tenant_a")["entities"] == 1
        assert store.stats("tenant_b")["entities"] == 1


class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_empty_stub(self):
        extractor = EntityExtractor()
        entities, relations = await extractor.extract(
            "Tesla reported revenue of $23.35B in Q3 2023.",
            source_document="tesla.pdf",
        )
        # Stub implementation — returns empty until v3.0
        assert entities == []
        assert relations == []

    def test_extraction_prompt_has_required_fields(self):
        assert "entities" in EntityExtractor.EXTRACTION_PROMPT
        assert "relations" in EntityExtractor.EXTRACTION_PROMPT
        assert "COMPANY" in EntityExtractor.EXTRACTION_PROMPT


class TestGraphAugmentedRetriever:
    @pytest.mark.asyncio
    async def test_falls_back_when_graph_empty(self):
        retriever = GraphAugmentedRetriever()
        base_chunks = ["chunk1", "chunk2"]
        result = await retriever.retrieve_with_graph(
            "What subsidiaries does Tesla have?", base_chunks, tenant_id="t1"
        )
        assert result == base_chunks

    @pytest.mark.asyncio
    async def test_falls_back_with_populated_graph(self):
        store = InMemoryGraphStore()
        store.add_entity(Entity(id="e1", type="COMPANY", name="Tesla",
                               source_document="d.pdf", tenant_id="t1"))
        retriever = GraphAugmentedRetriever(graph_store=store)
        base_chunks = ["chunk1"]
        # Stub still returns base_chunks until full v3.0 traversal implemented
        result = await retriever.retrieve_with_graph(
            "Tesla subsidiaries?", base_chunks, tenant_id="t1"
        )
        assert result == base_chunks
