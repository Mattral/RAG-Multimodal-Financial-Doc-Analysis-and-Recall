"""Tests for knowledge graph — InMemoryGraphStore and real LLM EntityExtractor."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_system.components.knowledge_graph import (
    Entity,
    EntityExtractor,
    GraphAugmentedRetriever,
    InMemoryGraphStore,
    Relation,
)


class TestEntity:
    def test_creation(self):
        e = Entity(id="e1", type="COMPANY", name="Tesla", source_document="t.pdf")
        assert e.type == "COMPANY" and e.tenant_id == "default"

    def test_with_value(self):
        e = Entity(id="m1", type="METRIC", name="Revenue",
                   source_document="t.pdf", value="23.35B")
        assert e.value == "23.35B"


class TestInMemoryGraphStore:
    def test_add_entity(self):
        s = InMemoryGraphStore()
        s.add_entity(Entity(id="e1", type="COMPANY", name="Tesla",
                            source_document="d.pdf", tenant_id="acme"))
        assert "e1" in s._tenant_index["acme"]

    def test_one_hop_neighbors(self):
        s = InMemoryGraphStore()
        for eid, name in [("tesla","Tesla"),("pan","Panasonic")]:
            s.add_entity(Entity(id=eid, type="COMPANY", name=name, source_document="d.pdf"))
        s.add_relation(Relation(subject_id="pan", predicate="SUPPLIES_TO",
                                object_id="tesla", source_document="d.pdf"))
        n = s.get_neighbors("tesla", depth=1)
        assert len(n) == 1 and n[0].id == "pan"

    def test_two_hop_neighbors(self):
        s = InMemoryGraphStore()
        for eid in ["a","b","c"]:
            s.add_entity(Entity(id=eid, type="COMPANY", name=eid, source_document="d.pdf"))
        s.add_relation(Relation(subject_id="a",predicate="SUBSIDIARY_OF",object_id="b",source_document="d.pdf"))
        s.add_relation(Relation(subject_id="b",predicate="SUBSIDIARY_OF",object_id="c",source_document="d.pdf"))
        assert len(s.get_neighbors("a", depth=2)) == 2

    def test_predicate_filter(self):
        s = InMemoryGraphStore()
        for eid in ["a","b","c"]:
            s.add_entity(Entity(id=eid,type="COMPANY",name=eid,source_document="d.pdf"))
        s.add_relation(Relation(subject_id="a",predicate="SUBSIDIARY_OF",object_id="b",source_document="d.pdf"))
        s.add_relation(Relation(subject_id="a",predicate="SUPPLIES_TO",object_id="c",source_document="d.pdf"))
        r = s.get_neighbors("a", predicate="SUBSIDIARY_OF", depth=1)
        assert len(r) == 1 and r[0].id == "b"

    def test_search_by_name_case_insensitive(self):
        s = InMemoryGraphStore()
        s.add_entity(Entity(id="e1",type="COMPANY",name="Tesla Inc",
                            source_document="d.pdf",tenant_id="t1"))
        assert len(s.search_by_name("tesla", tenant_id="t1")) == 1

    def test_stats(self):
        s = InMemoryGraphStore()
        s.add_entity(Entity(id="e1",type="COMPANY",name="A",
                            source_document="d.pdf",tenant_id="acme"))
        assert s.stats("acme")["entities"] == 1

    def test_tenant_isolation(self):
        s = InMemoryGraphStore()
        s.add_entity(Entity(id="e1",type="COMPANY",name="A",source_document="d.pdf",tenant_id="a"))
        s.add_entity(Entity(id="e2",type="COMPANY",name="B",source_document="d.pdf",tenant_id="b"))
        assert s.stats("a")["entities"] == 1 and s.stats("b")["entities"] == 1


class TestEntityExtractorReal:
    @pytest.mark.asyncio
    async def test_well_formed_response_parsed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        body = json.dumps({
            "entities": [
                {"id":"e1","type":"COMPANY","name":"Tesla","value":None},
                {"id":"e2","type":"METRIC","name":"Revenue","value":"$23.35B"},
            ],
            "relations": [{"subject_id":"e1","predicate":"REPORTED_REVENUE","object_id":"e2"}]
        })
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices":[{"message":{"content": body}}]}
        with patch("httpx.AsyncClient") as mock_cls:
            c = AsyncMock()
            c.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value.__aenter__.return_value = c
            entities, relations = await EntityExtractor().extract(
                "Tesla reported total revenue of $23.35B for the third quarter of fiscal year 2023.",
                "tesla.pdf",
            )
        assert len(entities) == 2
        assert any(e.name == "Tesla" for e in entities)
        assert len(relations) == 1 and relations[0].predicate == "REPORTED_REVENUE"
        reset_config()

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self, monkeypatch):
        import httpx
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        with patch("httpx.AsyncClient") as mock_cls:
            c = AsyncMock()
            c.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_cls.return_value.__aenter__.return_value = c
            entities, relations = await EntityExtractor().extract("A"*100, "d.pdf")
        assert entities == [] and relations == []
        reset_config()

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.rag_system.config import reset_config
        reset_config()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices":[{"message":{"content":"not json!!!"}}]}
        with patch("httpx.AsyncClient") as mock_cls:
            c = AsyncMock()
            c.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value.__aenter__.return_value = c
            entities, relations = await EntityExtractor().extract("A"*100, "d.pdf")
        assert entities == [] and relations == []
        reset_config()

    @pytest.mark.asyncio
    async def test_short_text_skipped(self):
        entities, relations = await EntityExtractor().extract("Short.", "d.pdf")
        assert entities == [] and relations == []

    def test_prompt_has_required_fields(self):
        p = EntityExtractor.EXTRACTION_PROMPT
        assert all(x in p for x in ["ONLY","JSON","COMPANY","REPORTED_REVENUE"])


class TestGraphAugmentedRetriever:
    @pytest.mark.asyncio
    async def test_falls_back_when_graph_empty(self):
        r = GraphAugmentedRetriever()
        base = ["c1","c2"]
        assert await r.retrieve_with_graph("q", base) == base

    @pytest.mark.asyncio
    async def test_returns_base_with_populated_graph(self):
        s = InMemoryGraphStore()
        s.add_entity(Entity(id="e1",type="COMPANY",name="Tesla",
                            source_document="d.pdf",tenant_id="t1"))
        r = GraphAugmentedRetriever(graph_store=s)
        base = ["c1"]
        assert await r.retrieve_with_graph("Tesla subsidiaries", base, tenant_id="t1") == base
