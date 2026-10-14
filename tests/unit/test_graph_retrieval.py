"""Tests for graph construction, traversal and graph-augmented retrieval."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_system.components.base import RetrievedChunk
from src.rag_system.components.knowledge_graph import (
    Entity,
    EntityExtractor,
    GraphAugmentedRetriever,
    InMemoryGraphStore,
    Relation,
    canonical_entity_id,
)


def _chunk(cid, text, doc="tesla.pdf", page=1):
    return RetrievedChunk(text=text, score=0.0, source_document=doc, page_number=page, chunk_id=cid)


def _ent(etype, name, value=None, tenant="t1", doc="tesla.pdf"):
    return Entity(
        id=canonical_entity_id(etype, name, value, tenant),
        type=etype,
        name=name,
        source_document=doc,
        value=value,
        tenant_id=tenant,
    )


def _rel(subject, predicate, obj):
    return Relation(
        subject_id=subject.id, predicate=predicate, object_id=obj.id, source_document="tesla.pdf"
    )


@pytest.fixture
def tesla_graph():
    """Tesla -> gross margin 17.9% -> Q3 2023; Tesla -> capex; an unrelated Apple chunk."""
    s = InMemoryGraphStore()
    tesla = _ent("COMPANY", "Tesla")
    apple = _ent("COMPANY", "Apple", doc="apple.pdf")
    margin = _ent("METRIC", "Gross margin", "17.9%")
    q3 = _ent("DATE", "Q3 2023")
    capex = _ent("METRIC", "Capital expenditures", "$2,460M")
    s.add_chunk_extraction(
        _chunk("c_margin", "Total GAAP gross margin 17.9% in Q3 2023."),
        [tesla, margin, q3],
        [_rel(tesla, "REPORTED_METRIC", margin), _rel(margin, "FOR_PERIOD", q3)],
        tenant_id="t1",
    )
    # Mentions neither "Tesla" nor the query terms; reachable only through the graph.
    s.add_chunk_extraction(
        _chunk("c_capex", "Capital expenditures (2,460) in the quarter."), [capex], [], "t1"
    )
    s.add_relation(_rel(tesla, "REPORTED_METRIC", capex))
    s.add_chunk_extraction(
        _chunk("c_apple", "Apple products gross margin 36.5%.", doc="apple.pdf"), [apple], [], "t1"
    )
    return s


class TestCanonicalIds:
    def test_same_entity_from_two_chunks_is_one_node(self):
        s = InMemoryGraphStore()
        for cid in ("c1", "c2"):
            s.add_chunk_extraction(
                _chunk(cid, f"Tesla text {cid}"), [_ent("COMPANY", "Tesla")], [], tenant_id="t1"
            )
        assert s.stats("t1")["entities"] == 1
        assert s.chunk_count(canonical_entity_id("COMPANY", "Tesla", tenant_id="t1")) == 2

    def test_metric_id_includes_value_and_tenant(self):
        a = canonical_entity_id("METRIC", "Revenue", "$23.35B", "t1")
        assert a != canonical_entity_id("METRIC", "Revenue", "$25.17B", "t1")
        assert a != canonical_entity_id("METRIC", "Revenue", "$23.35B", "t2")
        assert canonical_entity_id("company", "Tesla, Inc.") == "default/COMPANY:tesla-inc"


class TestTraversal:
    def test_find_entities_matches_whole_words(self, tesla_graph):
        found = tesla_graph.find_entities_in_text("What was Tesla's Q3 2023 margin?", "t1")
        assert {e.name for e in found} == {"Tesla", "Q3 2023"}
        s = InMemoryGraphStore()
        s.add_entity(_ent("COMPANY", "Meta"))
        assert s.find_entities_in_text("check the metadata", "t1") == []

    def test_neighborhood_hops(self, tesla_graph):
        tesla = canonical_entity_id("COMPANY", "Tesla", tenant_id="t1")
        hops = tesla_graph.neighborhood([tesla], max_hops=2)
        assert hops[tesla] == 0
        assert hops[canonical_entity_id("METRIC", "Gross margin", "17.9%", "t1")] == 1
        assert hops[canonical_entity_id("DATE", "Q3 2023", tenant_id="t1")] == 2
        assert canonical_entity_id("COMPANY", "Apple", tenant_id="t1") not in hops

    def test_hubs_are_reached_but_not_expanded(self):
        s = InMemoryGraphStore()
        seed, hub = _ent("COMPANY", "Tesla"), _ent("DATE", "Q3 2023")
        s.add_entity(seed)
        s.add_entity(hub)
        s.add_relation(_rel(seed, "FOR_PERIOD", hub))
        for i in range(5):
            other = _ent("COMPANY", f"Company {i}")
            s.add_entity(other)
            s.add_relation(_rel(other, "FOR_PERIOD", hub))
        hops = s.neighborhood([seed.id], max_hops=2, max_degree=3)
        assert set(hops) == {seed.id, hub.id}


class TestGraphAugmentedRetrieval:
    @pytest.mark.asyncio
    async def test_adds_graph_chunks_and_ranks_multi_entity_matches_first(self, tesla_graph):
        r = GraphAugmentedRetriever(graph_store=tesla_graph, max_hops=1)
        query = "Tesla gross margin in Q3 2023"
        base = [_chunk("b1", "Unrelated base result.", doc="other.pdf")]
        out = await r.retrieve_with_graph(query, base, top_k=5, tenant_id="t1")
        ids = [c.chunk_id for c in out]
        assert "b1" in ids and "c_margin" in ids
        assert "c_apple" not in ids
        # Reached by Tesla, the margin metric and the period, so it ranks first.
        assert r.graph_chunks(query, "t1")[0].chunk_id == "c_margin"
        margin = next(c for c in out if c.chunk_id == "c_margin")
        assert margin.metadata["retrieval"] == "graph"
        assert "Tesla" in margin.metadata["graph_entities"]

    def test_traversal_reaches_chunks_not_named_in_query(self, tesla_graph):
        one_hop = GraphAugmentedRetriever(graph_store=tesla_graph, max_hops=1)
        assert "c_capex" in [c.chunk_id for c in one_hop.graph_chunks("Tesla results", "t1")]
        no_hops = GraphAugmentedRetriever(graph_store=tesla_graph, max_hops=0)
        assert [c.chunk_id for c in no_hops.graph_chunks("Tesla results", "t1")] == ["c_margin"]

    @pytest.mark.asyncio
    async def test_period_alone_does_not_anchor_retrieval(self, tesla_graph):
        r = GraphAugmentedRetriever(graph_store=tesla_graph)
        assert r.graph_chunks("What happened in Q3 2023?", "t1") == []
        base = [_chunk("b1", "x")]
        assert (
            await r.retrieve_with_graph("What happened in Q3 2023?", base, tenant_id="t1") is base
        )

    def test_period_boosts_matching_chunk_without_adding_new_ones(self):
        s = InMemoryGraphStore()
        tesla, q3, q4 = _ent("COMPANY", "Tesla"), _ent("DATE", "Q3 2023"), _ent("DATE", "Q4 2023")
        s.add_chunk_extraction(_chunk("c_q4", "Tesla Q4 2023 results"), [tesla, q4], [], "t1")
        s.add_chunk_extraction(_chunk("c_q3", "Tesla Q3 2023 results"), [tesla, q3], [], "t1")
        other = _ent("COMPANY", "Apple")
        s.add_chunk_extraction(_chunk("c_apple_q3", "Apple Q3 2023"), [other, q3], [], "t1")
        ranked = GraphAugmentedRetriever(graph_store=s).graph_chunks("Tesla in Q3 2023", "t1")
        assert [c.chunk_id for c in ranked] == ["c_q3", "c_q4"]  # Apple's Q3 chunk not added

    @pytest.mark.asyncio
    async def test_no_matching_entity_returns_base_unchanged(self, tesla_graph):
        r = GraphAugmentedRetriever(graph_store=tesla_graph)
        base = [_chunk("b1", "x")]
        assert await r.retrieve_with_graph("NVIDIA data center", base, tenant_id="t1") is base

    @pytest.mark.asyncio
    async def test_other_tenant_sees_nothing(self, tesla_graph):
        r = GraphAugmentedRetriever(graph_store=tesla_graph)
        base = [_chunk("b1", "x")]
        assert await r.retrieve_with_graph("Tesla gross margin", base, tenant_id="t2") is base

    @pytest.mark.asyncio
    async def test_duplicate_of_base_chunk_appears_once(self, tesla_graph):
        r = GraphAugmentedRetriever(graph_store=tesla_graph)
        base = [_chunk("c_margin", "Total GAAP gross margin 17.9% in Q3 2023.")]
        out = await r.retrieve_with_graph("Tesla gross margin Q3 2023", base, 5, "t1")
        assert [c.text for c in out].count(base[0].text) == 1

    @pytest.mark.asyncio
    async def test_retrieve_wraps_base_retriever(self, tesla_graph):
        base_retriever = MagicMock()
        base_retriever.name = "hybrid_rrf_retriever"
        base_retriever.retrieve = AsyncMock(return_value=[_chunk("b1", "base")])
        r = GraphAugmentedRetriever(graph_store=tesla_graph, base_retriever=base_retriever)
        out = await r.retrieve("Tesla gross margin", top_k=3, tenant_id="t1")
        base_retriever.retrieve.assert_awaited_once()
        assert {"b1", "c_margin"} <= {c.chunk_id for c in out}
        assert r.name == "graph_augmented(hybrid_rrf_retriever)"

    @pytest.mark.asyncio
    async def test_graph_weight_controls_how_far_graph_chunks_rise(self, tesla_graph):
        base = [_chunk(f"b{i}", f"base {i}", doc="other.pdf") for i in range(3)]
        query = "Tesla gross margin in Q3 2023"
        equal = await GraphAugmentedRetriever(
            graph_store=tesla_graph, graph_weight=1.0
        ).retrieve_with_graph(query, base, top_k=5, tenant_id="t1")
        assert equal[0].chunk_id in {"b0", "c_margin"}  # rank-1 of each list tie
        light = await GraphAugmentedRetriever(
            graph_store=tesla_graph, graph_weight=0.5
        ).retrieve_with_graph(query, base, top_k=5, tenant_id="t1")
        ids = [c.chunk_id for c in light]
        assert ids[0] == "b0" and ids.index("c_margin") > 0  # base keeps the top slot


def test_weighted_rrf_defaults_to_equal_weights():
    from src.rag_system.components.retriever import _reciprocal_rank_fusion

    a, b = [_chunk("a", "A")], [_chunk("b", "B")]
    assert [c.chunk_id for c in _reciprocal_rank_fusion([a, b], k=60)][0] == "a"  # stable tie
    assert [c.chunk_id for c in _reciprocal_rank_fusion([a, b], k=60, weights=[0.5, 1.0])] == [
        "b",
        "a",
    ]


class TestExtractorParsingAndProviders:
    def test_parse_assigns_canonical_ids_and_resolves_relations(self):
        raw = json.dumps(
            {
                "entities": [
                    {"id": "e1", "type": "company", "name": "Tesla"},
                    {"id": "e2", "type": "METRIC", "name": "Revenue", "value": "$23.35B"},
                    {"id": "e3", "type": "DATE", "name": "Q3 2023"},
                ],
                "relations": [
                    {"subject_id": "e1", "predicate": "REPORTED_REVENUE", "object_id": "e2"},
                    {"subject_id": "e2", "predicate": "FOR_PERIOD", "object_id": "e3"},
                    {"subject_id": "e1", "predicate": "CITES", "object_id": "missing"},
                ],
            }
        )
        entities, relations = EntityExtractor().parse(raw, "tesla.pdf", tenant_id="t1")
        assert entities[0].id == "t1/COMPANY:tesla" and entities[0].type == "COMPANY"
        assert [r.predicate for r in relations] == ["REPORTED_REVENUE", "FOR_PERIOD"]

    @pytest.mark.asyncio
    async def test_uses_gemini_when_configured(self, monkeypatch):
        from src.rag_system.config import reset_config

        monkeypatch.setenv("LLM_CONFIG__PROVIDER", "gemini")
        monkeypatch.setenv("LLM_CONFIG__MODEL", "gemini-3.8-flash")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        reset_config()
        body = json.dumps(
            {"entities": [{"id": "e1", "type": "COMPANY", "name": "Tesla"}], "relations": []}
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": body}]}}
            ]
        }
        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value.__aenter__.return_value = client
            entities, _ = await EntityExtractor().extract("Tesla " + "x" * 80, "tesla.pdf")
        url = client.post.call_args.args[0]
        assert "gemini-3.8-flash:generateContent" in url and "key=" not in url
        assert client.post.call_args.kwargs["headers"] == {"x-goog-api-key": "test-google-key"}
        gen_cfg = client.post.call_args.kwargs["json"]["generationConfig"]
        assert gen_cfg["responseMimeType"] == "application/json"
        assert [e.name for e in entities] == ["Tesla"]
        reset_config()


def _mock_store():
    store = MagicMock()
    store.initialize = AsyncMock()
    store.upsert = AsyncMock()
    return store


class TestPipelineWiring:
    @pytest.mark.asyncio
    async def test_ingest_builds_graph_and_retriever_is_wrapped(self, monkeypatch, tmp_path):
        from src.rag_system.components.base import DocumentElement
        from src.rag_system.config import reset_config
        from src.rag_system.pipeline import RAGPipeline

        monkeypatch.setenv("ENABLE_KNOWLEDGE_GRAPH", "true")
        reset_config()
        pdf = tmp_path / "tesla.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        element = DocumentElement(
            type="text",
            text="Tesla gross margin 17.9% in Q3 2023.",
            source_document="tesla.pdf",
            page_number=4,
            content_hash="h1",
        )
        parser = MagicMock()
        parser.parse_batch = AsyncMock(return_value=[element])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=([_ent("COMPANY", "Tesla", tenant="acme")], []))
        base_retriever = MagicMock()
        base_retriever.name = "hybrid_rrf_retriever"

        with (
            patch("src.rag_system.components.parser.build_parser", return_value=parser),
            patch("src.rag_system.components.vision.build_vision_describer", return_value=None),
        ):
            pipeline = await RAGPipeline.create(
                embedder=embedder,
                vector_store=_mock_store(),
                retriever=base_retriever,
                entity_extractor=extractor,
                generator=MagicMock(),
            )
        assert isinstance(pipeline._retriever, GraphAugmentedRetriever)

        result = await pipeline.ingest([str(pdf)], tenant_id="acme", process_vision=False)
        extractor.extract.assert_awaited_once()
        assert result["knowledge_graph"]["entities"] == 1
        hits = pipeline._retriever.graph_chunks("Tesla margin", "acme")
        assert [c.chunk_id for c in hits] == ["h1"]
        assert hits[0].page_number == 4
        reset_config()

    @pytest.mark.asyncio
    async def test_flag_off_leaves_retriever_unwrapped(self, monkeypatch):
        from src.rag_system.config import reset_config
        from src.rag_system.pipeline import RAGPipeline

        monkeypatch.delenv("ENABLE_KNOWLEDGE_GRAPH", raising=False)
        reset_config()
        base_retriever = MagicMock()
        with (
            patch("src.rag_system.components.parser.build_parser", return_value=MagicMock()),
            patch("src.rag_system.components.vision.build_vision_describer", return_value=None),
        ):
            pipeline = await RAGPipeline.create(
                embedder=MagicMock(),
                vector_store=_mock_store(),
                retriever=base_retriever,
                generator=MagicMock(),
            )
        assert pipeline._retriever is base_retriever
        assert pipeline._graph_store is None
        reset_config()
