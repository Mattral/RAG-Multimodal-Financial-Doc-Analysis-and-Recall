"""Knowledge graph extraction and graph-augmented retrieval.

Guideline §7: 'Knowledge Graph layer (optional Neo4j or in-memory +
LLM extraction of entities/relations: companies, metrics, events, filings).'

How it fits together (enable with ENABLE_KNOWLEDGE_GRAPH=true):

* Ingest: ``EntityExtractor`` asks the configured LLM for entities and relations
  in each chunk; ``InMemoryGraphStore.add_chunk_extraction`` merges them into
  one graph per tenant and links every entity to the chunks it came from.
* Query: ``GraphAugmentedRetriever`` finds entities named in the query, walks
  their neighbours (company -> metrics -> periods ...), collects the chunks
  linked to that neighbourhood, and fuses them with the base (hybrid)
  retrieval results via Reciprocal Rank Fusion. With no matching entities it
  returns the base results unchanged.

The store is in-memory: the graph is rebuilt on ingest and is lost on
restart. Swap in Neo4j / Kuzu behind the same interface for persistence.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import structlog

from src.rag_system.components.base import BaseRetriever, RetrievedChunk
from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)


@dataclass
class Entity:
    """A named entity extracted from a financial document."""

    id: str
    type: str  # COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION, REGULATION
    name: str
    source_document: str
    page_number: Optional[int] = None
    value: Optional[str] = None
    tenant_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """A directed relationship between two entities."""

    subject_id: str
    predicate: str  # SUBSIDIARY_OF, REPORTED_REVENUE, REPORTED_METRIC, FOR_PERIOD, ...
    object_id: str
    source_document: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def canonical_entity_id(
    entity_type: str, name: str, value: Optional[str] = None, tenant_id: str = "default"
) -> str:
    """Stable id so the same entity from different chunks/filings becomes one node.

    METRIC ids include the value: "Revenue = $23.35B" and "Revenue = $25.17B"
    are different facts, while the same fact quoted in a 10-Q and a deck merges.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    etype = (entity_type or "UNKNOWN").upper()
    if etype == "METRIC" and value:
        slug += "=" + re.sub(r"[^a-z0-9.%$]+", "", str(value).lower())
    return f"{tenant_id}/{etype}:{slug}"


class InMemoryGraphStore:
    """In-memory graph store for development and single-node deployments.

    Replace with Neo4j, NetworkX + persistence, or Kuzu for production.
    Interface is identical.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._tenant_index: Dict[str, List[str]] = {}
        # Adjacency: entity id -> {(predicate, neighbour id, is_outgoing)}
        self._edges: Dict[str, Set[Tuple[str, str, bool]]] = defaultdict(set)
        # Entity -> linked chunk keys; chunk key -> chunk ("<tenant>/<chunk id>")
        self._entity_chunks: Dict[str, Set[str]] = defaultdict(set)
        self._chunks: Dict[str, RetrievedChunk] = {}

    # ── Building the graph ───────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self._entities:
            return  # first mention wins; later mentions only add chunk links
        self._entities[entity.id] = entity
        self._tenant_index.setdefault(entity.tenant_id, []).append(entity.id)

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)
        self._edges[relation.subject_id].add((relation.predicate, relation.object_id, True))
        self._edges[relation.object_id].add((relation.predicate, relation.subject_id, False))

    def link_chunk(self, entity_id: str, chunk: RetrievedChunk, tenant_id: str = "default") -> None:
        """Record that ``entity_id`` is mentioned in ``chunk``."""
        key = f"{tenant_id}/{chunk.chunk_id or hash(chunk.text)}"
        self._chunks.setdefault(key, chunk)
        self._entity_chunks[entity_id].add(key)

    def add_chunk_extraction(
        self,
        chunk: RetrievedChunk,
        entities: Iterable[Entity],
        relations: Iterable[Relation],
        tenant_id: str = "default",
    ) -> None:
        """Merge one chunk's extracted entities/relations and link them to the chunk."""
        for entity in entities:
            self.add_entity(entity)
            self.link_chunk(entity.id, chunk, tenant_id)
        for relation in relations:
            self.add_relation(relation)

    # ── Querying the graph ───────────────────────────────────────────────────

    def get_neighbors(
        self, entity_id: str, predicate: Optional[str] = None, depth: int = 1
    ) -> List[Entity]:
        """Entities within ``depth`` hops. ``predicate`` filters outgoing edges."""
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for eid in frontier:
                for pred, other, outgoing in self._edges.get(eid, ()):
                    if outgoing and predicate is not None and pred != predicate:
                        continue
                    if other not in visited:
                        next_frontier.add(other)
            frontier = next_frontier
            visited.update(frontier)
        return [self._entities[eid] for eid in visited - {entity_id} if eid in self._entities]

    def neighborhood(
        self, seed_ids: Iterable[str], max_hops: int = 1, max_degree: int = 50
    ) -> Dict[str, int]:
        """Breadth-first hop distance from the nearest seed, for every reachable entity.

        Hub entities with more than ``max_degree`` edges (e.g. a period shared by
        every company) are reached but not expanded further.
        """
        hops = {sid: 0 for sid in seed_ids if sid in self._entities}
        frontier = list(hops)
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for eid in frontier:
                edges = self._edges.get(eid, ())
                if hop > 1 and len(edges) > max_degree:
                    continue
                for _pred, other, _outgoing in edges:
                    if other not in hops and other in self._entities:
                        hops[other] = hop
                        next_frontier.append(other)
            frontier = next_frontier
        return hops

    def chunks_for_entity(self, entity_id: str) -> List[Tuple[str, RetrievedChunk]]:
        return [(key, self._chunks[key]) for key in self._entity_chunks.get(entity_id, ())]

    def chunk_count(self, entity_id: str) -> int:
        return len(self._entity_chunks.get(entity_id, ()))

    def search_by_name(self, name: str, tenant_id: str = "default") -> List[Entity]:
        name_lower = name.lower()
        return [
            self._entities[eid]
            for eid in self._tenant_index.get(tenant_id, [])
            if name_lower in self._entities[eid].name.lower()
        ]

    def find_entities_in_text(
        self, text: str, tenant_id: str = "default", min_name_length: int = 3
    ) -> List[Entity]:
        """Entities of ``tenant_id`` whose name appears in ``text`` as whole words."""
        haystack = f" {re.sub(r'[^a-z0-9%$.]+', ' ', text.lower())} "
        found = []
        for eid in self._tenant_index.get(tenant_id, []):
            entity = self._entities[eid]
            needle = re.sub(r"[^a-z0-9%$.]+", " ", entity.name.lower()).strip()
            if len(needle) >= min_name_length and f" {needle} " in haystack:
                found.append(entity)
        return found

    def stats(self, tenant_id: str = "default") -> Dict[str, int]:
        return {
            "entities": len(self._tenant_index.get(tenant_id, [])),
            "relations": len(self._relations),
            "chunks": sum(1 for key in self._chunks if key.startswith(f"{tenant_id}/")),
        }


class EntityExtractor:
    """LLM-powered entity and relation extractor for financial documents.

    Sends each text chunk to the configured LLM (OpenAI or Gemini, from
    LLM_CONFIG__PROVIDER) with a structured-output prompt and parses the JSON
    response into Entity/Relation objects with canonical ids.

    Extracted entities: COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION, REGULATION
    Extracted relations: SUBSIDIARY_OF, REPORTED_REVENUE, REPORTED_METRIC, FOR_PERIOD,
                         SUPPLIES_TO, CITES, MANAGED_BY, ISSUED_BY, REGULATED_BY, COMPETES_WITH
    """

    EXTRACTION_PROMPT = """\
Extract all named entities and relationships from this financial document excerpt.

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "entities": [
    {"id": "e1", "type": "COMPANY", "name": "Tesla Inc", "value": null},
    {"id": "e2", "type": "METRIC",  "name": "Revenue",   "value": "$23.35B"},
    {"id": "e3", "type": "DATE",    "name": "Q3 2023",   "value": null}
  ],
  "relations": [
    {"subject_id": "e1", "predicate": "REPORTED_REVENUE", "object_id": "e2"},
    {"subject_id": "e2", "predicate": "FOR_PERIOD", "object_id": "e3"}
  ]
}

Entity types: COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION, REGULATION
Relation predicates: SUBSIDIARY_OF, REPORTED_REVENUE, REPORTED_METRIC, FOR_PERIOD, SUPPLIES_TO, CITES, MANAGED_BY, ISSUED_BY, REGULATED_BY, COMPETES_WITH

Use DATE entities for reporting periods (e.g. "Q3 2023", "fiscal 2023"). Link each METRIC
to the company that reported it (REPORTED_REVENUE or REPORTED_METRIC) and to its period
(FOR_PERIOD). Use the company's common name (e.g. "Tesla", "Microsoft").

Text:
{text}"""

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model

    def _get_api_key(self) -> str:
        cfg = get_config()
        return cfg.get_openai_key()

    async def _complete(self, prompt: str) -> str:
        """One JSON-mode completion from the configured provider."""
        import httpx

        cfg = get_config()
        llm = cfg.llm_config
        async with httpx.AsyncClient(timeout=60) as client:
            if llm.provider in ("gemini", "google"):
                model = self._model or llm.model
                generation_config: Dict[str, Any] = {
                    "temperature": 0.0,
                    "maxOutputTokens": 2000,
                    "responseMimeType": "application/json",
                }
                if llm.thinking_budget_tokens is not None:
                    generation_config["maxOutputTokens"] += llm.thinking_budget_tokens
                    generation_config["thinkingConfig"] = {
                        "thinkingBudget": llm.thinking_budget_tokens
                    }
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": cfg.get_google_key()},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": generation_config,
                    },
                )
                response.raise_for_status()
                parts = response.json()["candidates"][0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts if not p.get("thought"))

            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._get_api_key()}"},
                json={
                    "model": self._model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def parse(
        self, raw: str, source_document: str, tenant_id: str = "default"
    ) -> Tuple[List[Entity], List[Relation]]:
        """Turn the model's JSON into entities with canonical ids and resolved relations."""
        import json

        data = json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())

        entities: List[Entity] = []
        by_local_id: Dict[str, Entity] = {}
        for e in data.get("entities", []):
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            etype = str(e.get("type") or "UNKNOWN").upper()
            value = e.get("value")
            entity = Entity(
                id=canonical_entity_id(etype, name, value, tenant_id),
                type=etype,
                name=name,
                source_document=source_document,
                value=value,
                tenant_id=tenant_id,
            )
            entities.append(entity)
            by_local_id[str(e.get("id", name))] = entity

        relations: List[Relation] = []
        for r in data.get("relations", []):
            s = by_local_id.get(str(r.get("subject_id", "")))
            o = by_local_id.get(str(r.get("object_id", "")))
            if s and o and s.id != o.id:
                relations.append(
                    Relation(
                        subject_id=s.id,
                        predicate=str(r.get("predicate") or "RELATED_TO"),
                        object_id=o.id,
                        source_document=source_document,
                        confidence=0.85,
                    )
                )
        return entities, relations

    async def extract(
        self,
        text: str,
        source_document: str,
        tenant_id: str = "default",
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text via LLM structured output."""
        if not text or len(text.strip()) < 50:
            return [], []

        try:
            # NOTE: EXTRACTION_PROMPT embeds a literal JSON example with its
            # own braces, so .format(text=...) would raise KeyError trying
            # to interpret those braces as placeholders. Use a direct
            # string replace of the single {text} marker instead.
            prompt = self.EXTRACTION_PROMPT.replace("{text}", text[:3000])
            entities, relations = self.parse(
                await self._complete(prompt), source_document, tenant_id
            )
            logger.info(
                "kg_extraction_complete",
                source=source_document,
                num_entities=len(entities),
                num_relations=len(relations),
            )
            return entities, relations

        except Exception as exc:
            logger.warning("kg_extraction_failed", source=source_document, error=str(exc)[:120])
            return [], []


class GraphAugmentedRetriever(BaseRetriever):
    """Fuses a base retriever's results with chunks found by graph traversal.

    1. Find entities named in the query (whole-word match on entity names).
    2. Expand to their neighbours up to ``max_hops`` (metrics, periods, related
       companies), without expanding through hub entities.
    3. Score each linked chunk by the entities that reach it: an entity at hop
       h contributes 1/(1+h), divided by log2(2 + number of chunks it appears
       in) so entities mentioned everywhere count for less. Chunks reached by
       several query entities (e.g. "Tesla" and "Q3 2023") rank highest.
    4. Fuse the top graph chunks with the base results using Reciprocal Rank
       Fusion, the same method the hybrid retriever uses.

    Falls back to the base results unchanged when the graph has no entity
    matching the query or no linked chunks.
    """

    def __init__(
        self,
        graph_store: Optional[InMemoryGraphStore] = None,
        base_retriever: Optional[BaseRetriever] = None,
        max_hops: Optional[int] = None,
        max_graph_chunks: Optional[int] = None,
        max_degree: int = 50,
        rrf_k: Optional[int] = None,
    ) -> None:
        cfg = get_config().retriever_config
        self._graph = graph_store or InMemoryGraphStore()
        self._base = base_retriever
        self._max_hops = cfg.graph_max_hops if max_hops is None else max_hops
        self._max_graph_chunks = (
            cfg.graph_max_chunks if max_graph_chunks is None else max_graph_chunks
        )
        self._max_degree = max_degree
        self._rrf_k = cfg.rrf_k if rrf_k is None else rrf_k

    @property
    def name(self) -> str:
        base = self._base.name if self._base else "none"
        return f"graph_augmented({base})"

    @property
    def graph_store(self) -> InMemoryGraphStore:
        return self._graph

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        base_chunks: List[RetrievedChunk] = []
        if self._base is not None:
            base_chunks = await self._base.retrieve(
                query, top_k=top_k, filters=filters, tenant_id=tenant_id
            )
        return await self.retrieve_with_graph(
            query, base_chunks, top_k=top_k, tenant_id=tenant_id or "default"
        )

    def graph_chunks(self, query: str, tenant_id: str = "default") -> List[RetrievedChunk]:
        """Chunks reached from the query's entities, best first (no fusion)."""
        seeds = self._graph.find_entities_in_text(query, tenant_id)
        if not seeds:
            return []
        hops = self._graph.neighborhood(
            [s.id for s in seeds], max_hops=self._max_hops, max_degree=self._max_degree
        )

        scores: Dict[str, float] = defaultdict(float)
        reached_by: Dict[str, Set[str]] = defaultdict(set)
        chunks: Dict[str, RetrievedChunk] = {}
        for entity_id, hop in hops.items():
            linked = self._graph.chunks_for_entity(entity_id)
            if not linked:
                continue
            weight = (1.0 / (1 + hop)) / math.log2(2 + len(linked))
            name = self._graph._entities[entity_id].name
            for key, chunk in linked:
                scores[key] += weight
                reached_by[key].add(name)
                chunks[key] = chunk

        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[: self._max_graph_chunks]
        return [
            RetrievedChunk(
                **{
                    **chunks[k].model_dump(),
                    "score": scores[k],
                    "metadata": {
                        **chunks[k].metadata,
                        "retrieval": "graph",
                        "graph_entities": sorted(reached_by[k]),
                    },
                }
            )
            for k in ranked
        ]

    async def retrieve_with_graph(
        self,
        query: str,
        base_chunks: List[Any],
        top_k: int = 5,
        tenant_id: str = "default",
    ) -> List[Any]:
        if self._graph.stats(tenant_id)["entities"] == 0:
            logger.debug("graph_augmented_retrieval_skipped", reason="empty_graph")
            return base_chunks

        graph_results = self.graph_chunks(query, tenant_id)
        if not graph_results:
            logger.debug("graph_augmented_retrieval_skipped", reason="no_linked_entities")
            return base_chunks

        from src.rag_system.components.retriever import _reciprocal_rank_fusion

        fused = _reciprocal_rank_fusion([list(base_chunks), graph_results], k=self._rrf_k)
        limit = max(top_k, len(base_chunks))
        logger.info(
            "graph_augmented_retrieval",
            tenant_id=tenant_id,
            base=len(base_chunks),
            graph=len(graph_results),
            returned=min(limit, len(fused)),
        )
        return fused[:limit]
