"""Knowledge Graph extraction and graph-augmented retrieval stub.

Guideline §7: 'Knowledge Graph layer (optional Neo4j or in-memory +
LLM extraction of entities/relations: companies, metrics, events, filings).'

Status: Interface + in-memory stub in v2.0. Full implementation in v3.0.
Enable via: ENABLE_KNOWLEDGE_GRAPH=true
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)


@dataclass
class Entity:
    """A named entity extracted from a financial document."""
    id: str
    type: str           # COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION
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
    predicate: str      # SUBSIDIARY_OF, REPORTED_REVENUE, SUPPLIES_TO, CITES, MANAGED_BY
    object_id: str
    source_document: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryGraphStore:
    """In-memory graph store for development and testing.

    Replace with Neo4j, NetworkX + persistence, or Kuzu for production.
    Interface is identical.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._tenant_index: Dict[str, List[str]] = {}

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        if entity.tenant_id not in self._tenant_index:
            self._tenant_index[entity.tenant_id] = []
        self._tenant_index[entity.tenant_id].append(entity.id)

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)

    def get_neighbors(
        self, entity_id: str, predicate: Optional[str] = None, depth: int = 1
    ) -> List[Entity]:
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for eid in frontier:
                for rel in self._relations:
                    if rel.subject_id == eid:
                        if (predicate is None or rel.predicate == predicate) and rel.object_id not in visited:
                            next_frontier.add(rel.object_id)
                    elif rel.object_id == eid and rel.subject_id not in visited:
                        next_frontier.add(rel.subject_id)
            frontier = next_frontier
            visited.update(frontier)
        return [self._entities[eid] for eid in visited - {entity_id} if eid in self._entities]

    def search_by_name(self, name: str, tenant_id: str = "default") -> List[Entity]:
        name_lower = name.lower()
        return [
            self._entities[eid]
            for eid in self._tenant_index.get(tenant_id, [])
            if name_lower in self._entities[eid].name.lower()
        ]

    def stats(self, tenant_id: str = "default") -> Dict[str, int]:
        return {
            "entities": len(self._tenant_index.get(tenant_id, [])),
            "relations": len(self._relations),
        }


class EntityExtractor:
    """LLM-powered entity and relation extractor for financial documents.

    Sends each text chunk to the LLM with a structured-output prompt and
    parses the JSON response into Entity/Relation objects.

    Extracted entities: COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION, REGULATION
    Extracted relations: SUBSIDIARY_OF, REPORTED_REVENUE, SUPPLIES_TO, CITES,
                          MANAGED_BY, ISSUED_BY, REGULATED_BY, COMPETES_WITH
    """

    EXTRACTION_PROMPT = """\
Extract all named entities and relationships from this financial document excerpt.

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "entities": [
    {"id": "e1", "type": "COMPANY", "name": "Tesla Inc", "value": null},
    {"id": "e2", "type": "METRIC",  "name": "Revenue",   "value": "$23.35B"}
  ],
  "relations": [
    {"subject_id": "e1", "predicate": "REPORTED_REVENUE", "object_id": "e2"}
  ]
}

Entity types: COMPANY, PERSON, METRIC, DATE, PRODUCT, LOCATION, REGULATION
Relation predicates: SUBSIDIARY_OF, REPORTED_REVENUE, SUPPLIES_TO, CITES, MANAGED_BY, ISSUED_BY, REGULATED_BY, COMPETES_WITH

Text:
{text}"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def _get_api_key(self) -> str:
        cfg = get_config()
        return cfg.get_openai_key()

    async def extract(
        self,
        text: str,
        source_document: str,
        tenant_id: str = "default",
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text via LLM structured output."""
        import hashlib
        import json
        import re

        import httpx

        if not text or len(text.strip()) < 50:
            return [], []

        try:
            api_key = self._get_api_key()
            # NOTE: EXTRACTION_PROMPT embeds a literal JSON example with its
            # own braces, so .format(text=...) would raise KeyError trying
            # to interpret those braces as placeholders. Use a direct
            # string replace of the single {text} marker instead.
            prompt = self.EXTRACTION_PROMPT.replace("{text}", text[:3000])
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]

            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            data = json.loads(raw)

            entities: List[Entity] = []
            entity_map: Dict[str, Entity] = {}
            for e in data.get("entities", []):
                eid = e.get("id", hashlib.sha256(e.get("name","").encode()).hexdigest()[:8])
                entity = Entity(
                    id=f"{source_document}:{eid}",
                    type=e.get("type", "UNKNOWN"),
                    name=e.get("name", ""),
                    source_document=source_document,
                    value=e.get("value"),
                    tenant_id=tenant_id,
                )
                entities.append(entity)
                entity_map[eid] = entity

            relations: List[Relation] = []
            for r in data.get("relations", []):
                s, o = r.get("subject_id",""), r.get("object_id","")
                if s in entity_map and o in entity_map:
                    relations.append(Relation(
                        subject_id=entity_map[s].id,
                        predicate=r.get("predicate", "RELATED_TO"),
                        object_id=entity_map[o].id,
                        source_document=source_document,
                        confidence=0.85,
                    ))

            logger.info("kg_extraction_complete", source=source_document,
                       num_entities=len(entities), num_relations=len(relations))
            return entities, relations

        except Exception as exc:
            logger.warning("kg_extraction_failed", source=source_document, error=str(exc)[:120])
            return [], []


class GraphAugmentedRetriever:
    """Combines vector search with graph traversal.

    EntityExtractor performs real LLM-based extraction. The graph-traversal
    augmentation logic itself (merging graph-discovered chunks into base
    retrieval results) is still v3.0 — falls back to standard vector
    retrieval until that traversal logic ships.
    """

    def __init__(self, graph_store: Optional[InMemoryGraphStore] = None) -> None:
        self._graph = graph_store or InMemoryGraphStore()

    async def retrieve_with_graph(
        self,
        query: str,
        base_chunks: List[Any],
        top_k: int = 5,
        tenant_id: str = "default",
    ) -> List[Any]:
        stats = self._graph.stats(tenant_id)
        if stats["entities"] == 0:
            logger.debug("graph_augmented_retrieval_skipped", reason="empty_graph")
            return base_chunks
        # Full traversal logic in v3.0
        return base_chunks
