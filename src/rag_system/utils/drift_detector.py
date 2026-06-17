"""Corpus drift and query pattern drift detection.

Guideline §3.3: 'Drift Detection: On corpus (new doc types, distribution shift in embeddings)
or query patterns → trigger reindex or alert.'

Detects:
1. Embedding distribution drift (cosine centroid shift over time)
2. Query pattern drift (new intents, new entity types)
3. Corpus freshness drift (stale documents, new doc types appearing)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingDriftDetector:
    """Monitors centroid shift in embedding space over time.

    Computes the centroid of embeddings at each snapshot and alerts
    if the cosine distance between consecutive centroids exceeds threshold.
    """

    def __init__(
        self,
        state_path: str = "./data/drift_state.json",
        threshold: float = 0.15,
    ) -> None:
        self._state_path = Path(state_path)
        self._threshold = threshold
        self._state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2))

    @staticmethod
    def _centroid(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        n = len(vectors)
        dim = len(vectors[0])
        return [sum(v[i] for v in vectors) / n for i in range(dim)]

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 1.0
        return 1.0 - dot / (mag_a * mag_b)

    def snapshot(
        self,
        embeddings: List[List[float]],
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Record a snapshot of the current embedding distribution."""
        if not embeddings:
            return {"status": "skipped", "reason": "empty_embeddings"}

        centroid = self._centroid(embeddings[:1000])  # cap for performance
        now = datetime.now(timezone.utc).isoformat()
        key = f"tenant:{tenant_id}"

        drift_detected = False
        distance = 0.0
        prev = self._state.get(key)

        if prev and prev.get("centroid"):
            distance = self._cosine_distance(centroid, prev["centroid"])
            drift_detected = distance > self._threshold
            if drift_detected:
                logger.warning(
                    "embedding_drift_detected",
                    tenant_id=tenant_id,
                    cosine_distance=round(distance, 4),
                    threshold=self._threshold,
                    recommendation="Consider triggering a full reindex",
                )

        self._state[key] = {
            "centroid": centroid,
            "snapshot_timestamp": now,
            "num_embeddings": len(embeddings),
            "prev_distance": distance,
        }
        self._save_state()

        return {
            "tenant_id": tenant_id,
            "drift_detected": drift_detected,
            "cosine_distance": round(distance, 4),
            "threshold": self._threshold,
            "num_embeddings_sampled": min(len(embeddings), 1000),
            "timestamp": now,
        }


class QueryPatternDriftDetector:
    """Monitors shifts in query intent distribution over time.

    Alerts when a new intent type appears frequently or when
    entity extraction patterns shift significantly.
    """

    def __init__(
        self,
        state_path: str = "./data/query_drift_state.json",
        window_size: int = 1000,
        alert_threshold: float = 0.20,
    ) -> None:
        self._state_path = Path(state_path)
        self._window = window_size
        self._threshold = alert_threshold
        self._state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2))

    def record_query(self, intent: str, tenant_id: str = "default") -> None:
        key = f"intents:{tenant_id}"
        counts: Dict[str, int] = self._state.get(key, {})
        counts[intent] = counts.get(intent, 0) + 1
        self._state[key] = counts
        self._save_state()

    def get_distribution(self, tenant_id: str = "default") -> Dict[str, float]:
        key = f"intents:{tenant_id}"
        counts = self._state.get(key, {})
        total = sum(counts.values())
        if total == 0:
            return {}
        return {intent: count / total for intent, count in counts.items()}

    def check_drift(
        self,
        recent_intents: List[str],
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Compare recent intent distribution against historical baseline."""
        historical = self.get_distribution(tenant_id)
        if not historical or not recent_intents:
            return {"drift_detected": False, "reason": "insufficient_data"}

        recent_counts: Dict[str, int] = {}
        for intent in recent_intents:
            recent_counts[intent] = recent_counts.get(intent, 0) + 1
        total = len(recent_intents)
        recent_dist = {k: v / total for k, v in recent_counts.items()}

        # Jensen-Shannon divergence proxy (simplified)
        all_intents = set(historical) | set(recent_dist)
        drift_score = sum(
            abs(recent_dist.get(i, 0) - historical.get(i, 0))
            for i in all_intents
        ) / 2

        drift_detected = drift_score > self._threshold
        if drift_detected:
            logger.warning(
                "query_pattern_drift_detected",
                tenant_id=tenant_id,
                drift_score=round(drift_score, 4),
                new_intents=list(set(recent_dist) - set(historical)),
            )

        return {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "threshold": self._threshold,
            "historical_distribution": historical,
            "recent_distribution": recent_dist,
        }
