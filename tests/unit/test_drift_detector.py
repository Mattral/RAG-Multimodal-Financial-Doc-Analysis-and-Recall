"""Unit tests for drift detection utilities."""

from src.rag_system.utils.drift_detector import (
    EmbeddingDriftDetector,
    QueryPatternDriftDetector,
)


class TestEmbeddingDriftDetector:
    def test_first_snapshot_no_drift(self, tmp_path):
        d = EmbeddingDriftDetector(state_path=str(tmp_path / "drift.json"))
        result = d.snapshot([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], tenant_id="t1")
        assert result["drift_detected"] is False
        assert result["cosine_distance"] == 0.0

    def test_identical_snapshots_no_drift(self, tmp_path):
        d = EmbeddingDriftDetector(state_path=str(tmp_path / "drift.json"))
        vecs = [[0.1, 0.9, 0.2]] * 10
        d.snapshot(vecs, tenant_id="t1")
        result = d.snapshot(vecs, tenant_id="t1")
        assert result["drift_detected"] is False
        assert result["cosine_distance"] < 0.01

    def test_orthogonal_embeddings_high_drift(self, tmp_path):
        d = EmbeddingDriftDetector(state_path=str(tmp_path / "drift.json"), threshold=0.05)
        d.snapshot([[1.0, 0.0, 0.0]] * 10, tenant_id="t1")
        result = d.snapshot([[0.0, 1.0, 0.0]] * 10, tenant_id="t1")
        assert result["drift_detected"] is True
        assert result["cosine_distance"] > 0.9

    def test_empty_embeddings_skipped(self, tmp_path):
        d = EmbeddingDriftDetector(state_path=str(tmp_path / "drift.json"))
        result = d.snapshot([], tenant_id="t1")
        assert result["status"] == "skipped"

    def test_tenant_isolation(self, tmp_path):
        d = EmbeddingDriftDetector(state_path=str(tmp_path / "drift.json"), threshold=0.05)
        d.snapshot([[1.0, 0.0]] * 5, tenant_id="t1")
        d.snapshot([[1.0, 0.0]] * 5, tenant_id="t2")
        # t2 has different history than t1 — should not cross-contaminate
        r1 = d.snapshot([[0.0, 1.0]] * 5, tenant_id="t1")
        r2 = d.snapshot([[1.0, 0.0]] * 5, tenant_id="t2")
        assert r1["drift_detected"] is True
        assert r2["drift_detected"] is False

    def test_state_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "drift.json")
        d1 = EmbeddingDriftDetector(state_path=path, threshold=0.05)
        d1.snapshot([[1.0, 0.0, 0.0]] * 10, tenant_id="t1")
        d2 = EmbeddingDriftDetector(state_path=path, threshold=0.05)
        result = d2.snapshot([[0.0, 1.0, 0.0]] * 10, tenant_id="t1")
        assert result["drift_detected"] is True


class TestQueryPatternDriftDetector:
    def test_record_and_retrieve_distribution(self, tmp_path):
        d = QueryPatternDriftDetector(state_path=str(tmp_path / "qdrift.json"))
        for _ in range(8):
            d.record_query("factual", "t1")
        for _ in range(2):
            d.record_query("numeric", "t1")
        dist = d.get_distribution("t1")
        assert abs(dist["factual"] - 0.8) < 0.01
        assert abs(dist["numeric"] - 0.2) < 0.01

    def test_empty_distribution_returns_empty(self, tmp_path):
        d = QueryPatternDriftDetector(state_path=str(tmp_path / "qdrift.json"))
        assert d.get_distribution("unknown") == {}

    def test_no_drift_when_stable(self, tmp_path):
        d = QueryPatternDriftDetector(
            state_path=str(tmp_path / "qdrift.json"), alert_threshold=0.20
        )
        for _ in range(100):
            d.record_query("factual", "t1")
        result = d.check_drift(["factual"] * 50, "t1")
        assert result["drift_detected"] is False

    def test_drift_detected_on_shift(self, tmp_path):
        d = QueryPatternDriftDetector(
            state_path=str(tmp_path / "qdrift.json"), alert_threshold=0.10
        )
        for _ in range(100):
            d.record_query("factual", "t1")
        # Sudden shift to all agentic queries
        result = d.check_drift(["agentic"] * 50, "t1")
        assert result["drift_detected"] is True

    def test_insufficient_data_no_crash(self, tmp_path):
        d = QueryPatternDriftDetector(state_path=str(tmp_path / "qdrift.json"))
        result = d.check_drift([], "t1")
        assert result["drift_detected"] is False
