"""Configuration tests — upgraded for v2.0 Pydantic v2 BaseSettings."""

import pytest
from pydantic import ValidationError

from src.rag_system.config import (
    CacheConfig,
    Config,
    LLMConfig,
    MultiTenancyConfig,
    PDFParsingConfig,
    RateLimitConfig,
    RerankerConfig,
    RetrieverConfig,
    SecurityConfig,
    VectorStoreConfig,
    VisionConfig,
    get_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def reset():
    reset_config()
    yield
    reset_config()


class TestVisionConfig:
    def test_default_values(self):
        c = VisionConfig()
        assert c.model == "gpt-4o"
        assert c.max_tokens == 1500
        assert 0.0 <= c.temperature <= 2.0

    def test_temperature_validation_upper_bound(self):
        with pytest.raises(ValidationError):
            VisionConfig(temperature=2.5)

    def test_temperature_validation_lower_bound(self):
        with pytest.raises(ValidationError):
            VisionConfig(temperature=-0.1)

    def test_frozen(self):
        c = VisionConfig()
        with pytest.raises(ValidationError):
            c.model = "mutated"  # type: ignore

    def test_fallback_model_optional(self):
        c = VisionConfig(fallback_model=None)
        assert c.fallback_model is None


class TestPDFParsingConfig:
    def test_defaults(self):
        c = PDFParsingConfig()
        assert c.max_characters == 4000
        assert c.infer_table_structure is True
        assert c.extract_images is True

    def test_primary_parser_options(self):
        c = PDFParsingConfig(primary_parser="docling")
        assert c.primary_parser == "docling"
        with pytest.raises(ValidationError):
            PDFParsingConfig(primary_parser="invalid_parser")


class TestVectorStoreConfig:
    def test_defaults(self):
        c = VectorStoreConfig()
        assert c.provider == "deeplake"
        assert c.embedding_dim == 1536
        assert c.enable_hybrid_search is True

    def test_provider_options(self):
        for p in ["deeplake", "pgvector", "qdrant", "milvus", "chroma"]:
            c = VectorStoreConfig(provider=p)
            assert c.provider == p


class TestRetrieverConfig:
    def test_defaults(self):
        c = RetrieverConfig()
        assert c.strategy == "hybrid"
        assert c.top_k_dense == 20
        assert c.enable_reranker is True

    def test_strategy_options(self):
        for s in ["dense", "hybrid", "graph_augmented"]:
            c = RetrieverConfig(strategy=s)
            assert c.strategy == s


class TestRerankerConfig:
    def test_defaults(self):
        c = RerankerConfig()
        assert c.provider == "cross_encoder"
        assert c.top_n == 5


class TestRateLimitConfig:
    def test_defaults(self):
        c = RateLimitConfig()
        assert c.enabled is True
        assert c.requests_per_second == 10.0
        assert c.burst_size == 20

    def test_rps_lower_bound(self):
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_second=0.0)

    def test_rps_minimum_valid(self):
        c = RateLimitConfig(requests_per_second=0.1)
        assert c.requests_per_second == 0.1


class TestCacheConfig:
    def test_defaults(self):
        c = CacheConfig()
        assert c.enabled is True
        assert c.semantic_cache_threshold == 0.92
        assert c.embedding_cache_ttl_seconds == 86400

    def test_backend_options(self):
        for b in ["redis", "memory"]:
            c = CacheConfig(backend=b)
            assert c.backend == b


class TestSecurityConfig:
    def test_defaults(self):
        c = SecurityConfig()
        assert c.enable_pii_redaction is True
        assert c.enable_guardrails is True
        assert "PERSON" in c.pii_entities
        assert "US_SSN" in c.pii_entities

    def test_audit_log_backend_options(self):
        for b in ["postgres", "s3", "file"]:
            c = SecurityConfig(audit_log_backend=b)
            assert c.audit_log_backend == b


class TestMultiTenancyConfig:
    def test_defaults(self):
        c = MultiTenancyConfig()
        assert c.enabled is True
        assert c.default_tenant == "default"
        assert c.default_queries_per_day == 1000

    def test_isolation_levels(self):
        for level in ["namespace", "collection", "filter"]:
            c = MultiTenancyConfig(isolation_level=level)
            assert c.isolation_level == level


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.model == "gpt-4o-mini"
        assert c.complex_query_model == "gpt-4o"
        assert c.enable_model_routing is True

    def test_provider_options(self):
        for p in ["openai", "anthropic", "azure_openai", "together", "local"]:
            c = LLMConfig(provider=p)
            assert c.provider == p


class TestMainConfig:
    def test_defaults(self, monkeypatch):
        # conftest.py sets ENVIRONMENT=testing session-wide; clear it here to
        # verify the schema's true default value in isolation.
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        c = Config()
        assert c.environment == "development"
        assert c.batch_size == 10
        assert c.num_workers == 4
        assert c.query_mode == "hybrid"

    def test_is_production_false_by_default(self):
        c = Config()
        assert c.is_production is False

    def test_is_production_true(self):
        c = Config(environment="production")
        assert c.is_production is True

    def test_is_debug_true_in_dev(self):
        c = Config(environment="development")
        assert c.is_debug is True

    def test_missing_api_key_raises(self, monkeypatch):
        from src.rag_system.utils.exceptions import ConfigurationError

        # conftest.py sets OPENAI_API_KEY session-wide for the rest of the
        # suite; clear it here to verify the genuinely-missing-key path.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c = Config()
        with pytest.raises(ConfigurationError):
            c.get_openai_key()

    def test_api_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-xyz")
        reset_config()
        c = Config()
        assert c.get_openai_key() == "sk-test-key-xyz"

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_env_nested_override(self, monkeypatch):
        monkeypatch.setenv("LLM_CONFIG__MODEL", "gpt-4o")
        reset_config()
        c = Config()
        assert c.llm_config.model == "gpt-4o"
        reset_config()

    def test_environment_options(self):
        for env in ["development", "staging", "production"]:
            c = Config(environment=env)
            assert c.environment == env

    def test_invalid_environment_raises(self):
        with pytest.raises(ValidationError):
            Config(environment="invalid")
