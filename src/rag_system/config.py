"""Centralized configuration using Pydantic v2 BaseSettings.

Supports environment variables, .env files, multi-tenancy,
secrets management, and all enterprise features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VisionConfig(BaseModel):
    provider: Literal["openai", "qwen2-vl", "pixtral", "internvl"] = "openai"
    model: str = "gpt-4o"
    fallback_model: Optional[str] = "gpt-4-vision-preview"
    max_tokens: int = 1500
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    timeout_seconds: int = 120
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    detail_level: Literal["low", "high", "auto"] = "auto"
    model_config = {"frozen": True}


class PDFParsingConfig(BaseModel):
    primary_parser: Literal["unstructured", "docling", "marker"] = "unstructured"
    fallback_parser: Optional[Literal["unstructured"]] = "unstructured"
    max_characters: int = 4000
    new_after_n_chars: int = 3800
    combine_text_under_n_chars: int = 2000
    infer_table_structure: bool = True
    extract_images: bool = True
    extract_page_numbers: bool = True
    preserve_layout: bool = True
    model_config = {"frozen": True}


class VectorStoreConfig(BaseModel):
    provider: Literal["deeplake", "pgvector", "qdrant", "milvus", "chroma"] = "deeplake"
    dataset_path: Optional[str] = None
    connection_string: Optional[str] = None
    collection_name: str = "rag_financial"
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "local", "voyage", "cohere"] = "openai"
    embedding_dim: int = 1536
    enable_hybrid_search: bool = True
    enable_deep_memory: bool = False
    index_type: str = "hnsw"
    model_config = {"frozen": True}


class RetrieverConfig(BaseModel):
    strategy: Literal["dense", "hybrid", "graph_augmented"] = "hybrid"
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 10
    rrf_k: int = 60
    enable_metadata_filters: bool = True
    enable_reranker: bool = True
    model_config = {"frozen": True}


class RerankerConfig(BaseModel):
    provider: Literal["cohere", "bge", "cross_encoder", "none"] = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    batch_size: int = 32
    model_config = {"frozen": True}


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_second: float = Field(10.0, ge=0.1)
    burst_size: int = 20
    retry_on_429: bool = True
    retry_max_attempts: int = 5
    retry_backoff_factor: float = 2.0
    model_config = {"frozen": True}


class CacheConfig(BaseModel):
    enabled: bool = True
    backend: Literal["redis", "memory"] = "redis"
    redis_url: str = "redis://localhost:6379/0"
    embedding_cache_ttl_seconds: int = 86400
    query_cache_ttl_seconds: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    model_config = {"frozen": True}


class ObservabilityConfig(BaseModel):
    otlp_endpoint: Optional[str] = None
    prometheus_port: int = 8001
    service_name: str = "rag-financial-multimodal"
    service_version: str = "2.0.0"
    trace_sampling_rate: float = Field(1.0, ge=0.0, le=1.0)
    enable_cost_metrics: bool = True
    enable_quality_metrics: bool = True
    slo_p99_latency_ms: float = 8000.0
    shutdown_drain_timeout_seconds: float = Field(
        30.0, ge=0.0, le=300.0,
        description="Max seconds to wait for in-flight requests to drain on SIGTERM "
                    "before the process exits. Should be set lower than the K8s "
                    "terminationGracePeriodSeconds to leave headroom for cleanup.",
    )
    model_config = {"frozen": True}


class SecurityConfig(BaseModel):
    enable_pii_redaction: bool = True
    pii_entities: List[str] = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER",
    ]
    enable_financial_redaction: bool = True
    enable_audit_log: bool = True
    audit_log_backend: Literal["postgres", "s3", "file"] = "file"
    audit_log_path: str = "./audit_logs"
    enable_guardrails: bool = True
    guardrail_numeric_check: bool = True
    api_key_hash_algorithm: str = "bcrypt"
    model_config = {"frozen": True}


class MultiTenancyConfig(BaseModel):
    enabled: bool = True
    isolation_level: Literal["namespace", "collection", "filter"] = "filter"
    default_tenant: str = "default"
    enable_rbac: bool = True
    enable_quotas: bool = True
    default_queries_per_day: int = 1000
    default_tokens_per_month: int = 10_000_000
    model_config = {"frozen": True}


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "azure_openai", "together", "local"] = "openai"
    model: str = "gpt-4o-mini"
    fallback_model: Optional[str] = "gpt-3.5-turbo"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = 2048
    timeout_seconds: int = 60
    complex_query_model: str = "gpt-4o"
    enable_model_routing: bool = True
    model_config = {"frozen": True}


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    include_trace_id: bool = True
    include_span_id: bool = True
    include_memory_metrics: bool = True
    log_file: Optional[str] = None
    model_config = {"frozen": True}


class Config(BaseSettings):
    """Main enterprise configuration for the RAG Financial system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="allow",
    )

    # API Keys
    openai_api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(None, alias="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[SecretStr] = Field(None, alias="COHERE_API_KEY")
    activeloop_token: Optional[SecretStr] = Field(None, alias="ACTIVELOOP_TOKEN")
    voyage_api_key: Optional[SecretStr] = Field(None, alias="VOYAGE_API_KEY")

    # Sub-configs
    vision_config: VisionConfig = Field(default_factory=VisionConfig)
    pdf_parsing_config: PDFParsingConfig = Field(default_factory=PDFParsingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retriever_config: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker_config: RerankerConfig = Field(default_factory=RerankerConfig)
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    observability_config: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
    multi_tenancy_config: MultiTenancyConfig = Field(default_factory=MultiTenancyConfig)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)

    # Pipeline
    batch_size: int = Field(10, ge=1)
    num_workers: int = Field(4, ge=1)
    query_mode: Literal["simple", "hybrid", "agentic"] = "hybrid"

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug_mode: bool = False

    # Feature flags
    enable_langgraph_agentic: bool = False
    enable_knowledge_graph: bool = False
    enable_colpali: bool = False

    @field_validator("llm_config", mode="before")
    @classmethod
    def coerce_llm_config(cls, v: Any) -> LLMConfig:
        return LLMConfig(**v) if isinstance(v, dict) else v

    @field_validator("vision_config", mode="before")
    @classmethod
    def coerce_vision_config(cls, v: Any) -> VisionConfig:
        return VisionConfig(**v) if isinstance(v, dict) else v

    @field_validator("pdf_parsing_config", mode="before")
    @classmethod
    def coerce_pdf_config(cls, v: Any) -> PDFParsingConfig:
        return PDFParsingConfig(**v) if isinstance(v, dict) else v

    @field_validator("vector_store_config", mode="before")
    @classmethod
    def coerce_vector_config(cls, v: Any) -> VectorStoreConfig:
        return VectorStoreConfig(**v) if isinstance(v, dict) else v

    @field_validator("cache_config", mode="before")
    @classmethod
    def coerce_cache_config(cls, v: Any) -> CacheConfig:
        return CacheConfig(**v) if isinstance(v, dict) else v

    @field_validator("security_config", mode="before")
    @classmethod
    def coerce_security_config(cls, v: Any) -> SecurityConfig:
        return SecurityConfig(**v) if isinstance(v, dict) else v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        return self.debug_mode or self.environment == "development"

    def get_openai_key(self) -> str:
        from src.rag_system.utils.exceptions import ConfigurationError
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY not set", config_key="openai_api_key")
        return self.openai_api_key.get_secret_value()


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    global _config
    _config = None
