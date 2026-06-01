"""Centralized configuration using Pydantic V2 and BaseSettings."""

import os
from typing import Literal, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator, SecretStr, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from phi.utils.log import logger

# --- SUB-CONFIGURATION MODELS ---

class VisionConfig(BaseModel):
    """Configuration for vision processing (GPT-4V)."""
    model_config = ConfigDict(frozen=True)

    model: str = Field(default="gpt-4-vision-preview", description="Vision model to use")
    max_tokens: int = Field(default=1000, description="Max tokens for vision response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    timeout_seconds: int = Field(default=120, description="Request timeout in seconds")
    retry_max_attempts: int = Field(default=3, description="Max retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")


class PDFParsingConfig(BaseModel):
    """Configuration for PDF parsing and chunk serialization patterns."""
    model_config = ConfigDict(frozen=True)

    max_characters: int = Field(default=4000, description="Max characters per chunk")
    new_after_n_chars: int = Field(default=3800, description="New chunk after N chars")
    combine_text_under_n_chars: int = Field(default=2000, description="Combine chunks under N chars")
    infer_table_structure: bool = Field(default=True, description="Infer table structure")
    extract_images: bool = Field(default=False, description="Extract images from PDF")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (DeepLake)."""
    model_config = ConfigDict(frozen=True)

    dataset_path: str = Field(..., description="DeepLake dataset path (hub://org/dataset or local path)")
    runtime_type: str = Field(default="tensor_db", description="DeepLake runtime type")
    read_only: bool = Field(default=False, description="Read-only mode")
    overwrite: bool = Field(default=False, description="Overwrite existing dataset")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    embedding_dim: int = Field(default=1536, description="Embedding dimension")
    enable_deep_memory: bool = Field(default=False, description="Enable Deep Memory feature")


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting and token buckets."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_second: float = Field(default=10.0, ge=0.1, description="Requests per second")
    burst_size: int = Field(default=20, description="Burst size for token bucket")
    retry_on_429: bool = Field(default=True, description="Retry on HTTP 429")
    retry_max_attempts: int = Field(default=5, description="Max retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")


class LoggingConfig(BaseModel):
    """Configuration for structured tracing outputs."""
    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    include_trace_id: bool = Field(default=True, description="Include trace_id in logs")
    include_span_id: bool = Field(default=True, description="Include span_id in logs")
    include_memory_metrics: bool = Field(default=True, description="Include memory metrics")


# --- MAIN CENTRALIZED SYSTEM SETTINGS CONFIGURATION ---

class Config(BaseSettings):
    """Main centralized type-safe settings engine for the system."""
    
    # Locate .env gracefully before compilation step
    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Required API Keys
    openai_api_key: SecretStr = Field(
        ..., description="OpenAI API key", alias="OPENAI_API_KEY"
    )
    activeloop_token: Optional[SecretStr] = Field(
        None, description="Activeloop token required for hub:// vectors", alias="ACTIVELOOP_TOKEN"
    )

    # Target Language Models Config
    llm_model: str = Field(default="gpt-4o", description="Primary model execution target")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM sampling temperature")

    # Composite Config Blocks
    vision_config: VisionConfig = Field(default_factory=VisionConfig)
    pdf_parsing_config: PDFParsingConfig = Field(default_factory=PDFParsingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime Execution Optimization Properties
    batch_size: int = Field(default=10, ge=1, description="Batch size for concurrent processing")
    num_workers: int = Field(default=4, ge=1, description="Number of async engine worker threads")
    enable_cache: bool = Field(default=True, description="Enable response caching mechanisms")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL parameters in seconds")

    # Workspace Context Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Workspace deployment environment parameter"
    )
    debug_mode: bool = Field(default=False, description="Activate diagnostic level logs")

    # --- MODERN VALIDATION HOOKS (Pydantic v2 Compliant) ---

    @field_validator("vision_config", mode="before")
    @classmethod
    def validate_vision_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return VisionConfig(**v)
        return v

    @field_validator("pdf_parsing_config", mode="before")
    @classmethod
    def validate_pdf_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return PDFParsingConfig(**v)
        return v

    @field_validator("vector_store_config", mode="before")
    @classmethod
    def validate_vector_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return VectorStoreConfig(**v)
        return v

    @field_validator("rate_limit_config", mode="before")
    @classmethod
    def validate_rate_limit_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return RateLimitConfig(**v)
        return v

    @field_validator("logging_config", mode="before")
    @classmethod
    def validate_logging_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return LoggingConfig(**v)
        return v

    # --- HEALTH CHECKS & HELPER RESOLVERS ---

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        return self.debug_mode or self.environment == "development"

    @property
    def reveal_openai_key(self) -> str:
        """Safely extracts OpenAI token without printing object references in debug dumps."""
        return self.openai_api_key.get_secret_value()

    @property
    def reveal_activeloop_token(self) -> Optional[str]:
        """Safely extracts Activeloop token, falling back to None."""
        return self.activeloop_token.get_secret_value() if self.activeloop_token else None


# --- GLOBAL RESOURCE SINGLETON HOOKS ---

_config: Optional[Config] = None

def get_config() -> Config:
    """Gets or registers the active Config instance safely."""
    global _config
    if _config is None:
        try:
            _config = Config()
            # Dynamic warning validation logic: Hub datasets require token authorization
            if _config.vector_store_config.dataset_path.startswith("hub://") and not _config.activeloop_token:
                logger.warning(
                    "DeepLake path set to cloud storage ('hub://') but no 'ACTIVELOOP_TOKEN' is loaded. "
                    "Write permissions or private read workflows might fail during pipeline execution."
                )
        except Exception as e:
            logger.error(f"Failed to compile application runtime settings properties: {e}")
            raise ValueError(f"Settings Parsing Error: Ensure .env is properly formatted. Internal: {e}")
    return _config


def reset_config() -> None:
    """Flushes session configuration cache cleanly (useful for dynamic testing environments)."""
    global _config
    _config = None
