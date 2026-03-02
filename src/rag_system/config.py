"""Centralized configuration using Pydantic BaseSettings."""

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator, SecretStr
from pydantic_settings import BaseSettings


class VisionConfig(BaseModel):
    """Configuration for vision processing (GPT-4V)."""

    model: str = Field(default="gpt-4-vision-preview", description="Vision model to use")
    max_tokens: int = Field(default=1000, description="Max tokens for vision response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    timeout_seconds: int = Field(default=120, description="Request timeout in seconds")
    retry_max_attempts: int = Field(default=3, description="Max retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")

    class Config:
        frozen = True


class PDFParsingConfig(BaseModel):
    """Configuration for PDF parsing."""

    max_characters: int = Field(default=4000, description="Max characters per chunk")
    new_after_n_chars: int = Field(default=3800, description="New chunk after N chars")
    combine_text_under_n_chars: int = Field(default=2000, description="Combine chunks under N chars")
    infer_table_structure: bool = Field(default=True, description="Infer table structure")
    extract_images: bool = Field(default=False, description="Extract images from PDF")

    class Config:
        frozen = True


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (DeepLake)."""

    dataset_path: Optional[str] = Field(default=None, description="DeepLake dataset path (hub://org/dataset)")
    runtime_type: str = Field(default="tensor_db", description="DeepLake runtime type")
    read_only: bool = Field(default=False, description="Read-only mode")
    overwrite: bool = Field(default=False, description="Overwrite existing dataset")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    embedding_dim: int = Field(default=1536, description="Embedding dimension")
    enable_deep_memory: bool = Field(default=False, description="Enable Deep Memory feature")

    class Config:
        frozen = True


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_second: float = Field(default=10.0, ge=0.1, description="Requests per second")
    burst_size: int = Field(default=20, description="Burst size for token bucket")
    retry_on_429: bool = Field(default=True, description="Retry on HTTP 429")
    retry_max_attempts: int = Field(default=5, description="Max retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")

    class Config:
        frozen = True


class LoggingConfig(BaseModel):
    """Configuration for structured logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    include_trace_id: bool = Field(default=True, description="Include trace_id in logs")
    include_span_id: bool = Field(default=True, description="Include span_id in logs")
    include_memory_metrics: bool = Field(default=True, description="Include memory metrics")

    class Config:
        frozen = True


class Config(BaseSettings):
    """Main configuration for the RAG system."""

    # API Keys (required)
    openai_api_key: Optional[SecretStr] = Field(
        None, description="OpenAI API key", alias="OPENAI_API_KEY"
    )
    activeloop_token: Optional[SecretStr] = Field(
        None, description="Activeloop token", alias="ACTIVELOOP_TOKEN"
    )

    # Model Configuration
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")

    # Component Configurations
    vision_config: VisionConfig = Field(default_factory=VisionConfig)
    pdf_parsing_config: PDFParsingConfig = Field(default_factory=PDFParsingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)

    # Pipeline Configuration
    batch_size: int = Field(default=10, ge=1, description="Batch size for processing")
    num_workers: int = Field(default=4, ge=1, description="Number of async workers")
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment type"
    )
    debug_mode: bool = Field(default=False, description="Debug mode (verbose logging)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

    @validator("vision_config", pre=True, always=True)
    def validate_vision_config(cls, v: any) -> VisionConfig:
        """Ensure vision_config is a VisionConfig instance."""
        if isinstance(v, dict):
            return VisionConfig(**v)
        return v

    @validator("pdf_parsing_config", pre=True, always=True)
    def validate_pdf_config(cls, v: any) -> PDFParsingConfig:
        """Ensure pdf_parsing_config is a PDFParsingConfig instance."""
        if isinstance(v, dict):
            return PDFParsingConfig(**v)
        return v

    @validator("vector_store_config", pre=True, always=True)
    def validate_vector_config(cls, v: any) -> VectorStoreConfig:
        """Ensure vector_store_config is a VectorStoreConfig instance."""
        if isinstance(v, dict):
            return VectorStoreConfig(**v)
        return v

    @validator("rate_limit_config", pre=True, always=True)
    def validate_rate_limit_config(cls, v: any) -> RateLimitConfig:
        """Ensure rate_limit_config is a RateLimitConfig instance."""
        if isinstance(v, dict):
            return RateLimitConfig(**v)
        return v

    @validator("logging_config", pre=True, always=True)
    def validate_logging_config(cls, v: any) -> LoggingConfig:
        """Ensure logging_config is a LoggingConfig instance."""
        if isinstance(v, dict):
            return LoggingConfig(**v)
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode or self.environment == "development"


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the global config instance.

    Returns:
        Config: The global configuration instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global config instance (mainly for testing)."""
    global _config
    _config = None
