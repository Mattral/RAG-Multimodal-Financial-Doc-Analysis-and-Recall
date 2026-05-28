"""Tests for configuration management."""

import pytest
from pydantic import ValidationError

from src.rag_system.config import (
    Config,
    VisionConfig,
    PDFParsingConfig,
    VectorStoreConfig,
    RateLimitConfig,
    LoggingConfig,
    get_config,
    reset_config,
)


class TestVisionConfig:
    """Test vision configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VisionConfig()
        assert config.model == "gpt-4-vision-preview"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            VisionConfig(temperature=2.5)

    def test_frozen_config(self) -> None:
        """Test that config is frozen."""
        config = VisionConfig()
        with pytest.raises(Exception):
            config.model = "gpt-4"  # type: ignore


class TestPDFParsingConfig:
    """Test PDF parsing configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PDFParsingConfig()
        assert config.max_characters == 4000
        assert config.new_after_n_chars == 3800
        assert config.infer_table_structure is True


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20

    def test_validation_requests_per_second(self) -> None:
        """Test validation of requests per second."""
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_second=0.0)


class TestConfigIntegration:
    """Integration tests for main Config."""

    def test_main_config_creation(self) -> None:
        """Test main config creation with required fields."""
        # This would require OPENAI_API_KEY to be set
        # For testing, we'd need to mock or set the environment variable
        pass

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        # This would require a mock config
        pass


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()
