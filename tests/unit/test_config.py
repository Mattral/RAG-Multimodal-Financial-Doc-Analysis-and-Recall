"""Config validation and env override tests."""
import os
import pytest
from src.rag_system.config import Config, reset_config


def test_env_override_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_CONFIG__MODEL", "gpt-4o")
    reset_config()
    cfg = Config()
    assert cfg.llm_config.model == "gpt-4o"
    reset_config()


def test_env_override_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    reset_config()
    cfg = Config()
    assert cfg.is_production is True
    reset_config()


def test_nested_config_frozen():
    cfg = Config()
    with pytest.raises(Exception):
        cfg.llm_config.model = "hacked"  # type: ignore


def test_multi_tenancy_defaults():
    cfg = Config()
    assert cfg.multi_tenancy_config.enabled is True
    assert cfg.multi_tenancy_config.default_tenant == "default"
