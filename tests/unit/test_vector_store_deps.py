"""A configured vector store whose client package is missing must fail at startup.

DeepLake used to catch ImportError inside upsert and log a warning, so ingestion
reported success while every vector was dropped.
"""

from __future__ import annotations

import sys

import pytest

from filing_qa.components.vector_store import build_vector_store
from filing_qa.utils.exceptions import ConfigurationError


@pytest.mark.parametrize(
    "provider, module, hint",
    [
        ("deeplake", "deeplake", "pip install 'deeplake>=3.9,<4'"),
        ("qdrant", "qdrant_client", "pip install 'qdrant-client>=1.9.0'"),
        ("pgvector", "asyncpg", "pip install asyncpg pgvector"),
    ],
)
def test_missing_package_raises_at_construction(monkeypatch, provider, module, hint):
    # A None entry in sys.modules makes `import module` raise ImportError.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ConfigurationError) as info:
        build_vector_store(provider)
    message = str(info.value)
    assert f"'{provider}'" in message
    assert f"'{module}'" in message
    assert hint in message


def test_memory_store_needs_no_optional_package():
    assert build_vector_store("memory").name == "in_memory"
