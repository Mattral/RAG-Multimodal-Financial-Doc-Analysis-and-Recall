"""Fail at construction when a configured vector store's package is missing."""

from __future__ import annotations

import importlib

from filing_qa.utils.exceptions import ConfigurationError


def require_package(module: str, provider: str, install: str) -> None:
    """Raise ConfigurationError if ``module`` cannot be imported.

    A vector store that cannot import its client must not start: logging a warning
    and skipping writes silently loses every ingested vector.
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise ConfigurationError(
            f"Vector store provider '{provider}' needs the '{module}' package, which is "
            f"not installed. Install it with: {install}",
            config_key="VECTOR_STORE_CONFIG__PROVIDER",
        ) from exc
