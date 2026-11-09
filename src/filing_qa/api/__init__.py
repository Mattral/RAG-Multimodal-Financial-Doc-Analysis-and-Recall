"""FastAPI application with auth middleware, OpenAPI docs, and health probes."""

from filing_qa.api.app import create_app

__all__ = ["create_app"]
