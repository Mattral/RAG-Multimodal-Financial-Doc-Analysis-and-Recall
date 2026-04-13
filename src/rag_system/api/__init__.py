"""FastAPI application with auth middleware, OpenAPI docs, and health probes."""
from src.rag_system.api.app import create_app

__all__ = ["create_app"]
