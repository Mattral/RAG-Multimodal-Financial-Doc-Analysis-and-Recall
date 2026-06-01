"""Tests for CLI interface."""

import pytest
from typer.testing import CliRunner

# Import CLI app
from src.rag_system.cli import app


runner = CliRunner()


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self) -> None:
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RAG system" in result.stdout

    def test_version_command(self) -> None:
        """Test version command."""
        result = runner.invoke(app, ["version"])
        # Version command should succeed
        assert result.exit_code == 0
        assert "v1.0.0" in result.stdout

    def test_ingest_help(self) -> None:
        """Test ingest command help."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.stdout.lower()

    def test_query_help(self) -> None:
        """Test query command help."""
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "query" in result.stdout.lower()
