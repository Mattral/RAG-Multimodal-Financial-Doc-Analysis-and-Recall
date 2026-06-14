"""CLI tests — upgraded for v2.0 enterprise CLI."""
import pytest
from typer.testing import CliRunner
from src.rag_system.cli import app

runner = CliRunner()


class TestCLIHelp:
    """Top-level help and version flags."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower()
        assert "query" in result.output.lower()

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "2.0.0" in result.output

    def test_ingest_help(self):
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--tenant" in result.output

    def test_query_help(self):
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "--top-k" in result.output

    def test_evaluate_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_health_help(self):
        result = runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0


class TestCLIIngest:
    """Ingest command validation."""

    def test_ingest_missing_file_exits_nonzero(self, tmp_path):
        nonexistent = str(tmp_path / "no_such_file.pdf")
        result = runner.invoke(app, ["ingest", nonexistent])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Files not found" in result.output

    def test_ingest_no_files_shows_error(self):
        result = runner.invoke(app, ["ingest"])
        # Missing required argument should fail
        assert result.exit_code != 0


class TestCLIQuery:
    """Query command validation."""

    def test_query_requires_question(self):
        result = runner.invoke(app, ["query"])
        assert result.exit_code != 0

    def test_query_json_flag_exists(self):
        result = runner.invoke(app, ["query", "--help"])
        assert "--json" in result.output

    def test_query_show_sources_flag_exists(self):
        result = runner.invoke(app, ["query", "--help"])
        assert "--show-sources" in result.output
