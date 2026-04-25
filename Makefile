# ─────────────────────────────────────────────────────────────────────────────
# RAG Financial Multimodal — Makefile
# Targets: setup, dev, test, lint, format, docs, ingest-sample, query, serve
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
PYTHON        := python3.11
VENV          := .venv
BIN           := $(VENV)/bin
SRC           := src/rag_system
TESTS         := tests

.PHONY: help setup dev install test test-unit test-integration test-load \
        lint format typecheck security eval serve docs clean docker-up \
        docker-down ingest-sample query

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  RAG Financial Multimodal — Available Targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup          Install deps + create .env from .env.example"
	@echo "  make dev            Start full stack with Docker (API + Redis + observability)"
	@echo "  make install        Install Python deps into virtualenv"
	@echo "  make test           Run all tests with coverage"
	@echo "  make test-unit      Run unit tests only (fast)"
	@echo "  make test-integration  Run integration tests"
	@echo "  make lint           Ruff lint check"
	@echo "  make format         Black + isort auto-format"
	@echo "  make typecheck      mypy type checking"
	@echo "  make security       Trivy + pip-audit security scan"
	@echo "  make eval           Run quality evaluation against golden dataset"
	@echo "  make serve          Start FastAPI server locally"
	@echo "  make docs           Build and serve MkDocs documentation"
	@echo "  make docker-up      docker compose up -d (API + Redis)"
	@echo "  make docker-down    docker compose down"
	@echo "  make ingest-sample  Ingest sample PDF (downloads Tesla 10-Q)"
	@echo "  make query Q='...'  Query the pipeline (e.g. make query Q='What was revenue?')"
	@echo "  make clean          Remove build artifacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "▸ Running setup script..."
	bash scripts/setup.sh

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"
	@echo "✅ Installed. Activate: source $(VENV)/bin/activate"

# ── Development ───────────────────────────────────────────────────────────────
dev:
	docker compose --profile observability up -d
	@echo "✅ Stack started:"
	@echo "   API:       http://localhost:8000/docs"
	@echo "   Grafana:   http://localhost:3000"
	@echo "   Jaeger:    http://localhost:16686"
	@echo "   Prometheus:http://localhost:9090"

docker-up:
	docker compose up -d
	@echo "✅ API running at http://localhost:8000"

docker-down:
	docker compose down

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(BIN)/pytest $(TESTS)/ -v \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=70 \
		-p no:warnings

test-unit:
	$(BIN)/pytest $(TESTS)/unit/ $(TESTS)/test_*.py -v --tb=short -p no:warnings

test-integration:
	$(BIN)/pytest $(TESTS)/integration/ -v --tb=short -p no:warnings

test-load:
	@echo "▸ Starting load test (requires running API at http://localhost:8000)"
	$(BIN)/locust -f $(TESTS)/load/locustfile.py \
		--host=http://localhost:8000 \
		--users 50 --spawn-rate 5 --run-time 60s --headless

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(BIN)/ruff check $(SRC)/ $(TESTS)/ evals/
	@echo "✅ Lint passed"

format:
	$(BIN)/black $(SRC)/ $(TESTS)/ evals/ --line-length=100
	$(BIN)/isort $(SRC)/ $(TESTS)/ evals/ --profile=black
	@echo "✅ Formatted"

format-check:
	$(BIN)/black $(SRC)/ $(TESTS)/ --check --line-length=100
	$(BIN)/isort $(SRC)/ $(TESTS)/ --check-only --profile=black

typecheck:
	$(BIN)/mypy $(SRC)/config.py $(SRC)/components/base.py --ignore-missing-imports
	@echo "✅ Type check passed"

security:
	@echo "▸ Running pip-audit..."
	$(BIN)/pip-audit --requirement requirements.txt || true
	@echo "▸ Running Trivy filesystem scan..."
	trivy fs . --exit-code 0 --severity HIGH,CRITICAL || true
	@echo "✅ Security scan complete"

pre-commit-install:
	$(BIN)/pre-commit install
	@echo "✅ Pre-commit hooks installed"

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	$(BIN)/python -m evals.run_evals \
		--dataset evals/golden_datasets/financial_qa.jsonl \
		--fail-on-regression \
		--output evals/ci_report.json
	@echo "✅ Eval complete — see evals/ci_report.json"

seed-eval:
	$(BIN)/python scripts/seed_golden_dataset.py
	@echo "✅ Golden dataset seeded"

# ── API Server ────────────────────────────────────────────────────────────────
serve:
	$(BIN)/uvicorn src.rag_system.api.app:create_app \
		--factory --host 0.0.0.0 --port 8000 --reload
	
# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest-sample:
	@echo "▸ Downloading Tesla Q3 2023 investor update..."
	curl -sL "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf" \
		-o /tmp/tesla_q3_2023.pdf || echo "⚠ Download failed — place a PDF at /tmp/tesla_q3_2023.pdf"
	$(BIN)/rag-financial ingest /tmp/tesla_q3_2023.pdf --tenant demo
	@echo "✅ Sample document ingested"

query:
	@[ "$(Q)" ] || ( echo "Usage: make query Q='Your question here'"; exit 1 )
	$(BIN)/rag-financial query "$(Q)" --tenant demo --show-sources

# ── Documentation ─────────────────────────────────────────────────────────────
docs:
	$(BIN)/mkdocs serve --dev-addr=0.0.0.0:8080
	@echo "Docs at http://localhost:8080"

docs-build:
	$(BIN)/mkdocs build --site-dir site/
	@echo "✅ Docs built at site/"

# ── Utilities ─────────────────────────────────────────────────────────────────
generate-api-key:
	$(BIN)/python scripts/generate_api_key.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov coverage.xml site/ dist/ build/ *.egg-info/
	@echo "✅ Cleaned"

# ── CI helpers ────────────────────────────────────────────────────────────────
ci: format-check lint typecheck test
	@echo "✅ All CI checks passed"
