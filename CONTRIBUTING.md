# Contributing Guide

Thank you for your interest in contributing to RAG Financial Multimodal!

## Quick Start for Contributors

```bash
git clone https://github.com/your-org/rag-financial-multimodal
cd rag-financial-multimodal
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pip install pytest pytest-asyncio pytest-cov pytest-mock black ruff mypy
cp .env.example .env   # set OPENAI_API_KEY
```

## Development Workflow

1. **Create a branch** from `develop` (not `main`):
   ```bash
   git checkout develop && git pull
   git checkout -b feat/your-feature-name
   ```

2. **Write code** following the standards below.

3. **Run the full check suite** before pushing:
   ```bash
   ruff check src/ tests/          # lint
   black --check src/ tests/       # formatting
   mypy src/rag_system/config.py   # types
   pytest tests/unit/ -v           # unit tests
   pytest tests/integration/ -v    # integration tests
   ```

4. **Open a PR** against `develop`. PRs require:
   - All CI checks green
   - At least one approving review
   - No regression on eval gate (faithfulness drop < 5%)

## Architecture Principles

- **Dependency injection, not global state.** Every component is passed in; never import and instantiate inside business logic.
- **Async-first.** All I/O (HTTP calls, file reads, vector store ops) must be `async def`. Use `asyncio.to_thread()` for blocking libs.
- **Config from `get_config()`.** Never hardcode model names, URLs, or keys.
- **Every new component implements its ABC.** New parsers inherit `BaseParser`, new vector stores inherit `BaseVectorStore`, etc.
- **Tests for every new component.** Minimum: happy path, error path, edge case (empty input). Aim for 80%+ branch coverage on new code.

## Adding a New Component

### Example: Adding a new vector store backend

1. **Implement the ABC** in `src/rag_system/components/vector_store/__init__.py`:
   ```python
   class MyVectorStore(BaseVectorStore):
       @property
       def name(self) -> str: return "my_store"
       async def initialize(self, ...): ...
       async def upsert(self, ...): ...
       async def search(self, ...): ...
       async def delete(self, ...): ...
   ```

2. **Register in the factory:**
   ```python
   def build_vector_store(provider=None):
       if name == "my_store": return MyVectorStore()
       ...
   ```

3. **Add config option** to `VectorStoreConfig.provider` in `config.py`.

4. **Write unit tests** in `tests/unit/test_vector_store.py`.

5. **Update `README.md`** architecture table and `.env.example`.

## Code Style

- **Formatter:** `black` (line length 100)
- **Linter:** `ruff` (see `[tool.ruff]` in `pyproject.toml`)
- **Types:** All public functions must have type annotations
- **Docstrings:** Google style for all public classes and methods
- **Imports:** `isort` with `profile = "black"`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(retriever): add Qdrant vector store adapter
fix(guardrails): handle empty context list in numeric check
docs(readme): update deployment section for Helm v3
test(pot): add timeout edge case
refactor(pipeline): extract _redact_elements to separate method
```

## Adding to the Golden Dataset

High-quality eval samples are critical. To add new ones:

1. Find a publicly available financial filing (10-K/10-Q/earnings release).
2. Identify a question with a verifiable numerical answer.
3. Add to `evals/golden_datasets/financial_qa.jsonl` as:
   ```json
   {
     "question": "...",
     "ground_truth": "...",
     "source_documents": ["filename.pdf"],
     "expected_page": 12,
     "expected_numeric_values": ["$23.35B", "9%"],
     "tags": ["revenue", "yoy", "quarterly"]
   }
   ```
4. Run `rag-financial evaluate` to confirm the new sample doesn't regress metrics.

## Reporting Issues

- **Bugs:** Open a GitHub issue with the `bug` label. Include: Python version, OS, error traceback, minimal repro steps.
- **Security vulnerabilities:** See [SECURITY.md](SECURITY.md) — do **not** open a public issue.
- **Feature requests:** Open a GitHub issue with the `enhancement` label and link to the roadmap item if applicable.
