# Production-Grade Multimodal Financial RAG – Refactoring Roadmap

**Status:** In Progress  
**Target:** Achieve production-grade engineering standards compatible with OpenAI, Anthropic, and Google DeepMind architectures.

---

## Executive Summary

This repository will be refactored from an educational proof-of-concept into a high-reliability, mission-critical RAG system. The refactoring addresses four critical architectural gaps identified in `feedback.md`.

---

## Phase 1: Foundation Upgrade (Current Phase)

### ✅ Completed
- [x] Python 3.11 virtualenv setup (`.venv311`)
- [x] Dependency resolution and installation
- [x] Test suite passing (13 tests, 0 failures)
- [x] Config management refactoring (Pydantic v2 compatibility)

### 🔄 In Progress
- [ ] `.gitignore` setup (standard Python + ML + IDE artifacts)
- [ ] `roadmap.md` creation (tracking progress)

### ⏳ To Do
- [ ] Type safety audit (`mypy --strict`)
- [ ] Documentation standardization

---

## Phase 2: Component Refactoring

### 2.1 Program-of-Thought (PoT) Execution Engine
**File:** `src/rag_system/components/pot_executor.py`  
**Status:** Not Started

**Goals:**
- [ ] Implement secure Python sandbox for financial calculations
- [ ] Extract and parse LLM-generated code blocks (markdown syntax)
- [ ] Execute isolated calculations with timeout protection
- [ ] Return grounded numerical results with error handling
- [ ] Support financial templates: Percentage Change, CAGR, ROI

**Pattern Support:**
```
Percentage Change = (V_new - V_old) / V_old × 100
CAGR = (V_final / V_initial)^(1/n) - 1
ROI = (Gain - Cost) / Cost × 100
```

**Dependencies:**
- `structlog` for logging
- `pydantic` for result model
- Custom timeout management

**Compliance:**
- 100% type hints (mypy --strict)
- Async-ready architecture
- Full docstring coverage

---

### 2.2 Layout-Aware Semantic Parser
**File:** `src/rag_system/components/layout_parser.py`  
**Status:** Not Started

**Goals:**
- [ ] Preserve spatial coordinates and layout hierarchies from PDFs
- [ ] Group related elements (tables, captions, adjacent paragraphs)
- [ ] Generate HTML markdown wrappers for structured content
- [ ] Maintain visual context during chunking
- [ ] Prevent fragmentation of multi-page tables and figures

**Architecture:**
```
Raw PDF
  ├─ Extract layout metadata (coordinates, bounding boxes)
  ├─ Identify logical sections (tables, charts, narratives)
  ├─ Group spatially adjacent elements
  └─ Generate unified HTML markdown chunks
```

**Inputs:**
- Document elements from `PDFParser`
- Layout metadata (coordinates, fonts, styles)

**Outputs:**
- Layout-preserving chunks with semantic grouping
- HTML markdown with CSS class annotations

**Compliance:**
- 100% type hints
- Async-compatible for batch processing
- Retry logic for large documents

---

### 2.3 CLI Interface (Typer-based)
**File:** `src/rag_system/cli.py`  
**Status:** Not Started

**Goals:**
- [ ] Replace notebook-based execution with CLI commands
- [ ] Implement `ingest` command for document processing
- [ ] Implement `query` command for retrieval + response
- [ ] Real-time progress reporting and logging
- [ ] Support multi-turn conversation mode

**Commands:**

#### Ingest
```bash
python -m src.rag_system.cli ingest /path/to/document.pdf
```
- Parse document asynchronously
- Extract charts and images
- Generate visual descriptions (GPT-4V)
- Index into vector store
- Output processing summary

#### Query
```bash
python -m src.rag_system.cli query "What is the CAGR from 2020-2024?"
```
- Retrieve relevant context
- Route to PoT executor if numerical
- Generate grounded response
- Display with citations

**Compliance:**
- 100% type hints
- Comprehensive error handling
- Structured logging via `structlog`

---

## Phase 3: Evaluation & Monitoring

### 3.1 CI/CD Evaluation Harness
**File:** `tests/test_rag_pipeline.py`  
**Status:** Not Started

**Goals:**
- [ ] Integrate DeepEval or Ragas
- [ ] Define Faithfulness metric (groundedness)
- [ ] Define Context Recall metric
- [ ] Define Answer Relevancy metric
- [ ] Output metrics to `results/eval_report.json`
- [ ] Block deployments on metric regression

**Metrics:**

| Metric | Formula | Threshold |
|--------|---------|-----------|
| **Faithfulness** | \|C_grounded\| / \|C_total\| | ≥ 0.90 |
| **Context Recall** | \|S_retrieved ∩ S_relevant\| / \|S_relevant\| | ≥ 0.85 |
| **Answer Relevancy** | Cosine(query_embedding, answer_embedding) | ≥ 0.80 |

**Compliance:**
- Integration with pytest
- Automated CI/CD blocking on failures
- Human-readable HTML reports

---

### 3.2 Multimodal VLM Integration (Future)
**Status:** Not Started (Phase 4)

**Goals:**
- [ ] Replace GPT-4V text descriptions with native VLM embeddings
- [ ] Integrate ColPali or ColQwen for direct image-to-vector indexing
- [ ] Preserve visual features (layouts, fonts, chart lines) natively
- [ ] Improve retrieval speed and accuracy

---

## Phase 4: Testing & Documentation

### 4.1 Type Safety Audit
**File:** `.mypy.ini` (new), all component files  
**Status:** Not Started

**Goals:**
- [ ] Configure mypy for strict mode
- [ ] Audit existing code for type compliance
- [ ] Add comprehensive type hints
- [ ] Document type patterns for contributors

---

### 4.2 Documentation
**Files:** `docs/architecture.md`, `docs/api.md`  
**Status:** Not Started

**Goals:**
- [ ] High-level architecture overview with diagrams
- [ ] API reference for all public modules
- [ ] Developer setup guide
- [ ] Contributing guidelines

---

## Artifact Management

### .gitignore (See dedicated section below)
Covers:
- Python artifacts (`__pycache__`, `.pyc`, `.egg-info`)
- Virtual environments (`.venv*`, `venv`)
- IDE configurations (`.vscode`, `.idea`, `*.swp`)
- ML artifacts (`*.pt`, `*.pth`, `*.pkl`)
- Temporary files (`*.log`, `.DS_Store`)
- Results and cache (`htmlcov`, `.pytest_cache`, `results/`)

---

## Timeline & Milestones

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Phase 1 Foundation | ✅ Completed | Done |
| Phase 2a PoT Executor | Next 2-3 days | Not Started |
| Phase 2b Layout Parser | Next 4-5 days | Not Started |
| Phase 2c CLI Interface | Next 3-4 days | Not Started |
| Phase 3a Evaluation Harness | Next 2-3 days | Not Started |
| Phase 4 Type Safety + Docs | Next 5-7 days | Not Started |
| **Full Production Ready** | **~3-4 weeks** | In Progress |

---

## Success Criteria

- [ ] All tests pass (pytest with coverage ≥ 80%)
- [ ] Type safety audit complete (`mypy --strict` passes)
- [ ] CLI interface fully functional
- [ ] PoT executor handles all financial templates
- [ ] Layout parser preserves structural context
- [ ] Evaluation metrics meet thresholds
- [ ] Documentation complete
- [ ] Code review and approval

---

## Notes & Decisions

1. **Pydantic v2 Migration**: Config system now uses Pydantic v2 with relaxed required fields for testing.
2. **Async-First Architecture**: All I/O operations designed for async/await execution.
3. **Rate Limiting**: Existing token-bucket rate limiter integrated into all API calls.
4. **Retry Policy**: Exponential backoff applied to external services.
5. **Structured Logging**: All components use `structlog` for JSON-formatted logs.

---

## References

- `feedback.md` – Detailed architectural gap analysis
- `src/rag_system/config.py` – Configuration system (Pydantic v2)
- `tests/` – Test suite baseline
- `pyproject.toml` – Project metadata and build configuration
