#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RAG Financial Multimodal — One-shot setup script
# Supports: Ubuntu 22.04/24.04, Debian 12, macOS 13+
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
PYTHON=${PYTHON:-python3.11}

echo "╔══════════════════════════════════════════════════════╗"
echo "║   RAG Financial Multimodal  v2.0  Setup              ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 1. Detect OS and install system deps ─────────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "▸ Installing Linux system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr tesseract-ocr-eng \
        libmagic1 curl git gcc g++ python3.11 python3.11-venv \
        2>/dev/null || echo "  (some packages may already be installed)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "▸ Installing macOS system dependencies (Homebrew)..."
    which brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }
    brew install poppler tesseract || true
fi

# ── 2-3. Python environment and dependencies (from uv.lock) ──────────────────
echo "▸ Installing Python dependencies from uv.lock..."
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
uv sync --frozen --all-extras --python 3.11
source .venv/bin/activate

# ── 4. Environment file ───────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "▸ Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠  ACTION REQUIRED: Edit .env and set your OPENAI_API_KEY"
    echo "   nano .env"
else
    echo "▸ .env already exists — skipping"
fi

# ── 5. Create data directories ────────────────────────────────────────────────
mkdir -p data/vectorstore audit_logs logs

# ── 6. Verify installation ────────────────────────────────────────────────────
echo ""
echo "▸ Verifying installation..."
python -c "from filing_qa.config import get_config; cfg = get_config(); print(f'  Config OK — environment: {cfg.environment}')"
python -c "from filing_qa.components.base import DocumentElement; print('  Components OK')"
python -c "from filing_qa.utils.exceptions import RAGException; print('  Exceptions OK')"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  Setup complete!                                  ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Activate:  source .venv/bin/activate                ║"
echo "║  Configure: nano .env  (set OPENAI_API_KEY)          ║"
echo "║  Ingest:    filing-qa ingest your_doc.pdf        ║"
echo "║  Query:     filing-qa query 'What was revenue?'  ║"
echo "║  Serve:     filing-qa serve                      ║"
echo "║  Test:      pytest tests/unit/ -v                    ║"
echo "╚══════════════════════════════════════════════════════╝"
