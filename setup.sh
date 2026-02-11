#!/usr/bin/env bash
set -euo pipefail

# Minimal setup script for development environment
# Installs system deps (poppler, tesseract) and a Python virtualenv,
# then installs Python packages from requirements.txt

if [ "$(id -u)" -ne 0 ]; then
  echo "Some system packages may require sudo. Re-run with sudo if needed."
fi

echo "Installing system packages (poppler-utils, tesseract-ocr)..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y poppler-utils tesseract-ocr
else
  echo "apt-get not found. Please install poppler-utils and tesseract-ocr manually."
fi

PYENV_DIR=".venv"
PYTHON_BIN="python3"

echo "Creating virtual environment in ${PYENV_DIR} (using ${PYTHON_BIN})..."
${PYTHON_BIN} -m venv "${PYENV_DIR}"
echo "Activating virtualenv and upgrading pip..."
# shellcheck disable=SC1091
source "${PYENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "Installing Python packages from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Please create or provide the file."
fi

echo "Setup complete. Activate the virtualenv with: source ${PYENV_DIR}/bin/activate"

exit 0
