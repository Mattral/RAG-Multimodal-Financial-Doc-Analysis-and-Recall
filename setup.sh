#!/usr/bin/env bash
# Delegate to scripts/setup.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/setup.sh" "$@"
