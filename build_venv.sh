#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_venv.sh — create a local uv virtual environment for development
# ---------------------------------------------------------------------------
set -euo pipefail

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "📦 Creating virtual environment..."
uv venv .venv

echo "📦 Installing dependencies (local marker-pdf sibling)..."
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

echo ""
echo "✅ Done! Activate with: source .venv/bin/activate"
echo "   Then run: marker_mcp --help"
