#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_venv.sh — create the conda environment for local development
#
# Creates a conda env named "marker-mcp" using environment.yml.
# Conda handles CUDA/PyTorch binary packages much more reliably than pip.
#
# Requires: conda (Anaconda or Miniconda)
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="marker-mcp"
CONDA_CMD="conda"

# Prefer mamba when available (much faster solver)
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
    echo "🐍 Using mamba for faster dependency resolution"
fi

if ! command -v conda &>/dev/null; then
    echo "❌ conda not found. Install Anaconda or Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if env already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Conda env '${ENV_NAME}' already exists. Updating..."
    ${CONDA_CMD} env update -n "${ENV_NAME}" -f environment.yml --prune
else
    echo "📦 Creating conda env '${ENV_NAME}'..."
    ${CONDA_CMD} env create -n "${ENV_NAME}" -f environment.yml
fi

echo ""
echo "✅ Done!"
echo "   Activate with: conda activate ${ENV_NAME}"
echo "   Then run:      marker_mcp --help"
