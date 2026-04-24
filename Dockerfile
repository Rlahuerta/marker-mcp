# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Marker MCP Server — single CUDA-capable image
#
# CPU-only: docker compose --profile cpu-only up
# GPU:      docker compose up
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -Lsf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
COPY pyproject.toml ./

# Install marker-pdf[full] from PyPI (not the local sibling — no uv.sources in Docker)
RUN pip install --no-cache-dir marker-pdf[full] fastmcp click

# Copy application source and install package in no-deps mode
COPY marker_mcp/ ./marker_mcp/

# Install only this package, all deps already present
RUN pip install --no-cache-dir --no-deps .

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/root/.cache/torch \
    HF_HOME=/root/.cache/huggingface \
    MARKER_MCP_HOST=0.0.0.0 \
    MARKER_MCP_PORT=8000

EXPOSE 8000

ENTRYPOINT ["marker_mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
