#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_mcp.sh — start the Marker MCP server (local stdio or SSE)
# ---------------------------------------------------------------------------
set -euo pipefail

TRANSPORT="${MARKER_MCP_TRANSPORT:-stdio}"
HOST="${MARKER_MCP_HOST:-0.0.0.0}"
PORT="${MARKER_MCP_PORT:-8000}"

# Source a local .env file when present
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a && source .env && set +a
fi

# Detect GPU availability
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "🖥  GPU detected — using CUDA"
else
    echo "💻 No GPU detected — using CPU"
    export CUDA_VISIBLE_DEVICES=""
fi

case "$TRANSPORT" in
    stdio)
        echo "🚀 Starting Marker MCP server (stdio transport)"
        exec marker_mcp --transport stdio
        ;;
    sse)
        echo "🚀 Starting Marker MCP server (SSE, ${HOST}:${PORT})"
        exec marker_mcp --transport sse --host "$HOST" --port "$PORT"
        ;;
    http)
        echo "🚀 Starting Marker MCP server (HTTP, ${HOST}:${PORT})"
        exec marker_mcp --transport http --host "$HOST" --port "$PORT"
        ;;
    *)
        echo "❌ Unknown transport: ${TRANSPORT}. Use stdio, sse, or http."
        exit 1
        ;;
esac
