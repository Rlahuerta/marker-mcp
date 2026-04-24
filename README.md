# Marker MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that exposes the
[Marker](https://github.com/datalab-to/marker) document conversion library as callable tools
for LLM clients such as Claude Desktop.

## Tools

| Tool | Description |
|------|-------------|
| `convert_document` | Convert a document at a local file path |
| `convert_document_from_content` | Convert a document from base64-encoded bytes |
| `convert_documents_batch` | Batch-convert multiple documents |
| `get_converter_status` | Check model initialisation status |

Supports PDF, DOCX, PPTX, XLSX, HTML, EPUB and image files.  
Output formats: `markdown`, `json`, `html`, `chunks`.

---

## Quick start — local (stdio)

### 1. Install

```bash
# From this repository
pip install -e .

# Or from PyPI (when published)
pip install marker-mcp
```

### 2. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "marker": {
      "command": "marker_mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Restart Claude Desktop — the Marker tools will appear in the tool picker.

> **Note:** Models (~5 GB) are downloaded on first run. Subsequent calls are fast.

---

## Quick start — Docker

### GPU (default)

```bash
docker compose up marker-mcp-gpu
```

### CPU only

```bash
docker compose --profile cpu-only up marker-mcp-cpu
```

The server will be available at `http://localhost:8000`.

Configure Claude Desktop:

```json
{
  "mcpServers": {
    "marker-docker": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

---

## Local SSE/HTTP (development)

```bash
# SSE transport (compatible with SSE-aware MCP clients)
marker_mcp --transport sse --host 0.0.0.0 --port 8000

# Streamable HTTP (fastmcp 2.x)
marker_mcp --transport http --host 0.0.0.0 --port 8000
```

Or use the provided script:

```bash
MARKER_MCP_TRANSPORT=sse bash start_mcp.sh
```

---

## Development setup

Uses **conda** for local development — conda handles CUDA/PyTorch binary packages
more reliably than pip/uv.

```bash
# Create the conda env "marker-mcp" (uses environment.yml)
bash build_venv.sh

conda activate marker-mcp
marker_mcp --help
```

The `environment.yml` installs the sibling `../marker` repo in editable mode by default.
Swap the commented lines to use the published `marker-pdf` package from PyPI instead.

> **Tip:** Install [mamba](https://mamba.readthedocs.io) for a much faster conda solver:
> `conda install -n base -c conda-forge mamba`

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MARKER_MCP_TRANSPORT` | `stdio` | Transport: `stdio`, `sse`, `http` |
| `MARKER_MCP_HOST` | `0.0.0.0` | Bind host (SSE/HTTP only) |
| `MARKER_MCP_PORT` | `8000` | Bind port (SSE/HTTP only) |
| `GOOGLE_API_KEY` | *(none)* | Required for `use_llm=True` with Gemini |
| `OPENAI_API_KEY` | *(none)* | Required for `use_llm=True` with OpenAI-compatible LLM |
| `CUDA_VISIBLE_DEVICES` | *(auto)* | Set to `""` to force CPU-only mode |

---

## Architecture

```
marker_mcp/
├── __init__.py
├── conversion_service.py   # async-safe conversion service; loads models at import time
└── mcp_server.py           # FastMCP server + 4 tools + CLI entry point
```

- **`conversion_service.py`**: Calls `create_model_dict()` at module load time.
  All blocking Marker calls are wrapped in `asyncio.to_thread()` so the event loop is never blocked.
- **`mcp_server.py`**: Defines tools with `@mcp.tool` (fastmcp) and a Click CLI for transport selection.

---

## License

MIT
