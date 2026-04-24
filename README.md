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
| `CUDA_VISIBLE_DEVICES` | *(auto)* | Set to `""` to force CPU-only mode |
| `MARKER_LLM_SERVICE` | *(auto)* | Override LLM service class explicitly |
| `GOOGLE_API_KEY` | *(none)* | Gemini API key — auto-selects `GoogleGeminiService` |
| `OLLAMA_BASE_URL` | *(none)* | Ollama URL — auto-selects `OllamaService` |
| `OLLAMA_MODEL` | `llama3.2-vision` | Ollama model name |
| `OPENAI_API_KEY` | *(none)* | OpenAI key — auto-selects `OpenAIService` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `CLAUDE_API_KEY` | *(none)* | Anthropic key — auto-selects `ClaudeService` |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Claude model name |

---

## LLM-Enhanced Conversion

Marker can use a vision-capable LLM alongside its built-in ML models to significantly improve
conversion accuracy — better table reconstruction across pages, inline math, form field
extraction, and image-to-text descriptions.

Pass `use_llm=True` to any conversion tool to enable it:

```
convert_document(filepath="/path/to/report.pdf", use_llm=True)
```

The LLM service is configured entirely via **environment variables** — you do not need to
change the tool call. The service is auto-detected based on which key is present:

| Priority | Env var set | Service used |
|----------|-------------|--------------|
| 1 (explicit) | `MARKER_LLM_SERVICE` | whatever class you specify |
| 2 | `OLLAMA_BASE_URL` or `OLLAMA_MODEL` | Ollama (local) |
| 3 | `OPENAI_API_KEY` | OpenAI / compatible |
| 4 | `CLAUDE_API_KEY` | Anthropic Claude |
| 5 (default) | `GOOGLE_API_KEY` | Google Gemini |

---

### Ollama (local, privacy-first) — recommended for self-hosted use

Ollama runs entirely on your machine — no data leaves your network.

#### 1. Install Ollama and pull a vision model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a multimodal vision model (required for PDF image understanding)
ollama pull llama3.2-vision        # 7 B — good balance of speed and quality
ollama pull llava                  # alternative
ollama pull minicpm-v              # smaller, fast on CPU
```

> **A vision model is required.** Text-only models (llama3, mistral, etc.) will
> not work because Marker sends page images to the LLM.

#### 2. Configure the server

**Local (conda) — create a `.env` file:**

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision
```

Load it before starting the server:

```bash
conda activate marker-mcp
set -a && source .env && set +a
marker_mcp --transport stdio
```

Or use `start_mcp.sh` which sources `.env` automatically:

```bash
bash start_mcp.sh
```

**Docker — edit `docker-compose.yml` or pass env vars:**

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434 \
OLLAMA_MODEL=llama3.2-vision \
docker compose up marker-mcp-gpu
```

> `host.docker.internal` resolves to the host machine from inside Docker on Linux/macOS.
> On Linux you may need `--add-host=host.docker.internal:host-gateway` in `docker-compose.yml`.

#### 3. Use the tool

```
convert_document(filepath="/path/to/document.pdf", use_llm=True)
```

---

### Google Gemini (default)

Highest accuracy. Requires an API key from [Google AI Studio](https://aistudio.google.com/apikey)
(free tier available).

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key
```

No further configuration needed — Gemini is the default when `GOOGLE_API_KEY` is set.

---

### OpenAI / any OpenAI-compatible API

Useful with GPT-4o or a self-hosted compatible endpoint (vLLM, LM Studio, Ollama's OpenAI
compatibility layer, etc.).

```bash
# .env — OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o          # default: gpt-4o-mini

# .env — local vLLM or LM Studio
OPENAI_API_KEY=any-value     # many local servers accept any non-empty string
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=your-model-name
```

> **Tip:** To use Ollama's OpenAI-compatible endpoint instead of the native API:
> ```bash
> OPENAI_API_KEY=ollama
> OPENAI_BASE_URL=http://localhost:11434/v1
> OPENAI_MODEL=llama3.2-vision
> ```

---

### Anthropic Claude

```bash
# .env
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-3-5-sonnet-20241022   # default
```

---

### Explicit service override

If you need to override the auto-detection (e.g., multiple keys are set):

```bash
# .env
MARKER_LLM_SERVICE=marker.services.ollama.OllamaService
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava
```

Available service paths:

| Service | Import path |
|---------|-------------|
| Ollama | `marker.services.ollama.OllamaService` |
| Gemini | `marker.services.gemini.GoogleGeminiService` |
| OpenAI | `marker.services.openai.OpenAIService` |
| Claude | `marker.services.claude.ClaudeService` |
| Google Vertex | `marker.services.vertex.GoogleVertexService` |

---

### When to use `use_llm=True`

| Use case | Benefit |
|----------|---------|
| PDFs with complex tables spanning multiple pages | Table merging + cleanup |
| Documents with inline LaTeX / math | Accurate math rendering |
| Scanned forms | Form field extraction |
| PDFs with embedded images | Image-to-text descriptions |
| Low-quality scans | Combined with `force_ocr=True` |

> **Performance note:** LLM-enhanced conversion is significantly slower than standard
> conversion. Expect 30–120 seconds per page depending on the model and hardware.
> For bulk processing use `convert_documents_batch` which processes files sequentially
> without blocking the event loop.

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
