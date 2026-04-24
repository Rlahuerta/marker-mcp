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
# activate conda root env (if isn't already active)
source ~/anaconda3/bin/activate root

# Create the conda env "marker-mcp" (uses environment.yml)
bash build_venv.sh

# (or) Create env from curated spec (first time)
conda env create -f environment.yml

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
| `OLLAMA_MODEL` | `llama3.2-vision` | Ollama model name (use `gemma4:31b` for best quality) |
| `OLLAMA_NO_CLOUD` | *(unset)* | Set to `1` to disable Ollama Cloud routing |
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

Ollama runs entirely on your machine — no data leaves your network. It also offers
**Ollama Cloud** to offload inference to ollama.com servers when local hardware is limited.

> **A vision/multimodal model is required.** Marker sends page images (base64) to the LLM
> for table reconstruction, math, and form extraction. Text-only models will not work.

#### Recommended models

| Model | Size | VRAM (Q4) | Vision | Notes |
|-------|------|-----------|--------|-------|
| `gemma4:31b` | ~62 GB | ~16 GB | ✅ | **Best quality** — Google's latest multimodal, April 2025 |
| `llama3.2-vision` | ~7 B | ~6 GB | ✅ | Default — good balance of speed and quality |
| `llava` | ~7 B | ~6 GB | ✅ | Established alternative |
| `minicpm-v` | ~4 B | ~4 GB | ✅ | Smallest option, fast on CPU |

**Gemma4** (`gemma4:31b`) is Google DeepMind's newest generation model (released April 2025).
It supports variable aspect ratios, configurable visual token budgets, and delivers
significantly higher accuracy than llama3.2-vision — especially on complex tables, math,
and multi-column layouts. It is fully API-compatible with Marker's `OllamaService` with
no code changes required.

VRAM requirements for `gemma4:31b` by quantization:

| Quantization | VRAM | Typical GPU |
|---|---|---|
| FP16 | ~31 GB | A100, H100 |
| Q8_0 | ~30 GB | A100 (40 GB) |
| Q5_0 | ~18 GB | RTX 6000 Ada |
| **Q4_0** | **~16 GB** | **RTX 4090 / 3090** ← recommended |
| Q3_K_M | ~12 GB | RTX 3080 Ti |
| Q2_K | ~8 GB | RTX 3070 |

#### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### 2. Pull a vision model

```bash
# High quality (needs ~16 GB VRAM with Q4 quantization)
ollama pull gemma4:31b

# Lighter option (~6 GB VRAM)
ollama pull llama3.2-vision
```

#### 3a. Local Ollama (self-hosted)

Create a `.env` file in the project root:

```bash
# .env — local Ollama with gemma4
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b
```

Load it before starting the server:

```bash
conda activate marker-mcp
bash start_mcp.sh          # sources .env automatically
# or: set -a && source .env && set +a && marker_mcp --transport stdio
```

**Docker** — the container connects to the host Ollama instance:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434 \
OLLAMA_MODEL=gemma4:31b \
docker compose up marker-mcp-gpu
```

> On Linux you may need to add `extra_hosts: ["host.docker.internal:host-gateway"]`
> to the service in `docker-compose.yml`.

#### 3b. Ollama Cloud (no local GPU required)

Ollama Cloud routes inference to ollama.com servers — useful when local VRAM is
insufficient to run `gemma4:31b`. Requires an [ollama.com](https://ollama.com) account.

```bash
# Sign in once
ollama signin

# Cloud model names have a -cloud suffix
ollama pull gemma4:31b-cloud     # routes to ollama.com
```

Configure the server to use the cloud model:

```bash
# .env — Ollama Cloud
OLLAMA_BASE_URL=http://localhost:11434   # still routes through local Ollama daemon
OLLAMA_MODEL=gemma4:31b-cloud
```

> **Privacy note:** With Ollama Cloud, document content is sent to ollama.com servers.
> Use local Ollama when working with sensitive documents.
>
> To disable cloud routing entirely: `export OLLAMA_NO_CLOUD=1`

#### 4. Use the tool

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

## Setup with AI coding tools

### GitHub Copilot (VS Code)

GitHub Copilot supports MCP servers through a `mcp.json` file in your workspace or
user profile. [Full reference](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration).

#### Local (stdio via conda)

First, find the absolute path to the `marker_mcp` binary in your conda env:

```bash
conda activate marker-mcp
which marker_mcp
# Example: /home/user/.conda/envs/marker-mcp/bin/marker_mcp
```

Create or edit `.vscode/mcp.json` in your workspace (or run **MCP: Open User Configuration**
from the VS Code Command Palette for a global config):

```json
{
  "servers": {
    "marker": {
      "type": "stdio",
      "command": "/home/user/.conda/envs/marker-mcp/bin/marker_mcp",
      "envFile": "${workspaceFolder}/.env"
    }
  }
}
```

> **Tip:** `envFile` automatically loads your `.env` (Ollama, OpenAI, etc.) so you don't
> need to duplicate env vars in the config.

#### Docker (SSE)

Start the container first (`docker compose up marker-mcp-gpu`), then add to `mcp.json`:

```json
{
  "servers": {
    "marker": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

---

### opencode

[opencode](https://opencode.ai) reads MCP server config from `~/.config/opencode/config.json`
(global) or `opencode.json` in your project root.
[Full reference](https://opencode.ai/docs/mcp-servers).

#### Local (stdio via conda)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "marker": {
      "type": "local",
      "command": ["/home/user/.conda/envs/marker-mcp/bin/marker_mcp"],
      "enabled": true,
      "environment": {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "gemma4:31b-cloud"
      }
    }
  }
}
```

Replace the environment values (or the entire `environment` block) with your preferred LLM
service variables from the [Environment variables](#environment-variables) table.

#### Docker (remote)

Start the container first, then:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "marker": {
      "type": "remote",
      "url": "http://localhost:8000/sse",
      "enabled": true
    }
  }
}
```

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
