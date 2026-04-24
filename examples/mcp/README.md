# Claude Desktop MCP Configuration

> **Quick reference.** For full documentation see the top-level `README.md`.

---

## 1. Local install (conda + stdio) — recommended

### Create the conda environment

```bash
cd /path/to/marker-mcp

# Create (or update) the conda env named "marker-mcp"
bash build_venv.sh

# Activate it
conda activate marker-mcp

# Verify the entry point is available
marker_mcp --help
```

The script uses [`environment.yml`](../../environment.yml) which installs PyTorch via the
`pytorch` and `nvidia` conda channels (reliable CUDA binary packages) and then pip-installs
the local sibling `../marker` repo in editable mode alongside `fastmcp` and `click`.

> **Tip:** Install [mamba](https://mamba.readthedocs.io) for a faster conda solver:
> `conda install -n base -c conda-forge mamba`

### Find the full path to `marker_mcp`

Claude Desktop needs the **absolute path** to the executable:

```bash
conda activate marker-mcp
which marker_mcp
# e.g. /home/youruser/anaconda3/envs/marker-mcp/bin/marker_mcp
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "marker": {
      "command": "/home/youruser/anaconda3/envs/marker-mcp/bin/marker_mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Replace the path with the output of `which marker_mcp` above, then restart Claude Desktop.

---

## 2. Docker (SSE)

No conda required — the Docker image installs everything via pip internally.

### Start the server

```bash
# GPU (default)
docker compose up marker-mcp-gpu

# CPU only
docker compose --profile cpu-only up marker-mcp-cpu
```

Models (~5 GB) are cached in a named Docker volume (`marker_models`) and survive rebuilds.

### Configure Claude Desktop

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

## 3. Local SSE (development / testing)

Useful for testing the server without Claude Desktop:

```bash
conda activate marker-mcp

# SSE transport
MARKER_MCP_TRANSPORT=sse bash start_mcp.sh

# Or directly
marker_mcp --transport sse --host 127.0.0.1 --port 8000
```

---

## Config file reference

See [`claude_desktop_config.json`](claude_desktop_config.json) for copy-paste snippets
for both the stdio (conda) and SSE (Docker) setups.
