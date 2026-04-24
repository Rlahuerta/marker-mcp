# Claude Desktop MCP Configuration

Paste the relevant block from `claude_desktop_config.json` into your Claude Desktop
`claude_desktop_config.json` (usually `~/Library/Application Support/Claude/claude_desktop_config.json`).

## Local (stdio)

```json
"marker-local-stdio": {
  "command": "marker_mcp",
  "args": ["--transport", "stdio"]
}
```

Requires `marker_mcp` to be on your PATH:
```bash
pip install -e /path/to/marker-mcp
```

## Docker (SSE)

Start the server first:
```bash
# GPU
docker compose up marker-mcp-gpu

# CPU only
docker compose --profile cpu-only up marker-mcp-cpu
```

Then add to Claude Desktop:
```json
"marker-docker-sse": {
  "url": "http://localhost:8000/sse"
}
```
