"""Async-safe document conversion service wrapping the Marker library.

Models are initialised at module load time so that subsequent tool calls are fast.
All blocking conversion calls are offloaded to a thread via asyncio.to_thread().

LLM service selection is driven entirely by environment variables so that the
MCP tool signatures stay simple (just use_llm=True) while the backend is
fully configurable. See README.md § LLM-Enhanced Conversion for details.
"""

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# ---------------------------------------------------------------------------
# LLM service configuration — resolved from environment variables.
#
# MARKER_LLM_SERVICE   import path of the service class (overrides all others)
#
# Ollama (local):
#   OLLAMA_BASE_URL    base URL            (default: http://localhost:11434)
#   OLLAMA_MODEL       vision model name   (default: llama3.2-vision)
#
# OpenAI / compatible:
#   OPENAI_API_KEY     API key
#   OPENAI_BASE_URL    base URL            (default: https://api.openai.com/v1)
#   OPENAI_MODEL       model name          (default: gpt-4o-mini)
#
# Gemini (default when GOOGLE_API_KEY is set):
#   GOOGLE_API_KEY     Gemini API key
#
# Claude:
#   CLAUDE_API_KEY     API key
#   CLAUDE_MODEL       model name          (default: claude-3-5-sonnet-20241022)
# ---------------------------------------------------------------------------

def _llm_options_from_env() -> dict:
    """Build the LLM-related options dict from environment variables."""
    opts: dict = {}

    explicit_service = os.environ.get("MARKER_LLM_SERVICE")
    if explicit_service:
        opts["llm_service"] = explicit_service

    # Ollama — detected by OLLAMA_BASE_URL or OLLAMA_MODEL being set
    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_model = os.environ.get("OLLAMA_MODEL")
    if (ollama_url or ollama_model) and not explicit_service:
        opts["llm_service"] = "marker.services.ollama.OllamaService"
    if ollama_url:
        opts["ollama_base_url"] = ollama_url
    if ollama_model:
        opts["ollama_model"] = ollama_model

    # OpenAI-compatible — detected by OPENAI_API_KEY being set
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_BASE_URL")
    openai_model = os.environ.get("OPENAI_MODEL")
    if openai_key and not explicit_service and "llm_service" not in opts:
        opts["llm_service"] = "marker.services.openai.OpenAIService"
    if openai_key:
        opts["openai_api_key"] = openai_key
    if openai_url:
        opts["openai_base_url"] = openai_url
    if openai_model:
        opts["openai_model"] = openai_model

    # Claude
    claude_key = os.environ.get("CLAUDE_API_KEY")
    claude_model = os.environ.get("CLAUDE_MODEL")
    if claude_key and not explicit_service and "llm_service" not in opts:
        opts["llm_service"] = "marker.services.claude.ClaudeService"
    if claude_key:
        opts["claude_api_key"] = claude_key
    if claude_model:
        opts["claude_model_name"] = claude_model

    # Gemini is the marker default — just needs GOOGLE_API_KEY in env, no extra opts needed.

    return opts


# ---------------------------------------------------------------------------
# Module-level model initialisation — fail fast with a clear error message.
# ---------------------------------------------------------------------------
_MODELS: Optional[dict] = None

try:
    _MODELS = create_model_dict()
    print("✅ Marker models loaded successfully.")
except Exception as _exc:
    print(f"❌ Failed to load Marker models: {_exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_converter(options: dict) -> PdfConverter:
    """Build a configured PdfConverter from an options dict."""
    config_parser = ConfigParser(options)
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    return PdfConverter(
        config=config_dict,
        artifact_dict=_MODELS,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def _convert_sync(filepath: str, options: dict) -> str:
    """Synchronous conversion — always called via asyncio.to_thread()."""
    if _MODELS is None:
        raise RuntimeError(
            "Marker models are not available. Check startup logs for initialisation errors."
        )
    converter = _build_converter(options)
    rendered = converter(filepath)
    text, _, _ = text_from_rendered(rendered)
    return str(text)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def convert_file(filepath: str, options: Optional[dict] = None) -> str:
    """Convert a document at the given file path and return the converted text."""
    opts = {"output_format": "markdown", **(options or {})}
    return await asyncio.to_thread(_convert_sync, filepath, opts)


async def convert_bytes(content: bytes, filename: str, options: Optional[dict] = None) -> str:
    """Convert a document from raw bytes.

    Writes the bytes to a temporary file, converts it, then cleans up.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return await convert_file(tmp_path, options)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def convert_bytes_batch(files: list[dict]) -> list[dict]:
    """Batch-convert multiple documents.

    Each entry must contain:
        - filename (str): original filename including extension
        - content_base64 (str): base64-encoded file bytes
        - options (dict, optional): per-file conversion options

    Returns a list of result dicts: {filename, success, content, error}.
    """
    results: list[dict] = []
    for file in files:
        filename = file.get("filename", "document.pdf")
        try:
            content = base64.b64decode(file.get("content_base64", ""))
            text = await convert_bytes(content, filename, file.get("options"))
            results.append({"filename": filename, "success": True, "content": text, "error": None})
        except Exception as exc:
            results.append({"filename": filename, "success": False, "content": None, "error": str(exc)})
    return results


def get_status() -> dict:
    """Return the converter readiness status (synchronous, safe to call from anywhere)."""
    return {
        "initialized": _MODELS is not None,
        "status": "ready" if _MODELS is not None else "failed",
        "message": (
            "Marker models loaded and ready."
            if _MODELS is not None
            else "Model initialisation failed — check startup logs."
        ),
    }
