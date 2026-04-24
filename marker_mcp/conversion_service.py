"""Async-safe document conversion service wrapping the Marker library.

Models are initialised at module load time so that subsequent tool calls are fast.
All blocking conversion calls are offloaded to a thread via asyncio.to_thread().
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
