"""FastMCP server exposing Marker document conversion as MCP tools.

Transports:
  stdio  — for local use with Claude Desktop (default)
  sse    — HTTP + Server-Sent Events for networked / Docker deployments
  http   — Streamable HTTP (fastmcp 2.x)

Usage:
  marker_mcp                                              # stdio
  marker_mcp --transport sse --host 0.0.0.0 --port 8000  # SSE
"""

from __future__ import annotations

import base64 as _b64
import importlib
import os
import sys

import click
from fastmcp import FastMCP
import marker_mcp.conversion_service as svc

mcp = FastMCP(name="Marker Document Conversion Service")


def _conversion_service():
    return svc


def _configure_ocr_device(ocr_device: str | None) -> None:
    """Apply the OCR device selector before the conversion service is used."""
    _configure_runtime_overrides(ocr_device=ocr_device)


def _configure_runtime_overrides(
    ocr_device: str | None = None,
    model_dtype: str | None = None,
) -> None:
    """Apply startup-only conversion-service overrides and reload lazily."""
    should_reload = False

    if ocr_device is not None:
        os.environ["MARKER_MCP_OCR_DEVICE"] = ocr_device
        should_reload = True

    if model_dtype is not None:
        os.environ["MARKER_MCP_MODEL_DTYPE"] = model_dtype
        should_reload = True

    if should_reload and "marker_mcp.conversion_service" in sys.modules:
        importlib.reload(sys.modules["marker_mcp.conversion_service"])


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool
async def convert_document(
    filepath: str,
    output_format: str = "markdown",
    page_range: str | None = None,
    max_pages_per_chunk: int | None = None,
    max_page_height_px: int | None = None,
    gpu_memory_profile: str | None = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    use_llm: bool = False,
) -> str:
    """Convert a document at a local file path.

    Supports PDF, DOCX, PPTX, XLSX, HTML, EPUB and image files.
    Returns the converted text in the requested format.

    Args:
        filepath: Absolute or relative path to the document file.
        output_format: Output format — "markdown" (default), "json", "html", or "chunks".
        page_range: Page range to convert, e.g. "0,5-10,20". Null converts all pages.
        max_pages_per_chunk: Optional PDF chunk size for sequential page-range batching.
        max_page_height_px: Experimental PDF page-strip height for rasterized per-page tiling.
        gpu_memory_profile: Optional GPU memory tuning profile, e.g. "low-vram".
        force_ocr: Force OCR on all pages even if a text layer exists.
        paginate_output: Separate pages with horizontal rules containing page numbers.
        use_llm: Use an LLM for higher accuracy (requires GOOGLE_API_KEY or compatible service).
    """
    options = _build_options(
        output_format,
        page_range,
        force_ocr,
        paginate_output,
        use_llm,
        max_pages_per_chunk=max_pages_per_chunk,
        max_page_height_px=max_page_height_px,
        gpu_memory_profile=gpu_memory_profile,
    )
    return await _conversion_service().convert_file(filepath, options)


@mcp.tool
async def convert_document_result(
    filepath: str,
    output_format: str = "markdown",
    page_range: str | None = None,
    max_pages_per_chunk: int | None = None,
    max_page_height_px: int | None = None,
    gpu_memory_profile: str | None = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    use_llm: bool = False,
) -> dict:
    """Convert a document and return text plus structured metadata and warnings."""
    options = _build_options(
        output_format,
        page_range,
        force_ocr,
        paginate_output,
        use_llm,
        max_pages_per_chunk=max_pages_per_chunk,
        max_page_height_px=max_page_height_px,
        gpu_memory_profile=gpu_memory_profile,
    )
    return await _conversion_service().convert_file_result(filepath, options)


@mcp.tool
async def convert_document_from_content(
    content_base64: str,
    filename: str,
    output_format: str = "markdown",
    page_range: str | None = None,
    max_pages_per_chunk: int | None = None,
    max_page_height_px: int | None = None,
    gpu_memory_profile: str | None = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    use_llm: bool = False,
) -> str:
    """Convert a document supplied as base64-encoded bytes.

    Use this when the file content is already in memory rather than on disk.

    Args:
        content_base64: Base64-encoded bytes of the document file.
        filename: Original filename including extension, e.g. "report.pdf".
        output_format: Output format — "markdown", "json", "html", or "chunks".
        page_range: Page range, e.g. "0,5-10,20". Null for all pages.
        max_pages_per_chunk: Optional PDF chunk size for sequential page-range batching.
        max_page_height_px: Experimental PDF page-strip height for rasterized per-page tiling.
        gpu_memory_profile: Optional GPU memory tuning profile, e.g. "low-vram".
        force_ocr: Force OCR on all pages.
        paginate_output: Separate pages with horizontal rules.
        use_llm: Use LLM for higher accuracy.
    """
    options = _build_options(
        output_format,
        page_range,
        force_ocr,
        paginate_output,
        use_llm,
        max_pages_per_chunk=max_pages_per_chunk,
        max_page_height_px=max_page_height_px,
        gpu_memory_profile=gpu_memory_profile,
    )
    content = _b64.b64decode(content_base64)
    return await _conversion_service().convert_bytes(content, filename, options)


@mcp.tool
async def convert_documents_batch(files: list[dict]) -> dict:
    """Batch-convert multiple documents.

    Each entry in `files` must contain:
        - filename (str): original filename with extension
        - content_base64 (str): base64-encoded file bytes
        - options (dict, optional): per-file options (output_format, force_ocr, etc.)

    Returns:
        {
            results: [{filename, success, content, error}, ...],
            summary: {total, successful, failed}
        }
    """
    results = await _conversion_service().convert_bytes_batch(files)
    successful = sum(1 for r in results if r["success"])
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful,
        },
    }


@mcp.tool
async def get_converter_status() -> dict:
    """Return the converter readiness status.

    Call this before attempting conversion to verify the service is initialised.

    Returns:
        {initialized: bool, status: "ready" | "failed", message: str}
    """
    return _conversion_service().get_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_options(
    output_format: str,
    page_range: str | None,
    force_ocr: bool,
    paginate_output: bool,
    use_llm: bool,
    max_pages_per_chunk: int | None = None,
    max_page_height_px: int | None = None,
    gpu_memory_profile: str | None = None,
) -> dict:
    opts: dict = {"output_format": output_format}
    if page_range:
        opts["page_range"] = page_range
    if max_pages_per_chunk is not None:
        opts["max_pages_per_chunk"] = max_pages_per_chunk
    if max_page_height_px is not None:
        opts["max_page_height_px"] = max_page_height_px
    if gpu_memory_profile:
        opts["gpu_memory_profile"] = gpu_memory_profile
    if force_ocr:
        opts["force_ocr"] = True
    if paginate_output:
        opts["paginate_output"] = True
    if use_llm:
        opts["use_llm"] = True
        # Inject LLM service configuration from environment variables.
        opts.update(_conversion_service()._llm_options_from_env())
    return opts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "http"]),
    show_default=True,
    help="MCP transport protocol.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Bind host (SSE/HTTP only).",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Bind port (SSE/HTTP only).",
)
@click.option(
    "--ocr-device",
    default=None,
    type=click.Choice(["auto", "cpu", "cuda", "nvidia", "amd", "rocm", "mps"]),
    help=(
        "OCR runtime device for Marker's local models. "
        "Use 'cpu' for a CPU-only OCR path and 'amd'/'rocm' for ROCm-backed PyTorch."
    ),
)
@click.option(
    "--model-dtype",
    default=None,
    type=click.Choice(["float16", "float32", "bfloat16"]),
    help="Experimental Marker model dtype override used when loading local models.",
)
def mcp_server_cli(
    transport: str,
    host: str,
    port: int,
    ocr_device: str | None,
    model_dtype: str | None,
) -> None:
    """Start the Marker MCP server."""
    _configure_runtime_overrides(ocr_device=ocr_device, model_dtype=model_dtype)
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    mcp_server_cli()
