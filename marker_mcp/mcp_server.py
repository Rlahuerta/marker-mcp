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

import click
from fastmcp import FastMCP

import marker_mcp.conversion_service as svc
from marker_mcp.conversion_service import _llm_options_from_env

mcp = FastMCP(name="Marker Document Conversion Service")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool
async def convert_document(
    filepath: str,
    output_format: str = "markdown",
    page_range: str | None = None,
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
        force_ocr: Force OCR on all pages even if a text layer exists.
        paginate_output: Separate pages with horizontal rules containing page numbers.
        use_llm: Use an LLM for higher accuracy (requires GOOGLE_API_KEY or compatible service).
    """
    options = _build_options(output_format, page_range, force_ocr, paginate_output, use_llm)
    return await svc.convert_file(filepath, options)


@mcp.tool
async def convert_document_from_content(
    content_base64: str,
    filename: str,
    output_format: str = "markdown",
    page_range: str | None = None,
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
        force_ocr: Force OCR on all pages.
        paginate_output: Separate pages with horizontal rules.
        use_llm: Use LLM for higher accuracy.
    """
    options = _build_options(output_format, page_range, force_ocr, paginate_output, use_llm)
    content = _b64.b64decode(content_base64)
    return await svc.convert_bytes(content, filename, options)


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
    results = await svc.convert_bytes_batch(files)
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
    return svc.get_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_options(
    output_format: str,
    page_range: str | None,
    force_ocr: bool,
    paginate_output: bool,
    use_llm: bool,
) -> dict:
    opts: dict = {"output_format": output_format}
    if page_range:
        opts["page_range"] = page_range
    if force_ocr:
        opts["force_ocr"] = True
    if paginate_output:
        opts["paginate_output"] = True
    if use_llm:
        opts["use_llm"] = True
        # Inject LLM service configuration from environment variables.
        opts.update(_llm_options_from_env())
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
def mcp_server_cli(transport: str, host: str, port: int) -> None:
    """Start the Marker MCP server."""
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    mcp_server_cli()
