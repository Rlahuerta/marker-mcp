"""Integration tests — require real Marker models and an actual PDF file.

Run with:
    conda run -n marker-mcp pytest tests/test_integration.py -v -m integration

These tests are skipped by default in CI and plain `pytest` runs.
"""

import base64
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures — use the real models (no mocking)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def load_real_models():
    """Reload conversion_service with real models for this module."""
    import importlib
    import sys
    from unittest.mock import patch

    # Remove cached (mocked) module so it reloads with real models
    for mod in list(sys.modules.keys()):
        if mod.startswith("marker_mcp"):
            del sys.modules[mod]

    # Import fresh — this will call the real create_model_dict()
    import marker_mcp.conversion_service  # noqa: F401
    yield

    # Restore mocked modules after the integration tests finish
    for mod in list(sys.modules.keys()):
        if mod.startswith("marker_mcp"):
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_sample_pdf():
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found at {SAMPLE_PDF}")
    return str(SAMPLE_PDF)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_convert_file_basic():
    """Convert the sample PDF and verify we get non-empty markdown."""
    import marker_mcp.conversion_service as svc

    pdf_path = _require_sample_pdf()
    result = await svc.convert_file(pdf_path)

    assert isinstance(result, str)
    assert len(result) > 100, "Converted output seems too short"
    # arXiv papers always have an abstract
    assert any(word in result.lower() for word in ["abstract", "introduction", "#"]), (
        "Expected markdown headings or common academic paper sections"
    )


@pytest.mark.integration
async def test_convert_bytes_basic():
    """Convert PDF bytes and verify output."""
    import marker_mcp.conversion_service as svc

    pdf_path = _require_sample_pdf()
    pdf_bytes = Path(pdf_path).read_bytes()
    result = await svc.convert_bytes(pdf_bytes, "sample.pdf")

    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
async def test_get_status_shows_ready():
    """After real model loading, status must be ready."""
    import marker_mcp.conversion_service as svc

    status = svc.get_status()
    assert status["initialized"] is True
    assert status["status"] == "ready"


@pytest.mark.integration
async def test_mcp_tool_convert_document():
    """End-to-end: call the MCP tool with the real server and models."""
    import json
    import marker_mcp.mcp_server  # reimported after module reset

    from fastmcp import Client

    pdf_path = _require_sample_pdf()
    mcp = marker_mcp.mcp_server.mcp

    async with Client(mcp) as client:
        result = await client.call_tool("convert_document", {"filepath": pdf_path})

    assert result
    text = result[0].text if hasattr(result[0], "text") else str(result[0])
    assert len(text) > 100


@pytest.mark.integration
async def test_batch_conversion():
    """Batch-convert a file twice and verify both succeed."""
    import marker_mcp.conversion_service as svc

    pdf_path = _require_sample_pdf()
    encoded = base64.b64encode(Path(pdf_path).read_bytes()).decode()

    results = await svc.convert_bytes_batch([
        {"filename": "copy1.pdf", "content_base64": encoded},
        {"filename": "copy2.pdf", "content_base64": encoded},
    ])

    assert len(results) == 2
    for r in results:
        assert r["success"] is True, f"Batch item failed: {r['error']}"
        assert len(r["content"]) > 100
