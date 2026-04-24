"""Shared fixtures for the marker-mcp test suite.

Models are heavy — this conftest injects sys.modules mocks for ALL marker ML
submodules BEFORE `marker_mcp.conversion_service` is first imported.  This:
  - Avoids loading torch / transformers / torchvision (slow, can fail on mismatched envs)
  - Keeps unit tests fast and CI-friendly
  - Still allows integration tests to reload with real models (see test_integration.py)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build mock models dict
# ---------------------------------------------------------------------------
_MOCK_MODELS = {
    "layout_model": MagicMock(name="layout_model"),
    "texify_model": MagicMock(name="texify_model"),
    "recognition_model": MagicMock(name="recognition_model"),
    "table_rec_model": MagicMock(name="table_rec_model"),
    "detection_model": MagicMock(name="detection_model"),
    "edit_model": MagicMock(name="edit_model"),
}

# ---------------------------------------------------------------------------
# Inject sys.modules mocks for all marker ML submodules BEFORE import.
# This must happen at module (conftest) load time, before pytest collects tests.
# ---------------------------------------------------------------------------

def _inject_marker_mocks() -> None:
    """Stub out the marker ML chain so no real models or torch ops are loaded."""
    # marker.models
    mock_models_mod = MagicMock(name="marker.models")
    mock_models_mod.create_model_dict = MagicMock(return_value=_MOCK_MODELS)

    # marker.config.parser — ConfigParser returns a sensible mock instance
    mock_config_instance = MagicMock(name="ConfigParserInstance")
    mock_config_instance.generate_config_dict.return_value = {"output_format": "markdown"}
    mock_config_instance.get_processors.return_value = None
    mock_config_instance.get_renderer.return_value = None
    mock_config_instance.get_llm_service.return_value = None
    mock_config_parser_mod = MagicMock(name="marker.config.parser")
    mock_config_parser_mod.ConfigParser.return_value = mock_config_instance

    # marker.converters.pdf — PdfConverter(filepath) returns a mock rendered object
    mock_rendered = MagicMock(name="RenderedObject")
    mock_converter_instance = MagicMock(name="PdfConverterInstance", return_value=mock_rendered)
    mock_pdf_mod = MagicMock(name="marker.converters.pdf")
    mock_pdf_mod.PdfConverter.return_value = mock_converter_instance

    # marker.output
    mock_output_mod = MagicMock(name="marker.output")
    mock_output_mod.text_from_rendered = MagicMock(
        return_value=("# Mocked Output\n\nParagraph.", {}, {})
    )

    modules_to_inject = {
        "marker.models": mock_models_mod,
        "marker.config": MagicMock(name="marker.config"),
        "marker.config.parser": mock_config_parser_mod,
        "marker.converters": MagicMock(name="marker.converters"),
        "marker.converters.pdf": mock_pdf_mod,
        "marker.output": mock_output_mod,
    }

    for name, mock in modules_to_inject.items():
        sys.modules.setdefault(name, mock)


if "marker_mcp.conversion_service" not in sys.modules:
    _inject_marker_mocks()

import marker_mcp.conversion_service as _svc_module  # noqa: E402
import marker_mcp.mcp_server as _server_module  # noqa: E402

# Ensure _MODELS is always our mock dict (not None or accidentally the real one)
_svc_module._MODELS = _MOCK_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_minimal_pdf() -> bytes:
    """Build a minimal but spec-valid 1-page PDF from scratch."""
    content_stream = b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"

    raw_objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            b"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1"
            b" /BaseFont /Helvetica >> >> >>\n"
            b"   /Contents 4 0 R >>"
        ),
        4: (
            b"<< /Length "
            + str(len(content_stream)).encode()
            + b" >>\nstream\n"
            + content_stream
            + b"endstream"
        ),
    }

    out = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}

    for obj_num in sorted(raw_objects):
        offsets[obj_num] = len(out)
        out += f"{obj_num} 0 obj\n".encode()
        out += raw_objects[obj_num]
        out += b"\nendobj\n"

    xref_offset = len(out)
    num_objs = max(raw_objects) + 1

    out += b"xref\n"
    out += f"0 {num_objs}\n".encode()
    out += b"0000000000 65535 f \n"
    for i in range(1, num_objs):
        offset = offsets.get(i, 0)
        out += f"{offset:010d} 00000 n \n".encode()

    out += b"trailer\n"
    out += f"<< /Size {num_objs} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_offset}\n".encode()
    out += b"%%EOF\n"

    return out


@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """Return bytes of a minimal valid 1-page PDF."""
    return _make_minimal_pdf()


@pytest.fixture
def tmp_pdf_file(tmp_path, minimal_pdf_bytes) -> Path:
    """Write the minimal PDF to a temporary file and return its path."""
    p = tmp_path / "test_document.pdf"
    p.write_bytes(minimal_pdf_bytes)
    return p


@pytest.fixture
def sample_pdf_path() -> Path | None:
    """Path to the bundled arXiv sample PDF (used by integration tests)."""
    p = FIXTURES_DIR / "sample.pdf"
    return p if p.exists() else None


@pytest.fixture(autouse=True)
def reset_mock_models():
    """Ensure _MODELS is the mock dict before each unit test."""
    _svc_module._MODELS = _MOCK_MODELS
    yield
    _svc_module._MODELS = _MOCK_MODELS


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Remove all LLM-related env vars so tests start from a clean slate."""
    for var in [
        "MARKER_LLM_SERVICE",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "CLAUDE_API_KEY",
        "CLAUDE_MODEL",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield

