"""Ollama Cloud integration tests for the Marker MCP server.

Verifies that marker-mcp can produce quality markdown from challenging PDFs
using a remote LLM via Ollama Cloud (e.g. ``gemma4:31b-cloud``).

Skip conditions (evaluated at module load):
  - ``.env`` file not found in the project root
  - ``OLLAMA_BASE_URL`` or ``OLLAMA_MODEL`` not set in ``.env``
  - ``OLLAMA_MODEL`` does not end in ``-cloud``
  - Ollama daemon is not reachable at ``OLLAMA_BASE_URL``

Setup::

    # Sign in to Ollama Cloud (once)
    ollama signin

    # Pull the cloud-routed model
    ollama pull gemma4:31b-cloud

    # .env in project root
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=gemma4:31b-cloud

Run::

    conda run -n marker-mcp pytest tests/test_ollama_cloud_integration.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest
from tests import conftest as test_support

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
COMPLEX_PDF = FIXTURES_DIR / "complex_document.pdf"
ENV_FILE = Path(__file__).parent.parent / ".env"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers — .env parsing and skip detection
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file and return a dict (no shell expansion)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _check_skip_reason() -> str | None:
    """Return a human-readable skip reason, or None if tests may run."""
    if not ENV_FILE.exists():
        return f".env not found at {ENV_FILE}"

    env = _load_dotenv(ENV_FILE)
    base_url = env.get("OLLAMA_BASE_URL", "").strip()
    model = env.get("OLLAMA_MODEL", "").strip()

    if not base_url:
        return "OLLAMA_BASE_URL not set in .env"
    if not model:
        return "OLLAMA_MODEL not set in .env"
    if not model.endswith("-cloud"):
        return (
            f"OLLAMA_MODEL={model!r} — must end in '-cloud' to run Ollama Cloud tests"
        )

    # Verify the local Ollama daemon is reachable
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
    except Exception as exc:
        return f"Ollama daemon not reachable at {base_url}: {exc}"

    return None


_SKIP_REASON = _check_skip_reason()

# Apply marks at module level — skipped if any condition above fires
pytestmark: list = [pytest.mark.integration, pytest.mark.ollama_cloud]
if _SKIP_REASON:
    pytestmark.append(pytest.mark.skip(reason=_SKIP_REASON))


# ---------------------------------------------------------------------------
# Module-scoped fixture — real models + Ollama Cloud env
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def load_real_models_and_env():
    """Reload marker_mcp with real models and Ollama Cloud env vars active."""
    env = _load_dotenv(ENV_FILE)

    # Inject OLLAMA_* vars so the reloaded conversion_service picks them up
    saved: dict[str, str | None] = {}
    for key in ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "MARKER_LLM_SERVICE", "CUDA_VISIBLE_DEVICES"):
        saved[key] = os.environ.get(key)
        if key in env:
            os.environ[key] = env[key]

    test_support._clear_marker_mcp_modules()
    test_support._clear_marker_mock_modules()
    test_support._release_torch_resources()

    # Fresh import — calls the real create_model_dict()
    try:
        import marker_mcp.conversion_service  # noqa: F401
    except Exception as exc:
        test_support._restore_mocked_marker_mcp()
        pytest.skip(f"Could not load real Marker models: {exc}")

    yield

    # Restore env and discard loaded modules
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val

    test_support._restore_mocked_marker_mcp()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_fixture(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"Fixture PDF not found: {path}")
    return str(path)


def _llm_options() -> dict:
    import marker_mcp.mcp_server as server

    return server._build_options("markdown", None, False, False, True)


def _run_real_python(script: str, extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    stdout = proc.stdout.strip()
    stdout_lines = [line for line in stdout.splitlines() if line.strip()]
    payload = stdout_lines[-1] if stdout_lines else ""

    if payload.startswith("SKIP:"):
        pytest.skip(payload.removeprefix("SKIP:").strip())

    if proc.returncode != 0:
        raise AssertionError(
            "Integration subprocess failed.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Expected JSON output from subprocess, got:\n{stdout}") from exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_ollama_cloud_status():
    """After module setup with Ollama Cloud env, the service must report ready."""
    import marker_mcp.conversion_service as svc

    status = svc.get_status()
    assert status["initialized"] is True, f"Models not initialized: {status}"
    assert status["status"] == "ready"


async def test_ollama_cloud_convert_simple():
    """Convert a standard arXiv PDF with Ollama Cloud LLM enhancement.

    The sample.pdf is arXiv:2301.00001 — a compact academic paper that exercises
    the standard conversion path with LLM table / heading clean-up.
    """
    import marker_mcp.conversion_service as svc

    pdf_path = _require_fixture(SAMPLE_PDF)
    result = await svc.convert_file(pdf_path, _llm_options())

    assert isinstance(result, str), "Expected string output"
    assert len(result) > 200, (
        f"Output suspiciously short ({len(result)} chars) — conversion may have failed"
    )
    assert "#" in result, "Expected at least one markdown heading"
    # Any coherent English text is a good sign
    assert any(
        word in result.lower() for word in ["abstract", "introduction", "the", "a"]
    ), "Output contains no readable text"


async def test_ollama_cloud_convert_complex_document():
    """Convert a visually complex PDF (ViT paper) with Ollama Cloud.

    The Vision Transformer paper (arXiv:2010.11929) contains:
    - Multi-column layout
    - Architecture comparison tables
    - Mathematical notation
    - Figure captions with complex formatting

    This is the primary stress test for the cloud model quality.
    """
    import marker_mcp.conversion_service as svc

    pdf_path = _require_fixture(COMPLEX_PDF)
    result = await svc.convert_file(pdf_path, _llm_options())

    assert isinstance(result, str)
    assert len(result) > 500, (
        f"Complex document output too short ({len(result)} chars)"
    )

    lower = result.lower()
    # The ViT paper must mention these concepts
    assert any(
        word in lower for word in ["image", "transformer", "attention", "vision", "patch"]
    ), "Expected ViT paper content in output"
    assert "#" in result, "Expected markdown headings in converted output"


async def test_ollama_cloud_via_mcp_tool():
    """End-to-end: call the convert_document MCP tool with use_llm=True.

    Validates the full stack — FastMCP server → conversion_service → Marker
    → Ollama Cloud LLM — using the in-process Client.
    """
    import marker_mcp.mcp_server

    from fastmcp import Client

    pdf_path = _require_fixture(SAMPLE_PDF)
    mcp = marker_mcp.mcp_server.mcp

    async with Client(mcp) as client:
        result = await client.call_tool(
            "convert_document",
            {"filepath": pdf_path, "use_llm": True},
        )

    assert result.content, "MCP tool returned empty result"
    first = result.content[0]
    text = first.text if hasattr(first, "text") else str(first)
    assert len(text) > 200, f"MCP tool output too short: {len(text)} chars"
    assert "#" in text, "Expected markdown output from MCP tool"


async def test_ollama_cloud_convert_bytes_path():
    """Test byte-based conversion with Ollama Cloud (exercises base64 path)."""
    import marker_mcp.conversion_service as svc

    pdf_path = _require_fixture(SAMPLE_PDF)
    pdf_bytes = Path(pdf_path).read_bytes()

    result = await svc.convert_bytes(pdf_bytes, "sample.pdf", _llm_options())

    assert isinstance(result, str)
    assert len(result) > 200
    assert "#" in result


async def test_ollama_cloud_example_cli_path():
    """End-to-end: run examples/convert_pdf.py with --use-llm against Ollama Cloud."""
    pdf_path = _require_fixture(SAMPLE_PDF)
    env = _load_dotenv(ENV_FILE)
    data = _run_real_python(
        textwrap.dedent(
            f"""
            import importlib.util
            import json
            from pathlib import Path
            try:
                example_path = Path("examples/convert_pdf.py").resolve()
                spec = importlib.util.spec_from_file_location("examples.convert_pdf", example_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
            except Exception as exc:
                print(f"SKIP:Could not import example CLI: {{exc}}")
                raise SystemExit(0)

            import asyncio

            output_path = Path("tests/fixtures/sample-ollama-cloud-example.md").resolve()
            args = module.build_parser().parse_args([
                {pdf_path!r},
                "--page-range", "0-0",
                "--use-llm",
                "-o", str(output_path),
            ])

            async def main():
                written = await module.convert_pdf(args)
                text = written.read_text(encoding="utf-8")
                print(json.dumps({{
                    "output_exists": written.exists(),
                    "length": len(text),
                    "has_heading": "#" in text,
                    "uses_cloud_model": {env.get("OLLAMA_MODEL", "").strip()!r}.endswith("-cloud"),
                }}))
                written.unlink(missing_ok=True)

            asyncio.run(main())
            """
        )
    )

    assert data["output_exists"] is True
    assert data["length"] > 100
    assert data["has_heading"] is True
    assert data["uses_cloud_model"] is True
