"""Integration tests — require real Marker models and an actual PDF file.

Run with:
    conda run -n marker-mcp pytest tests/test_integration.py -v -m integration

These tests run the real conversion path in subprocesses so they don't leak GPU
or module state into the rest of the pytest session.
"""

import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"

pytestmark = pytest.mark.integration


def _require_sample_pdf() -> str:
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found at {SAMPLE_PDF}")
    return str(SAMPLE_PDF)


def _run_real_python(script: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

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


@pytest.mark.integration
async def test_convert_file_basic():
    pdf_path = _require_sample_pdf()
    data = _run_real_python(
        textwrap.dedent(
            f"""
            import json
            try:
                import marker_mcp.conversion_service as svc
            except Exception as exc:
                print(f"SKIP:Could not load real Marker models: {{exc}}")
                raise SystemExit(0)

            import asyncio

            async def main():
                result = await svc.convert_file({pdf_path!r})
                print(json.dumps({{
                    "is_str": isinstance(result, str),
                    "length": len(result),
                    "has_expected_text": any(word in result.lower() for word in ["abstract", "introduction", "#"]),
                }}))

            asyncio.run(main())
            """
        )
    )

    assert data["is_str"] is True
    assert data["length"] > 100, "Converted output seems too short"
    assert data["has_expected_text"] is True


@pytest.mark.integration
async def test_convert_bytes_basic():
    pdf_path = _require_sample_pdf()
    data = _run_real_python(
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            try:
                import marker_mcp.conversion_service as svc
            except Exception as exc:
                print(f"SKIP:Could not load real Marker models: {{exc}}")
                raise SystemExit(0)

            import asyncio

            async def main():
                pdf_bytes = Path({pdf_path!r}).read_bytes()
                result = await svc.convert_bytes(pdf_bytes, "sample.pdf")
                print(json.dumps({{
                    "is_str": isinstance(result, str),
                    "length": len(result),
                }}))

            asyncio.run(main())
            """
        )
    )

    assert data["is_str"] is True
    assert data["length"] > 100


@pytest.mark.integration
async def test_get_status_shows_ready():
    data = _run_real_python(
        textwrap.dedent(
            """
            import json
            try:
                import marker_mcp.conversion_service as svc
            except Exception as exc:
                print(f"SKIP:Could not load real Marker models: {exc}")
                raise SystemExit(0)

            print(json.dumps(svc.get_status()))
            """
        )
    )

    assert data["initialized"] is True
    assert data["status"] == "ready"


@pytest.mark.integration
async def test_mcp_tool_convert_document():
    pdf_path = _require_sample_pdf()
    data = _run_real_python(
        textwrap.dedent(
            f"""
            import asyncio
            import json
            try:
                import marker_mcp.mcp_server
            except Exception as exc:
                print(f"SKIP:Could not load real Marker models: {{exc}}")
                raise SystemExit(0)

            from fastmcp import Client

            async def main():
                mcp = marker_mcp.mcp_server.mcp
                async with Client(mcp) as client:
                    result = await client.call_tool("convert_document", {{"filepath": {pdf_path!r}}})
                first = result.content[0]
                text = first.text if hasattr(first, "text") else str(first)
                print(json.dumps({{"length": len(text)}}))

            asyncio.run(main())
            """
        )
    )

    assert data["length"] > 100


@pytest.mark.integration
async def test_batch_conversion():
    pdf_path = _require_sample_pdf()
    data = _run_real_python(
        textwrap.dedent(
            f"""
            import asyncio
            import base64
            import json
            from pathlib import Path
            try:
                import marker_mcp.conversion_service as svc
            except Exception as exc:
                print(f"SKIP:Could not load real Marker models: {{exc}}")
                raise SystemExit(0)

            async def main():
                encoded = base64.b64encode(Path({pdf_path!r}).read_bytes()).decode()
                results = await svc.convert_bytes_batch([
                    {{"filename": "copy1.pdf", "content_base64": encoded}},
                    {{"filename": "copy2.pdf", "content_base64": encoded}},
                ])
                print(json.dumps({{
                    "count": len(results),
                    "successes": [r["success"] for r in results],
                    "lengths": [len(r["content"]) if r["content"] else 0 for r in results],
                    "errors": [r["error"] for r in results],
                }}))

            asyncio.run(main())
            """
        )
    )

    assert data["count"] == 2
    assert data["successes"] == [True, True], f"Batch item failed: {data['errors']}"
    assert all(length > 100 for length in data["lengths"])
