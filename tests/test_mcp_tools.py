"""Tests for MCP tools via the fastmcp in-process Client."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_mcp():
    import marker_mcp.mcp_server as server

    return server.mcp


async def _call(tool_name: str, **kwargs):
    """Call a tool through the in-process fastmcp Client and return the text payload."""
    async with Client(_current_mcp()) as client:
        result = await client.call_tool(tool_name, kwargs)
    # result is a CallToolResult; .content is a list of content items
    assert result.content, f"Tool '{tool_name}' returned no content"
    first = result.content[0]
    return first.text if hasattr(first, "text") else str(first)


# ---------------------------------------------------------------------------
# get_converter_status
# ---------------------------------------------------------------------------

class TestGetConverterStatus:
    async def test_returns_ready_status(self):
        text = await _call("get_converter_status")
        # The tool returns a dict serialised to JSON string by fastmcp
        data = json.loads(text)
        assert data["initialized"] is True
        assert data["status"] == "ready"

    async def test_returns_failed_when_models_none(self):
        import marker_mcp.conversion_service as svc
        svc._MODELS = None
        text = await _call("get_converter_status")
        data = json.loads(text)
        assert data["initialized"] is False
        assert data["status"] == "failed"
        # reset_mock_models autouse fixture restores _MODELS after this test

    async def test_status_has_expected_keys(self):
        text = await _call("get_converter_status")
        data = json.loads(text)
        assert {"initialized", "status", "message"} <= data.keys()


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    async def test_all_four_tools_registered(self):
        async with Client(_current_mcp()) as client:
            tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {
            "convert_document",
            "convert_document_from_content",
            "convert_documents_batch",
            "get_converter_status",
        } <= names

    async def test_tools_have_descriptions(self):
        async with Client(_current_mcp()) as client:
            tools = await client.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    async def test_structured_result_tool_registered(self):
        async with Client(_current_mcp()) as client:
            tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "convert_document_result" in names


# ---------------------------------------------------------------------------
# convert_document
# ---------------------------------------------------------------------------

class TestConvertDocument:
    async def test_delegates_to_convert_file(self, tmp_pdf_file):
        with patch("marker_mcp.conversion_service.convert_file", return_value="# Result") as mock_cf:
            text = await _call("convert_document", filepath=str(tmp_pdf_file))
        assert text == "# Result"
        mock_cf.assert_called_once()

    async def test_passes_output_format(self, tmp_pdf_file):
        with patch("marker_mcp.conversion_service.convert_file", return_value="<h1>ok</h1>") as mock_cf:
            await _call("convert_document", filepath=str(tmp_pdf_file), output_format="html")
        opts = mock_cf.call_args[0][1]
        assert opts["output_format"] == "html"

    async def test_page_range_forwarded(self, tmp_pdf_file):
        with patch("marker_mcp.conversion_service.convert_file", return_value="ok") as mock_cf:
            await _call("convert_document", filepath=str(tmp_pdf_file), page_range="0-2")
        opts = mock_cf.call_args[0][1]
        assert opts["page_range"] == "0-2"

    async def test_force_ocr_forwarded(self, tmp_pdf_file):
        with patch("marker_mcp.conversion_service.convert_file", return_value="ok") as mock_cf:
            await _call("convert_document", filepath=str(tmp_pdf_file), force_ocr=True)
        opts = mock_cf.call_args[0][1]
        assert opts.get("force_ocr") is True

    async def test_max_pages_per_chunk_forwarded(self, tmp_pdf_file):
        with patch("marker_mcp.conversion_service.convert_file", return_value="ok") as mock_cf:
            await _call("convert_document", filepath=str(tmp_pdf_file), max_pages_per_chunk=8)
        opts = mock_cf.call_args[0][1]
        assert opts["max_pages_per_chunk"] == 8


class TestConvertDocumentResult:
    async def test_returns_text_metadata_and_asset_paths(self, tmp_pdf_file):
        payload = {
            "text": "# Result",
            "metadata": {"page_count": 1},
            "warnings": [],
            "assets": {
                "images": [
                    {
                        "filename": "page-1.png",
                        "path": "artifacts/images/page-1.png",
                    }
                ]
            },
        }

        with patch(
            "marker_mcp.conversion_service.convert_file_result",
            return_value=payload,
            create=True,
        ) as mock_result:
            text = await _call("convert_document_result", filepath=str(tmp_pdf_file))

        assert json.loads(text) == payload
        mock_result.assert_called_once()

    async def test_max_pages_per_chunk_forwarded(self, tmp_pdf_file):
        payload = {
            "text": "# Result",
            "metadata": {},
            "warnings": [],
            "assets": {},
        }

        with patch(
            "marker_mcp.conversion_service.convert_file_result",
            return_value=payload,
            create=True,
        ) as mock_result:
            await _call("convert_document_result", filepath=str(tmp_pdf_file), max_pages_per_chunk=8)

        opts = mock_result.call_args[0][1]
        assert opts["max_pages_per_chunk"] == 8

    async def test_exports_marker_figures_as_serializable_assets(self, tmp_pdf_file):
        markdown = "![Figure](_page_0_Picture_2.jpeg)"

        with (
            patch("marker_mcp.conversion_service._build_converter") as mock_build,
            patch(
                "marker_mcp.conversion_service.text_from_rendered",
                return_value=(
                    markdown,
                    {"page_count": 1},
                    {"_page_0_Picture_2.jpeg": Image.new("RGB", (2, 2), color="white")},
                ),
            ),
        ):
            mock_build.return_value = MagicMock(return_value=MagicMock())
            text = await _call("convert_document_result", filepath=str(tmp_pdf_file))

        data = json.loads(text)
        assert data["metadata"] == {"page_count": 1}
        [asset] = data["assets"]["images"]
        assert asset["filename"] == "_page_0_Picture_2.jpeg"
        assert asset.get("path", "").endswith("_page_0_Picture_2.jpeg") or "content_base64" in asset


# ---------------------------------------------------------------------------
# convert_document_from_content
# ---------------------------------------------------------------------------

class TestConvertDocumentFromContent:
    async def test_decodes_base64_and_delegates(self, minimal_pdf_bytes):
        encoded = base64.b64encode(minimal_pdf_bytes).decode()

        with patch("marker_mcp.conversion_service.convert_bytes", return_value="# From bytes") as mock_cb:
            text = await _call(
                "convert_document_from_content",
                content_base64=encoded,
                filename="test.pdf",
            )

        assert text == "# From bytes"
        call_bytes, call_filename = mock_cb.call_args[0][:2]
        assert call_bytes == minimal_pdf_bytes
        assert call_filename == "test.pdf"

    async def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            await _call(
                "convert_document_from_content",
                content_base64="not-valid-base64!!!",
                filename="test.pdf",
            )


# ---------------------------------------------------------------------------
# convert_documents_batch
# ---------------------------------------------------------------------------

class TestConvertDocumentsBatch:
    async def test_empty_batch(self):
        text = await _call("convert_documents_batch", files=[])
        data = json.loads(text)
        assert data["summary"]["total"] == 0
        assert data["results"] == []

    async def test_batch_summary_counts(self, minimal_pdf_bytes):
        good_b64 = base64.b64encode(minimal_pdf_bytes).decode()

        async def fake_convert(content, filename, options=None):
            if filename == "fail.pdf":
                raise RuntimeError("Simulated failure")
            return "# ok"

        with patch("marker_mcp.conversion_service.convert_bytes", side_effect=fake_convert):
            text = await _call(
                "convert_documents_batch",
                files=[
                    {"filename": "good.pdf", "content_base64": good_b64},
                    {"filename": "fail.pdf", "content_base64": good_b64},
                ],
            )

        data = json.loads(text)
        assert data["summary"]["total"] == 2
        assert data["summary"]["successful"] == 1
        assert data["summary"]["failed"] == 1
