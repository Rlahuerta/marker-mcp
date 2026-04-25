"""Unit tests for the async conversion_service API with mocked Marker internals."""

import asyncio
import base64
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import marker_mcp.conversion_service as svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_converter(return_text: str = "# Hello World\n\nTest content."):
    """Return a mock PdfConverter that produces a known string."""
    rendered_mock = MagicMock()
    converter_mock = MagicMock(return_value=rendered_mock)

    with patch("marker_mcp.conversion_service.text_from_rendered", return_value=(return_text, {}, {})):
        pass  # just documenting the expected patch target

    return converter_mock, rendered_mock, return_text


# ---------------------------------------------------------------------------
# OCR device override
# ---------------------------------------------------------------------------

class TestOcrDeviceOverride:
    def test_resolve_cpu(self):
        assert svc._resolve_ocr_device_override("cpu") == "cpu"

    def test_resolve_amd_aliases_to_cuda(self):
        assert svc._resolve_ocr_device_override("amd") == "cuda"
        assert svc._resolve_ocr_device_override("rocm") == "cuda"

    def test_resolve_auto(self):
        assert svc._resolve_ocr_device_override("auto") == svc._AUTO_DEVICE_SENTINEL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Unsupported MARKER_MCP_OCR_DEVICE"):
            svc._resolve_ocr_device_override("directml")

    def test_apply_sets_torch_device(self, monkeypatch):
        monkeypatch.setenv("MARKER_MCP_OCR_DEVICE", "cpu")
        monkeypatch.delenv("TORCH_DEVICE", raising=False)

        svc._apply_ocr_device_override_from_env()

        assert os.environ["TORCH_DEVICE"] == "cpu"

    def test_apply_auto_clears_torch_device(self, monkeypatch):
        monkeypatch.setenv("MARKER_MCP_OCR_DEVICE", "auto")
        monkeypatch.setenv("TORCH_DEVICE", "cuda")

        svc._apply_ocr_device_override_from_env()

        assert "TORCH_DEVICE" not in os.environ


# ---------------------------------------------------------------------------
# _build_converter
# ---------------------------------------------------------------------------

class TestBuildConverter:
    def test_builds_pdf_converter_with_config_parser_outputs(self):
        parser = MagicMock()
        parser.generate_config_dict.return_value = {"output_format": "json"}
        parser.get_processors.return_value = ["proc"]
        parser.get_renderer.return_value = "renderer"
        parser.get_llm_service.return_value = "llm"

        with (
            patch("marker_mcp.conversion_service.ConfigParser", return_value=parser) as mock_parser_cls,
            patch("marker_mcp.conversion_service.PdfConverter", return_value="converter") as mock_pdf_converter,
        ):
            result = svc._build_converter({"output_format": "json"})

        assert result == "converter"
        mock_parser_cls.assert_called_once_with({"output_format": "json"})
        mock_pdf_converter.assert_called_once_with(
            config={"output_format": "json", "pdftext_workers": 1},
            artifact_dict=svc._MODELS,
            processor_list=["proc"],
            renderer="renderer",
            llm_service="llm",
        )


# ---------------------------------------------------------------------------
# convert_file
# ---------------------------------------------------------------------------

class TestConvertFile:
    async def test_calls_convert_sync_in_thread(self, tmp_pdf_file):
        expected = "# Converted\n\nContent here."
        with (
            patch("marker_mcp.conversion_service._build_converter") as mock_build,
            patch("marker_mcp.conversion_service.text_from_rendered", return_value=(expected, {}, {})),
        ):
            mock_converter = MagicMock(return_value=MagicMock())
            mock_build.return_value = mock_converter
            result = await svc.convert_file(str(tmp_pdf_file))

        assert result == expected

    async def test_default_output_format_is_markdown(self, tmp_pdf_file):
        with (
            patch("marker_mcp.conversion_service._build_converter") as mock_build,
            patch("marker_mcp.conversion_service.text_from_rendered", return_value=("out", {}, {})),
        ):
            mock_build.return_value = MagicMock(return_value=MagicMock())
            await svc.convert_file(str(tmp_pdf_file))
            call_opts = mock_build.call_args[0][0]

        assert call_opts["output_format"] == "markdown"

    async def test_raises_when_models_none(self, tmp_pdf_file):
        svc._MODELS = None
        with pytest.raises(RuntimeError, match="not available"):
            await svc.convert_file(str(tmp_pdf_file))
        # fixture reset_mock_models will restore _MODELS after test

    async def test_accepts_custom_options(self, tmp_pdf_file):
        with (
            patch("marker_mcp.conversion_service._build_converter") as mock_build,
            patch("marker_mcp.conversion_service.text_from_rendered", return_value=("html", {}, {})),
        ):
            mock_build.return_value = MagicMock(return_value=MagicMock())
            await svc.convert_file(str(tmp_pdf_file), {"output_format": "html", "force_ocr": True})
            opts = mock_build.call_args[0][0]

        assert opts["output_format"] == "html"
        assert opts["force_ocr"] is True


# ---------------------------------------------------------------------------
# convert_bytes
# ---------------------------------------------------------------------------

class TestConvertBytes:
    async def test_cleans_up_temp_file(self, minimal_pdf_bytes):
        temp_paths: list[str] = []

        original_convert_file = svc.convert_file

        async def spy_convert_file(filepath, options=None):
            temp_paths.append(filepath)
            return "mocked output"

        with patch("marker_mcp.conversion_service.convert_file", side_effect=spy_convert_file):
            result = await svc.convert_bytes(minimal_pdf_bytes, "test.pdf")

        assert result == "mocked output"
        assert len(temp_paths) == 1
        assert not os.path.exists(temp_paths[0]), "Temp file should be deleted after conversion"

    async def test_uses_correct_suffix(self, minimal_pdf_bytes):
        captured: list[str] = []

        async def spy(filepath, options=None):
            captured.append(filepath)
            return "ok"

        with patch("marker_mcp.conversion_service.convert_file", side_effect=spy):
            await svc.convert_bytes(minimal_pdf_bytes, "doc.docx")

        assert captured[0].endswith(".docx")

    async def test_temp_file_cleaned_on_error(self, minimal_pdf_bytes):
        temp_paths: list[str] = []

        async def failing_convert(filepath, options=None):
            temp_paths.append(filepath)
            raise ValueError("Simulated conversion error")

        with patch("marker_mcp.conversion_service.convert_file", side_effect=failing_convert):
            with pytest.raises(ValueError, match="Simulated"):
                await svc.convert_bytes(minimal_pdf_bytes, "test.pdf")

        assert not os.path.exists(temp_paths[0]), "Temp file must be cleaned up even on error"

    async def test_cleanup_ignores_oserror(self, minimal_pdf_bytes):
        temp_paths: list[str] = []

        async def spy_convert(filepath, options=None):
            temp_paths.append(filepath)
            return "ok"

        with (
            patch("marker_mcp.conversion_service.convert_file", side_effect=spy_convert),
            patch("marker_mcp.conversion_service.os.unlink", side_effect=OSError("busy")),
        ):
            result = await svc.convert_bytes(minimal_pdf_bytes, "test.pdf")

        assert result == "ok"
        assert len(temp_paths) == 1
        Path(temp_paths[0]).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# convert_bytes_batch
# ---------------------------------------------------------------------------

class TestConvertBytesBatch:
    async def test_empty_list_returns_empty(self):
        results = await svc.convert_bytes_batch([])
        assert results == []

    async def test_successful_conversion(self, minimal_pdf_bytes):
        encoded = base64.b64encode(minimal_pdf_bytes).decode()

        async def fake_convert(content, filename, options=None):
            return f"# {filename}"

        with patch("marker_mcp.conversion_service.convert_bytes", side_effect=fake_convert):
            results = await svc.convert_bytes_batch([
                {"filename": "a.pdf", "content_base64": encoded},
                {"filename": "b.pdf", "content_base64": encoded},
            ])

        assert len(results) == 2
        assert results[0] == {"filename": "a.pdf", "success": True, "content": "# a.pdf", "error": None}
        assert results[1] == {"filename": "b.pdf", "success": True, "content": "# b.pdf", "error": None}

    async def test_bad_base64_returns_error_entry(self):
        results = await svc.convert_bytes_batch([
            {"filename": "bad.pdf", "content_base64": "!!!not-valid-base64!!!"},
        ])
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["error"] is not None
        assert results[0]["content"] is None
        assert results[0]["filename"] == "bad.pdf"

    async def test_partial_failure_does_not_abort_batch(self, minimal_pdf_bytes):
        good_b64 = base64.b64encode(minimal_pdf_bytes).decode()
        call_count = 0

        async def sometimes_fail(content, filename, options=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure")
            return f"# {filename}"

        with patch("marker_mcp.conversion_service.convert_bytes", side_effect=sometimes_fail):
            results = await svc.convert_bytes_batch([
                {"filename": "ok1.pdf", "content_base64": good_b64},
                {"filename": "fail.pdf", "content_base64": good_b64},
                {"filename": "ok3.pdf", "content_base64": good_b64},
            ])

        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    async def test_default_filename_fallback(self):
        """Entry with no filename key falls back to 'document.pdf'."""
        async def fake_convert(content, filename, options=None):
            return filename

        with patch("marker_mcp.conversion_service.convert_bytes", side_effect=fake_convert):
            results = await svc.convert_bytes_batch([
                {"content_base64": base64.b64encode(b"x").decode()},
            ])

        assert results[0]["filename"] == "document.pdf"


# ---------------------------------------------------------------------------
# module import behavior
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_model_init_failure_keeps_models_none_and_logs(self):
        marker_models = sys.modules["marker.models"]
        original_create_model_dict = marker_models.create_model_dict

        try:
            marker_models.create_model_dict = MagicMock(side_effect=RuntimeError("boom"))
            with patch("builtins.print") as mock_print:
                importlib.reload(svc)

            assert svc._MODELS is None
            mock_print.assert_any_call("❌ Failed to load Marker models: boom")
        finally:
            marker_models.create_model_dict = original_create_model_dict
            importlib.reload(svc)
