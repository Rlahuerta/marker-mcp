"""Unit tests for the async conversion_service API with mocked Marker internals."""

import asyncio
import base64
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from PIL import Image

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


def _make_test_figure() -> Image.Image:
    """Create a tiny in-memory image that mimics a Marker figure export."""
    return Image.new("RGB", (2, 2), color="white")


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
    def test_builds_pdf_converter_with_default_pdftext_workers(self):
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
            config={"output_format": "json", "pdftext_workers": 8},
            artifact_dict=svc._MODELS,
            processor_list=["proc"],
            renderer="renderer",
            llm_service="llm",
        )

    def test_preserves_explicit_pdftext_workers(self):
        parser = MagicMock()
        parser.generate_config_dict.return_value = {"output_format": "json", "pdftext_workers": 3}
        parser.get_processors.return_value = ["proc"]
        parser.get_renderer.return_value = "renderer"
        parser.get_llm_service.return_value = "llm"

        with (
            patch("marker_mcp.conversion_service.ConfigParser", return_value=parser),
            patch("marker_mcp.conversion_service.PdfConverter", return_value="converter") as mock_pdf_converter,
        ):
            result = svc._build_converter({"output_format": "json", "pdftext_workers": 3})

        assert result == "converter"
        mock_pdf_converter.assert_called_once_with(
            config={"output_format": "json", "pdftext_workers": 3},
            artifact_dict=svc._MODELS,
            processor_list=["proc"],
            renderer="renderer",
            llm_service="llm",
        )

    def test_applies_low_vram_gpu_profile_to_converter_config(self):
        parser = MagicMock()
        parser.generate_config_dict.return_value = {
            "output_format": "json",
            "gpu_memory_profile": "low-vram",
        }
        parser.get_processors.return_value = ["proc"]
        parser.get_renderer.return_value = "renderer"
        parser.get_llm_service.return_value = "llm"

        with (
            patch("marker_mcp.conversion_service.ConfigParser", return_value=parser),
            patch("marker_mcp.conversion_service.PdfConverter", return_value="converter") as mock_pdf_converter,
        ):
            result = svc._build_converter({"output_format": "json", "gpu_memory_profile": "low-vram"})

        assert result == "converter"
        mock_pdf_converter.assert_called_once_with(
            config={
                "output_format": "json",
                "layout_batch_size": 2,
                "detection_batch_size": 1,
                "recognition_batch_size": 16,
                "ocr_error_batch_size": 4,
                "table_rec_batch_size": 4,
                "equation_batch_size": 4,
                "pdftext_workers": 8,
            },
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

    async def test_retries_with_cpu_when_cuda_oom(self, tmp_pdf_file, monkeypatch):
        original_models = {"device": "cuda"}
        cpu_models = {"device": "cpu"}
        restored_models = {"device": "cuda-restored"}
        svc._MODELS = original_models
        seen_attempts: list[tuple[str | None, dict]] = []

        def fake_convert(filepath, options):
            seen_attempts.append((os.environ.get("TORCH_DEVICE"), svc._MODELS))
            if len(seen_attempts) == 1:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return "# CPU fallback output"

        monkeypatch.delenv("TORCH_DEVICE", raising=False)

        with (
            patch("marker_mcp.conversion_service._convert_sync", side_effect=fake_convert),
            patch(
                "marker_mcp.conversion_service._create_models",
                side_effect=[cpu_models, restored_models],
            ) as mock_create,
        ):
            result = await svc.convert_file(str(tmp_pdf_file))

        assert result == "# CPU fallback output"
        assert seen_attempts == [(None, original_models), ("cpu", cpu_models)]
        assert mock_create.call_args_list == [call(device="cpu"), call(device=None)]
        assert svc._MODELS is restored_models

    async def test_warns_and_falls_back_when_llm_conversion_fails(self, tmp_pdf_file):
        attempts: list[dict] = []

        def fake_convert(filepath, options):
            attempts.append(dict(options))
            if options.get("use_llm"):
                raise RuntimeError("LLM service unavailable")
            return "# Fallback output"

        with patch("marker_mcp.conversion_service._convert_sync", side_effect=fake_convert):
            with pytest.warns(UserWarning, match="LLM"):
                result = await svc.convert_file(str(tmp_pdf_file), {"use_llm": True})

        assert result == "# Fallback output"
        assert attempts[0]["use_llm"] is True
        assert "use_llm" not in attempts[1]

    async def test_run_conversion_with_fallbacks_releases_memory_after_success(self, tmp_pdf_file):
        def fake_convert(filepath, options):
            return "# ok"

        with patch("marker_mcp.conversion_service._release_torch_memory") as mock_release:
            result = await svc._run_conversion_with_fallbacks(str(tmp_pdf_file), {}, fake_convert)

        assert result == "# ok"
        mock_release.assert_called_once()

    async def test_uses_page_tiling_when_max_page_height_px_set(self, tmp_pdf_file):
        payload = {"text": "# Tiled", "metadata": {}, "warnings": [], "assets": {}}

        with patch(
            "marker_mcp.conversion_service._convert_file_with_page_tiling",
            new=AsyncMock(return_value=payload),
        ) as mock_tiled:
            result = await svc.convert_file(str(tmp_pdf_file), {"max_page_height_px": 1600})

        assert result == "# Tiled"
        mock_tiled.assert_awaited_once()


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

    async def test_convert_bytes_retries_after_cuda_oom(self, minimal_pdf_bytes):
        call_options: list[dict | None] = []

        async def fake_convert_file(filepath, options=None):
            call_options.append(options.copy() if options else None)
            if len(call_options) == 1:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return "# CPU fallback output"

        with patch("marker_mcp.conversion_service.convert_file", side_effect=fake_convert_file):
            result = await svc.convert_bytes(minimal_pdf_bytes, "test.pdf")

        assert result == "# CPU fallback output"
        assert len(call_options) == 2


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
    def test_chunking_dependency_pypdf_available(self):
        import pypdf

        assert pypdf is not None

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


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

class TestChunkHelpers:
    def test_chunk_page_ranges_for_large_documents(self):
        assert svc._chunk_page_ranges(total_pages=9, max_pages_per_chunk=4) == [
            "0-3",
            "4-7",
            "8-8",
        ]

    def test_tile_vertical_bounds_for_large_pages(self):
        assert svc._tile_vertical_bounds(height=900, max_tile_height=400, overlap_px=96) == [
            (0, 400),
            (304, 704),
            (608, 900),
        ]

    def test_page_tile_overlap_scales_down_for_small_tiles(self):
        assert svc._page_tile_overlap_px(250) == 41
        assert svc._page_tile_overlap_px(400) == 66
        assert svc._page_tile_overlap_px(1600) == 96

    def test_parse_and_chunk_explicit_page_range(self):
        assert svc._parse_page_range("0,2,4-6") == [0, 2, 4, 5, 6]
        assert svc._chunk_requested_page_range("0,2,4-6", max_pages_per_chunk=2) == [
            "0,2",
            "4-5",
            "6",
        ]

    def test_merge_tiled_texts_trims_duplicate_overlap_lines(self):
        assert svc._merge_tiled_texts(
            [
                "# Page 1\nShared line",
                "Shared line\nContinuation",
                "Continuation\nFinal line",
            ]
        ) == "# Page 1\nShared line\n\nContinuation\n\nFinal line"

    def test_merge_tiled_texts_strips_per_tile_page_separators(self):
        assert svc._merge_tiled_texts(
            [
                "{0}------------------------------------------------\n\n# Page 1\nShared line",
                "{0}------------------------------------------------\n\nShared line\nContinuation",
            ]
        ) == "# Page 1\nShared line\n\nContinuation"

    def test_cleanup_tiled_page_text_prefers_fuller_overlap_block(self):
        assert svc._cleanup_tiled_page_text(
            "\n\n".join(
                [
                    "# Abstract",
                    "Any claim of precise risk carries a margin of error.",
                    (
                        "Any claim of precise risk carries a margin of error, and that margin "
                        "itself is uncertain in an infinite regress of doubt."
                    ),
                ]
            )
        ) == (
            "# Abstract\n\n"
            "Any claim of precise risk carries a margin of error, and that margin itself is "
            "uncertain in an infinite regress of doubt."
        )

    def test_cleanup_tiled_page_text_drops_repeated_phrase_gibberish(self):
        assert svc._cleanup_tiled_page_text(
            "\n\n".join(
                [
                    "# Intro",
                    "the contract of the contract of the contract of the contract of the contract",
                    "A clean paragraph remains.",
                ]
            )
        ) == "# Intro\n\nA clean paragraph remains."

    def test_merge_chunk_texts_preserves_chunk_order(self):
        assert svc._merge_chunk_texts(["# Chunk 1", "# Chunk 2", "# Chunk 3"]) == (
            "# Chunk 1\n\n# Chunk 2\n\n# Chunk 3"
        )

    def test_gpu_oom_tile_height_attempts_descend_to_floor(self):
        assert svc._gpu_oom_tile_height_attempts() == [1600, 800, 400]


class TestChunkedConversionOrchestration:
    async def test_convert_file_chunks_large_pdfs_when_max_pages_per_chunk_set(self, tmp_pdf_file):
        calls: list[dict] = []
        chunk_outputs = {
            "0-1": "# Chunk 1",
            "2-3": "# Chunk 2",
            "4-4": "# Chunk 3",
        }

        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            calls.append(
                {
                    "filepath": filepath,
                    "options": dict(options),
                    "sync_converter": sync_converter,
                    "warnings_list": warnings_list,
                }
            )
            return chunk_outputs[options["page_range"]]

        with (
            patch("marker_mcp.conversion_service._get_pdf_page_count", return_value=5, create=True),
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
        ):
            result = await svc.convert_file(
                str(tmp_pdf_file),
                {"max_pages_per_chunk": 2, "force_ocr": True},
            )

        assert result == "# Chunk 1\n\n# Chunk 2\n\n# Chunk 3"
        assert [call["options"]["page_range"] for call in calls] == ["0-1", "2-3", "4-4"]
        assert all(call["options"]["force_ocr"] is True for call in calls)
        assert all("max_pages_per_chunk" not in call["options"] for call in calls)
        assert all(call["sync_converter"] is svc._convert_sync for call in calls)

    async def test_convert_file_in_chunks_releases_memory_between_chunks(self, tmp_pdf_file):
        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            return f"# {options['page_range']}"

        with (
            patch("marker_mcp.conversion_service._get_pdf_page_count", return_value=5, create=True),
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
            patch("marker_mcp.conversion_service._release_torch_memory") as mock_release,
        ):
            await svc._convert_file_in_chunks(
                str(tmp_pdf_file),
                {"max_pages_per_chunk": 2},
                svc._convert_sync,
            )

        assert mock_release.call_count == 3

    async def test_convert_file_chunks_explicit_page_range_when_max_pages_per_chunk_set(self, tmp_pdf_file):
        calls: list[dict] = []
        chunk_outputs = {
            "0-1": "# Chunk 1",
            "2-3": "# Chunk 2",
            "4-5": "# Chunk 3",
        }

        async def fake_run(filepath, options, sync_converter, warnings_list=None):
            calls.append(
                {
                    "filepath": filepath,
                    "options": dict(options),
                    "sync_converter": sync_converter,
                }
            )
            return chunk_outputs[options["page_range"]]

        with patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run):
            result = await svc.convert_file(
                str(tmp_pdf_file),
                {"page_range": "0-5", "max_pages_per_chunk": 2, "force_ocr": True},
            )

        assert result == "# Chunk 1\n\n# Chunk 2\n\n# Chunk 3"
        assert [call["options"]["page_range"] for call in calls] == ["0-1", "2-3", "4-5"]
        assert all(call["options"]["force_ocr"] is True for call in calls)
        assert all("max_pages_per_chunk" not in call["options"] for call in calls)
        assert all(call["sync_converter"] is svc._convert_sync for call in calls)

    async def test_single_page_chunk_uses_gpu_tiling_before_cpu_fallback(self, tmp_pdf_file):
        chunk_outputs = {
            "1": {
                "text": "# Page 2 via tiling",
                "metadata": {"page_count": 1},
                "warnings": [],
                "assets": {},
            },
            "2": {
                "text": "# Page 3",
                "metadata": {"page_count": 1},
                "warnings": [],
                "assets": {},
            },
        }
        run_calls: list[dict] = []

        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            run_calls.append(
                {
                    "page_range": options.get("page_range"),
                    "gpu_memory_profile": options.get("gpu_memory_profile"),
                    "allow_cpu_fallback": allow_cpu_fallback,
                }
            )
            if options.get("page_range") == "1" and allow_cpu_fallback is False:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return chunk_outputs[options["page_range"]]

        tiled_payload = {
            "text": "# Page 2 via tiling",
            "metadata": {"page_count": 1},
            "warnings": [],
            "assets": {},
        }

        with (
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
            patch(
                "marker_mcp.conversion_service._convert_file_with_page_tiling",
                new=AsyncMock(return_value=tiled_payload),
            ) as mock_tiling,
        ):
            result = await svc.convert_file_result(
                str(tmp_pdf_file),
                {"page_range": "1-2", "max_pages_per_chunk": 1},
            )

        assert result["text"] == "# Page 2 via tiling\n\n# Page 3"
        assert result["metadata"]["page_count"] == 2
        assert run_calls == [
            {"page_range": "1", "gpu_memory_profile": None, "allow_cpu_fallback": False},
            {"page_range": "1", "gpu_memory_profile": "low-vram", "allow_cpu_fallback": False},
            {"page_range": "2", "gpu_memory_profile": None, "allow_cpu_fallback": False},
        ]
        mock_tiling.assert_awaited_once()

    async def test_single_page_chunk_persists_low_vram_gpu_profile_after_success(self, tmp_pdf_file):
        run_calls: list[dict] = []

        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            run_calls.append(
                {
                    "page_range": options.get("page_range"),
                    "gpu_memory_profile": options.get("gpu_memory_profile"),
                    "allow_cpu_fallback": allow_cpu_fallback,
                }
            )
            if options.get("page_range") == "1" and options.get("gpu_memory_profile") is None:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return {
                "text": f"# Page {options['page_range']}",
                "metadata": {"page_count": 1},
                "warnings": [],
                "assets": {},
            }

        with (
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
            patch(
                "marker_mcp.conversion_service._convert_file_with_page_tiling",
                new=AsyncMock(),
            ) as mock_tiling,
        ):
            result = await svc.convert_file_result(
                str(tmp_pdf_file),
                {"page_range": "1-2", "max_pages_per_chunk": 1},
            )

        assert result["text"] == "# Page 1\n\n# Page 2"
        assert result["metadata"]["page_count"] == 2
        assert run_calls == [
            {"page_range": "1", "gpu_memory_profile": None, "allow_cpu_fallback": False},
            {"page_range": "1", "gpu_memory_profile": "low-vram", "allow_cpu_fallback": False},
            {"page_range": "2", "gpu_memory_profile": "low-vram", "allow_cpu_fallback": False},
        ]
        mock_tiling.assert_not_awaited()

    async def test_single_page_chunk_retries_smaller_gpu_tiles_before_cpu(self, tmp_pdf_file):
        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            if allow_cpu_fallback is False:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return {
                "text": "# CPU fallback",
                "metadata": {"page_count": 1},
                "warnings": [],
                "assets": {},
            }

        tiling_calls: list[int] = []

        async def fake_tiling(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            tiling_calls.append(options["max_page_height_px"])
            if options["max_page_height_px"] == 1600:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return {
                "text": "# Page via smaller tiling",
                "metadata": {"page_count": 1},
                "warnings": [],
                "assets": {},
            }

        with (
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
            patch("marker_mcp.conversion_service._convert_file_with_page_tiling", side_effect=fake_tiling),
        ):
            result = await svc.convert_file_result(
                str(tmp_pdf_file),
                {"page_range": "1", "max_pages_per_chunk": 1},
            )

        assert result["text"] == "# Page via smaller tiling"
        assert result["metadata"]["page_count"] == 1
        assert tiling_calls == [1600, 800]

    async def test_convert_file_result_merges_chunked_structured_payloads(self, tmp_pdf_file):
        chunk_results = {
            "0-1": {
                "text": "# Chunk 1",
                "metadata": {"page_count": 2, "title": "Long PDF"},
                "warnings": ["chunk-1 warning"],
                "assets": {
                    "images": [
                        {"filename": "page-1.png", "path": "artifacts/images/page-1.png"},
                    ]
                },
            },
            "2-3": {
                "text": "# Chunk 2",
                "metadata": {"page_count": 2, "author": "Ada"},
                "warnings": ["chunk-2 warning"],
                "assets": {
                    "images": [
                        {"filename": "page-3.png", "path": "artifacts/images/page-3.png"},
                    ],
                    "attachments": [
                        {"filename": "table.csv", "path": "artifacts/files/table.csv"},
                    ],
                },
            },
        }

        async def fake_run(filepath, options, sync_converter, warnings_list=None):
            return chunk_results[options["page_range"]]

        with (
            patch("marker_mcp.conversion_service._get_pdf_page_count", return_value=4, create=True),
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
        ):
            result = await svc.convert_file_result(
                str(tmp_pdf_file),
                {"max_pages_per_chunk": 2},
            )

        assert result == {
            "text": "# Chunk 1\n\n# Chunk 2",
            "metadata": {
                "page_count": 4,
                "title": "Long PDF",
                "author": "Ada",
            },
            "warnings": ["chunk-1 warning", "chunk-2 warning"],
            "assets": {
                "images": [
                    {"filename": "page-1.png", "path": "artifacts/images/page-1.png"},
                    {"filename": "page-3.png", "path": "artifacts/images/page-3.png"},
                ],
                "attachments": [
                    {"filename": "table.csv", "path": "artifacts/files/table.csv"},
                ],
            },
        }


class TestPageTilingOrchestration:
    async def test_convert_file_result_uses_page_tiling_when_configured(self, tmp_pdf_file):
        payload = {
            "text": "# Tiled",
            "metadata": {"page_count": 2},
            "warnings": [],
            "assets": {},
        }

        with patch(
            "marker_mcp.conversion_service._convert_file_with_page_tiling",
            new=AsyncMock(return_value=payload),
        ) as mock_tiled:
            result = await svc.convert_file_result(str(tmp_pdf_file), {"max_page_height_px": 1600})

        assert result["text"] == "# Tiled"
        assert result["metadata"]["page_count"] == 2
        mock_tiled.assert_awaited_once()

    async def test_convert_file_with_page_tiling_releases_memory_between_tiles(self, tmp_pdf_file):
        tile_result = {
            "text": "# Tile",
            "metadata": {},
            "warnings": [],
            "assets": {},
        }

        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            return tile_result

        with (
            patch(
                "marker_mcp.conversion_service._render_pdf_page_image",
                return_value=Image.new("RGB", (400, 900), color="white"),
            ),
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
            patch("marker_mcp.conversion_service._release_torch_memory") as mock_release,
        ):
            result = await svc._convert_file_with_page_tiling(
                str(tmp_pdf_file),
                {"page_range": "0", "max_page_height_px": 400},
                svc._convert_sync_result,
            )

        assert result["metadata"]["page_count"] == 1
        assert mock_release.call_count == 4

    async def test_convert_file_with_page_tiling_reinserts_page_separators_once_per_page(self, tmp_pdf_file):
        tile_options: list[dict] = []
        tile_index = 0

        async def fake_run(filepath, options, sync_converter, warnings_list=None, allow_cpu_fallback=True):
            nonlocal tile_index
            tile_options.append(dict(options))
            tile_index += 1
            return {
                "text": "{0}------------------------------------------------\n\n# Tile "
                f"{tile_index}",
                "metadata": {},
                "warnings": [],
                "assets": {},
            }

        with (
            patch(
                "marker_mcp.conversion_service._render_pdf_page_image",
                side_effect=lambda *args, **kwargs: Image.new("RGB", (400, 300), color="white"),
            ),
            patch("marker_mcp.conversion_service._run_conversion_with_fallbacks", side_effect=fake_run),
        ):
            result = await svc._convert_file_with_page_tiling(
                str(tmp_pdf_file),
                {"page_range": "0-1", "max_page_height_px": 400, "paginate_output": True},
                svc._convert_sync_result,
            )

        assert all("paginate_output" not in opts for opts in tile_options)
        assert result["text"] == (
            "{0}------------------------------------------------\n\n# Tile 1\n\n"
            "{1}------------------------------------------------\n\n# Tile 2"
        )


class TestStructuredFigureExport:
    async def test_convert_file_result_serializes_marker_figures_into_assets(self, tmp_pdf_file):
        figure = _make_test_figure()
        markdown = "![Figure](_page_0_Picture_2.jpeg)"

        with (
            patch("marker_mcp.conversion_service._build_converter") as mock_build,
            patch(
                "marker_mcp.conversion_service.text_from_rendered",
                return_value=(
                    markdown,
                    {"page_count": 1, "title": "Figure doc"},
                    {"_page_0_Picture_2.jpeg": figure},
                ),
            ),
        ):
            mock_build.return_value = MagicMock(return_value=MagicMock())
            result = await svc.convert_file_result(str(tmp_pdf_file))

        assert result["text"] == markdown
        assert result["metadata"] == {"page_count": 1, "title": "Figure doc"}
        assert "_page_0_Picture_2.jpeg" not in result["metadata"]

        [asset] = result["assets"]["images"]
        assert asset["filename"] == "_page_0_Picture_2.jpeg"
        assert asset.get("path", "").endswith("_page_0_Picture_2.jpeg") or "content_base64" in asset
        if "content_base64" in asset:
            assert base64.b64decode(asset["content_base64"])

        assert json.loads(json.dumps(result)) == result
