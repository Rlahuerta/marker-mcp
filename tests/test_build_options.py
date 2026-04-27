"""Unit tests for _build_options() in mcp_server and get_status() in conversion_service."""

import runpy
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import marker_mcp.conversion_service as svc
import marker_mcp.mcp_server as server
from marker_mcp.mcp_server import _build_options


class TestBuildOptions:
    def test_defaults(self, clean_llm_env):
        opts = _build_options("markdown", None, False, False, False)
        assert opts == {"output_format": "markdown"}

    def test_output_format_html(self, clean_llm_env):
        opts = _build_options("html", None, False, False, False)
        assert opts["output_format"] == "html"

    def test_page_range_included_when_set(self, clean_llm_env):
        opts = _build_options("markdown", "0,5-10", False, False, False)
        assert opts["page_range"] == "0,5-10"

    def test_page_range_omitted_when_none(self, clean_llm_env):
        opts = _build_options("markdown", None, False, False, False)
        assert "page_range" not in opts

    def test_force_ocr_true(self, clean_llm_env):
        opts = _build_options("markdown", None, True, False, False)
        assert opts["force_ocr"] is True

    def test_force_ocr_false_omitted(self, clean_llm_env):
        opts = _build_options("markdown", None, False, False, False)
        assert "force_ocr" not in opts

    def test_paginate_output_true(self, clean_llm_env):
        opts = _build_options("markdown", None, False, True, False)
        assert opts["paginate_output"] is True

    def test_use_llm_sets_flag(self, clean_llm_env):
        opts = _build_options("markdown", None, False, False, True)
        assert opts["use_llm"] is True

    def test_use_llm_injects_ollama_config(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b")
        opts = _build_options("markdown", None, False, False, True)
        assert opts["use_llm"] is True
        assert opts["llm_service"] == "marker.services.ollama.OllamaService"
        assert opts["ollama_model"] == "gemma4:31b"

    def test_use_llm_false_no_llm_keys(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        opts = _build_options("markdown", None, False, False, False)
        # use_llm=False → LLM config not injected
        assert "use_llm" not in opts
        assert "llm_service" not in opts

    def test_all_options_combined(self, clean_llm_env):
        opts = _build_options("json", "1-3", True, True, False)
        assert opts == {
            "output_format": "json",
            "page_range": "1-3",
            "force_ocr": True,
            "paginate_output": True,
        }


class TestGetStatus:
    def test_ready_when_models_loaded(self):
        # _MODELS is set to mock dict by conftest.autouse fixture
        result = svc.get_status()
        assert result["initialized"] is True
        assert result["status"] == "ready"
        assert "ready" in result["message"].lower()

    def test_failed_when_models_none(self):
        original = svc._MODELS
        svc._MODELS = None
        try:
            result = svc.get_status()
            assert result["initialized"] is False
            assert result["status"] == "failed"
        finally:
            svc._MODELS = original

    def test_status_has_required_keys(self):
        result = svc.get_status()
        assert set(result.keys()) == {"initialized", "status", "message"}


class TestMcpServerCli:
    def test_stdio_transport_runs_stdio(self):
        runner = CliRunner()
        with patch.object(server.mcp, "run") as mock_run:
            result = runner.invoke(server.mcp_server_cli, ["--transport", "stdio"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(transport="stdio")

    def test_http_transport_passes_host_and_port(self):
        runner = CliRunner()
        with patch.object(server.mcp, "run") as mock_run:
            result = runner.invoke(
                server.mcp_server_cli,
                ["--transport", "http", "--host", "127.0.0.1", "--port", "9001"],
            )

        assert result.exit_code == 0
        mock_run.assert_called_once_with(transport="http", host="127.0.0.1", port=9001)

    def test_ocr_device_sets_env_and_reloads_service(self, monkeypatch):
        runner = CliRunner()
        monkeypatch.delenv("MARKER_MCP_OCR_DEVICE", raising=False)

        with (
            patch.object(server.mcp, "run") as mock_run,
            patch("importlib.reload") as mock_reload,
        ):
            result = runner.invoke(server.mcp_server_cli, ["--transport", "stdio", "--ocr-device", "cpu"])

        assert result.exit_code == 0
        assert server.os.environ["MARKER_MCP_OCR_DEVICE"] == "cpu"
        mock_reload.assert_called_once()
        mock_run.assert_called_once_with(transport="stdio")

    def test_model_dtype_sets_env_and_reloads_service(self, monkeypatch):
        runner = CliRunner()
        monkeypatch.delenv("MARKER_MCP_MODEL_DTYPE", raising=False)

        with (
            patch.object(server.mcp, "run") as mock_run,
            patch("importlib.reload") as mock_reload,
        ):
            result = runner.invoke(
                server.mcp_server_cli,
                ["--transport", "stdio", "--model-dtype", "bfloat16"],
            )

        assert result.exit_code == 0
        assert server.os.environ["MARKER_MCP_MODEL_DTYPE"] == "bfloat16"
        mock_reload.assert_called_once()
        mock_run.assert_called_once_with(transport="stdio")

    def test_main_guard_invokes_click_command(self):
        existing = sys.modules.pop("marker_mcp.mcp_server", None)
        try:
            with patch("click.core.Command.main", return_value=None) as mock_main:
                runpy.run_module("marker_mcp.mcp_server", run_name="__main__")
        finally:
            if existing is not None:
                sys.modules["marker_mcp.mcp_server"] = existing

        mock_main.assert_called_once()
