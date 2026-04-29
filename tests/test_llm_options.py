"""Unit tests for _llm_options_from_env() in conversion_service."""

import pytest

from marker_mcp.conversion_service import _llm_options_from_env


class TestNoEnvVars:
    def test_returns_empty_dict(self, clean_llm_env):
        result = _llm_options_from_env()
        assert result == {}


class TestExplicitService:
    def test_marker_llm_service_is_passed_through(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("MARKER_LLM_SERVICE", "marker.services.openai.OpenAIService")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.openai.OpenAIService"

    def test_marker_llm_service_overrides_ollama(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("MARKER_LLM_SERVICE", "my.custom.Service")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        result = _llm_options_from_env()
        # Explicit service wins; ollama_base_url is still set (useful config),
        # but llm_service is the explicit one.
        assert result["llm_service"] == "my.custom.Service"
        assert result.get("ollama_base_url") == "http://localhost:11434"


class TestOllamaDetection:
    def test_ollama_base_url_only(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker_mcp.ollama_service.OllamaService"
        assert result["ollama_base_url"] == "http://localhost:11434"
        assert "ollama_model" not in result

    def test_ollama_model_only(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker_mcp.ollama_service.OllamaService"
        assert result["ollama_model"] == "gemma4:31b-cloud"
        assert "ollama_base_url" not in result

    def test_ollama_both_vars(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker_mcp.ollama_service.OllamaService"
        assert result["ollama_base_url"] == "http://localhost:11434"
        assert result["ollama_model"] == "gemma4:31b-cloud"

    def test_ollama_batch_env_vars(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        monkeypatch.setenv("OLLAMA_BATCH_SIZE", "6")
        monkeypatch.setenv("OLLAMA_BATCH_WAIT_MS", "120")
        result = _llm_options_from_env()
        assert result["ollama_batch_size"] == 6
        assert result["ollama_batch_wait_ms"] == 120

    def test_ollama_beats_openai_when_both_set(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker_mcp.ollama_service.OllamaService"
        # OpenAI key is still passed through (for potential use)
        assert result.get("openai_api_key") == "sk-test"


class TestOpenAIDetection:
    def test_openai_api_key_only(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.openai.OpenAIService"
        assert result["openai_api_key"] == "sk-test-key"

    def test_openai_with_custom_base_url(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-v3")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.openai.OpenAIService"
        assert result["openai_base_url"] == "https://api.deepseek.com/v1"
        assert result["openai_model"] == "deepseek-v3"

    def test_openai_model_without_key_not_set_as_service(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        result = _llm_options_from_env()
        # No OPENAI_API_KEY → no llm_service set
        assert "llm_service" not in result
        assert result.get("openai_model") == "gpt-4o"


class TestClaudeDetection:
    def test_claude_api_key(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.claude.ClaudeService"
        assert result["claude_api_key"] == "sk-ant-test"

    def test_claude_with_model(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
        monkeypatch.setenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.claude.ClaudeService"
        assert result["claude_model_name"] == "claude-3-5-sonnet-20241022"

    def test_claude_does_not_override_openai(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.openai.OpenAIService"
        # Claude key still passed through
        assert result.get("claude_api_key") == "sk-ant-test"


class TestServicePriority:
    def test_explicit_beats_everything(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("MARKER_LLM_SERVICE", "custom.MyService")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant")
        result = _llm_options_from_env()
        assert result["llm_service"] == "custom.MyService"

    def test_ollama_beats_openai_and_claude(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker_mcp.ollama_service.OllamaService"

    def test_openai_beats_claude(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant")
        result = _llm_options_from_env()
        assert result["llm_service"] == "marker.services.openai.OpenAIService"
