"""Tests for ModelAPIFactory."""

import pytest
import os
from src.api.model_api_factory import ModelAPIFactory
from src.api.openai_model_api import OpenAIModelAPI
from src.api.gemini_model_api import GeminiModelAPI
from src.api.anthropic_model_api import AnthropicModelAPI
from src.api.llama_model_api import LlamaModelAPI

# Skip tests that require API keys if the keys are not set in the environment
skip_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"
)
skip_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set"
)


@pytest.mark.unit
class TestModelAPIFactory:
    """Test suite for ModelAPIFactory."""

    def test_get_provider_from_model_openai(self):
        """Test provider detection for OpenAI models."""
        assert ModelAPIFactory.get_provider_from_model("gpt-4") == "openai"
        assert ModelAPIFactory.get_provider_from_model("gpt-4o-mini") == "openai"
        assert ModelAPIFactory.get_provider_from_model("o1-preview") == "openai"
        assert ModelAPIFactory.get_provider_from_model("o3-mini") == "openai"

    def test_get_provider_from_model_anthropic(self):
        """Test provider detection for Anthropic models."""
        assert ModelAPIFactory.get_provider_from_model("claude-3-opus") == "anthropic"
        assert ModelAPIFactory.get_provider_from_model("claude-sonnet") == "anthropic"

    def test_get_provider_from_model_google(self):
        """Test provider detection for Google models."""
        assert ModelAPIFactory.get_provider_from_model("gemini-pro") == "google"
        assert ModelAPIFactory.get_provider_from_model("gemini-2.0-flash") == "google"

    def test_get_provider_from_model_openrouter(self):
        """Test provider detection for OpenRouter models."""
        assert (
            ModelAPIFactory.get_provider_from_model("meta-llama/llama-3.1")
            == "openrouter"
        )
        assert (
            ModelAPIFactory.get_provider_from_model("deepseek/deepseek-chat")
            == "openrouter"
        )
        assert ModelAPIFactory.get_provider_from_model("qwen/qwen-2") == "openrouter"

    def test_get_provider_from_model_fallback(self):
        """Test fallback to llama for unknown models."""
        assert ModelAPIFactory.get_provider_from_model("unknown-model") == "llama"

    @skip_openai
    def test_get_model_api_openai(self):
        """Test getting OpenAI API instance."""
        api = ModelAPIFactory.get_model_api(provider="openai", model="gpt-4o-mini")
        assert isinstance(api, OpenAIModelAPI)
        assert api.model == "gpt-4o-mini"

    @skip_google
    def test_get_model_api_gemini(self):
        """Test getting Gemini API instance."""
        api = ModelAPIFactory.get_model_api(provider="google", model="gemini-2.0-flash")
        assert isinstance(api, GeminiModelAPI)
        assert api.model == "gemini-2.0-flash"

    def test_get_model_api_anthropic(self):
        """Test getting Anthropic API instance."""
        api = ModelAPIFactory.get_model_api(
            provider="anthropic", model="claude-3-sonnet"
        )
        assert isinstance(api, AnthropicModelAPI)
        assert api.model == "claude-3-sonnet"

    def test_get_model_api_llama(self):
        """Test getting Llama API instance."""
        api = ModelAPIFactory.get_model_api(provider="meta", model="llama-3.1")
        assert isinstance(api, LlamaModelAPI)
        assert api.model == "llama-3.1"

    @skip_openai
    def test_get_model_api_auto_detect_provider(self):
        """Test automatic provider detection from model name."""
        api = ModelAPIFactory.get_model_api(provider=None, model="gpt-4o-mini")
        assert isinstance(api, OpenAIModelAPI)

    def test_get_model_api_invalid_provider(self):
        """Test error handling for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelAPIFactory.get_model_api(provider="invalid", model="test")

    def test_register_model(self):
        """Test registering a new model provider."""

        class CustomAPI:
            def __init__(self, model=None):
                self.model = model

        ModelAPIFactory.register_model("custom", CustomAPI)
        api = ModelAPIFactory.get_model_api(provider="custom", model="custom-model")
        assert isinstance(api, CustomAPI)
