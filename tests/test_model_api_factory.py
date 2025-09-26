import pytest
from unittest.mock import MagicMock
from src.model_api_factory import ModelAPIFactory

@pytest.fixture
def mock_registry(mocker):
    """A fixture to have a clean, mocked registry for each test."""

    # Store original registry to restore it later
    original_registry = ModelAPIFactory._model_registry.copy()

    # Create mock objects for each API
    mock_openai = mocker.MagicMock()
    mock_llama = mocker.MagicMock()
    mock_gemini = mocker.MagicMock()
    mock_anthropic = mocker.MagicMock()

    # Clear the factory's registry and register our mocks
    ModelAPIFactory._model_registry = {}
    ModelAPIFactory.register_model('openai', mock_openai)
    ModelAPIFactory.register_model('meta', mock_llama)
    ModelAPIFactory.register_model('google', mock_gemini)
    ModelAPIFactory.register_model('anthropic', mock_anthropic)
    ModelAPIFactory.register_model('deepseek', mock_llama)
    ModelAPIFactory.register_model('openrouter', mock_llama)

    yield {
        'openai': mock_openai,
        'meta': mock_llama,
        'google': mock_gemini,
        'anthropic': mock_anthropic,
        'deepseek': mock_llama,
        'openrouter': mock_llama
    }

    # Restore original registry after the test completes
    ModelAPIFactory._model_registry = original_registry

@pytest.mark.parametrize("model_name, expected_provider", [
    ("gpt-4o", "openai"),
    ("meta-llama/Llama-3", "meta"),
    ("gemini-1.5-flash", "google"),
    ("claude-3-opus-20240229", "anthropic"),
    ("deepseek-coder", "deepseek"),
    ("google/gemini", "openrouter"),
    ("unknown-model", "llama") # Fallback case
])
def test_get_provider_from_model(model_name, expected_provider):
    """Tests that the factory correctly deduces the provider from the model string."""
    assert ModelAPIFactory.get_provider_from_model(model_name) == expected_provider

def test_get_model_api_by_provider(mock_registry):
    """Tests that the factory returns the correct model API instance for each provider."""
    for provider, mock_api in mock_registry.items():
        instance = ModelAPIFactory.get_model_api(provider=provider)
        # The factory should call the registered mock with the default model
        mock_api.assert_called_with(model='meta-llama/llama-3.1-405b-instruct:free')
        # The factory should return the result of that call
        assert instance == mock_api.return_value
        # Reset mock for the next iteration, especially for shared mocks
        mock_api.reset_mock()

def test_get_model_api_with_kwargs(mock_registry):
    """Tests that the factory passes additional keyword arguments to the model API constructor."""
    api_key = "test-api-key"
    mock_openai = mock_registry['openai']

    ModelAPIFactory.get_model_api(provider='openai', api_key=api_key)

    # Check that the constructor was called with the model and the extra kwarg
    mock_openai.assert_called_once_with(model='meta-llama/llama-3.1-405b-instruct:free', api_key=api_key)

def test_get_model_api_unknown_provider(mock_registry):
    """Tests that the factory raises a ValueError for an unrecognized provider."""
    with pytest.raises(ValueError, match="Unknown provider or model: non_existent_provider"):
        ModelAPIFactory.get_model_api(provider='non_existent_provider')

def test_custom_model_registration(mocker):
    """Tests that a custom model can be registered and instantiated by the factory."""
    # Use a clean registry for this test
    original_registry = ModelAPIFactory._model_registry.copy()
    ModelAPIFactory._model_registry = {}

    CustomModelAPI = mocker.MagicMock()
    ModelAPIFactory.register_model('custom', CustomModelAPI)

    instance = ModelAPIFactory.get_model_api(provider='custom')

    assert instance == CustomModelAPI.return_value
    CustomModelAPI.assert_called_once_with(model='meta-llama/llama-3.1-405b-instruct:free')

    # Restore original registry
    ModelAPIFactory._model_registry = original_registry