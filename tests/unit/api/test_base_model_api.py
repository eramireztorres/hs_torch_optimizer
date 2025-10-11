"""Tests for BaseModelAPI."""

import os
from unittest.mock import patch

import pytest

from src.api.base_model_api import BaseModelAPI


@pytest.mark.unit
class TestBaseModelAPI:
    """Test suite for BaseModelAPI."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""

        class TestAPI(BaseModelAPI):
            def get_api_key_from_env(self):
                return None

            def get_response(self, prompt, **kwargs):
                pass

        api = TestAPI(api_key="test-key-123")
        assert api.api_key == "test-key-123"

    @patch.dict(os.environ, {"TEST_API_KEY": "env-key-456"})
    def test_init_without_api_key(self):
        """Test initialization without API key uses environment."""

        class TestAPI(BaseModelAPI):
            def get_api_key_from_env(self):
                return os.getenv("TEST_API_KEY")

            def get_response(self, prompt, **kwargs):
                pass

        api = TestAPI()
        assert api.api_key == "env-key-456"

    def test_get_response_not_implemented(self):
        """Test that get_response must be implemented."""
        # Test that you cannot instantiate BaseModelAPI without implementing abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            api = BaseModelAPI(api_key="test")

    def test_get_api_key_from_env_not_implemented(self):
        """Test that get_api_key_from_env must be implemented."""
        # Test that abstract method enforcement works
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            api = BaseModelAPI(api_key="test")
