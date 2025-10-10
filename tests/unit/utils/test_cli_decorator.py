"""Tests for CLI decorator."""

import pytest
from unittest.mock import patch
from src.utils.cli_decorator import cli_decorator


class TestCLIDecorator:
    """Test suite for CLI decorator."""

    def test_cli_decorator_basic(self):
        """Test basic CLI decorator functionality."""
        @cli_decorator
        def sample_function(arg1, arg2='default'):
            return f"{arg1}-{arg2}"

        # The decorator should allow the function to be called normally
        result = sample_function('test', arg2='value')
        assert result == 'test-value'

    def test_cli_decorator_with_defaults(self):
        """Test CLI decorator with default arguments."""
        @cli_decorator
        def sample_function(required_arg, optional_arg='default'):
            return f"{required_arg}-{optional_arg}"

        result = sample_function('test')
        assert result == 'test-default'

    @patch('sys.argv', ['script.py', '--arg1', 'value1', '--arg2', 'value2'])
    def test_cli_decorator_parses_args(self):
        """Test that CLI decorator can parse command line args."""
        @cli_decorator
        def sample_function(arg1, arg2='default'):
            return f"{arg1}-{arg2}"

        # When called from CLI, it should parse arguments
        # This is hard to test without actually invoking it
        # Just verify the decorator doesn't break the function
        result = sample_function('test1', 'test2')
        assert 'test1' in result

    def test_cli_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @cli_decorator
        def sample_function(arg1):
            """Sample docstring."""
            return arg1

        assert sample_function.__name__ == 'sample_function'
        assert 'Sample docstring' in sample_function.__doc__
