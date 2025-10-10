"""Tests for ErrorCorrector."""

import pytest
from unittest.mock import Mock, MagicMock
from src.core.error_corrector import ErrorCorrector


@pytest.mark.unit
class TestErrorCorrector:
    """Test suite for ErrorCorrector."""

    def test_init(self, temp_dir):
        """Test initialization."""
        prompt_path = temp_dir / "error_prompt.txt"
        prompt_path.write_text("Fix this code: ${faulty_code}\nError: ${error_msg}")

        mock_llm = Mock()
        corrector = ErrorCorrector(mock_llm, str(prompt_path))

        assert corrector.llm_model == mock_llm
        assert "${faulty_code}" in corrector.prompt_template

    def test_get_error_fix(self, temp_dir):
        """Test getting error fix from LLM."""
        prompt_path = temp_dir / "error_prompt.txt"
        prompt_path.write_text("Fix this code: ${faulty_code}\nError: ${error_msg}")

        mock_llm = Mock()
        mock_llm.get_response = Mock(return_value="Fixed code here")

        corrector = ErrorCorrector(mock_llm, str(prompt_path))
        result = corrector.get_error_fix("bad code", "syntax error")

        assert result == "Fixed code here"
        mock_llm.get_response.assert_called_once()

    def test_get_error_fix_substitutes_variables(self, temp_dir):
        """Test that error fix correctly substitutes template variables."""
        prompt_path = temp_dir / "error_prompt.txt"
        prompt_path.write_text("Code: ${faulty_code}\nError: ${error_msg}")

        mock_llm = Mock()
        mock_llm.get_response = Mock(return_value="fixed")

        corrector = ErrorCorrector(mock_llm, str(prompt_path))
        corrector.get_error_fix("import torch", "ImportError")

        # Check that the substituted prompt was passed
        call_args = mock_llm.get_response.call_args[0][0]
        assert "import torch" in call_args
        assert "ImportError" in call_args

    def test_load_prompt_file_not_found(self):
        """Test error when prompt file doesn't exist."""
        mock_llm = Mock()
        with pytest.raises(FileNotFoundError):
            ErrorCorrector(mock_llm, "/nonexistent/prompt.txt")

    def test_get_error_fix_with_multiline_code(self, temp_dir):
        """Test error fix with multiline code."""
        prompt_path = temp_dir / "error_prompt.txt"
        prompt_path.write_text("Fix: ${faulty_code}\nError: ${error_msg}")

        faulty_code = """
def broken():
    x = 1
    return x +
"""
        error_msg = "SyntaxError: invalid syntax"

        mock_llm = Mock()
        mock_llm.get_response = Mock(return_value="def fixed():\n    return 1")

        corrector = ErrorCorrector(mock_llm, str(prompt_path))
        result = corrector.get_error_fix(faulty_code, error_msg)

        assert "fixed" in result
