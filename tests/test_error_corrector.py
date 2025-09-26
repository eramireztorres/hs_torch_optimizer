import pytest
from unittest.mock import MagicMock, mock_open
from src.error_corrector import ErrorCorrector

@pytest.fixture
def mock_llm_api():
    """Provides a mock for the BaseModelAPI."""
    return MagicMock()

@pytest.fixture
def mock_prompt_file(mocker):
    """Mocks the file reading for the prompt template."""
    prompt_content = "Faulty Code: ${faulty_code}\nError: ${error_msg}"
    return mocker.patch('builtins.open', mock_open(read_data=prompt_content))

def test_error_corrector_initialization(mock_llm_api, mock_prompt_file):
    """
    Tests that the ErrorCorrector class initializes correctly by loading the
    prompt template from the specified path.
    """
    prompt_path = "dummy/path/prompt.txt"
    corrector = ErrorCorrector(llm_model=mock_llm_api, prompt_path=prompt_path)

    # Verify that the file was opened at the correct path
    mock_prompt_file.assert_called_once_with(prompt_path, 'r')

    # Verify that the prompt template was loaded
    assert corrector.prompt_template == "Faulty Code: ${faulty_code}\nError: ${error_msg}"

def test_get_error_fix(mock_llm_api, mock_prompt_file):
    """
    Tests that the get_error_fix method correctly formats the prompt,
    calls the LLM API, and returns the response.
    """
    # Arrange
    prompt_path = "dummy/path/prompt.txt"
    corrector = ErrorCorrector(llm_model=mock_llm_api, prompt_path=prompt_path)

    faulty_code = "print('hello world'"
    error_msg = "SyntaxError: unexpected EOF while parsing"
    expected_prompt = "Faulty Code: print('hello world'\nError: SyntaxError: unexpected EOF while parsing"
    expected_fix = "print('hello world')"

    mock_llm_api.get_response.return_value = expected_fix

    # Act
    actual_fix = corrector.get_error_fix(faulty_code, error_msg)

    # Assert
    # Check that the LLM's get_response method was called with the correctly formatted prompt
    mock_llm_api.get_response.assert_called_once_with(expected_prompt)

    # Check that the method returned the expected fix from the LLM
    assert actual_fix == expected_fix

def test_get_error_fix_with_no_placeholders_in_prompt(mock_llm_api, mocker):
    """
    Tests that get_error_fix handles templates without placeholders gracefully
    using safe_substitute.
    """
    # Arrange
    prompt_content = "A static prompt with no placeholders."
    mocker.patch('builtins.open', mock_open(read_data=prompt_content))

    corrector = ErrorCorrector(llm_model=mock_llm_api, prompt_path="any/path.txt")

    mock_llm_api.get_response.return_value = "some response"

    # Act
    fix = corrector.get_error_fix("some code", "some error")

    # Assert
    mock_llm_api.get_response.assert_called_once_with(prompt_content)
    assert fix == "some response"