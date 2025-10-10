# Test Suite for hs-torch-optimizer

This directory contains comprehensive tests for the hs-torch-optimizer project.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and pytest configuration
├── unit/                       # Unit tests for individual modules
│   ├── api/                   # Tests for API modules
│   ├── core/                  # Tests for core optimization logic
│   ├── dynamic_models/        # Tests for dynamic model system
│   ├── training/              # Tests for training and data loading
│   └── utils/                 # Tests for utilities
├── integration/               # Integration tests
│   ├── test_end_to_end.py    # Full pipeline tests
│   └── test_cli_integration.py # CLI tests
└── fixtures/                  # Test data and fixtures

```

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
```

### Run Specific Test Files

```bash
# Run tests for a specific module
pytest tests/unit/core/test_llm_code_cleaner.py

# Run a specific test function
pytest tests/unit/core/test_llm_code_cleaner.py::TestLLMCodeCleaner::test_clean_code_with_python_fence
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Run with Verbose Output

```bash
pytest -v
```

### Run in Parallel (faster)

```bash
pip install pytest-xdist
pytest -n auto
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, test multiple components)
- `@pytest.mark.slow` - Slow tests (can be skipped for quick runs)
- `@pytest.mark.gpu` - Tests that require GPU (skip if GPU not available)

## Writing New Tests

### Test File Naming

- Unit tests: `test_<module_name>.py`
- Test classes: `class Test<ClassName>`
- Test functions: `def test_<description>()`

### Using Fixtures

Fixtures are defined in `conftest.py` and are automatically available to all tests:

```python
def test_example(sample_classification_data, temp_dir):
    """Test using fixtures."""
    # sample_classification_data and temp_dir are automatically provided
    assert 'X_train' in sample_classification_data
    assert temp_dir.exists()
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked LLM."""
    mock_llm = Mock()
    mock_llm.get_response = Mock(return_value="mocked response")

    # Use mock_llm in your test
    result = mock_llm.get_response("test prompt")
    assert result == "mocked response"
```

### Testing Exceptions

```python
import pytest

def test_exception():
    """Test that exception is raised."""
    with pytest.raises(ValueError, match="expected error message"):
        # Code that should raise ValueError
        raise ValueError("expected error message")
```

## Continuous Integration

Tests are automatically run on:
- Push to main branch
- Pull requests
- Scheduled nightly builds

## Coverage Goals

- Overall coverage: > 80%
- Core modules: > 90%
- API modules: > 70%
- Integration tests: Cover all main workflows

## Common Issues

### CUDA Errors in Tests

If you encounter CUDA errors:

```bash
# Run tests on CPU only
CUDA_VISIBLE_DEVICES="" pytest
```

### Slow Tests

To skip slow tests:

```bash
pytest -m "not slow"
```

### Import Errors

Make sure the package is installed in editable mode:

```bash
pip install -e .
```

## Contributing

When adding new features:

1. Write unit tests for new modules
2. Add integration tests if feature affects multiple components
3. Ensure all tests pass: `pytest`
4. Check coverage: `pytest --cov=src`
5. Format code: `black tests/`
