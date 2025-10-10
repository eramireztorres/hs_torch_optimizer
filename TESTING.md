# Testing Guide for hs-torch-optimizer

## Overview

This project has a comprehensive test suite covering unit tests, integration tests, and end-to-end scenarios. The tests are organized to ensure code quality, catch regressions, and facilitate confident refactoring.

## Test Statistics

- **Total Tests**: 91 tests
- **Test Coverage**: Targeting >80% overall
- **Test Categories**:
  - Unit tests: ~70 tests
  - Integration tests: ~6 tests
  - Test fixtures: 15+ shared fixtures

## Quick Start

### Install Test Dependencies

```bash
# Install package with test dependencies
pip install -e ".[test]"

# Or install development dependencies (includes linting, formatting)
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m "not gpu"         # Skip GPU-dependent tests

# Run specific test file
pytest tests/unit/core/test_llm_code_cleaner.py

# Run with verbose output
pytest -v
```

### Using Make

```bash
make test               # Run all tests
make test-unit         # Unit tests only
make test-fast         # Skip slow tests
make test-coverage     # With coverage report
make coverage-report   # Open HTML report
```

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── README.md                            # Testing documentation
├── unit/                                # Unit tests (fast, isolated)
│   ├── api/                            # API module tests
│   │   ├── test_base_model_api.py
│   │   └── test_model_api_factory.py
│   ├── core/                           # Core logic tests
│   │   ├── test_error_corrector.py
│   │   ├── test_llm_code_cleaner.py
│   │   ├── test_model_history_manager.py
│   │   └── test_model_type_registry.py
│   ├── dynamic_models/                 # Dynamic model tests
│   │   └── test_dynamic_model_updater.py
│   ├── training/                       # Training module tests
│   │   ├── test_data_loader.py
│   │   └── test_model_trainer.py
│   └── utils/                          # Utility tests
│       └── test_cli_decorator.py
└── integration/                         # Integration tests (slower)
    ├── test_end_to_end.py
    └── test_cli_integration.py
```

## Key Features

### 1. Comprehensive Fixtures (conftest.py)

Shared fixtures available to all tests:

- `temp_dir`: Temporary directory for test files
- `sample_classification_data`: 100-sample classification dataset
- `sample_regression_data`: 100-sample regression dataset
- `sample_image_data`: 100-sample 28x28 image dataset
- `sample_csv_data`: CSV file with mixed features
- `sample_joblib_data`: Joblib-formatted data
- `simple_model`: Simple PyTorch model
- `mock_llm_response`: Mock LLM output
- `mock_model_code`: Sample dynamic model code
- `sample_metrics`: Sample training metrics

### 2. Test Markers

Tests are categorized with markers:

```python
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Tests multiple components
@pytest.mark.slow          # Longer-running tests
@pytest.mark.gpu          # Requires GPU hardware
```

### 3. Code Coverage

- Coverage reports generated in `htmlcov/`
- XML coverage for CI/CD integration
- Excludes test files and vendored code
- Target: >80% overall coverage

### 4. Continuous Integration

GitHub Actions workflow (`.github/workflows/tests.yml`):
- Runs on: Ubuntu, macOS, Windows
- Python versions: 3.9, 3.10, 3.11, 3.12
- Automated coverage reporting to Codecov
- Linting with flake8
- Format checking with black

## Writing New Tests

### Test File Structure

```python
"""Tests for ModuleName."""

import pytest
from src.module import ClassToTest


class TestClassName:
    """Test suite for ClassName."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = ClassToTest()
        result = obj.method()
        assert result == expected_value

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="error message"):
            ClassToTest().invalid_operation()

    def test_with_fixture(self, sample_classification_data):
        """Test using fixture."""
        # Fixtures are automatically provided
        assert 'X_train' in sample_classification_data
```

### Using Mocks

```python
from unittest.mock import Mock, patch

def test_with_mock_llm():
    """Test with mocked LLM."""
    mock_llm = Mock()
    mock_llm.get_response = Mock(return_value="response")

    # Use mock in test
    result = my_function(mock_llm)
    assert result == expected

    # Verify mock was called
    mock_llm.get_response.assert_called_once()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("gpt-4", "openai"),
    ("claude-3", "anthropic"),
    ("gemini-2", "google"),
])
def test_provider_detection(input, expected):
    """Test provider detection with multiple inputs."""
    assert get_provider(input) == expected
```

## Test Coverage by Module

| Module | Coverage Target | Status |
|--------|----------------|--------|
| API modules | >70% | ✓ |
| Core modules | >90% | ✓ |
| Dynamic models | >85% | ✓ |
| Training modules | >85% | ✓ |
| Integration | All workflows | ✓ |

## Common Testing Patterns

### 1. Testing File Operations

```python
def test_file_operation(temp_dir):
    """Test file operations with temp directory."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("content")
    assert file_path.exists()
    # Automatically cleaned up after test
```

### 2. Testing Model Training

```python
def test_model_training(simple_model, sample_classification_data):
    """Test model training."""
    trainer = NNModelTrainer(
        model=simple_model,
        X_train=sample_classification_data['X_train'],
        y_train=sample_classification_data['y_train'],
        X_test=sample_classification_data['X_test'],
        y_test=sample_classification_data['y_test']
    )
    trainer.train_model(epochs=2)
    metrics = trainer.evaluate_model()
    assert 'accuracy' in metrics
```

### 3. Testing Dynamic Code

```python
def test_dynamic_model_loading(temp_dir, sample_classification_data):
    """Test loading dynamic model."""
    model_path = temp_dir / "model.py"
    model_path.write_text(model_code)

    updater = DynamicModelUpdater(str(model_path))
    model, error = updater.run_dynamic_model(
        sample_classification_data['X_train'],
        sample_classification_data['y_train']
    )

    assert model is not None
    assert error is None
```

## Troubleshooting

### CUDA Errors

If you encounter CUDA errors in tests:

```bash
# Force CPU-only testing
CUDA_VISIBLE_DEVICES="" pytest
```

### Slow Tests

Speed up test runs:

```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### Import Errors

Ensure package is installed in editable mode:

```bash
pip install -e .
```

### Coverage Not Working

```bash
# Reinstall coverage tools
pip install --force-reinstall pytest-cov coverage
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on others
2. **Clear Names**: Use descriptive test names that explain what is being tested
3. **One Assertion Focus**: Each test should focus on one aspect of behavior
4. **Use Fixtures**: Leverage fixtures for common setup/teardown
5. **Mock External Dependencies**: Mock API calls, file I/O when possible
6. **Fast Tests**: Keep unit tests fast (<1s each)
7. **Document Complex Tests**: Add docstrings explaining non-obvious test logic

## Contributing

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure all existing tests pass: `pytest`
3. Check coverage: `pytest --cov=src`
4. Format code: `black tests/`
5. Lint code: `flake8 tests/`
6. Update this documentation if needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mocking in Python](https://docs.python.org/3/library/unittest.mock.html)
