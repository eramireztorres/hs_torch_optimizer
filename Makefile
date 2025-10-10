.PHONY: test test-unit test-integration test-coverage clean install install-dev lint format

# Install package in editable mode
install:
	pip install -e .

# Install package with development dependencies
install-dev:
	pip install -e ".[dev]"

# Install test dependencies
install-test:
	pip install -e ".[test]"

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest -m unit

# Run only integration tests
test-integration:
	pytest -m integration

# Run fast tests (skip slow ones)
test-fast:
	pytest -m "not slow"

# Run tests with coverage
test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term

# Run tests in parallel
test-parallel:
	pytest -n auto

# Open coverage report
coverage-report:
	xdg-open htmlcov/index.html || open htmlcov/index.html

# Clean generated files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Format code with black
format:
	black src/ tests/

# Lint code
lint:
	flake8 src/ tests/
	black --check src/ tests/

# Type checking
type-check:
	mypy src/

# Run all checks (lint, type-check, tests)
check: lint type-check test

# Build package
build: clean
	python -m build

# Help
help:
	@echo "Available targets:"
	@echo "  install          - Install package in editable mode"
	@echo "  install-dev      - Install with development dependencies"
	@echo "  install-test     - Install test dependencies"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run only unit tests"
	@echo "  test-integration - Run only integration tests"
	@echo "  test-fast        - Run fast tests (skip slow ones)"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  test-parallel    - Run tests in parallel"
	@echo "  coverage-report  - Open HTML coverage report"
	@echo "  clean            - Clean generated files"
	@echo "  format           - Format code with black"
	@echo "  lint             - Lint code with flake8"
	@echo "  type-check       - Run type checking with mypy"
	@echo "  check            - Run all checks (lint, type-check, tests)"
	@echo "  build            - Build package"
