"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
import joblib


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 3, 100).astype(np.int64)

    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def sample_image_data():
    """Generate sample image dataset (28x28 grayscale)."""
    np.random.seed(42)
    X = np.random.randn(100, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 100).astype(np.int64)

    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV files."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })

    csv_path = temp_dir / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_joblib_data(temp_dir, sample_classification_data):
    """Create sample joblib file."""
    joblib_path = temp_dir / "data.joblib"
    joblib.dump(sample_classification_data, joblib_path)
    return joblib_path


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 20)
            self.fc2 = nn.Linear(20, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return """
import torch
import torch.nn as nn

def load_model(X_train, y_train):
    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))

    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
"""


@pytest.fixture
def mock_model_code():
    """Mock model code for dynamic model testing."""
    return """
import torch
import torch.nn as nn

def load_model(X_train, y_train):
    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    return SimpleNet()
"""


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return {
        'accuracy': 0.85,
        'loss': 0.42,
        'precision': 0.83,
        'recall': 0.87,
        'f1': 0.85
    }


@pytest.fixture
def sample_model_history(sample_metrics, mock_model_code):
    """Sample model history for testing."""
    return [
        {'model_code': mock_model_code, 'metrics': sample_metrics},
        {'model_code': mock_model_code, 'metrics': {**sample_metrics, 'accuracy': 0.87}}
    ]


@pytest.fixture(autouse=True)
def reset_cuda():
    """Reset CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
