"""Tests for DynamicModelUpdater."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.dynamic_models.dynamic_model_updater import (
    DynamicModelUpdater,
    DynamicRegressionModelUpdater,
    DynamicImageModelUpdater,
    DynamicImageRegressionModelUpdater
)


@pytest.mark.unit
class TestDynamicModelUpdater:
    """Test suite for DynamicModelUpdater."""

    def test_init(self, temp_dir):
        """Test initialization."""
        model_path = temp_dir / "test_model.py"
        model_path.touch()

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        assert updater.dynamic_file_path == str(model_path)

    def test_update_model_code(self, temp_dir):
        """Test updating model code."""
        model_path = temp_dir / "test_model.py"
        model_path.touch()

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        new_code = "def load_model(X_train, y_train):\n    return None"

        updater.update_model_code(new_code)

        assert model_path.read_text() == new_code

    def test_run_dynamic_model_success(self, temp_dir, sample_classification_data):
        """Test running dynamic model successfully."""
        model_path = temp_dir / "test_model.py"

        model_code = """
import torch.nn as nn

def load_model(X_train, y_train):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()
"""
        model_path.write_text(model_code)

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        model, error = updater.run_dynamic_model(
            sample_classification_data['X_train'],
            sample_classification_data['y_train']
        )

        assert model is not None
        assert error is None
        assert isinstance(model, nn.Module)

    def test_run_dynamic_model_returns_tuple(self, temp_dir, sample_classification_data):
        """Test running dynamic model that returns tuple."""
        model_path = temp_dir / "test_model.py"

        model_code = """
import torch
import torch.nn as nn

def load_model(X_train, y_train):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
"""
        model_path.write_text(model_code)

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        result, error = updater.run_dynamic_model(
            sample_classification_data['X_train'],
            sample_classification_data['y_train']
        )

        assert result is not None
        assert error is None
        # Result should be a tuple
        assert isinstance(result, tuple)

    def test_run_dynamic_model_missing_load_model(self, temp_dir, sample_classification_data):
        """Test error when load_model function is missing."""
        model_path = temp_dir / "test_model.py"
        model_path.write_text("# No load_model function")

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        model, error = updater.run_dynamic_model(
            sample_classification_data['X_train'],
            sample_classification_data['y_train']
        )

        assert model is None
        assert error is not None
        assert "'load_model()' function not found" in error

    def test_run_dynamic_model_syntax_error(self, temp_dir, sample_classification_data):
        """Test error handling for syntax errors."""
        model_path = temp_dir / "test_model.py"
        model_path.write_text("def load_model(X_train, y_train):\n    return )")  # Syntax error

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        model, error = updater.run_dynamic_model(
            sample_classification_data['X_train'],
            sample_classification_data['y_train']
        )

        assert model is None
        assert error is not None

    def test_run_dynamic_model_runtime_error(self, temp_dir, sample_classification_data):
        """Test error handling for runtime errors."""
        model_path = temp_dir / "test_model.py"

        model_code = """
def load_model(X_train, y_train):
    raise ValueError("Test error")
"""
        model_path.write_text(model_code)

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))
        model, error = updater.run_dynamic_model(
            sample_classification_data['X_train'],
            sample_classification_data['y_train']
        )

        assert model is None
        assert "Test error" in error

    def test_update_model_code_applies_patch(self, temp_dir):
        """Test that update_model_code applies the LabelSmoothing patch."""
        model_path = temp_dir / "test_model.py"
        model_path.touch()

        updater = DynamicModelUpdater(dynamic_file_path=str(model_path))

        buggy_code = """
class LabelSmoothingCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = loss * class_weights[target]
"""

        updater.update_model_code(buggy_code)
        patched_code = model_path.read_text()

        assert "class_weights.to(target.device)[target]" in patched_code

    def test_regression_updater(self, temp_dir):
        """Test DynamicRegressionModelUpdater uses correct default path."""
        # The regression updater should use a different default file
        updater = DynamicRegressionModelUpdater()
        assert "regression" in updater.dynamic_file_path.lower()

    def test_image_updater(self, temp_dir):
        """Test DynamicImageModelUpdater uses correct default path."""
        updater = DynamicImageModelUpdater()
        assert "image" in updater.dynamic_file_path.lower()

    def test_image_regression_updater(self, temp_dir):
        """Test DynamicImageRegressionModelUpdater uses correct default path."""
        updater = DynamicImageRegressionModelUpdater()
        assert "image" in updater.dynamic_file_path.lower()
        assert "regression" in updater.dynamic_file_path.lower()
