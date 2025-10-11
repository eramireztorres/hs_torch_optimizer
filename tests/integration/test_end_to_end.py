"""End-to-end integration tests."""

from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pytest

from src.main_controller import MainController
from src.optimization_config import OptimizationConfig


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndOptimization:
    """End-to-end integration tests for the optimization pipeline."""

    def test_full_optimization_pipeline_classification(
        self, temp_dir, sample_classification_data
    ):
        """Test full optimization pipeline with classification data."""
        # Save data to joblib
        data_path = temp_dir / "data.joblib"
        joblib.dump(sample_classification_data, data_path)

        history_path = temp_dir / "history.joblib"
        error_prompt_path = temp_dir / "error_prompt.txt"
        error_prompt_path.write_text("Fix: ${faulty_code}\nError: ${error_msg}")

        # Mock LLM to return valid model code
        mock_llm_response = """
import torch.nn as nn

def load_model(X_train, y_train):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()
"""

        with patch(
            "src.api.model_api_factory.ModelAPIFactory.get_model_api"
        ) as mock_api:
            mock_model = Mock()
            mock_model.get_response = Mock(return_value=mock_llm_response)
            mock_api.return_value = mock_model

            config = OptimizationConfig(
                joblib_file_path=str(data_path),
                model_provider="openai",
                history_file_path=str(history_path),
                model="gpt-4o-mini",
                is_regression=False,
                is_image=False,
                extra_info="Test run",
                batch_size=32,
                lr=0.001,
                epochs=2,
                metrics_source="validation",
                error_model=None,
                error_prompt_path=str(error_prompt_path),
                initial_model_path=None,
            )

            controller = MainController(config)
            controller.run(iterations=2)

        # Verify history was saved
        assert history_path.exists()
        history = joblib.load(history_path)
        assert len(history) > 0
        assert "model_code" in history[0]
        assert "metrics" in history[0]

    def test_full_optimization_pipeline_regression(
        self, temp_dir, sample_regression_data
    ):
        """Test full optimization pipeline with regression data."""
        # Save data to joblib
        data_path = temp_dir / "data.joblib"
        joblib.dump(sample_regression_data, data_path)

        history_path = temp_dir / "history.joblib"
        error_prompt_path = temp_dir / "error_prompt.txt"
        error_prompt_path.write_text("Fix: ${faulty_code}\nError: ${error_msg}")

        mock_llm_response = """
import torch.nn as nn

def load_model(X_train, y_train):
    class RegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    return RegressionModel()
"""

        with patch(
            "src.api.model_api_factory.ModelAPIFactory.get_model_api"
        ) as mock_api:
            mock_model = Mock()
            mock_model.get_response = Mock(return_value=mock_llm_response)
            mock_api.return_value = mock_model

            config = OptimizationConfig(
                joblib_file_path=str(data_path),
                model_provider="openai",
                history_file_path=str(history_path),
                model="gpt-4o-mini",
                is_regression=True,
                is_image=False,
                extra_info="Test regression",
                batch_size=32,
                lr=0.001,
                epochs=2,
                metrics_source="validation",
                error_model=None,
                error_prompt_path=str(error_prompt_path),
                initial_model_path=None,
            )

            controller = MainController(config)
            controller.run(iterations=2)

        # Verify history was saved
        assert history_path.exists()
        history = joblib.load(history_path)
        assert len(history) > 0

    def test_error_correction_integration(self, temp_dir, sample_classification_data):
        """Test that error correction works in the pipeline."""
        data_path = temp_dir / "data.joblib"
        joblib.dump(sample_classification_data, data_path)

        history_path = temp_dir / "history.joblib"
        error_prompt_path = temp_dir / "error_prompt.txt"
        error_prompt_path.write_text("Fix: ${faulty_code}\nError: ${error_msg}")

        # First return bad code, then good code on error correction
        bad_code = (
            "def load_model(X_train, y_train):\n    return None  # This will fail"
        )

        good_code = """
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

        with patch(
            "src.api.model_api_factory.ModelAPIFactory.get_model_api"
        ) as mock_api:
            mock_model = Mock()
            # Return bad code first, then good code
            mock_model.get_response = Mock(side_effect=[bad_code, good_code, good_code])
            mock_api.return_value = mock_model

            config = OptimizationConfig(
                joblib_file_path=str(data_path),
                model_provider="openai",
                history_file_path=str(history_path),
                model="gpt-4o-mini",
                is_regression=False,
                is_image=False,
                extra_info="Test error correction",
                batch_size=32,
                lr=0.001,
                epochs=1,
                metrics_source="validation",
                error_model="gpt-4o-mini",
                error_prompt_path=str(error_prompt_path),
                initial_model_path=None,
            )

            controller = MainController(config)
            controller.run(iterations=1)

        # Should complete successfully despite initial bad code
        assert history_path.exists()
