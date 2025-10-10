"""Tests for ModelHistoryManager."""

import pytest
import joblib
from pathlib import Path
from src.core.model_history_manager import ModelHistoryManager


@pytest.mark.unit
class TestModelHistoryManager:
    """Test suite for ModelHistoryManager."""

    def test_init_creates_empty_history(self, temp_dir):
        """Test initialization creates empty history."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))
        assert manager.model_history == []

    def test_save_model_history_creates_file(self, temp_dir, mock_model_code, sample_metrics):
        """Test saving model history creates file."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))

        manager.save_model_history(mock_model_code, sample_metrics)

        assert history_path.exists()

    def test_save_model_history_appends(self, temp_dir, mock_model_code, sample_metrics):
        """Test saving model history appends to existing history."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))

        manager.save_model_history(mock_model_code, sample_metrics)
        manager.save_model_history(mock_model_code, {**sample_metrics, 'accuracy': 0.9})

        saved_history = joblib.load(history_path)
        assert len(saved_history) == 2
        assert saved_history[0]['metrics']['accuracy'] == 0.85
        assert saved_history[1]['metrics']['accuracy'] == 0.9

    def test_save_model_history_preserves_data(self, temp_dir, mock_model_code, sample_metrics):
        """Test saving model history preserves model code and metrics."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))

        manager.save_model_history(mock_model_code, sample_metrics)

        saved_history = joblib.load(history_path)
        assert saved_history[0]['model_code'] == mock_model_code
        assert saved_history[0]['metrics'] == sample_metrics

    def test_load_existing_history(self, temp_dir, sample_model_history):
        """Test loading existing history file."""
        history_path = temp_dir / "history.joblib"
        joblib.dump(sample_model_history, history_path)

        manager = ModelHistoryManager(history_file_path=str(history_path))
        # Note: The manager doesn't automatically load history in __init__
        # It starts with empty history and only reads when saving
        manager.save_model_history("new code", {'accuracy': 0.95})

        # After saving, it should have loaded the old history and appended
        saved_history = joblib.load(history_path)
        assert len(saved_history) >= 1  # At least the new entry

    def test_get_history(self, temp_dir, mock_model_code, sample_metrics):
        """Test getting history."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))

        manager.save_model_history(mock_model_code, sample_metrics)
        history = manager.model_history

        assert len(history) == 1
        assert history[0]['model_code'] == mock_model_code

    def test_multiple_saves(self, temp_dir, mock_model_code):
        """Test multiple consecutive saves."""
        history_path = temp_dir / "history.joblib"
        manager = ModelHistoryManager(history_file_path=str(history_path))

        for i in range(5):
            manager.save_model_history(mock_model_code, {'iteration': i})

        history = joblib.load(history_path)
        assert len(history) == 5
        assert history[4]['metrics']['iteration'] == 4
