"""Tests for DataLoader."""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.training.data_loader import DataLoader


@pytest.mark.unit
class TestDataLoader:
    """Test suite for DataLoader."""

    def test_load_joblib_pre_split(self, sample_joblib_data):
        """Test loading pre-split joblib data."""
        data = DataLoader.load_data(str(sample_joblib_data))

        assert 'X_train' in data
        assert 'y_train' in data
        assert 'X_test' in data
        assert 'y_test' in data
        assert data['is_pre_split'] is True

    def test_load_joblib_unsplit(self, temp_dir):
        """Test loading unsplit joblib data."""
        unsplit_data = {
            'X': np.random.randn(100, 10),
            'y': np.random.randint(0, 2, 100)
        }
        joblib_path = temp_dir / "unsplit.joblib"
        joblib.dump(unsplit_data, joblib_path)

        data = DataLoader.load_data(str(joblib_path))

        assert 'X' in data or 'X_train' in data  # May be split automatically
        assert 'y' in data or 'y_train' in data
        assert 'is_pre_split' in data

    def test_load_single_csv(self, sample_csv_data):
        """Test loading single CSV file."""
        data = DataLoader.load_data(str(sample_csv_data))

        assert 'X' in data or 'X_train' in data
        assert 'y' in data or 'y_train' in data

    def test_load_csv_directory_pre_split(self, temp_dir):
        """Test loading CSV directory with pre-split files."""
        # Create directory with split CSV files
        csv_dir = temp_dir / "csv_data"
        csv_dir.mkdir()

        pd.DataFrame(np.random.randn(80, 10)).to_csv(csv_dir / "X_train.csv", index=False)
        pd.DataFrame(np.random.randint(0, 2, 80)).to_csv(csv_dir / "y_train.csv", index=False)
        pd.DataFrame(np.random.randn(20, 10)).to_csv(csv_dir / "X_test.csv", index=False)
        pd.DataFrame(np.random.randint(0, 2, 20)).to_csv(csv_dir / "y_test.csv", index=False)

        data = DataLoader.load_data(str(csv_dir))

        assert 'X_train' in data
        assert 'y_train' in data
        assert 'X_test' in data
        assert 'y_test' in data
        assert data['is_pre_split'] is True

    def test_load_csv_directory_unsplit(self, temp_dir):
        """Test loading CSV directory with unsplit files."""
        csv_dir = temp_dir / "csv_data"
        csv_dir.mkdir()

        pd.DataFrame(np.random.randn(100, 10)).to_csv(csv_dir / "X.csv", index=False)
        pd.DataFrame(np.random.randint(0, 2, 100)).to_csv(csv_dir / "y.csv", index=False)

        data = DataLoader.load_data(str(csv_dir))

        assert 'X' in data or 'X_train' in data
        assert 'y' in data or 'y_train' in data

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(Exception):  # Should raise some exception
            DataLoader.load_data("/nonexistent/file.joblib")

    def test_handles_categorical_columns(self, temp_dir):
        """Test handling of categorical columns."""
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'cat1': ['A', 'B'] * 50,
            'cat2': ['X', 'Y', 'Z'] * 33 + ['X'],
            'target': np.random.randint(0, 2, 100)
        })

        csv_path = temp_dir / "categorical.csv"
        df.to_csv(csv_path, index=False)

        data = DataLoader.load_data(str(csv_path))

        # Should handle categorical columns
        assert data is not None

    def test_converts_data_to_numpy(self, sample_csv_data):
        """Test that data is converted to numpy arrays."""
        data = DataLoader.load_data(str(sample_csv_data))

        # Get X and y (may be split or unsplit)
        X_key = 'X_train' if 'X_train' in data else 'X'
        y_key = 'y_train' if 'y_train' in data else 'y'

        assert isinstance(data[X_key], np.ndarray)
        assert isinstance(data[y_key], np.ndarray)

    def test_preserves_float32_dtype(self, temp_dir):
        """Test that float32 dtype is preserved."""
        data_dict = {
            'X_train': np.random.randn(80, 10).astype(np.float32),
            'y_train': np.random.randint(0, 2, 80),
            'X_test': np.random.randn(20, 10).astype(np.float32),
            'y_test': np.random.randint(0, 2, 20)
        }

        joblib_path = temp_dir / "float32.joblib"
        joblib.dump(data_dict, joblib_path)

        loaded_data = DataLoader.load_data(str(joblib_path))

        assert loaded_data['X_train'].dtype == np.float32
