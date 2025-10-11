"""CLI integration tests."""

import subprocess
from pathlib import Path

import joblib
import pytest


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_torch_optimize_help(self):
        """Test that torch_optimize --help works."""
        result = subprocess.run(
            ["torch_optimize", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "torch_optimize" in result.stdout

    def test_torch_optimize_missing_data(self):
        """Test error handling when data path is missing."""
        result = subprocess.run(
            ["torch_optimize", "--data", "/nonexistent/data.csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with error
        assert (
            result.returncode != 0
            or "error" in result.stderr.lower()
            or "error" in result.stdout.lower()
        )

    @pytest.mark.slow
    def test_torch_optimize_full_run(self, temp_dir, sample_classification_data):
        """Test full torch_optimize CLI run (mocked LLM)."""
        # This would require mocking the LLM API calls
        # Skip for now as it's complex to mock in subprocess
        pytest.skip("Requires complex mocking of API calls in subprocess")
