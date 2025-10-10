"""Tests for LLMCodeCleaner."""

import pytest
from src.core.llm_code_cleaner import LLMCodeCleaner


class TestLLMCodeCleaner:
    """Test suite for LLMCodeCleaner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = LLMCodeCleaner()

    def test_clean_code_with_python_fence(self):
        """Test cleaning code with Python markdown fence."""
        code = """```python
def load_model(X_train, y_train):
    import torch
    return torch.nn.Linear(10, 3)
```"""
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned
        assert "import torch" in cleaned
        assert "return torch.nn.Linear" in cleaned

    def test_clean_code_with_generic_fence(self):
        """Test cleaning code with generic markdown fence."""
        code = """```
def load_model(X_train, y_train):
    import torch
    return torch.nn.Linear(10, 3)
```"""
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned
        assert "import torch" in cleaned

    def test_clean_code_with_multiple_fences(self):
        """Test cleaning code with multiple markdown fences."""
        code = """```python
import torch
```

Some text

```python
def test():
    pass
```"""
        cleaned = self.cleaner.clean_code(code)
        # Should remove all fences
        assert '```' not in cleaned

    def test_clean_code_without_fences(self):
        """Test cleaning code without markdown fences."""
        code = "def load_model(X_train, y_train):\n    import torch\n    return torch.nn.Linear(10, 3)"
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned
        assert "import torch" in cleaned

    def test_clean_code_with_leading_whitespace(self):
        """Test cleaning code with leading whitespace."""
        code = "   def load_model(X_train, y_train):\n       import torch\n       return torch.nn.Linear(10, 3)"
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned

    def test_clean_code_with_trailing_whitespace(self):
        """Test cleaning code with trailing whitespace."""
        code = "import torch\ndef test():\n    pass   \n\n"
        cleaned = self.cleaner.clean_code(code)
        assert not cleaned.endswith('\n')

    def test_clean_empty_code(self):
        """Test cleaning empty code."""
        cleaned = self.cleaner.clean_code("")
        assert cleaned == ""

    def test_clean_none_code(self):
        """Test cleaning None input raises TypeError."""
        with pytest.raises(TypeError, match="llm_code must be a string"):
            self.cleaner.clean_code(None)

    def test_clean_code_preserves_indentation(self):
        """Test that code cleaning preserves proper indentation."""
        code = """```python
def load_model(X_train, y_train):
    if True:
        return 1
```"""
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned
        assert "    if True:" in cleaned
        assert "        return 1" in cleaned

    def test_clean_code_returns_empty_when_no_load_model(self):
        """Test that code cleaning returns empty string when load_model not found."""
        code = """```python
def other_function():
    pass
```"""
        cleaned = self.cleaner.clean_code(code)
        assert cleaned == ""

    def test_clean_code_extracts_only_load_model(self):
        """Test that only load_model function is extracted."""
        code = """```python
def load_model(X_train, y_train):
    return None

def other_function():
    print("Should not be included")
```"""
        cleaned = self.cleaner.clean_code(code)
        assert "def load_model" in cleaned
        assert "return None" in cleaned
        assert "other_function" not in cleaned
        assert "Should not be included" not in cleaned
