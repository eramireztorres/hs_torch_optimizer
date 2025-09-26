import pytest
from src.llm_code_cleaner import LLMCodeCleaner

def test_clean_code_with_valid_input():
    """
    Tests that the clean_code method correctly extracts the 'load_model' function
    and its indented block from a standard Python code string.
    """
    code = """
def load_model():
    print("This is the model")
    return True

def another_function():
    print("This should be ignored")
"""
    expected_code = 'def load_model():\n    print("This is the model")\n    return True'
    assert LLMCodeCleaner.clean_code(code) == expected_code

def test_clean_code_with_markdown():
    """
    Tests that the clean_code method correctly handles code wrapped in Markdown
    backticks, a common format for LLM code responses.
    """
    code = """
```python
def load_model():
    # Model loading logic
    return "model"
```
"""
    expected_code = 'def load_model():\n    # Model loading logic\n    return "model"'
    assert LLMCodeCleaner.clean_code(code) == expected_code

def test_clean_code_no_load_model_function():
    """
    Tests that clean_code returns an empty string when the 'load_model'
    function is not present in the input code.
    """
    code = """
def some_other_function():
    return False
"""
    assert LLMCodeCleaner.clean_code(code) == ""

def test_clean_code_with_invalid_input_type():
    """
    Tests that clean_code raises a TypeError when the input is not a string,
    ensuring robust type checking.
    """
    with pytest.raises(TypeError):
        LLMCodeCleaner.clean_code(123)

def test_clean_code_with_tab_indentation():
    """
    Tests that the clean_code method correctly handles tab-indented code by
    expanding tabs to a consistent number of spaces.
    """
    code = "def load_model():\n\treturn 'model'"
    expected_code = "def load_model():\n    return 'model'"
    assert LLMCodeCleaner.clean_code(code) == expected_code

def test_clean_code_with_extra_function_after():
    """
    Tests that clean_code only extracts the 'load_model' function and stops
    before any subsequent, unrelated functions.
    """
    code = """
def load_model():
    return "main model"

def utility_function():
    return "utility"
"""
    expected_code = 'def load_model():\n    return "main model"'
    assert LLMCodeCleaner.clean_code(code) == expected_code

def test_clean_code_with_empty_input():
    """
    Tests that clean_code returns an empty string when provided with an empty
    string as input.
    """
    assert LLMCodeCleaner.clean_code("") == ""

def test_clean_code_with_comments_and_blank_lines():
    """
    Tests that clean_code correctly preserves comments and blank lines within
    the 'load_model' function block.
    """
    code = '''
def load_model():
    # This is a comment

    return "model with comments"
'''
    expected_code = 'def load_model():\n    # This is a comment\n\n    return "model with comments"'
    assert LLMCodeCleaner.clean_code(code) == expected_code

def test_complex_load_model_function():
    """
    Tests a more complex 'load_model' function with nested structures and
    multiple lines to ensure the extraction logic is robust.
    """
    raw_llm_code = """
    ```python
    def load_model():
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        base_models = [
            ('xgb', XGBClassifier()),
            ('lgbm', LGBMClassifier())
        ]

        meta_learner = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression())
        ])

        model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5
        )
        return model

    def extra_function():
        print("This should not be part of the output.")
    ```
    """
    cleaned_code = LLMCodeCleaner.clean_code(raw_llm_code)
    assert "def load_model():" in cleaned_code
    assert "StackingClassifier" in cleaned_code
    assert "extra_function" not in cleaned_code
    assert cleaned_code.strip().endswith("return model")