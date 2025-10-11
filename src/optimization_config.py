from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class OptimizationConfig:
    """
    Configuration class for neural network optimization process.

    This class encapsulates all the parameters needed for the model optimization
    process, providing validation, defaults, and better organization.

    Attributes:
        joblib_file_path: Path to the dataset file (.joblib, .csv, or directory)
        model_provider: LLM provider ('openai', 'llama', 'google', etc.)
        history_file_path: Path to store model training history
        model: Specific LLM model name (e.g., 'gpt-4o-mini')
        is_regression: Task type (True=regression, False=classification, None=auto-detect)
        is_image: Whether data contains images (None=auto-detect)
        extra_info: Additional context for the LLM
        batch_size: Batch size for training (must be positive)
        lr: Learning rate (must be between 0 and 1)
        epochs: Number of training epochs per iteration
        metrics_source: Source for evaluation metrics ('validation' or 'test')
        error_model: LLM model for error correction (None=use main model)
        error_prompt_path: Path to error correction prompt template
        initial_model_path: Path to initial model code file
    """

    joblib_file_path: str
    model_provider: str
    history_file_path: str
    model: Optional[str] = None
    is_regression: Optional[bool] = None
    is_image: Optional[bool] = None
    extra_info: str = "Not available"
    batch_size: int = 32
    lr: float = 0.001
    epochs: int = 10
    metrics_source: Literal["validation", "test"] = "validation"
    error_model: Optional[str] = None
    error_prompt_path: Optional[str] = None
    initial_model_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_batch_size()
        self._validate_learning_rate()
        self._validate_metrics_source()
        self._validate_epochs()

    def _validate_batch_size(self):
        """Ensure the batch size is valid (positive integer)."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

    def _validate_learning_rate(self):
        """Ensure the learning rate is within a reasonable range."""
        if not (0 < self.lr <= 1):
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.lr}")

    def _validate_metrics_source(self):
        """Ensure metrics_source is valid."""
        if self.metrics_source not in ["validation", "test"]:
            raise ValueError(
                f"metrics_source must be 'validation' or 'test', got {self.metrics_source}"
            )

    def _validate_epochs(self):
        """Ensure epochs is positive."""
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {self.epochs}")
