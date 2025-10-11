import os
from typing import Literal

from src.api.model_api_factory import ModelAPIFactory
from src.main_controller import MainController
from src.optimization_config import OptimizationConfig
from src.utils.cli_decorator import cli_decorator

# %%


@cli_decorator
def select_model_cli(
    data,
    model: str = "gpt-4.1-mini",
    model_provider: str = None,
    is_regression: Literal[None, "true", "false"] = None,
    history_file_path: str = "model_history.joblib",
    iterations: int = 10,
    extra_info: str = "Not available",
    batch_size: int = 32,
    lr: float = 0.001,
    epochs: int = 10,
    metrics_source: str = "validation",
    error_model: str = None,
    initial_model_path: str = None,
):
    """
    Command-line interface for the model optimization process.

    This function initializes and runs the main optimization controller, which iteratively
    trains, evaluates, and improves a model using an LLM.

    Args:
        data (str): Path to the dataset. It can be a `.joblib` file with pre-split data
            ('X_train', 'y_train', 'X_test', 'y_test'), a `.joblib` or folder of `.csv`
            files with unsplit data ('X', 'y'), or a single `.csv` file where the
            last column is the target.
        model (str, optional): The LLM model for generating improvements (e.g., 'gpt-4o-mini').
            Defaults to 'gpt-4o-mini'.
        model_provider (str, optional): The provider of the LLM ('openai', 'llama', 'google').
            If None, it's inferred from the model name. Defaults to None.
        is_regression (Literal[None, "true", "false"], optional): Specifies the task type.
            Set to "true" for regression, "false" for classification. If None, the type is
            inferred from the data. Defaults to None.
        history_file_path (str, optional): Path to the joblib file for storing model
            history. Defaults to 'model_history.joblib'.
        iterations (int, optional): Number of optimization iterations. Defaults to 10.
        extra_info (str, optional): Additional context for the LLM (e.g., "data has
            class imbalance"). Defaults to 'Not available'.
        batch_size (int, optional): Batch size for neural network training. Defaults to 32.
        lr (float, optional): Learning rate for neural network training. Defaults to 0.001.
        epochs (int, optional): Number of training epochs per iteration. Defaults to 10.
        metrics_source (str, optional): Data source for metrics ('validation' or 'test').
            Defaults to 'validation'.
        error_model (str, optional): The LLM model to use for error correction. If None,
            uses the main `model`. Defaults to None.
        initial_model_path (str, optional): Path to a Python file containing a `load_model`
            function to be used as the starting point for the optimization. If None, a
            default model is generated. Defaults to None.

    Raises:
        ValueError: If `metrics_source` is not 'validation' or 'test'.

    Example:
        ```bash
        torch_optimize --data path/to/data.csv --model gpt-4o-mini --iterations 5
        ```
    """

    error_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts/error_correction_prompt.txt"
    )

    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")
    print(f"Metrics source: {metrics_source}")

    if is_regression is not None:
        is_regression = is_regression == "true"

    config = OptimizationConfig(
        joblib_file_path=data,
        model_provider=model_provider,
        history_file_path=history_file_path,
        model=model,
        is_regression=is_regression,
        is_image=None,
        extra_info=extra_info,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        metrics_source=metrics_source,
        error_model=error_model,
        error_prompt_path=error_prompt_path,
        initial_model_path=initial_model_path,
    )

    controller = MainController(config)
    controller.run(iterations=iterations)


if __name__ == "__main__":
    select_model_cli()
