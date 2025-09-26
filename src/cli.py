from typing import Literal

import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_decorator import cli_decorator
from main_controller import MainController, ModelAPIFactory

#%%

@cli_decorator
def select_model_cli(data,
        model: str = 'gpt-4o-mini',
        model_provider: str = None,   
        is_regression: Literal[None, "true", "false"] = None,                  
        history_file_path: str = 'model_history.joblib',
        iterations: int = 10,
        extra_info: str = 'Not available',  
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 10,
        metrics_source: str = 'validation',
        error_model: str = None
        ):
    """
    Command-line interface function for selecting and running a model optimization process.
    This function initializes the optimization controller, trains and evaluates models, 
    and uses an LLM to propose improvements to the model architecture and hyperparameters.

    Args:
        - data (str): Path to the input dataset. This can be: 1. A `.joblib` file containing pre-split data dictionary with keys 'X_train', 'y_train', 'X_test', and 'y_test'. 2. A `.joblib` file or a folder with `.csv` files containing unsplit data with keys or filenames 'X' and 'y'. The program will perform a validation split if unsplit data is provided. 3. A single `.csv` file containing multiple feature columns and one target column (assumed to be the last column).

        - model (str, optional): The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters (e.g., 'gpt-4', 'llama-3.1'). Default is 'gpt-4o-mini'.

        - model_provider (str, optional): The provider of the LLM model (e.g., 'openai', 'llama', 'google'). If not provided, it will be inferred automatically based on the `model` argument. Default is None.

        - history_file_path (str, optional): Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is 'model_history.joblib'.

        - iterations (int, optional): The number of optimization iterations to perform. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is 10.

        - extra_info (str, optional): Additional context or information to pass to the LLM for better suggestions. Examples include class imbalance, noisy labels, or outlier data. Default is 'Not available'.

        - batch_size (int, optional): Batch size for training the neural network. Default is 32.

        - lr (float, optional): Learning rate for training the neural network. Default is 0.001.

        - epochs (int, optional): Number of training epochs for the neural network during each iteration. Default is 10.

        - metrics_source (str, optional): Source of the metrics to provide to the LLM for evaluation. Must be one of the following: 'validation': Metrics are derived from a validation split of the training data; 'test': Metrics are derived from the test data, if provided. Default is 'validation'.

    Raises:
        - ValueError: If the `metrics_source` argument is not 'validation' or 'test'.

    Example:
        ```bash
        torch_optimize -d path/to/data --model gpt-4o-mini --batch-size 64 --lr 0.0005 \
                          --epochs 20 --metrics-source validation --iterations 5
        ```

    Usage:
        This function is designed to be used as part of the command-line interface for the 
        `hs-torch-optimizer` project. It initializes the `MainController` with the provided 
        arguments, manages the optimization process, and leverages LLM feedback to improve 
        model architecture and hyperparameters dynamically.

    Notes:
        - If the input data is not pre-split, the program will automatically split it into 
          training and validation sets with an 80/20 ratio.
        - The function uses the specified LLM model and provider to suggest improvements 
          after each iteration.
        - Ensure that the data format matches the expected structure to avoid errors.
    """
    
    error_prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/error_correction_prompt.txt')
    
    if metrics_source not in ['validation', 'test']:
        raise ValueError("metrics_source must be 'validation' or 'test'")
    
    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")
    print(f"Metrics source: {metrics_source}")
    
    if is_regression is not None:
        is_regression = is_regression == 'true'

    controller = MainController(data, model_provider, history_file_path, 
                                model=model,
                                extra_info=extra_info, 
                                batch_size=batch_size, 
                                lr=lr, 
                                epochs=epochs,
                                is_regression=is_regression,
                                metrics_source=metrics_source,
                                error_model=error_model,
                                error_prompt_path=error_prompt_path)  
    
    controller.run(iterations=iterations)



if __name__ == "__main__":
    select_model_cli()

 
