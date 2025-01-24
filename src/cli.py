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
        history_file_path: str = 'model_history.joblib',
        iterations: int = 10,
        extra_info: str = 'Not available',  
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 10,
        metrics_source: str = 'validation'  # New argument
        ):
    """
    Args:
        - data (dict): A dictionary containing training and test data, with keys such as 'X_train', 'y_train', 
          'X_test', 'y_test'. These should be NumPy arrays or torch tensors representing the feature and target datasets 
          for model training and evaluation.
        - history_file_path (str, optional): Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is 'model_history.joblib'.
        - model (str, optional): The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to 'gpt-4o-mini'.
        - iterations (int, optional): The number of iterations to run, where each iteration involves training a model, evaluating its performance, and generating improvements. Default is 5.
        - extra_info (str, optional): Additional context or information to pass to the LLM, such as class imbalance or noisy labels. Default is 'Not available'.
        - batch_size (int, optional): Batch size for model training. Default is 32.
        - lr (float, optional): Learning rate for model training. Default is 0.001.
        - epochs (int, optional): Number of training epochs for the neural network. Default is 10.
        - metrics_source (str, optional): Source of metrics to show the LLM ('validation' or 'test'). Default is 'validation'.
    """
    if metrics_source not in ['validation', 'test']:
        raise ValueError("metrics_source must be 'validation' or 'test'")
    
    if not model_provider:
        model_provider = ModelAPIFactory.get_provider_from_model(model)

    print(f"Using model: {model} (provider: {model_provider})")
    print(f"Metrics source: {metrics_source}")

    controller = MainController(data, model_provider, history_file_path, 
                                model=model,
                                extra_info=extra_info, 
                                batch_size=batch_size, 
                                lr=lr, 
                                epochs=epochs,
                                metrics_source=metrics_source)  # Pass the argument
    
    controller.run(iterations=iterations)


# @cli_decorator
# def select_model_cli(data,
#         model: str = 'gpt-4o-mini',
#         model_provider: str = None,                     
#         history_file_path: str = 'model_history.joblib',
#         iterations: int = 10,
#         extra_info: str = 'Not available',  
#         batch_size: int = 32,
#         lr: float = 0.001,
#         epochs: int = 10
        
#         ):
#     """
#     - Selects and optimizes a neural network model for classification or regression using an LLM improver, 
#     iterating through models and hyperparameters based on the specified configuration.

#     Args:
    # - data (dict): A dictionary containing training and test data, with keys such as 'X_train', 'y_train', 
    #   'X_test', 'y_test'. These should be NumPy arrays or torch tensors representing the feature and target datasets 
    #   for model training and evaluation.
    # - history_file_path (str, optional): Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is 'model_history.joblib'.
    # - model (str, optional): The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to 'gpt-4o-mini'.
    # - iterations (int, optional): The number of iterations to run, where each iteration involves training a model, evaluating its performance, and generating improvements. Default is 5.
    # - extra_info (str, optional): Additional context or information to pass to the LLM, such as class imbalance or noisy labels. Default is 'Not available'.
    # - batch_size (int, optional): Batch size for model training. Default is 32.
    # - lr (float, optional): Learning rate for model training. Default is 0.001.
    # - epochs (int, optional): Number of training epochs for the neural network. Default is 10.

#     Returns:
#     - None. This function optimizes the model iteratively and stores the history of models and performance metrics in the specified history file. The final model and improvements are made based on the LLM's suggestions.

#     Raises:
#     - ValueError: If the LLM model specified cannot be initialized.
#     - FileNotFoundError: If the specified history file cannot be found or created.
#     - RuntimeError: If any issue arises during the model training or optimization process.
#     """
    
#     if not model_provider:
#         model_provider = ModelAPIFactory.get_provider_from_model(model)

#     print(f"Using model: {model} (provider: {model_provider})")

#     # Initialize and run the main controller with the extra_info passed in
#     controller = MainController(data, model_provider, history_file_path, 
#                                 model=model,
#                                 # is_regression_bool=is_regression_bool, 
#                                 # is_image=is_image_bool, 
#                                 extra_info=extra_info, 
#                                 batch_size=batch_size, lr=lr, epochs=epochs)
    
#     controller.run(iterations=iterations)



if __name__ == "__main__":
    select_model_cli()

 