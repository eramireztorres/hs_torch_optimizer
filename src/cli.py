from joblib import load
import sys
import os
sys.path.append(os.path.dirname(__file__))

from cli_decorator import cli_decorator
from main_controller import MainController, is_regression, is_image
from gpt import Gpt4AnswerGenerator
from llm_improver import NNLLMImprover, NNRegressionLLMImprover, NNImageLLMImprover, NNImageRegressionLLMImprover

#%%
api_key = os.getenv('OPENAI_API_KEY')

@cli_decorator
def select_model_cli(data,
                     
        history_file_path: str = 'model_history.joblib',
        model: str = 'gpt-4o-mini',
        iterations: int = 5,
        extra_info: str = 'Not available',  
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 10
        
        ):
    """
    - Selects and optimizes a neural network model for classification or regression using an LLM improver, 
    iterating through models and hyperparameters based on the specified configuration.

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

    Returns:
    - None. This function optimizes the model iteratively and stores the history of models and performance metrics in the specified history file. The final model and improvements are made based on the LLM's suggestions.

    Raises:
    - ValueError: If the LLM model specified cannot be initialized.
    - FileNotFoundError: If the specified history file cannot be found or created.
    - RuntimeError: If any issue arises during the model training or optimization process.
    """
    
    # Initialize the LLM generator
    generator = Gpt4AnswerGenerator(api_key, model=model)

    # Determine if the task is regression or classification
    is_regression_bool = is_regression(load(data)['y_train'])
    
    # Determine if the input data is images or flat feature vectors
    is_image_bool = is_image(load(data)['X_train'])

    # Check if it’s a regression task and assign the appropriate LLM improver
    if is_regression_bool:
        if is_image_bool:
            llm_improver = NNImageRegressionLLMImprover(generator)
        else:
            llm_improver = NNRegressionLLMImprover(generator)
    else:
        if is_image_bool:
            llm_improver = NNImageLLMImprover(generator)
        else:
            llm_improver = NNLLMImprover(generator)

    # Initialize and run the main controller with the extra_info passed in
    controller = MainController(data, llm_improver, history_file_path, 
                                is_regression_bool=is_regression_bool, 
                                is_image=is_image_bool, 
                                extra_info=extra_info, 
                                batch_size=batch_size, lr=lr, epochs=epochs)
    
    controller.run(iterations=iterations)





if __name__ == "__main__":
    select_model_cli()

 