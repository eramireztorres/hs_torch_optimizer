import joblib
import logging
import numpy as np
import re
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_trainer import NNModelTrainer, NNRegressionModelTrainer
from llm_improver import NNLLMImprover, NNRegressionLLMImprover, NNImageLLMImprover, NNImageRegressionLLMImprover
from model_history_manager import ModelHistoryManager
from dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater, DynamicImageModelUpdater, DynamicImageRegressionModelUpdater
from gpt import Gpt4AnswerGenerator



#%%

class MainController:
    def __init__(self, joblib_file_path, llm_improver, history_file_path, is_regression_bool=False, 
                 is_image=False, extra_info="Not available", batch_size=32, lr=0.001, epochs=10):
        """
        Initialize the MainController.
        """
        self.joblib_file_path = joblib_file_path
        self.llm_improver = llm_improver
        self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
        self.data = self._load_data()
        self.extra_info = extra_info  # Store the additional information
        self.model_trainer = None
        self.is_regression = is_regression_bool
        self.batch_size = self._validate_batch_size(batch_size)  # Validate batch size
        self.lr = self._validate_learning_rate(lr)  # Validate learning rate
        self.epochs = epochs

        # Choose between regression and classification, also handle image data
        if is_regression_bool:
            if is_image:
                self.dynamic_updater = DynamicImageRegressionModelUpdater()
            else:
                self.dynamic_updater = DynamicRegressionModelUpdater()
        else:
            if is_image:
                self.dynamic_updater = DynamicImageModelUpdater()
            else:
                self.dynamic_updater = DynamicModelUpdater()

    def _validate_batch_size(self, batch_size):
        """Ensure the batch size is valid (positive integer)."""
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        return batch_size

    def _validate_learning_rate(self, lr):
        """Ensure the learning rate is within a reasonable range."""
        if not (0 < lr <= 1):
            raise ValueError(f"Learning rate must be between 0 and 1, got {lr}")
        return lr

    def _load_data(self):
        """
        Load the training and test data from the joblib file.
        """
        try:
            data = joblib.load(self.joblib_file_path)
            logging.info(f"Data loaded successfully from {self.joblib_file_path}")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
            return None

    def run(self, iterations=5):
        """
        Run the training and improvement process for the specified number of iterations.
        """
        original_model_code = self._backup_original_model()
        if not original_model_code:
            logging.error("Failed to backup the original model. Exiting.")
            return

        try:
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1} ===")

                # Run the dynamically updated model
                
                # Add channel dimension if missing (for grayscale images)
                X_train, y_train = self.data['X_train'], self.data['y_train']
                
                if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
                    X_train = np.expand_dims(X_train, axis=1)
               
                
                model = self.dynamic_updater.run_dynamic_model(X_train=X_train, y_train=y_train)
                if model is None:
                    logging.error("No model returned by the dynamic model. Exiting.")
                    break

                print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

                # Train and evaluate the model
                self.model_trainer = self._get_model_trainer(model)
                self.model_trainer.train_model(epochs=self.epochs)
                metrics = self.model_trainer.evaluate_model()

                print(f"Metrics for iteration {iteration + 1}: {metrics}")

                # Log the model and its performance
                current_model_code = self._get_dynamic_model_code()
                self.history_manager.save_model_history(current_model_code, metrics)

                # Log the model history in LLMImprover
                self.llm_improver.log_model_history(current_model_code, metrics)

                # Get suggestions from the LLM for improvements, using extra_info
                improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics, extra_info=self.extra_info)

                # Clean up the returned code
                improved_code = self._clean_code(improved_code)

                if improved_code:
                    print(f"Improved model code for iteration {iteration + 1} received from LLM.")
                    self.dynamic_updater.update_model_code(improved_code)
                else:
                    logging.warning("No improvements suggested by the LLM in this iteration.")
                    print("No improvements suggested by the LLM in this iteration.")

        finally:
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")

    def _get_model_trainer(self, model):
        """
        Return the appropriate trainer based on whether it's a regression or classification task.
        """
        if self.is_regression:
            return NNRegressionModelTrainer(
                model=model,
                X_train=self.data['X_train'],
                y_train=self.data['y_train'],
                X_test=self.data['X_test'],
                y_test=self.data['y_test'], 
                batch_size=self.batch_size, lr=self.lr
            )
        else:
            return NNModelTrainer(
                model=model,
                X_train=self.data['X_train'],
                y_train=self.data['y_train'],
                X_test=self.data['X_test'],
                y_test=self.data['y_test'],
                batch_size=self.batch_size, lr=self.lr
            )

    def _clean_code(self, code):
        """
        Clean the LLM-generated code to remove unnecessary markdown formatting.
        """
        if not code:
            return ""
        # Remove markdown syntax and any language-specific tags
        code = re.sub(r'^```.*\n', '', code).strip().strip('```').strip()
        code = re.sub(r'^python\n', '', code).strip()
        return code

    def _get_dynamic_model_code(self):
        """
        Retrieve the current Python code from the dynamic model file.
        """
        try:
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read the dynamic model code: {e}")
            return ""

    def _backup_original_model(self):
        """
        Backup the original model code from dynamic_model.py.
        """
        try:
            print(f'DYNAMIC PATH: {self.dynamic_updater.dynamic_file_path}')
            with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
                original_model_code = f.read()
            return original_model_code
        except Exception as e:
            logging.error(f"Failed to backup original model: {e}")
            return None


# class MainController:
#     def __init__(self, joblib_file_path, llm_improver, history_file_path, is_regression_bool=False, 
#                  is_image=False, extra_info="Not available", batch_size=32, lr=0.001):
#         """
#         Initialize the MainController.

#         Args:
#             joblib_file_path (str): Path to the joblib file containing the training and test data.
#             llm_improver: The LLM model improver to query for model improvements.
#             history_file_path: Path to the file where model history will be stored.
#             is_regression_bool (bool): Whether the task is regression.
#             is_image (bool): Whether the input data consists of image data (2D arrays).
#             extra_info (str): Additional information to include in the LLM prompt (e.g., class imbalance, noisy labels).
#         """
#         self.joblib_file_path = joblib_file_path
#         self.llm_improver = llm_improver
#         self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
#         self.data = self._load_data()
#         self.extra_info = extra_info  # Store the additional information
#         self.model_trainer = None
#         self.is_regression = is_regression_bool
#         self.batch_size = batch_size
#         self.lr = lr

#         # Choose between regression and classification
#         if is_regression_bool:
#             if is_image:
#                 self.dynamic_updater = DynamicImageRegressionModelUpdater()
#             else:
#                 self.dynamic_updater = DynamicRegressionModelUpdater()
#         else:
#             if is_image:
#                 self.dynamic_updater = DynamicImageModelUpdater()
#             else:
#                 self.dynamic_updater = DynamicModelUpdater()

#     def _load_data(self):
#         """
#         Load the training and test data from the joblib file.

#         Returns:
#             dict: A dictionary containing X_train, y_train, X_test, and y_test.
#         """
#         try:
#             data = joblib.load(self.joblib_file_path)
#             logging.info(f"Data loaded successfully from {self.joblib_file_path}")
#             return data
#         except Exception as e:
#             logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
#             return None

#     def run(self, iterations=5):
#         """
#         Run the training and improvement process for the specified number of iterations.

#         Args:
#             iterations (int): Number of iterations to improve the model.
#         """
#         original_model_code = self._backup_original_model()
#         if not original_model_code:
#             logging.error("Failed to backup the original model. Exiting.")
#             return

#         try:
#             for iteration in range(iterations):
#                 print(f"\n=== Iteration {iteration + 1} ===")

#                 # Run the dynamically updated model
#                 model = self.dynamic_updater.run_dynamic_model(X_train=self.data['X_train'], y_train=self.data['y_train'])
#                 if model is None:
#                     logging.error("No model returned by the dynamic model. Exiting.")
#                     break

#                 print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

#                 # Train and evaluate the model (classification or regression)
#                 self.model_trainer = self._get_model_trainer(model)
#                 self.model_trainer.train_model()
#                 metrics = self.model_trainer.evaluate_model()

#                 print(f"Metrics for iteration {iteration + 1}: {metrics}")

#                 # Log the model and its performance
#                 current_model_code = self._get_dynamic_model_code()
#                 self.history_manager.save_model_history(current_model_code, metrics)

#                 # Log the model history in LLMImprover
#                 self.llm_improver.log_model_history(current_model_code, metrics)

#                 # Get suggestions from the LLM for improvements, using extra_info
#                 improved_code = self.llm_improver.get_model_suggestions(current_model_code, metrics, extra_info=self.extra_info)

#                 # Clean up the returned code
#                 improved_code = self._clean_code(improved_code)

#                 if improved_code:
#                     print(f"Improved model code for iteration {iteration + 1} received from LLM.")
#                     self.dynamic_updater.update_model_code(improved_code)
#                 else:
#                     logging.warning("No improvements suggested by the LLM in this iteration.")
#                     print("No improvements suggested by the LLM in this iteration.")

#         finally:
#             if original_model_code:
#                 self.dynamic_updater.update_model_code(original_model_code)
#                 print("Original model restored after iterations.")
#                 logging.info("Original model restored after iterations.")

#     def _get_model_trainer(self, model):
#         """
#         Return the appropriate trainer based on whether it's a regression or classification task.

#         Args:
#             model: The neural network model to train.

#         Returns:
#             The appropriate trainer class (NNModelTrainer or NNRegressionModelTrainer).
#         """
#         if self.is_regression:
#             return NNRegressionModelTrainer(
#                 model=model,
#                 X_train=self.data['X_train'],
#                 y_train=self.data['y_train'],
#                 X_test=self.data['X_test'],
#                 y_test=self.data['y_test'], 
#                 batch_size=self.batch_size, lr=self.lr
#             )
#         else:
#             return NNModelTrainer(
#                 model=model,
#                 X_train=self.data['X_train'],
#                 y_train=self.data['y_train'],
#                 X_test=self.data['X_test'],
#                 y_test=self.data['y_test'],
#                 batch_size=self.batch_size, lr=self.lr
#             )

#     def _clean_code(self, code):
#         """
#         Clean the LLM-generated code to remove unnecessary markdown formatting.
        
#         Args:
#             code: The generated Python code from the LLM.

#         Returns:
#             str: The cleaned-up Python code.
#         """
#         if not code:
#             return ""
        
#         # Remove markdown syntax and any language-specific tags
#         code = re.sub(r'^```.*\n', '', code).strip().strip('```').strip()
#         code = re.sub(r'^python\n', '', code).strip()
#         return code

#     def _get_dynamic_model_code(self):
#         """
#         Retrieve the current Python code from the dynamic model file.

#         Returns:
#             str: The code inside the dynamic model file.
#         """
#         try:
#             with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
#                 return f.read()
#         except Exception as e:
#             logging.error(f"Failed to read the dynamic model code: {e}")
#             return ""

#     def _backup_original_model(self):
#         """
#         Backup the original model code from dynamic_model.py.
#         """
#         try:
#             print(f'DYNAMIC PATH: {self.dynamic_updater.dynamic_file_path}')
            
#             with open(self.dynamic_updater.dynamic_file_path, 'r') as f:
#                 original_model_code = f.read()
#             return original_model_code
#         except Exception as e:
#             logging.error(f"Failed to backup original model: {e}")
#             return None


def is_regression(y_train):
    """
    Check if the target values suggest a regression problem.
    Regression typically has continuous target values (e.g., floats).
    This function checks if all values are exact integers, even if they are of type float.
    
    Args:
        y_train (array-like): The target values from the training set.

    Returns:
        bool: True if the problem is regression, False if it's classification.
    """
    # If the target array contains floats but all values are actually integers
    if np.issubdtype(y_train.dtype, np.floating):
        # Check if all float values are actually integers
        if np.all(np.equal(np.mod(y_train, 1), 0)):
            return False  # This suggests it's a classification problem with integer-like floats

    # Otherwise, treat it as a regression problem if it's not an integer-like float array
    return np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(y_train.dtype, np.integer) and not np.all(np.equal(np.mod(y_train, 1), 0))



def is_image(X_train):
    """
    Check if the input data contains 2D features (i.e., height and width dimensions).

    Args:
        X_train (np.ndarray): Training data.

    Returns:
        bool: True if the data contains 2D features (image-like), False otherwise.
    """
    # Check if input is a NumPy array
    if not isinstance(X_train, np.ndarray):
        raise ValueError("Input data is not a NumPy array.")

    # Check if the data has at least 3 dimensions (batch_size, height, width or channels, height, width)
    if len(X_train.shape) >= 3:
        height, width = X_train.shape[-2], X_train.shape[-1]
        # Validate that height and width are greater than 1 (indicating 2D features)
        if height > 1 and width > 1:
            return True

    return False

