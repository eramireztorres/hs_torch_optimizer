import joblib
import numpy as np
import torch
import logging

class ModelHistoryManager:
    def __init__(self, history_file_path='model_history.joblib'):
        """
        Initialize the ModelHistoryManager.

        Args:
            history_file_path (str): Path to the file where the model history will be saved.
        """
        self.history_file_path = history_file_path
        self.model_history = []

    def _convert_metrics_to_python_types(self, metrics):
        """
        Convert all values in the metrics dictionary to Python-native types.
        
        Args:
            metrics (dict): The performance metrics of the model.

        Returns:
            dict: The metrics with Python-native types.
        """
        return {key: float(value) if isinstance(value, (np.float32, np.float64, torch.Tensor)) else value
                for key, value in metrics.items()}

    def save_model_history(self, model_code, metrics):
        """
        Save the current model's code and metrics into the history.

        Args:
            model_code (str): The Python code of the model.
            metrics (dict): The performance metrics of the model.
        """
        # Convert metrics to native Python types before saving
        converted_metrics = self._convert_metrics_to_python_types(metrics)
        
        history_entry = {
            'model_code': model_code,
            'metrics': converted_metrics
        }
        self.model_history.append(history_entry)

        try:
            joblib.dump(self.model_history, self.history_file_path)
            logging.info(f"Model history saved to {self.history_file_path}")
        except Exception as e:
            logging.error(f"Failed to save model history: {e}")

    def load_model_history(self):
        """
        Load the model history from the joblib file.

        Returns:
            list: The history of models, including their code and performance metrics.
        """
        try:
            self.model_history = joblib.load(self.history_file_path)
            logging.info(f"Model history loaded from {self.history_file_path}")
        except FileNotFoundError:
            logging.warning(f"No existing history found at {self.history_file_path}. Starting with an empty history.")
            self.model_history = []
        except Exception as e:
            logging.error(f"Failed to load model history: {e}")
            self.model_history = []

        return self.model_history


# class ModelHistoryManager:
#     def __init__(self, history_file_path='model_history.joblib'):
#         """
#         Initialize the ModelHistoryManager.

#         Args:
#             history_file_path (str): Path to the file where the model history will be saved.
#         """
#         self.history_file_path = history_file_path
#         self.model_history = []

#     def save_model_history(self, model_code, metrics):
#         """
#         Save the current model's code and metrics into the history.

#         Args:
#             model_code (str): The Python code of the model.
#             metrics (dict): The performance metrics of the model.
#         """
#         history_entry = {
#             'model_code': model_code,
#             'metrics': metrics
#         }
#         self.model_history.append(history_entry)

#         try:
#             joblib.dump(self.model_history, self.history_file_path)
#             logging.info(f"Model history saved to {self.history_file_path}")
#         except Exception as e:
#             logging.error(f"Failed to save model history: {e}")

#     def load_model_history(self):
#         """
#         Load the model history from the joblib file.

#         Returns:
#             list: The history of models, including their code and performance metrics.
#         """
#         try:
#             self.model_history = joblib.load(self.history_file_path)
#             logging.info(f"Model history loaded from {self.history_file_path}")
#         except FileNotFoundError:
#             logging.warning(f"No existing history found at {self.history_file_path}. Starting with an empty history.")
#             self.model_history = []
#         except Exception as e:
#             logging.error(f"Failed to load model history: {e}")
#             self.model_history = []

#         return self.model_history
