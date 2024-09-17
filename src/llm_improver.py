import logging
import json
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(__file__))

prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompts/classification_prompt.txt')
prompt_regression_file_path = os.path.join(os.path.dirname(__file__), 'prompts/regression_prompt.txt')
prompt_image_file_path = os.path.join(os.path.dirname(__file__), 'prompts/image_classification_prompt.txt')
prompt_image_regression_file_path = os.path.join(os.path.dirname(__file__), 'prompts/image_regression_prompt.txt')


class NNLLMImprover:
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_file_path):
        """
        Initialize the improver for neural networks.

        Args:
            llm_model: The LLM model to query.
            model_history: History of previous models and metrics.
            prompt_file_path: Path to the prompt template.
        """
        self.llm_model = llm_model
        self.model_history = model_history if model_history else []
        self.prompt_file_path = prompt_file_path

    def get_model_suggestions(self, current_model_code, metrics, extra_info="Not available"):
        """
        Query the LLM for model improvements.
        """
        prompt = self._format_prompt(current_model_code, metrics, extra_info)
        try:
            improved_code = self.llm_model.get_response(prompt)
            return improved_code
        except Exception as e:
            logging.error(f"Error querying LLM: {e}")
            return None

    def log_model_history(self, model_code, metrics):
        """
        Log model history for future iterations.
        """
        self.model_history.append({'model_code': model_code, 'metrics': metrics})
        

    def _format_prompt(self, current_model_code, metrics, extra_info):
        """
        Format the prompt with the current model and metrics.
        """
        with open(self.prompt_file_path, 'r') as file:
            prompt_template = file.read()

        # Ensure that self.model_history and metrics are JSON serializable
        history_serializable = self._convert_to_python_types(self.model_history)
        metrics_serializable = self._convert_to_python_types(metrics)

        history_str = json.dumps(history_serializable, indent=2)
        metrics_str = json.dumps(metrics_serializable, indent=2)

        return prompt_template.format(
            current_model_code=current_model_code,
            metrics_str=metrics_str,
            history_str=history_str,
            extra_info=extra_info
        )


    def _convert_to_python_types(self, obj):
        """
        Recursively convert all elements in the object to Python-native types.
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj

    # def _format_prompt(self, current_model_code, metrics, extra_info):
    #     """
    #     Format the prompt with the current model and metrics.
    #     Converts all values in the metrics dictionary to Python-native types to avoid serialization issues.
    #     """
    #     with open(self.prompt_file_path, 'r') as file:
    #         prompt_template = file.read()
    
    #     # # Convert metrics to Python-native types
    #     # metrics = {key: float(value) if isinstance(value, (np.float32, np.float64, torch.Tensor)) else value 
    #     #            for key, value in metrics.items()}
    
    #     history_str = json.dumps(self.model_history, indent=2)
    #     metrics_str = json.dumps(metrics, indent=2)
        
    #     return prompt_template.format(current_model_code=current_model_code, metrics_str=metrics_str, history_str=history_str, extra_info=extra_info)


class NNRegressionLLMImprover(NNLLMImprover):
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_regression_file_path):
        super().__init__(llm_model, model_history, prompt_file_path)


class NNImageLLMImprover(NNLLMImprover):
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_image_file_path):
        super().__init__(llm_model, model_history, prompt_file_path)

class NNImageRegressionLLMImprover(NNLLMImprover):
    def __init__(self, llm_model, model_history=None, prompt_file_path=prompt_image_regression_file_path):
        super().__init__(llm_model, model_history, prompt_file_path)
