import importlib
import os
import logging
import sys
sys.path.append(os.path.dirname(__file__))

dynamic_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_model.py')
dynamic_regression_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_regression_model.py')
dynamic_image_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_image_model.py')
dynamic_image_regression_file_path = os.path.join(os.path.dirname(__file__), 'dynamic_image_regression_model.py')


#%%

class DynamicModelUpdater:
    def __init__(self, dynamic_file_path=dynamic_file_path):
        """
        Initialize the DynamicModelUpdater.

        Args:
            dynamic_file_path (str): The path to the Python file that will be dynamically updated.
        """
        self.dynamic_file_path = dynamic_file_path

    def update_model_code(self, new_model_code):
        """
        Update the `dynamic_model.py` file with the new model code provided by the LLM.

        Args:
            new_model_code (str): The Python code for the new model and hyperparameters.
        """
        try:
            with open(self.dynamic_file_path, 'w') as f:
                f.write(new_model_code)
            logging.info(f"Updated model code in {self.dynamic_file_path}")
        except Exception as e:
            logging.error(f"Failed to update the dynamic model file: {e}")


    def run_dynamic_model(self, X_train, y_train):
        """
        Run the dynamically updated `load_model()` method from the `dynamic_model.py` file.

        Args:
            X_train: Training data (features) to infer input dimensions.
            y_train: Training labels to infer output dimensions.

        Returns:
            model: The model returned by the dynamically updated `load_model()` function.
        """
        try:
            # Validate that the dynamic file path exists
            if not os.path.exists(self.dynamic_file_path):
                raise FileNotFoundError(f"Dynamic model file '{self.dynamic_file_path}' not found.")
            
            # Get the module name from the file path
            module_name = os.path.splitext(os.path.basename(self.dynamic_file_path))[0]
            
            # Invalidate cache and reload the module
            if module_name in sys.modules:
                del sys.modules[module_name]
            spec = importlib.util.spec_from_file_location(module_name, self.dynamic_file_path)
            dynamic_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = dynamic_module
            spec.loader.exec_module(dynamic_module)
            
            # Check if the loaded module contains the load_model function
            if not hasattr(dynamic_module, "load_model"):
                raise AttributeError(f"'load_model()' function not found in {self.dynamic_file_path}")
            
            # Call the load_model function with X_train and y_train to dynamically get the model
            model = dynamic_module.load_model(X_train, y_train)
            logging.info("Successfully loaded the dynamically updated model.")
            return model

        except Exception as e:
            logging.error(f"Failed to run the dynamic model: {e}")
            return None

class DynamicRegressionModelUpdater(DynamicModelUpdater):
    def __init__(self, dynamic_file_path=dynamic_regression_file_path):
        super().__init__(dynamic_file_path)
        
class DynamicImageModelUpdater(DynamicModelUpdater):
    def __init__(self, dynamic_file_path=dynamic_image_file_path):
        super().__init__(dynamic_file_path)

class DynamicImageRegressionModelUpdater(DynamicModelUpdater):
    def __init__(self, dynamic_file_path=dynamic_image_regression_file_path):
        super().__init__(dynamic_file_path)
