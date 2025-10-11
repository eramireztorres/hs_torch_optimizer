import logging
import numpy as np
import pandas as pd

from src.training.model_trainer import NNModelTrainer, NNRegressionModelTrainer
from src.core.llm_improver import NNLLMImprover, NNRegressionLLMImprover, NNImageLLMImprover, NNImageRegressionLLMImprover
from src.core.model_history_manager import ModelHistoryManager
from src.dynamic_models.dynamic_model_updater import DynamicModelUpdater, DynamicRegressionModelUpdater, DynamicImageModelUpdater, DynamicImageRegressionModelUpdater
from src.api.model_api_factory import ModelAPIFactory
from src.training.data_loader import DataLoader
from src.core.llm_code_cleaner import LLMCodeCleaner
from src.core.error_corrector import ErrorCorrector
from src.optimization_config import OptimizationConfig
from src.core.model_type_registry import create_default_registry, TaskType, DataType


#%%

class MainController:
    def __init__(self, config: OptimizationConfig, registry=None):
        """
        Initialize the MainController with an OptimizationConfig.

        Args:
            config: OptimizationConfig object containing all optimization parameters
            registry: Optional ModelTypeRegistry instance (uses default if not provided)
        """
        self.config = config
        self.registry = registry or create_default_registry()
        self.joblib_file_path = config.joblib_file_path
        self.history_manager = ModelHistoryManager(history_file_path=config.history_file_path)
        self.is_regression = config.is_regression
        self.is_image = config.is_image

        if config.initial_model_path:
            try:
                with open(config.initial_model_path, 'r') as f:
                    init_code = f.read()

                task_type = TaskType.REGRESSION if self.is_regression else TaskType.CLASSIFICATION
                data_type = DataType.IMAGE if self.is_image else DataType.TABULAR
                updater = self.registry.create_updater(task_type, data_type)
                updater.update_model_code(init_code)

            except Exception as e:
                print(f"Warning: could not load initial model from {config.initial_model_path}: {e}")

        self.data = self._load_data()
        self.extra_info = config.extra_info
        self.model_trainer = None

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.epochs = config.epochs

        self.llm_improver = self._initialize_llm_improver(config.model_provider, config.model)

        task_type = TaskType.REGRESSION if self.is_regression else TaskType.CLASSIFICATION
        data_type = DataType.IMAGE if self.is_image else DataType.TABULAR
        self.dynamic_updater = self.registry.create_updater(task_type, data_type)

        self.metrics_source = config.metrics_source

        self.error_corrector = None
        if config.error_model:
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(config.error_model),
                model=config.error_model
            )
        else:
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(config.model),
                model=config.model
            )
        self.error_corrector = ErrorCorrector(error_llm, config.error_prompt_path)

    def _initialize_llm_improver(self, model_provider, model):
        """
        Initialize the LLM improver dynamically using the registry.
        """
        llm_model = ModelAPIFactory.get_model_api(provider=model_provider, model=model)

        task_type = TaskType.REGRESSION if self.is_regression else TaskType.CLASSIFICATION
        data_type = DataType.IMAGE if self.is_image else DataType.TABULAR

        return self.registry.create_improver(task_type, data_type, llm_model)

    def _load_data(self):
        try:
            data = DataLoader.load_data(self.joblib_file_path)
            logging.info(f"Data loaded successfully from {self.joblib_file_path}")

            self.is_pre_split = data.pop('is_pre_split', True)

            if self.is_regression is None:
                if self.is_pre_split:
                    self.is_regression = is_regression(data['y_train'])
                else:
                    self.is_regression = is_regression(data['y'])

            if self.is_image is None:
                if self.is_pre_split:
                    self.is_image = is_image(data['X_train'])
                else:
                    self.is_image = is_image(data['X'])

            task_type = TaskType.REGRESSION if self.is_regression else TaskType.CLASSIFICATION
            data_type = DataType.IMAGE if self.is_image else DataType.TABULAR
            self.dynamic_updater = self.registry.create_updater(task_type, data_type)

            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
            return None


    def run(self, iterations=5, max_retries=1):
        """
        Run the training and improvement process for the specified number of iterations.
        Retries up to `max_retries` times if the LLM provides invalid code.
        """
        if not self.data:
            logging.error("Data not loaded. Exiting.")
            return

        original_model_code = self._backup_original_model()
        last_valid_model_code = original_model_code
        if not original_model_code:
            logging.error("Failed to backup the original model. Exiting.")
            return
    
        if self.metrics_source == "validation":
            from sklearn.model_selection import train_test_split
    
        try:
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1} ===")
    
                
                if self.metrics_source == "validation":
                    if self.is_pre_split:
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
                        )
                    else:
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X'], self.data['y'], test_size=0.2, random_state=42
                        )
                else:  # metrics_source == "test"
                    if self.is_pre_split:
                        X_train, y_train = self.data['X_train'], self.data['y_train']
                        X_val, y_val = self.data['X_test'], self.data['y_test']
                    else:
                        logging.warning("Unsplit data provided; overriding metrics_source to 'validation'.")
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X'], self.data['y'], test_size=0.2, random_state=42
                        )

    
                retries = 0
                model = None
              
                while retries < max_retries:
                    model, error_msg = self.dynamic_updater.run_dynamic_model(X_train=X_train, y_train=y_train)
                    
                    if model is not None:
                        break
                        
                    if self.error_corrector:
                        current_code = self._get_dynamic_model_code()
                        
                        
                        improved_code = self.error_corrector.get_error_fix(
                            current_code, error_msg
                        )
                        
                        print("\n=== CODE after ERROR correction ===")
                        print(improved_code)
                        
                    else:
                      
                        
                        improved_code = self.llm_improver.get_model_suggestions(
                            last_valid_model_code, {}, self.extra_info
                        )
                        
                        
                    if improved_code:                       
                        
                        cleaner = LLMCodeCleaner()
                        improved_code = cleaner.clean_code(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)
                        
                        
                        model, error_msg = self.dynamic_updater.run_dynamic_model(X_train=X_train, y_train=y_train)
                        
                    else:
                        logging.warning("No new suggestions received from LLM. Skipping retry.")
                        print("No new suggestions received from LLM. Skipping retry.")
                        continue
                    
                    retries += 1
                
                
                if model is None:
                    logging.error(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
                    print(f"Exceeded maximum retries ({max_retries}) for iteration {iteration + 1}. Skipping iteration.")
                    continue
    
                print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
                self.model_trainer = self._get_model_trainer(model, X_train, y_train, X_val, y_val)
                
                self.model_trainer.train_model(epochs=self.epochs)               
                metrics = self.model_trainer.evaluate_model()
    
                print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
                current_model_code = self._get_dynamic_model_code()
                self.history_manager.save_model_history(current_model_code, metrics)
                
   
                last_valid_model_code = current_model_code    
                self.llm_improver.log_model_history(current_model_code, metrics)
                
   
                improved_code = self.llm_improver.get_model_suggestions(
                    current_model_code, metrics, extra_info=self.extra_info
                )
                
   
                if improved_code:
                 
                    
                    cleaner = LLMCodeCleaner()
                    improved_code = cleaner.clean_code(improved_code)
                    
                    print(f"\n=== IMPROVED MODEL ITERATION {iteration + 1} ===")
                    print(improved_code)  # Display the suggested code in the console
                    
                    self.dynamic_updater.update_model_code(improved_code)
                else:
                    logging.warning("No improvements suggested by the LLM in this iteration.")
                    print("No improvements suggested by the LLM in this iteration.")
    
        finally:
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")


    def _get_model_trainer(self, model, X_train, y_train, X_val, y_val):
        """Return the appropriate trainer."""
        if self.is_regression:
            return NNRegressionModelTrainer(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                batch_size=self.batch_size,
                lr=self.lr
            )
        else:
            return NNModelTrainer(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                batch_size=self.batch_size,
                lr=self.lr
            )


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
    
    if np.issubdtype(y_train.dtype, np.floating):
        if np.all(np.equal(np.mod(y_train, 1), 0)):
            return False  # This suggests it's a classification problem with integer-like floats

    return np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(y_train.dtype, np.integer) and not np.all(np.equal(np.mod(y_train, 1), 0))


def is_image(X_train):
    """
    Check if the input data contains 2D features (i.e., height and width dimensions).

    Args:
        X_train (np.ndarray or pd.DataFrame): Training data.

    Returns:
        bool: True if the data contains 2D features (image-like), False otherwise.
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    elif not isinstance(X_train, np.ndarray):
        raise ValueError("Input data must be a NumPy array or a Pandas DataFrame.")

    if len(X_train.shape) >= 3:
        height, width = X_train.shape[-2], X_train.shape[-1]
        if height > 1 and width > 1:
            return True

    return False

