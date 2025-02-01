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
from model_api_factory import ModelAPIFactory
from data_loader import DataLoader  
from llm_code_cleaner import LLMCodeCleaner
from error_corrector import ErrorCorrector



#%%

class MainController:
    def __init__(self, joblib_file_path, model_provider, history_file_path, model=None, is_regression=None, 
                 is_image=None, extra_info="Not available", batch_size=32, lr=0.001, epochs=10, 
                 metrics_source='validation',
                 error_model=None, error_prompt_path=None):  
        """
        Initialize the MainController.
        """
        self.joblib_file_path = joblib_file_path
        self.history_manager = ModelHistoryManager(history_file_path=history_file_path)
        self.is_regression = is_regression
        self.is_image = is_image        
        
        self.data = self._load_data()
        self.extra_info = extra_info  # Store the additional information
        self.model_trainer = None

        self.batch_size = self._validate_batch_size(batch_size)  # Validate batch size
        self.lr = self._validate_learning_rate(lr)  # Validate learning rate
        self.epochs = epochs     
        
        
        # Dynamically initialize the LLM model
        self.llm_improver = self._initialize_llm_improver(model_provider, model)

        # Choose between regression and classification, also handle image data
        if is_regression:
            if is_image:
                self.dynamic_updater = DynamicImageRegressionModelUpdater()
            else:
                self.dynamic_updater = DynamicRegressionModelUpdater()
        else:
            if is_image:
                self.dynamic_updater = DynamicImageModelUpdater()
            else:
                self.dynamic_updater = DynamicModelUpdater()
                
        self.metrics_source = metrics_source  # Store metrics source
        
        # Add these members:
        self.error_corrector = None
        if error_model:
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(error_model),
                model=error_model
            )
        else:
            # Use the main model as the error model
            error_llm = ModelAPIFactory.get_model_api(
                provider=ModelAPIFactory.get_provider_from_model(model),  # Use the main model's provider
                model=model  # Use the main model
            )
        self.error_corrector = ErrorCorrector(error_llm, error_prompt_path)

    def _initialize_llm_improver(self, model_provider, model):
        """
        Initialize the LLM improver dynamically based on the provider.
        """
        llm_model = ModelAPIFactory.get_model_api(provider=model_provider, model=model)   
   
        # Check if it’s a regression task and assign the appropriate LLM improver
        if self.is_regression:
            if self.is_image:
                llm_improver = NNImageRegressionLLMImprover(llm_model)
            else:
                llm_improver = NNRegressionLLMImprover(llm_model)
        else:
            if self.is_image:
                llm_improver = NNImageLLMImprover(llm_model)
            else:
                llm_improver = NNLLMImprover(llm_model)
        
        return llm_improver

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

    # def _load_data(self):
    #     """
    #     Load the training and test data from the joblib file.
    #     """
    #     try:
    #         data = joblib.load(self.joblib_file_path)
    #         logging.info(f"Data loaded successfully from {self.joblib_file_path}")
    #         return data
    #     except Exception as e:
    #         logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
    #         return None
    
    # def _load_data(self):
    #     """
    #     Load the training and test data using the DataLoader.
    
    #     Returns:
    #         dict: A dictionary containing X_train, y_train, X_test, and y_test.
    #     """
    #     try:
    #         data = DataLoader.load_data(self.joblib_file_path)  # Handle both file and directory inputs
    #         logging.info(f"Data loaded successfully from {self.joblib_file_path}")           
                      
            
    #         # Choose between regression and classification, also handle image data
    #         if self.is_regression is None:
    #             self.is_regression = is_regression(data['y_train'])
    #         if self.is_image is None:
    #             self.is_image = is_image(data['X_train'])
                
    #         # Choose between regression and classification, also handle image data
    #         if self.is_regression:
    #             if is_image:
    #                 self.dynamic_updater = DynamicImageRegressionModelUpdater()
    #             else:
    #                 self.dynamic_updater = DynamicRegressionModelUpdater()
    #         else:
    #             if self.is_image:
    #                 self.dynamic_updater = DynamicImageModelUpdater()
    #             else:
    #                 self.dynamic_updater = DynamicModelUpdater()
                
    #         return data
    #     except Exception as e:
    #         logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
    #         return None
    
    def _load_data(self):
        try:
            data = DataLoader.load_data(self.joblib_file_path)
            logging.info(f"Data loaded successfully from {self.joblib_file_path}")
            
            # Extract the pre-split flag; if missing, assume data is pre-split.
            self.is_pre_split = data.pop('is_pre_split', True)
            
            # Determine if the task is regression based on the appropriate target key.
            if self.is_regression is None:
                if self.is_pre_split:
                    self.is_regression = is_regression(data['y_train'])
                else:
                    self.is_regression = is_regression(data['y'])
                    
            # Similarly, set is_image using the proper key.
            if self.is_image is None:
                if self.is_pre_split:
                    self.is_image = is_image(data['X_train'])
                else:
                    self.is_image = is_image(data['X'])
                    
            # (Optional) Reinitialize dynamic_updater based on the flags.
            if self.is_regression:
                if self.is_image:
                    self.dynamic_updater = DynamicImageRegressionModelUpdater()
                else:
                    self.dynamic_updater = DynamicRegressionModelUpdater()
            else:
                if self.is_image:
                    self.dynamic_updater = DynamicImageModelUpdater()
                else:
                    self.dynamic_updater = DynamicModelUpdater()
                    
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {self.joblib_file_path}: {e}")
            return None


    def run(self, iterations=5, max_retries=1):
        """
        Run the training and improvement process for the specified number of iterations.
        Retries up to `max_retries` times if the LLM provides invalid code.
        """
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
    
                # # Decide on metrics source: validation or test
                # if self.metrics_source == "validation":
                #     X_train, X_val, y_train, y_val = train_test_split(
                #         self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
                #     )
                # else:
                #     X_train, y_train = self.data['X_train'], self.data['y_train']
                #     X_val, y_val = self.data['X_test'], self.data['y_test']
                
                if self.metrics_source == "validation":
                    if self.is_pre_split:
                        # Pre-split data: further split the provided training set.
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
                        )
                    else:
                        # Unsplit data: perform one split now to create training and validation sets.
                        X_train, X_val, y_train, y_val = train_test_split(
                            self.data['X'], self.data['y'], test_size=0.2, random_state=42
                        )
                else:  # metrics_source == "test"
                    if self.is_pre_split:
                        X_train, y_train = self.data['X_train'], self.data['y_train']
                        X_val, y_val = self.data['X_test'], self.data['y_test']
                    else:
                        # If unsplit data is provided but metrics_source is "test", override to validation.
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
                    
                    print(f'self.error_corrector: {self.error_corrector}')
                        
                    if self.error_corrector:
                        # Get current faulty code
                        current_code = self._get_dynamic_model_code()
                        
                        
                        print("\n=== CURRENT Code before ERROR correction ===")
                        print(current_code)
                        
                        # Get error correction
                        improved_code = self.error_corrector.get_error_fix(
                            current_code, error_msg
                        )
                        
                        print("\n=== CURRENT Code after ERROR correction ===")
                        print(improved_code)
                        
                    else:
                        # Original fallback                       
                      
                        
                        improved_code = self.llm_improver.get_model_suggestions(
                            last_valid_model_code, {}, self.extra_info
                        )
                        
                        print("\n=== CURRENT Code after retry ===")
                        print(improved_code)
                        
                    
                    # Log the response from the LLM
                    if improved_code:
                        print("\n=== LLM Suggested Code ===")
                        print(improved_code)  # Display the suggested code in the console
                        logging.info(f"LLM suggested code:\n{improved_code}")
    
                        # Clean and update the dynamic model code
                        # improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
                        # improved_code = re.sub(r'^python\n', '', improved_code).strip()
                        
                        
                        cleaner = LLMCodeCleaner()
                        improved_code = cleaner.clean_code(improved_code)
                        self.dynamic_updater.update_model_code(improved_code)
                        
                        print("\n=== CLEANED Code ===")
                        print(improved_code)  # Display the suggested code in the console
                        logging.info(f"CLEANED code:\n{improved_code}")
                        
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
    
                # Select the appropriate trainer (regression or classification)
                self.model_trainer = self._get_model_trainer(model, X_train, y_train, X_val, y_val)
                
                # Train and evaluate the model                                              
                self.model_trainer.train_model(epochs=self.epochs)               
                metrics = self.model_trainer.evaluate_model()
    
                print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
                # Save model to history and joblib if output_models_path is provided
                current_model_code = self._get_dynamic_model_code()
                self.history_manager.save_model_history(current_model_code, metrics)
                
   
                # Update last valid model code
                last_valid_model_code = current_model_code    
                self.llm_improver.log_model_history(current_model_code, metrics)
                
                
                print("\n=== CURRENT Code before improvement ===")
                print(current_model_code)
    
                # Get improved model code from the LLM
                improved_code = self.llm_improver.get_model_suggestions(
                    current_model_code, metrics, extra_info=self.extra_info
                )
                
   
                if improved_code:
                    print("\n=== LLM Suggested Code ===")
                    print(improved_code)  # Display the suggested code in the console
                    logging.info(f"LLM suggested code:\n{improved_code}")
    
                    # improved_code = re.sub(r'^```.*\n', '', improved_code).strip().strip('```').strip()
                    # improved_code = re.sub(r'^python\n', '', improved_code).strip()
                    
                    
                    cleaner = LLMCodeCleaner()
                    improved_code = cleaner.clean_code(improved_code)
                    print(f'CLEANED CODED: \n {improved_code}')
                    
                    self.dynamic_updater.update_model_code(improved_code)
                else:
                    logging.warning("No improvements suggested by the LLM in this iteration.")
                    print("No improvements suggested by the LLM in this iteration.")
    
        finally:
            if original_model_code:
                self.dynamic_updater.update_model_code(original_model_code)
                print("Original model restored after iterations.")
                logging.info("Original model restored after iterations.")


    # def run(self, iterations=5):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     """
    #     original_model_code = self._backup_original_model()
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return
    
    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")
                
    #             # Retrieve the current model code
    #             current_model_code = self._get_dynamic_model_code()
    
    #             # Train and evaluate the current model
    #             X_train, y_train = self.data['X_train'], self.data['y_train']
    #             X_val, y_val = self.data['X_test'], self.data['y_test']
                
                
    #             print(f"X_train shape: {self.data['X_train'].shape}, y_train shape: {self.data['y_train'].shape}")
    #             print(f"X_test shape: {self.data['X_test'].shape}, y_test shape: {self.data['y_test'].shape}")
                
    #             print("Checking NaN values in loaded data...")
    #             print(f"NaN count in X_train: {np.isnan(self.data['X_train']).sum()}")
    #             print(f"NaN count in y_train: {np.isnan(self.data['y_train']).sum()}")
    #             print(f"NaN count in X_test: {np.isnan(self.data['X_test']).sum()}")
    #             print(f"NaN count in y_test: {np.isnan(self.data['y_test']).sum()}")

    
    #             # Run the dynamically updated model
    #             model = self.dynamic_updater.run_dynamic_model(X_train=X_train, y_train=y_train)
    #             if model is None:
    #                 logging.error("No model returned by the dynamic model.")
    #                 print("Retrying with a new suggestion from the LLM...")
    
    #                 # Retry obtaining a suggestion from the LLM
    #                 improved_code = self.llm_improver.get_model_suggestions(
    #                     current_model_code, {}, extra_info=self.extra_info
    #                 )
    #                 improved_code = self._clean_code(improved_code)
    
    #                 if improved_code:
    #                     print(f"Retrying with new improved code:\n{improved_code}")
    #                     self.dynamic_updater.update_model_code(improved_code)
    #                     continue  # Retry this iteration
    #                 else:
    #                     logging.warning("No improvements suggested by the LLM during retry.")
    #                     print("Skipping this iteration.")
    #                     continue  # Skip to the next iteration
    
    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")
    
    #             # Train and evaluate the model
    #             self.model_trainer = self._get_model_trainer(model, X_train, y_train, X_val, y_val)
    #             self.model_trainer.train_model(epochs=self.epochs)
    #             metrics = self.model_trainer.evaluate_model()
    
    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")
    
    #             # Save model history
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)
    #             self.llm_improver.log_model_history(current_model_code, metrics)
    
    #             # Get suggestions from LLM
    #             improved_code = self.llm_improver.get_model_suggestions(
    #                 current_model_code, metrics, extra_info=self.extra_info
    #             )
    #             improved_code = self._clean_code(improved_code)
    
    #             if improved_code:
    #                 print(f"Improved model code for iteration {iteration + 1} received from LLM.")
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")
    
    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")



    # def run(self, iterations=5):
    #     """
    #     Run the training and improvement process for the specified number of iterations.
    #     """
    #     original_model_code = self._backup_original_model()
    #     if not original_model_code:
    #         logging.error("Failed to backup the original model. Exiting.")
    #         return
        
    #     if self.metrics_source == "validation":
    #         from sklearn.model_selection import train_test_split
    #     try:
    #         for iteration in range(iterations):
    #             print(f"\n=== Iteration {iteration + 1} ===")

    #             # Decide metrics source
    #             if self.metrics_source == "validation":
    #                 # Create validation split
    #                 X_train, X_val, y_train, y_val = train_test_split(
    #                     self.data['X_train'], self.data['y_train'], test_size=0.2, random_state=42
    #                 )
    #             else:
    #                 # Use test data
    #                 X_train, y_train = self.data['X_train'], self.data['y_train']
    #                 X_val, y_val = self.data['X_test'], self.data['y_test']

    #             # Run the dynamically updated model
    #             model = self.dynamic_updater.run_dynamic_model(X_train=X_train, y_train=y_train)
    #             if model is None:
    #                 logging.error("No model returned by the dynamic model. Exiting.")
    #                 break

    #             print(f"Model for iteration {iteration + 1}: {model.__class__.__name__}")

    #             # Train and evaluate the model
    #             self.model_trainer = self._get_model_trainer(model, X_train, y_train, X_val, y_val)
    #             self.model_trainer.train_model(epochs=self.epochs)
    #             metrics = self.model_trainer.evaluate_model()

    #             print(f"Metrics for iteration {iteration + 1}: {metrics}")

    #             # Save history
    #             current_model_code = self._get_dynamic_model_code()
    #             self.history_manager.save_model_history(current_model_code, metrics)
    #             self.llm_improver.log_model_history(current_model_code, metrics)

    #             # Get suggestions from LLM
    #             improved_code = self.llm_improver.get_model_suggestions(
    #                 current_model_code, metrics, extra_info=self.extra_info
    #             )
    #             improved_code = self._clean_code(improved_code)

    #             if improved_code:
    #                 print(f"Improved model code for iteration {iteration + 1} received from LLM.")
    #                 self.dynamic_updater.update_model_code(improved_code)
    #             else:
    #                 logging.warning("No improvements suggested by the LLM in this iteration.")
    #                 print("No improvements suggested by the LLM in this iteration.")

    #     finally:
    #         if original_model_code:
    #             self.dynamic_updater.update_model_code(original_model_code)
    #             print("Original model restored after iterations.")
    #             logging.info("Original model restored after iterations.")

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

    # def _clean_code(self, code):
    #     """
    #     Clean the LLM-generated code to remove unnecessary markdown formatting.
    #     """
    #     if not code:
    #         return ""
    #     # Remove markdown syntax and any language-specific tags
    #     code = re.sub(r'^```.*\n', '', code).strip().strip('```').strip()
    #     code = re.sub(r'^python\n', '', code).strip()
    #     return code

    def _clean_code(self, code):
        """
        Clean the LLM-generated code to remove unnecessary markdown formatting.
        """
        if not code:
            logging.warning("Received empty code from LLM.")
            return ""
        try:
            code = re.sub(r'^```.*\n', '', code).strip().strip('```').strip()
            code = re.sub(r'^python\n', '', code).strip()
            return code
        except Exception as e:
            logging.error(f"Failed to clean the code: {e}")
            return code  # Return uncleaned code as fallback


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

