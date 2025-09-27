import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, Optional


class NNModelTrainer:
    """
    Trainer class for classification neural networks.

    This class handles the training, validation, evaluation, saving, and loading of a neural network model.
    The model can be provided as a single instance or as a tuple containing the model and optionally a custom
    optimizer and/or loss criterion.

    Args:
        model: Either an instance of torch.nn.Module or a tuple containing:
            - (model,): Only model is provided. The default criterion (CrossEntropyLoss) and optimizer (Adam)
              are used.
            - (model, criterion): A custom loss criterion is provided; optimizer defaults to Adam.
            - (model, optimizer, criterion): Custom optimizer and loss criterion are provided.
        X_train (np.ndarray): Training features as a NumPy array.
        y_train (Union[np.ndarray, list]): Training labels.
        X_test (np.ndarray): Test features as a NumPy array.
        y_test (Union[np.ndarray, list]): Test labels.
        batch_size (int): Batch size for training (default: 32).
        lr (float): Learning rate for the optimizer (default: 0.001).
        device (Optional[Union[str, torch.device]]): Device to use ('cuda' or 'cpu'); if None, auto-detects.
        patience (int): Number of epochs with no improvement after which training will be stopped (default: 5).
    """

    def __init__(
        self, 
        model: Union[nn.Module, Tuple],
        X_train: np.ndarray,
        y_train: Union[np.ndarray, list],
        X_test: np.ndarray,
        y_test: Union[np.ndarray, list],
        batch_size: int = 32,
        lr: float = 0.001,
        device: Optional[Union[str, torch.device]] = None,
        patience: int = 5
    ):
        
        
        # Unpack the model if provided as a tuple.
        if isinstance(model, tuple):
            if len(model) == 1:
                self.model = model[0]
                self.criterion = nn.CrossEntropyLoss()  # Default loss for classification
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.scheduler = None
            elif len(model) == 2:
                self.model, second_element = model
                if isinstance(second_element, optim.Optimizer):  # If optimizer is passed
                    self.optimizer = second_element
                    self.criterion = nn.CrossEntropyLoss()                    
                else:  # Assume second_element is criterion
                    self.criterion = second_element
                    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.scheduler = None
            elif len(model) == 3:
                self.model, self.optimizer, self.criterion = model
                self.scheduler = None
            elif len(model) == 4:
                self.model, self.optimizer, self.criterion, self.scheduler = model
            else:
                raise ValueError("Invalid model tuple length. Expected 1, 2, 3 or 4 elements.")

        else:
            self.model = model
            self.criterion = nn.CrossEntropyLoss()  # Default loss for classification
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = None  # No scheduler provided

        # Add channel dimension if missing (for grayscale images).
        if X_train.ndim == 3:  # shape: (num_samples, height, width)
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)
            
        # Ensure data is a NumPy array if it is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values


        # Split the training data into training and validation sets.
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), 
            test_size=0.2, 
            random_state=42
        )

        # Prepare test data.
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.batch_size = batch_size

        # Set device: if provided as a string, convert to torch.device.
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        # Early stopping parameters.
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def _validate_model(self) -> float:
        """
        Validate the model on the validation set.

        Returns:
            float: The loss on the validation set.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val.to(self.device))
            loss = self.criterion(outputs, self.y_val.to(self.device))
        return loss.item()

    def train_model(self, epochs: int = 10) -> None:
        """
        Train the model for a specified number of epochs with early stopping based on validation loss.

        Args:
            epochs (int): Maximum number of epochs to train.
        """
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            val_loss = self._validate_model()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Early stopping check.
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    def evaluate_model(self) -> dict:
        """
        Evaluate the trained model on the test data and return performance metrics.

        Returns:
            dict: A dictionary containing accuracy, per-class precision, recall, f1-score, and overall metrics.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.X_test.to(self.device)
            labels = self.y_test.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(labels, predicted),
            "precision_per_class": precision_score(labels, predicted, average=None).tolist(),
            "recall_per_class": recall_score(labels, predicted, average=None).tolist(),
            "f1_score_per_class": f1_score(labels, predicted, average=None).tolist(),
            "overall_precision": precision_score(labels, predicted, average="weighted"),
            "overall_recall": recall_score(labels, predicted, average="weighted"),
            "overall_f1_score": f1_score(labels, predicted, average="weighted"),
        }
        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model's state dictionary to a file.

        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load the model's state dictionary from a file.

        Args:
            filepath (str): Path to the file from which to load the model.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


# class NNModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, 
#                  batch_size=32, lr=0.001, device=None, patience=5):
#         """
#         Trainer class for classification neural networks.

#         Args:
#             model_tuple: A tuple that may contain the model, optimizer, and criterion.
#             X_train: Training data (features).
#             y_train: Training labels.
#             X_test: Test data (features).
#             y_test: Test labels.
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """

#         # Unpack the model_tuple depending on how many items it contains
#         if isinstance(model, tuple):
#             if len(model) == 1:
#                 self.model = model[0]
#                 self.criterion = nn.CrossEntropyLoss()  # Default criterion for classification
#                 self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             elif len(model) == 2:
#                 self.model, self.criterion = model
#                 self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             elif len(model) == 3:
#                 self.model, self.optimizer, self.criterion = model
#             else:
#                 raise ValueError("Invalid model tuple length.")
#         else:
#             self.model = model
#             self.criterion = nn.CrossEntropyLoss()  # Default criterion for classification
#             self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)

#         # Split the data into training and validation sets (corrected)
#         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
#             torch.Tensor(X_train), torch.LongTensor(y_train), test_size=0.2, random_state=42
#         )

#         # Assign test data directly
#         self.X_test = torch.Tensor(X_test)
#         self.y_test = torch.LongTensor(y_test)

#         self.batch_size = batch_size
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)

#         # Add a patience parameter for early stopping
#         self.patience = patience
#         self.best_loss = float('inf')
#         self.epochs_no_improve = 0

#     def _validate_model(self):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(self.X_val)
#             loss = self.criterion(outputs, self.y_val)
#         return loss.item()

#     def train_model(self, epochs=10):
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             self.model.train()
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             # Validation loss
#             val_loss = self._validate_model()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')

#             # Early stopping
#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 self.epochs_no_improve = 0
#             else:
#                 self.epochs_no_improve += 1

#             if self.epochs_no_improve >= self.patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break


#     def evaluate_model(self):
#         """
#         Evaluate the trained model on test data and return performance metrics, including per-class metrics.
#         """
#         if self.model is None:
#             raise ValueError("Model has not been trained.")
        
#         self.model.eval()  # Set the model to evaluation mode
#         with torch.no_grad():
#             inputs, labels = self.X_test.to(self.device), self.y_test.to(self.device)
#             outputs = self.model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             predicted = predicted.cpu().numpy()
#             labels = labels.cpu().numpy()
    
#         # Calculate metrics per class and global metrics
#         metrics = {
#             "accuracy": accuracy_score(labels, predicted),
#             "precision_per_class": precision_score(labels, predicted, average=None).tolist(),  # Convert to list
#             "recall_per_class": recall_score(labels, predicted, average=None).tolist(),  # Convert to list
#             "f1_score_per_class": f1_score(labels, predicted, average=None).tolist(),  # Convert to list
#             "overall_precision": precision_score(labels, predicted, average='weighted'),
#             "overall_recall": recall_score(labels, predicted, average='weighted'),
#             "overall_f1_score": f1_score(labels, predicted, average='weighted')
#         }
    
#         # Return both per-class and global metrics for better insights
#         return metrics

#     def save_model(self, filepath):
#         """
#         Save the trained model to a file.
#         """
#         torch.save(self.model.state_dict(), filepath)

#     def load_model(self, filepath):
#         """
#         Load a model from a file.
#         """
#         self.model.load_state_dict(torch.load(filepath))


# class NNModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, 
#                  batch_size=32, lr=0.001, device=None, patience=5):

#         """
#         Trainer class for classification neural networks.

#         Args:
#             model: The PyTorch model to train (or a tuple of model and criterion).
#             X_train: Training data (features).
#             y_train: Training labels.
#             X_test: Test data (features).
#             y_test: Test labels.
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """

#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)
        
#         # Handle the case where the model is returned as a tuple (model, criterion)
#         if isinstance(model, tuple):
#             print(f'Tuple MODEL: {model} \n')
#             self.model, self.criterion = model
#         else:
#             self.model = model
#             self.criterion = nn.CrossEntropyLoss()  # Default criterion for classification

#         self.X_train = torch.Tensor(X_train)
#         self.y_train = torch.LongTensor(y_train)
#         self.X_test = torch.Tensor(X_test)
#         self.y_test = torch.LongTensor(y_test)
#         self.batch_size = batch_size
#         self.lr = lr
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
#         # Add a patience parameter for early stopping
#         self.patience = patience
#         self.best_loss = float('inf')
#         self.epochs_no_improve = 0
        
#         # Split the data into training and validation sets
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         self.X_val = torch.Tensor(X_val).to(self.device)
#         self.y_val = torch.LongTensor(y_val).to(self.device)


#     def _validate_model(self):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(self.X_val)
#             loss = self.criterion(outputs, self.y_val)
#         return loss.item()

#     def train_model(self, epochs=10):
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             self.model.train()
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             # Validation loss
#             val_loss = self._validate_model()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')

#             # Early stopping
#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 self.epochs_no_improve = 0
#             else:
#                 self.epochs_no_improve += 1

#             if self.epochs_no_improve >= self.patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
    
#     # def train_model(self, epochs=10):
#     #     """
#     #     Train the classification model.
        
#     #     Args:
#     #         epochs (int): Number of epochs to train.
#     #     """
#     #     self.model.train()
#     #     dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#     #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#     #     for epoch in range(epochs):
#     #         running_loss = 0.0
#     #         for inputs, labels in dataloader:
#     #             inputs, labels = inputs.to(self.device), labels.to(self.device)
#     #             self.optimizer.zero_grad()
#     #             outputs = self.model(inputs)
#     #             loss = self.criterion(outputs, labels)
#     #             loss.backward()
#     #             self.optimizer.step()
#     #             running_loss += loss.item()
            
#     #         print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}')

    
    # def evaluate_model(self):
    #     """
    #     Evaluate the trained model on test data and return performance metrics, including per-class metrics.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model has not been trained.")
        
    #     self.model.eval()  # Set the model to evaluation mode
    #     with torch.no_grad():
    #         inputs, labels = self.X_test.to(self.device), self.y_test.to(self.device)
    #         outputs = self.model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         predicted = predicted.cpu().numpy()
    #         labels = labels.cpu().numpy()
    
    #     # Calculate metrics per class and global metrics
    #     metrics = {
    #         "accuracy": accuracy_score(labels, predicted),
    #         "precision_per_class": precision_score(labels, predicted, average=None).tolist(),  # Convert to list
    #         "recall_per_class": recall_score(labels, predicted, average=None).tolist(),  # Convert to list
    #         "f1_score_per_class": f1_score(labels, predicted, average=None).tolist(),  # Convert to list
    #         "overall_precision": precision_score(labels, predicted, average='weighted'),
    #         "overall_recall": recall_score(labels, predicted, average='weighted'),
    #         "overall_f1_score": f1_score(labels, predicted, average='weighted')
    #     }
    
    #     # Return both per-class and global metrics for better insights
    #     return metrics



#     # def evaluate_model(self):
#     #     """
#     #     Evaluate the trained classification model on test data and return performance metrics.
        
#     #     Returns:
#     #         dict: A dictionary containing accuracy, precision, recall, and F1 score.
#     #     """
#     #     self.model.eval()
#     #     with torch.no_grad():
#     #         inputs, labels = self.X_test.to(self.device), self.y_test.to(self.device)
#     #         outputs = self.model(inputs)
#     #         _, predicted = torch.max(outputs, 1)
#     #         predicted = predicted.cpu().numpy()
#     #         labels = labels.cpu().numpy()
        
#     #     metrics = {
#     #         "accuracy": accuracy_score(labels, predicted),
#     #         "precision": precision_score(labels, predicted, average='weighted'),
#     #         "recall": recall_score(labels, predicted, average='weighted'),
#     #         "f1_score": f1_score(labels, predicted, average='weighted')
#     #     }
#     #     return metrics
    
    # def save_model(self, filepath):
    #     """
    #     Save the trained model to a file.
    #     """
    #     torch.save(self.model.state_dict(), filepath)

    # def load_model(self, filepath):
    #     """
    #     Load a model from a file.
    #     """
    #     self.model.load_state_dict(torch.load(filepath))



class NNRegressionModelTrainer:
    """
    Trainer class for regression neural networks.

    This class handles training, validation, evaluation, saving, and loading for regression models.
    The model can be provided as a single instance or as a tuple containing the model and optionally a
    custom optimizer and/or loss criterion.

    Args:
        model: Either an instance of torch.nn.Module or a tuple containing:
            - (model,): Only the model is provided. The default criterion (MSELoss) and optimizer (Adam)
              are used.
            - (model, criterion): A custom loss criterion is provided; optimizer defaults to Adam.
            - (model, optimizer, criterion): Custom optimizer and loss criterion are provided.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray or list): Training targets (continuous values).
        X_test (np.ndarray): Test features.
        y_test (np.ndarray or list): Test targets (continuous values).
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        device (Optional[Union[str, torch.device]]): Device to use ('cuda' or 'cpu'). If None, auto-detects.
        patience (int): Number of epochs with no improvement before triggering early stopping.
    """

    def __init__(
        self,
        model: Union[nn.Module, Tuple],
        X_train: np.ndarray,
        y_train: Union[np.ndarray, list],
        X_test: np.ndarray,
        y_test: Union[np.ndarray, list],
        batch_size: int = 32,
        lr: float = 0.001,
        device: Optional[Union[str, torch.device]] = None,
        patience: int = 5
    ):
        
        
        # Ensure data is a NumPy array if it is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        
        
        # Unpack the model if provided as a tuple.
        if isinstance(model, tuple):
            if len(model) == 1:
                self.model = model[0]
                self.criterion = nn.MSELoss()
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.scheduler = None
            elif len(model) == 2:
                self.model, second_element = model
                if isinstance(second_element, nn.Module):  # If the second element is another model (unlikely)
                    raise ValueError("Unexpected model structure.")
                elif isinstance(second_element, optim.Optimizer):  # If optimizer is passed
                    self.optimizer = second_element
                    self.criterion = nn.MSELoss()
                else:  # Assume second_element is criterion
                    self.criterion = second_element
                    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.scheduler = None
            elif len(model) == 3:
                self.model, self.optimizer, self.criterion = model
                self.scheduler = None
            elif len(model) == 4:
                self.model, self.optimizer, self.criterion, self.scheduler = model
            else:
                raise ValueError("Invalid model tuple length. Expected 1, 2, 3 or 4 elements.")
                
        else:
            self.model = model
            self.criterion = nn.MSELoss()  # Default loss for regression
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = None

        # If input data are NumPy arrays and have 3 dimensions (e.g., grayscale images), add a channel dimension.
        if isinstance(X_train, np.ndarray) and X_train.ndim == 3:
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        # Set device: if a device is provided as a string or torch.device, use it; otherwise, auto-detect.
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Ensure input data are NumPy arrays for splitting.
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)


        # Split the training data into training and validation sets.
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Convert the NumPy arrays to PyTorch tensors and move them to the selected device.
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32).to(self.device)
        self.X_val = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)        
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)       
        
        
        # Check if it's single-output regression (target shape [batch_size]) or multi-output ([batch_size, n_outputs])
        if y_train_np.ndim == 1 or (y_train_np.ndim == 2 and y_train_np.shape[1] == 1):
            # For single-output regression, ensure targets have shape [batch_size, 1]
            self.y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(self.device)
            self.y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1).to(self.device)
            self.y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            # For multi-output regression, keep the original shape
            self.y_train = torch.tensor(y_train_np, dtype=torch.float32).to(self.device)
            self.y_val = torch.tensor(y_val_np, dtype=torch.float32).to(self.device)
            self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)


        



        self.batch_size = batch_size
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        # Early stopping parameters.
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def _validate_model(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: The validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            loss = self.criterion(outputs, self.y_val)
        return loss.item()

    def train_model(self, epochs: int = 10) -> None:
        """
        Train the model for a specified number of epochs with early stopping based on validation loss.

        Args:
            epochs (int): Maximum number of epochs to train.
        """  
        
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Check for NaNs in model outputs.
                if torch.isnan(outputs).any():
                    print(f"WARNING: NaN detected in model outputs at epoch {epoch + 1}")

                loss = self.criterion(outputs, targets)
                # Check for NaNs in loss.
                if torch.isnan(loss):
                    print(f"ERROR: Loss became NaN at epoch {epoch + 1}! Stopping training.")
                    return  # Stop training if loss is NaN

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            val_loss = self._validate_model()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Early stopping check.
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    def evaluate_model(self) -> dict:
        """
        Evaluate the trained regression model on test data and return performance metrics.

        Returns:
            dict: A dictionary containing the mean squared error and R^2 score.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            outputs_np = outputs.cpu().numpy()
            targets_np = self.y_test.cpu().numpy()

        metrics = {
            "mean_squared_error": mean_squared_error(targets_np, outputs_np),
            "r2_score": r2_score(targets_np, outputs_np)
        }
        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model's state dictionary to a file.

        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load the model's state dictionary from a file.

        Args:
            filepath (str): Path to the file from which to load the model.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


# class NNRegressionModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, 
#                  batch_size=32, lr=0.001, device=None, patience=5):
#         """
#         Trainer class for regression neural networks.

#         Args:
#             model_tuple: A tuple that may contain the model, optimizer, and criterion.
#             X_train: Training data (features).
#             y_train: Training labels (continuous values).
#             X_test: Test data (features).
#             y_test: Test labels (continuous values).
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """

#         # Unpack the model_tuple depending on how many items it contains
#         if isinstance(model, tuple):
#             if len(model) == 1:
#                 self.model = model[0]
#                 self.criterion = nn.MSELoss()  # Default criterion for regression
#                 self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             elif len(model) == 2:
#                 self.model, self.criterion = model
#                 self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             elif len(model) == 3:
#                 self.model, self.optimizer, self.criterion = model
#             else:
#                 raise ValueError("Invalid model tuple length.")
#         else:
#             self.model = model
#             self.criterion = nn.MSELoss()  # Default criterion for regression
#             self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)
            
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Ensure that the input data is converted to PyTorch tensors
#         if isinstance(X_train, np.ndarray):
#             X_train = torch.tensor(X_train, dtype=torch.float32)
#             X_test = torch.tensor(X_test, dtype=torch.float32)
#             y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure correct shape
#             y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # Ensure correct shape


#         # Split the data into training and validation sets
#         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         # Debugging NaNs before conversion
#         print("Checking NaNs in training data before tensor conversion:")
#         print(f"NaN count in X_train: {np.isnan(X_train).sum()}")
#         print(f"NaN count in y_train: {np.isnan(y_train).sum()}")
#         print(f"NaN count in X_test: {np.isnan(X_test).sum()}")
#         print(f"NaN count in y_test: {np.isnan(y_test).sum()}")
        
#         # Convert to tensors
#         self.X_train = torch.Tensor(X_train).to(self.device)
#         self.y_train = torch.Tensor(y_train).view(-1, 1).to(self.device)
#         self.X_test = torch.Tensor(X_test).to(self.device)
#         self.y_test = torch.Tensor(y_test).view(-1, 1).to(self.device)
        
#         # Debugging after tensor conversion
#         print("Checking NaNs after tensor conversion:")
#         print(f"NaN count in X_train tensor: {torch.isnan(self.X_train).sum().item()}")
#         print(f"NaN count in y_train tensor: {torch.isnan(self.y_train).sum().item()}")


#         # Convert test data
#         self.X_test = torch.Tensor(X_test).to(self.device)
#         self.y_test = torch.Tensor(y_test).view(-1, 1).to(self.device)

#         self.batch_size = batch_size
#         self.model = self.model.to(self.device)

#         # Add a patience parameter for early stopping
#         self.patience = patience
#         self.best_loss = float('inf')
#         self.epochs_no_improve = 0

#     def _validate_model(self):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(self.X_val)
#             loss = self.criterion(outputs, self.y_val)
#         return loss.item()

#     def train_model(self, epochs=10):
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             self.model.train()
#             for inputs, targets in dataloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
                
#                 if torch.isnan(outputs).sum().item() > 0:
#                     print(f"WARNING: NaN detected in model outputs at epoch {epoch + 1}")
                
#                 loss = self.criterion(outputs, targets)
                
#                 if torch.isnan(loss).sum().item() > 0:
#                     print(f"ERROR: Loss became NaN at epoch {epoch + 1}! Stopping training.")
#                     return  # Exit training to prevent further NaN propagation
                
                
#                 loss = self.criterion(outputs, targets)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             # Validation loss
#             val_loss = self._validate_model()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')

#             # Early stopping
#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 self.epochs_no_improve = 0
#             else:
#                 self.epochs_no_improve += 1

#             if self.epochs_no_improve >= self.patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break

#     def evaluate_model(self):
#         """
#         Evaluate the trained regression model on test data and return performance metrics.
        
#         Returns:
#             dict: A dictionary containing mean squared error and R^2 score.
#         """
#         self.model.eval()
#         with torch.no_grad():
#             inputs, targets = self.X_test.to(self.device), self.y_test.to(self.device)
#             outputs = self.model(inputs)
#             outputs = outputs.cpu().numpy()
#             targets = targets.cpu().numpy()

#         metrics = {
#             "mean_squared_error": mean_squared_error(targets, outputs),
#             "r2_score": r2_score(targets, outputs)
#         }
#         return metrics
    
#     def save_model(self, filepath):
#         """
#         Save the trained model to a file.
#         """
#         torch.save(self.model.state_dict(), filepath)

#     def load_model(self, filepath):
#         """
#         Load a model from a file.
#         """
#         self.model.load_state_dict(torch.load(filepath))


# class NNRegressionModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, batch_size=32, lr=0.001, device=None, patience=5):
#         """
#         Trainer class for regression neural networks.

#         Args:
#             model: The PyTorch model to train (or a tuple of model and criterion).
#             X_train: Training data (features).
#             y_train: Training labels (continuous values).
#             X_test: Test data (features).
#             y_test: Test labels (continuous values).
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """

#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)
        
#         # Handle the case where the model is returned as a tuple (model, criterion)
#         if isinstance(model, tuple):
#             self.model, self.criterion = model
#         else:
#             self.model = model
#             self.criterion = nn.MSELoss()  # Default criterion for regression

#         self.X_train = torch.Tensor(X_train)
#         self.y_train = torch.Tensor(y_train).view(-1, 1)  # Ensure y is a column vector
#         self.X_test = torch.Tensor(X_test)
#         self.y_test = torch.Tensor(y_test).view(-1, 1)
#         self.batch_size = batch_size
#         self.lr = lr
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         # Add a patience parameter for early stopping
#         self.patience = patience
#         self.best_loss = float('inf')
#         self.epochs_no_improve = 0
        
#         # Split the data into training and validation sets
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         self.X_val = torch.Tensor(X_val).to(self.device)
#         self.y_val = torch.Tensor(y_val).view(-1, 1).to(self.device)

#     def train_model(self, epochs=10):
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             self.model.train()
#             for inputs, targets in dataloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             # Validation loss
#             val_loss = self._validate_model()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')

#             # Early stopping
#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 self.epochs_no_improve = 0
#             else:
#                 self.epochs_no_improve += 1

#             if self.epochs_no_improve >= self.patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break

#     def _validate_model(self):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(self.X_val)
#             loss = self.criterion(outputs, self.y_val)
#         return loss.item()


    # def train_model(self, epochs=10):
    #     """
    #     Train the regression model.

    #     Args:
    #         epochs (int): Number of epochs to train.
    #     """
    #     self.model.train()
    #     dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         for inputs, targets in dataloader:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             self.optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs, targets)
    #             loss.backward()
    #             self.optimizer.step()
    #             running_loss += loss.item()

    #         print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}')

    # def evaluate_model(self):
    #     """
    #     Evaluate the trained regression model on test data and return performance metrics.
        
    #     Returns:
    #         dict: A dictionary containing mean squared error and R^2 score.
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         inputs, targets = self.X_test.to(self.device), self.y_test.to(self.device)
    #         outputs = self.model(inputs)
    #         outputs = outputs.cpu().numpy()
    #         targets = targets.cpu().numpy()

    #     metrics = {
    #         "mean_squared_error": mean_squared_error(targets, outputs),
    #         "r2_score": r2_score(targets, outputs)
    #     }
    #     return metrics
    
    # def save_model(self, filepath):
    #     """
    #     Save the trained model to a file.
    #     """
    #     torch.save(self.model.state_dict(), filepath)

    # def load_model(self, filepath):
    #     """
    #     Load a model from a file.
    #     """
    #     self.model.load_state_dict(torch.load(filepath))



# class NNModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, batch_size=32, lr=0.001, device=None):
#         """
#         Trainer class for classification neural networks.

#         Args:
#             model: The PyTorch model to train.
#             X_train: Training data (features).
#             y_train: Training labels.
#             X_test: Test data (features).
#             y_test: Test labels.
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """
        
#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)
        
#         self.model = model
#         self.X_train = torch.Tensor(X_train)
#         self.y_train = torch.LongTensor(y_train)
#         self.X_test = torch.Tensor(X_test)
#         self.y_test = torch.LongTensor(y_test)
#         self.batch_size = batch_size
#         self.lr = lr
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
#     def train_model(self, epochs=10):
#         """
#         Train the classification model.
        
#         Args:
#             epochs (int): Number of epochs to train.
#         """
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()
            
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}')

#     def evaluate_model(self):
#         """
#         Evaluate the trained classification model on test data and return performance metrics.
        
#         Returns:
#             dict: A dictionary containing accuracy, precision, recall, and F1 score.
#         """
#         self.model.eval()
#         with torch.no_grad():
#             inputs, labels = self.X_test.to(self.device), self.y_test.to(self.device)
#             outputs = self.model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             predicted = predicted.cpu().numpy()
#             labels = labels.cpu().numpy()
        
#         metrics = {
#             "accuracy": accuracy_score(labels, predicted),
#             "precision": precision_score(labels, predicted, average='weighted'),
#             "recall": recall_score(labels, predicted, average='weighted'),
#             "f1_score": f1_score(labels, predicted, average='weighted')
#         }
#         return metrics
    
#     def save_model(self, filepath):
#         """
#         Save the trained model to a file.
#         """
#         torch.save(self.model.state_dict(), filepath)

#     def load_model(self, filepath):
#         """
#         Load a model from a file.
#         """
#         self.model.load_state_dict(torch.load(filepath))


# class NNRegressionModelTrainer:
#     def __init__(self, model, X_train, y_train, X_test, y_test, batch_size=32, lr=0.001, device=None):
#         """
#         Trainer class for regression neural networks.

#         Args:
#             model: The PyTorch model to train.
#             X_train: Training data (features).
#             y_train: Training labels (continuous values).
#             X_test: Test data (features).
#             y_test: Test labels (continuous values).
#             batch_size (int): Batch size for training.
#             lr (float): Learning rate for the optimizer.
#             device: Device to use ('cuda' or 'cpu').
#         """

#         # Add channel dimension if missing (for grayscale images)
#         if len(X_train.shape) == 3:  # If shape is (batch_size, height, width)
#             X_train = np.expand_dims(X_train, axis=1)
#             X_test = np.expand_dims(X_test, axis=1)
        
#         self.model = model
#         self.X_train = torch.Tensor(X_train)
#         self.y_train = torch.Tensor(y_train).view(-1, 1)  # Ensure y is a column vector
#         self.X_test = torch.Tensor(X_test)
#         self.y_test = torch.Tensor(y_test).view(-1, 1)
#         self.batch_size = batch_size
#         self.lr = lr
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)
#         self.criterion = nn.MSELoss()  # Using Mean Squared Error Loss for regression
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

#     def train_model(self, epochs=10):
#         """
#         Train the regression model.

#         Args:
#             epochs (int): Number of epochs to train.
#         """
#         self.model.train()
#         dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             for inputs, targets in dataloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}')

#     def evaluate_model(self):
#         """
#         Evaluate the trained regression model on test data and return performance metrics.
        
#         Returns:
#             dict: A dictionary containing mean squared error and R^2 score.
#         """
#         self.model.eval()
#         with torch.no_grad():
#             inputs, targets = self.X_test.to(self.device), self.y_test.to(self.device)
#             outputs = self.model(inputs)
#             outputs = outputs.cpu().numpy()
#             targets = targets.cpu().numpy()

#         metrics = {
#             "mean_squared_error": mean_squared_error(targets, outputs),
#             "r2_score": r2_score(targets, outputs)
#         }
#         return metrics
    
#     def save_model(self, filepath):
#         """
#         Save the trained model to a file.
#         """
#         torch.save(self.model.state_dict(), filepath)

#     def load_model(self, filepath):
#         """
#         Load a model from a file.
#         """
#         self.model.load_state_dict(torch.load(filepath))
