def load_model(X_train, y_train):
    """
    Returns a simple feedforward neural network for regression.
    Useful for flat feature vector inputs.

    Args:
        X_train (np.ndarray or torch.Tensor): Training data to infer input dimensions.
        y_train (np.ndarray or torch.Tensor): Training labels (not used directly here, but passed for consistency).
    
    Returns:
        nn.Module: A simple regression neural network.
    """
    
    import torch.nn as nn
    
    class SimpleRegressionNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(SimpleRegressionNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single value for regression

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Infer input_dim from X_train (number of features per sample)
    input_dim = X_train.shape[1] if len(X_train.shape) > 1 else 1  # Handle edge cases where X_train is 1D
    hidden_dim = 64  # You can adjust this or make it flexible

    return SimpleRegressionNN(input_dim, hidden_dim)

