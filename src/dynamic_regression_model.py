def load_model(X_train, y_train):
    """
    Returns a robust feedforward neural network for regression, designed to prevent NaN loss issues.
    
    Args:
        X_train (np.ndarray or torch.Tensor): Training data to infer input dimensions.
        y_train (np.ndarray or torch.Tensor): Training labels.
    
    Returns:
        nn.Module: A more stable regression neural network.
    """
    
    import torch.nn as nn

    class StableRegressionNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(StableRegressionNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)  # Normalize activations
            self.act1 = nn.LeakyReLU(0.01)  # Prevent dead neurons
            
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.act2 = nn.LeakyReLU(0.01)
            
            self.dropout = nn.Dropout(0.2)  # Prevent overfitting
            
            
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            
            self.fc3 = nn.Linear(hidden_dim, output_dim)  
        def forward(self, x):
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            
            x = self.fc2(x)
            x = self.norm2(x)
            x = self.act2(x)
            
            x = self.dropout(x)
            
            x = self.fc3(x)  # No activation function here for regression
            return x

    # Infer input_dim from X_train
    input_dim = X_train.shape[1] if len(X_train.shape) > 1 else 1  
    hidden_dim = 128  # Increased hidden units for better learning

    # Initialize and return model
    model = StableRegressionNN(input_dim, hidden_dim)

    # **Apply Proper Weight Initialization**
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    
    return model


