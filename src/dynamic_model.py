def load_model(X_train, y_train, hidden_dim=64):
    """
    Returns a simple feedforward neural network for classification, 
    with dynamic input and output dimensions inferred from the dataset.
    
    Args:
        X_train: The training feature dataset.
        y_train: The training labels.
        hidden_dim (int): Number of hidden units in the hidden layer.
    
    Returns:
        nn.Module: A PyTorch neural network model.
    """
    import torch.nn as nn
    
    class SimpleClassificationNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleClassificationNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Infer input_dim from X_train
    input_dim = X_train.shape[1]  # Assuming flat feature vectors
    
    # Infer output_dim from the number of unique labels in y_train
    output_dim = len(set(y_train))  # Number of unique classes
    
    return SimpleClassificationNN(input_dim, hidden_dim, output_dim)
