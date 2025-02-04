def load_model(X_train, y_train, hidden_dim=64):
    import torch.nn as nn

    class SimpleRegressionNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleRegressionNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output multiple regression targets

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    input_dim = X_train.shape[1]  
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1


    return SimpleRegressionNN(input_dim, hidden_dim, output_dim)



