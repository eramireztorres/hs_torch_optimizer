def load_model(X_train, y_train, hidden_dim=128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    class SimpleClassificationNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleClassificationNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    input_dim = X_train.shape[1]
    
    # Ensure y_train is converted to a numpy array for unique class counting
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.cpu().numpy()
    else:
        y_train_np = np.array(y_train)

    output_dim = len(np.unique(y_train_np))
    model = SimpleClassificationNN(input_dim, hidden_dim, output_dim)

    # Automatically calculate class weights using inverse frequency
    class_counts = np.bincount(y_train_np)
    class_weights = 1.0 / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    class_weights = torch.FloatTensor(class_weights)

    # Ensure the weights match the number of classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)

    return model, optimizer, criterion

