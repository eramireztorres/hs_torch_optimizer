def load_model(X_train, y_train):
    """
    Returns a simple convolutional neural network (CNN) for image regression.
    Automatically infers input dimensions (image size and channels) from X_train.

    Args:
        X_train (np.ndarray or torch.Tensor): Training data to infer input dimensions.
        y_train (np.ndarray or torch.Tensor): Training labels (not used directly here, but passed for consistency).
    
    Returns:
        nn.Module: A CNN for image regression.
    """
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleImageRegressionNN(nn.Module):
        def __init__(self, num_channels, img_height, img_width):
            super(SimpleImageRegressionNN, self).__init__()
            self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)  # Use inferred channels
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Dynamically calculate the size of the flattened features after pooling
            self.fc1 = nn.Linear(64 * (img_height // 4) * (img_width // 4), 128)  # Adjust based on image size
            self.fc2 = nn.Linear(128, 1)  # Single output for regression

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Infer dimensions from X_train
    num_channels = X_train.shape[1]  # Grayscale (1) or RGB (3)
    img_height = X_train.shape[2]  # Image height (e.g., 64)
    img_width = X_train.shape[3]  # Image width (e.g., 64)

    # Return the dynamically created regression model
    return SimpleImageRegressionNN(num_channels, img_height, img_width)
