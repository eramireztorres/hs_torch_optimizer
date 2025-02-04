def load_model(X_train, y_train):
    """
    Returns a flexible convolutional neural network (CNN) for image classification.
    Automatically infers input dimensions and adjusts for unknown output classes.
    """
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    import numpy as np

    class FlexibleImageClassificationNN(nn.Module):
        def __init__(self, num_channels, img_height, img_width, num_classes):
            super(FlexibleImageClassificationNN, self).__init__()
            self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Dynamically calculate the size of the flattened features after pooling
            flattened_size = 64 * (img_height // 4) * (img_width // 4)
            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Infer dimensions from X_train
    num_channels = X_train.shape[1]  
    img_height = X_train.shape[2]
    img_width = X_train.shape[3]

    # Dynamically infer the number of output classes from y_train
    if isinstance(y_train, torch.Tensor):
        num_classes = len(torch.unique(y_train))
    elif isinstance(y_train, np.ndarray):
        num_classes = len(np.unique(y_train))
    else:
        num_classes = 10  # Default fallback if y_train isn't provided correctly

    # Ensure at least 2 classes
    num_classes = max(2, num_classes)

    return FlexibleImageClassificationNN(num_channels, img_height, img_width, num_classes)
