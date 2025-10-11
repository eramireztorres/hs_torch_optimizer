def load_model(X_train, y_train):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FlexibleImageRegressionNN(nn.Module):
        def __init__(self, input_shape, output_dim):
            super(FlexibleImageRegressionNN, self).__init__()
            num_channels, img_height, img_width = input_shape

            self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Dynamically compute flattened size
            with torch.no_grad():
                dummy_input = torch.zeros(1, num_channels, img_height, img_width)
                dummy_output = self._forward_conv(dummy_input)
                flattened_size = dummy_output.view(1, -1).size(1)

            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, output_dim)

        def _forward_conv(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x

        def forward(self, x):
            x = self._forward_conv(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Ensure input has 4 dimensions: (batch_size, channels, height, width)
    if len(X_train.shape) != 4:
        raise ValueError(
            "Expected input with 4 dimensions (batch_size, channels, height, width)"
        )

    input_shape = X_train.shape[1:]  # (channels, height, width)
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    return FlexibleImageRegressionNN(input_shape, output_dim)
