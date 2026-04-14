import torch.nn as nn
import torch.nn.functional as F


class CnnShape(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, padding = 1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding = 1)

        self.fc1 = nn.Linear(8 * 32 * 32, 32)
        self.fc2 = nn.Linear(32, 3)

        self.flatten = nn.Flatten()


    def forward(self, x, return_fm = False):
        # First convolution layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Second Convolution layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        fm = x

        # Fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Return feature map after second conv layer
        if return_fm:
            return fm

        return x        