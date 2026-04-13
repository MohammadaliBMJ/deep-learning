import torch
import torch.nn as nn


class CnnMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.convo_network = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, X):
        return self.convo_network(X)