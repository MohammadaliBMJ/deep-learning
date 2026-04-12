import torch
import torch.nn as nn

class XOR(nn.Module):
    def __init__(self, hidden_num):
        super().__init__()

        self.neural_network = nn.Sequential(
            nn.Linear(2, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, 2)
        )

    def forward(self, x):
        return self.neural_network(x)