import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from model import XOR


# XOR dataset
X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype = torch.float32)

y = torch.tensor([0, 1, 1, 0])

torch.manual_seed(10)

losses = {}

hidden_size = [1, 2, 3, 4, 5]
# Train the model
for h in hidden_size:
    model = XOR(h)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    losses[h] = []

    for i in range(500):
        pred = model(X)
        loss = loss_function(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store the loss for each model
        losses[h].append(loss.item())

    # Print the results for each model
    print(f"XOR model with {h} hidden neurons. Final predictions: {torch.argmax(pred, dim = 1)}")


steps = range(500)

plt.figure(figsize = (8, 5))
for h in hidden_size:
    plt.plot(steps, losses[h], label = f"h={h}", linewidth = 3, alpha = 0.5)

plt.xlabel("Number of Steps", fontsize = 14)
plt.ylabel("Loss", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.show()










