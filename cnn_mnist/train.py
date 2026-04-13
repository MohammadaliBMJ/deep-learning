import torch
import torch.nn as nn
import numpy as np
from model import CnnMnist
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root = "data",
    train = True,
    transform = transform,
    download = True
)
test_data = datasets.MNIST(
    root = "data",
    train = False,
    transform = transform,
    download = True
)

train_loader = DataLoader(
    train_data, 
    batch_size = 128, 
    shuffle = True, 
    num_workers = 4, 
    pin_memory = True
)
test_loader = DataLoader(
    test_data, 
    batch_size = 128, 
    num_workers = 4, 
    pin_memory = True
)

# Create the model
model = CnnMnist()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fun = nn.CrossEntropyLoss()

# Transfer to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train
epochs = range(20)
train_loss = []
test_loss = []

for i in epochs:
    batch_loss = 0
    # Train
    model.train()
    for images, labels in train_loader:
        # GPU
        images = images.to(device)
        labels = labels.to(device)

        predicts = model(images)
        loss = loss_fun(predicts, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store batch loss
        batch_loss += loss.item()

    # Compute epoch loss. Which is the average of the losses in the epoch
    epoch_loss = batch_loss / len(train_loader)
    train_loss.append(batch_loss / len(train_loader))
    # Print number of epochs
    if i % 5 == 0 or i == epochs[-1]:
        print(f"Epoch: {i} Completed. Loss = {epoch_loss:.4f}")
    
    # Test Loss
    model.eval()
    with torch.no_grad():
        batch_loss = 0
        for images, labels in test_loader:
            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)

            test_predicts = model(images)
            loss = loss_fun(test_predicts, labels)

            # Store loss
            batch_loss += loss.item()

    # Compute Epoch loss for test data
    epoch_loss = batch_loss / len(test_loader)
    test_loss.append(epoch_loss)

    


# Save test and train losses and we plot them in visualize.py
np.save("train_loss", np.array(train_loss))
np.save("test_loss", np.array(test_loss))

# Save the parameters of the model
torch.save(model.state_dict(), "parameters.pth")

