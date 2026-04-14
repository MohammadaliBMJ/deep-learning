import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ShapeDataset
from model import CnnShape
import numpy as np


# Load train and test data
train_data = ShapeDataset("data/train")
test_data = ShapeDataset("data/test")

train_load = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 4,
    pin_memory = True
)

test_load = DataLoader(
    test_data,
    batch_size = 64,
    num_workers = 4,
    pin_memory = True,
    shuffle = False
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
model = CnnShape()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)



train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

# Train
for i in range(20):
    model.train()
    batch_loss = 0
    total = 0
    batch_correct = 0

    # Train the model
    for images, labels in train_load:
        # Move to GPU
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)

        loss.backward()
        optimizer.step()

        # Add loss to batch_loss
        batch_loss += loss.item()

        # Accuracy
        predicts = torch.argmax(predictions, dim = 1)
        batch_correct += (labels == predicts).sum().item()
        total += labels.shape[0]

    # Epoch accuracy
    epoch_accuracy = batch_correct / total
    train_accuracy.append(epoch_accuracy)

    # Train_loss for this epoch
    epoch_loss = batch_loss / len(train_load)
    train_loss.append(epoch_loss)

    # Print progress
    if i % 5 == 0 or i == 19:
        print(f"Epochs: {i}, Train loss: {epoch_loss:.2f}, Accuracy: {epoch_accuracy:.2f}")

    model.eval()
    batch_loss = 0
    total = 0
    batch_correct = 0
    # Test Loss over epochs
    with torch.no_grad():
        for images, labels in test_load:
            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            batch_loss += loss.item()

            # Accuracy
            predicts = torch.argmax(predictions, dim = 1)
            batch_correct += (predicts == labels).sum().item()
            total += labels.shape[0]
    
    # Epoch accuracy
    epoch_accuracy = batch_correct / total
    test_accuracy.append(epoch_accuracy)

    # Epoch loss
    epoch_loss = batch_loss / len(test_load)
    test_loss.append(epoch_loss)

    # Epoch accuracy



# Save the losses. Will be used in visualize file
np.save("train_loss.npy", np.array(train_loss))
np.save("test_loss.npy", np.array(test_loss))

# Save accuracies
np.save("train_accuracy.npy", np.array(train_accuracy))
np.save("test_accuracy.npy", np.array(test_accuracy))

# Save parameters
torch.save(model.state_dict(), "parameters.pth")

