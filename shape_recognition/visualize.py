import numpy as np
import torch
from dataset import ShapeDataset
from torch.utils.data import DataLoader
from model import CnnShape
import matplotlib.pyplot as plt


def plot_loss_accuracy(train_loss: np.ndarray, test_loss:np.ndarray, 
              train_accuracy: np.ndarray, test_accuracy: np.ndarray):
    """
    Plot train and test loss, accuracy over epochs.

    Args:
        train_loss (np.ndarray): Train losses over the epochs.
        test_loss (np.ndarray): Test losses over the epochs. 
        train_accuracy (np.ndarray): Train accuracy over epochs.
        test_accuracy (np.ndarray): Test accuracy over epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    
    axes[0].plot(train_loss, color = "blue", linewidth = 3, alpha = 0.6, label = "Train Loss")
    axes[0].plot(test_loss, color = "red", linewidth = 3, alpha = 0.6, label = "Test Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(train_accuracy, color = "blue", linewidth = 3, alpha = 0.6, label = "Train Accuracy")
    axes[1].plot(test_accuracy, color = "red", linewidth = 3, alpha = 0.6, label = "Test Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_fm(model, images):
    """
    Plot the feature map of given images after the second conv layer.

    Args:
        model: Cnn Model.
        images: Images from 3 different shapes ('circle', 'square', 'triangle')
    """
    # Classes
    model_class = {
            0: "circle",
            1: "square",
            2: "triangle"
        }
    
    # Move to same device as the model
    device = next(model.parameters()).device
    images = images.to(device)

    fm = model(images, True)
    predictions = torch.argmax(model(images), dim = 1)

    # For plot move to cpu
    images = images.detach().cpu()
    fm = fm.detach().cpu()

    fig, axes = plt.subplots(images.size(0), 2, figsize = (8, images.size(0) * 2))

    for i in range(images.size(0)):
        class_prediction = model_class[predictions[i].item()]

        axes[i, 0].imshow(fm[i, 0], cmap = "gray")
        axes[i, 0].set_title("Feature Map")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(images[i, 0], cmap = "gray")
        axes[i, 1].set_title(f"Model prediction: {class_prediction}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # Load training and loss data
    train_loss = np.load("train_loss.npy")
    test_loss = np.load("test_loss.npy")

    train_accuracy = np.load("train_accuracy.npy")
    test_accuracy = np.load("test_accuracy.npy")

    # Load parameters
    parameters = torch.load("parameters.pth")
    
    # Load images
    images_dataset = ShapeDataset("data/test")
    data_loader = DataLoader(images_dataset, batch_size = 3, shuffle = True)
    images, _ = next(iter(data_loader))

    # Model with parameters
    model = CnnShape()
    model.load_state_dict(parameters)

    plot_fm(model, images)
    plot_loss_accuracy(train_loss, test_loss, train_accuracy, test_accuracy)

