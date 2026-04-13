import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CnnMnist
import torch
from typing import List



def plot_loss(train_loss:np.ndarray, test_loss: np.ndarray):
    """
    Plot the training and test loss.

    Args:
        train_loss (np.ndarray): Train loss values over epochs.
        test_loss (np.ndarray): Test loss values over epochs.
    """
    num_epochs = range(len(train_loss))

    plt.figure(figsize = (8, 5))
    plt.plot(
        num_epochs, 
        train_loss, 
        color = "blue", 
        linewidth = 3, 
        alpha = 0.6, 
        label = "Train Loss"
    )
    plt.plot(
        num_epochs, 
        test_loss, 
        color = "red", 
        linewidth = 3, 
        alpha = 0.6, 
        label = "Test Loss"
    )
    plt.tight_layout()
    plt.legend()
    plt.show()


def visualize_kernels(model):
    """
    Plot the kernels of the first Convolutional layer of the model.

    Args:
        model: CNN model.
    """
    # Select the first convolutional layer
    conv1 = model.convo_network[0]
    # Get the weights
    conv1_kernels = conv1.weight.data.cpu()
    # Number of kernels from shape (out_channel, in_channel, kernel_h, kernel_w)
    kernels_num = conv1_kernels.shape[0]

    fig, axes = plt.subplots(1, kernels_num, figsize = (kernels_num * 2, 5))
    for i in range(kernels_num):
        # First channel of each kernel.
        kernel = conv1_kernels[i, 0]
        axes[i].imshow(kernel, cmap = "gray")
        axes[i].set_title(f"Filter {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, images):
    """
    Plot the first and second feature maps of the model with the images and predictions.

    Args:
        model: CNN model.
        images: Input images from the MNIST dataset.
    """
    # Convolutional Layers of the model
    conv1 = model.convo_network[0]
    relu1 = model.convo_network[1]
    conv2 = model.convo_network[2]
    relu2 = model.convo_network[3]

    # Move the images to the same device as model
    device = next(model.parameters()).device
    images = images.to(device)

    # First feature maps
    feature_map1 = relu1(conv1(images))
    feature_map2 = relu2(conv2(feature_map1))

    # Model predictions for the images
    predictions = model(images).argmax(dim = 1).cpu()

    # For plotting we move the layers and images to  cpu
    images = images.detach().cpu()
    feature_map1 = feature_map1.detach().cpu()
    feature_map2 = feature_map2.detach().cpu()
    
    # Plot
    num_images = images.shape[0]
    fig, axes = plt.subplots(num_images, 3, figsize = (9, num_images * 3))

    for i in range(num_images):
        # First feature map
        axes[i, 0].imshow(feature_map1[i, 0], cmap = "gray")
        axes[i, 0].set_title("First Feature Map")
        axes[i, 0].axis("off")

        # Second feature map
        axes[i, 1].imshow(feature_map2[i, 0], cmap = "gray")
        axes[i, 1].set_title("Second Feature Map")
        axes[i, 1].axis("off")

        # Third feature map
        axes[i, 2].imshow(images[i, 0], cmap = "gray")
        axes[i, 2].set_title(f"Model Prediction = {predictions[i].item()}")
        axes[i, 2].axis("off")

    plt.tight_layout()    
    plt.show()


if __name__ == "__main__":
    
    # Load train and test loss
    train_loss = np.load("train_loss.npy")
    test_loss = np.load("test_loss.npy")

    # Load parameters
    model_parameters = torch.load("parameters.pth")

    # Create model
    model = CnnMnist()
    model.load_state_dict(model_parameters)

    # Load MNIST dataset
    data = datasets.MNIST(
        root = "data", 
        train = False, 
        transform = transforms.ToTensor()
    )
    data_loader = DataLoader(data, batch_size = 5, shuffle = True)

    #Get images from data_loader
    images, _ = next(iter(data_loader))

    # Call the functions
    plot_loss(train_loss, test_loss)
    visualize_kernels(model)
    visualize_predictions(model, images)
