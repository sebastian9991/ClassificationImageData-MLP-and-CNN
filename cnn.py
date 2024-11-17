import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.image as mpimg

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomRotation
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm

"""SOURCE: Covnets google colab McGill COMP 551"""
def compute_accuracy(model, data_loader, device):
    """Compute the accuracy after a forward pass."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)  # Get logits
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices

            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total * 100  # Convert to percentage
    return accuracy

"""SOURCE: Covnets google colab McGill COMP 551"""
class CNN_basic(nn.Module): #I'm assuming greyscale images
    def __init__(self, num_filters, filter_size, pool_size, num_classes,strides, padding):
        """Constructur for Basic CNN."""
        super().__init__()
        self.model = nn.Sequential()
        #2CNN Layers
        self.model.append(nn.Conv2d(in_channels = 1, out_channels=num_filters, kernel_size=filter_size, stride=strides, padding=padding))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d(pool_size))


        self.model.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, stride=strides, padding=padding))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d(pool_size))
        
        #Flattening
        self.model.append(nn.Flatten()) #What is the size of flatten?

        self.model.append(nn.Linear(256, 256)) #Hidden Layer
        self.model.append(nn.ReLU())
        self.model.append(nn.Linear(256, 11))
        ## Softmax
    
    def forward(self, x):
        return self.model(x)
    
def train(model, optimizer, train_dataset, test_dataset, num_epochs, batch_size,  device):
    """
    Train function
    :param model: The architecture
    :param optimizer: Our optimzer object
    :param train_dataset: training set
    :param test_dataset: test_set
    :param num_epochs: epochs
    :param device: Device defined by torch
    """
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader):
            # Move tensors to the configured device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(images)
    
            # Calculate loss
            loss = criterion(outputs, labels)
    
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    history["loss"].append(running_loss/len(train_loader))
    history["accuracy"].append(compute_accuracy(model, train_loader, device))
    history["val_accuracy"].append(compute_accuracy(model, test_loader, device))
    print("Training Done.")
    return history