import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


## Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print ("Using device", device)


##Loading the dataset
transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      
        std=[0.229, 0.224, 0.225]
        )
    ]
)
data_dir = "../dataset/garbage_classification"
datasets = datasets.ImageFolder(root=data_dir,transform=transforms)