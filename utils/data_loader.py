from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from collections import Counter

def get_data_loaders(data_dir, batch_size=32):
    # Augmentations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(160, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # Class balancing using WeightedRandomSampler
    targets = [label for _, label in train_dataset.samples]
    class_counts = Counter(targets)
    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes)

    # Weights per sample
    class_weights = [1.0 / class_counts[i] for i in targets]
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader,class_counts,num_classes
