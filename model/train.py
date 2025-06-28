import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_arch import WasteClassifierNN
from utils.data_loader import get_data_loaders
from collections import Counter


## Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print ("Using device", device)


data_dir = "./dataset/split_garbage"
train_loader,val_loader = get_data_loaders(data_dir=data_dir)

targets = [label for _, label in train_loader.dataset.imgs]
class_counts = Counter(targets)
num_classes = len(train_loader.dataset.classes)
print("Number of classes:", num_classes)
weights = [1.0 / class_counts[i] for i in range(num_classes)]
total = sum(weights)
weights = [w / total for w in weights]
class_weights = torch.FloatTensor(weights).to(device)



# Defining the model , loss fiunction and optimizer
model = WasteClassifierNN().to(device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(),lr=1e-4)


# training the model
num_epochs = 20

for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")
    model.train()

    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Confusion Matrix Data"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = train_loader.dataset.classes

    print("\n🧾 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Save the trained model
torch.save(model.state_dict(), "waste_classifier.pth")
print("Model saved as waste_classifier.pth")

