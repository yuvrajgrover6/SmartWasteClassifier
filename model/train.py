import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Custom Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_arch import WasteClassifierNN
from utils.data_loader import get_data_loaders

# Optional: FocalLoss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data
data_dir = "./dataset/split_garbage"
train_loader, val_loader, class_counts, num_classes = get_data_loaders(data_dir=data_dir)

# Compute class weights
weights = [1.0 / class_counts[i] for i in range(num_classes)]
total = sum(weights)
weights = [w / total for w in weights]
class_weights = torch.FloatTensor(weights).to(device)

# Model
model = WasteClassifierNN(input_shape=(3, 160, 160), dropout=0.22588767769727539).to(device)

# Choose loss function
use_focal_loss = False  # Toggle this to True to use Focal Loss
if use_focal_loss:
    criterion = FocalLoss(gamma=2.0, weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer and Scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.00011702442681098273,
    weight_decay=0.0008819524113585738
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training settings
num_epochs = 45
train_losses, val_losses, val_accuracies = [], [], []

# Live plot setup
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

line1, = ax1.plot([], [], label='Train Loss')
line2, = ax1.plot([], [], label='Validation Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
ax1.grid(True)

line3, = ax2.plot([], [], label='Validation Accuracy', color='green')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy Over Epochs")
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Training Loop
for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_training_loss = running_loss / len(train_loader)
    train_losses.append(avg_training_loss)
    print(f"Epoch {epoch+1} Loss: {avg_training_loss:.4f}")

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
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
    scheduler.step(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # Classification Report
    print("\n🧾 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_loader.dataset.classes, zero_division=0))

    # Update live plot
    line1.set_xdata(range(1, len(train_losses) + 1))
    line1.set_ydata(train_losses)
    line2.set_xdata(range(1, len(val_losses) + 1))
    line2.set_ydata(val_losses)
    line3.set_xdata(range(1, len(val_accuracies) + 1))
    line3.set_ydata(val_accuracies)

    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    plt.pause(0.01)

# Save plots and model
plt.ioff()
plt.savefig("training_plots.png")
plt.show()
torch.save(model.state_dict(), "waste_classifier.pth")
print("Model saved as waste_classifier.pth")
