import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

def objective(trial):
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./dataset/split_garbage"
    train_loader, val_loader, class_counts, num_classes = get_data_loaders(data_dir=data_dir)

    # Class Weights
    weights = [1.0 / class_counts[i] for i in range(num_classes)]
    total = sum(weights)
    weights = [w / total for w in weights]
    class_weights = torch.FloatTensor(weights).to(device)

    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])

    # Model
    model = WasteClassifierNN(input_shape=(3, 160, 160), dropout=dropout).to(device)
    if use_focal_loss:
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    model.train()
    for epoch in range(15):  # increased epochs for better optimization
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # more trials for better tuning

    print("\n\u2705 Best Hyperparameters:")
    print(study.best_params)

    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))
