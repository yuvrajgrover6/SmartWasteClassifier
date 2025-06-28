import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_arch import WasteClassifierNN
from utils.data_loader import get_data_loaders


## Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print ("Using device", device)


data_dir = "./dataset/garbage_classification"
train_loader,val_loader = get_data_loaders(data_dir=data_dir)



# Defining the model , loss fiunction and optimizer
model = WasteClassifierNN().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)


# training the model
num_epochs = 5

for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")
    model.train()

    running_loss = 0.0
    
    for images,labels in train_loader:
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
    
    with torch.no_grad():
        for images,labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100*correct/total
    avg_val_loss = val_loss/len(val_loader)        
    print(f" Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "waste_classifier.pth")
print("🧠 Model saved as waste_classifier.pth")

