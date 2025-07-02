import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Custom model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_arch import WasteClassifierNN  # Updated model with dropout support

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model with same dropout used during training (e.g., from Optuna best_params)
model = WasteClassifierNN(input_shape=(3, 160, 160), dropout=0.22588767769727539).to(device)  # <-- Adjust dropout to best one
model.load_state_dict(torch.load("waste_classifier.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")

# Transforms (must match your training `val_transform`)
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # match training size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Prediction function
def predict_image(image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

# Update this list based on your dataset's folders
class_names = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "white-glass", "trash"
]

# Path to test image
test_image = "cloth2.webp"


# Predict
predicted = predict_image(test_image, class_names=class_names)
print("🔍 Predicted class:", predicted)
