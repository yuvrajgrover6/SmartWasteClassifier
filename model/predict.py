import torch
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_arch import WasteClassifierNN

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)



model = WasteClassifierNN().to(device=device)
model.load_state_dict(torch.load("waste_classifier.pth",map_location=device))
model.eval()
print("Model loaded successfully")



# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def predict_image(image_path,class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _,predicted = torch.max(outputs,1)
    predicted_class = class_names[predicted.item()]
    return predicted_class


class_names = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes"
]

test_image = "./sample.png"

predicted = predict_image(test_image,class_names=class_names)
print(predicted)