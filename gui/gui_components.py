import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_arch import WasteClassifierNN

def load_model(model_path="waste_classifier.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = WasteClassifierNN(input_shape=(3, 160, 160), dropout=0.22588767769727539).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = [
        "battery", "biological", "brown-glass", "cardboard", "clothes",
        "green-glass", "metal", "paper", "plastic", "shoes", "white-glass", "trash"
    ]
    return model, device, class_names

def transform_frame():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def predict_class(model, input_tensor, class_names):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
