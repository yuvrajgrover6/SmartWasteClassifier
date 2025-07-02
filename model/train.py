# train.py — YOLOv8 training script

from functools import cache
from ultralytics import YOLO

# Load model (choose yolov8n.yaml or yolov8s.yaml based on performance vs speed)
model = YOLO('yolov5n.yaml')  # or yolov8s.yaml

# Train
model.train(
    data='dataset/waste_dataset.yaml',
    imgsz=320,
    epochs=5,
    batch=32,
    device='mps',
    name='debug_yolov8',
    pretrained=True,
    cache='disk'  # avoids RAM overload
)


