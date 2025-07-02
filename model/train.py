# train.py — YOLOv8 training script

from ultralytics import YOLO

# Load model (choose yolov8n.yaml or yolov8s.yaml based on performance vs speed)
model = YOLO('yolov8n.yaml')  # or yolov8s.yaml

# Train
model.train(
    data='dataset/waste_dataset.yaml',
    imgsz=640,
    epochs=50,
    batch=16,
    device='mps',  # 'cuda' for GPU, 'cpu' otherwise
    name='waste_yolov8_model',
    pretrained=True
)
