# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from gui_components import load_model, draw_boxes

# Load YOLO model
model, device = load_model()

# Webcam init
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # use CAP_DSHOW on Windows

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("📷 Camera initialized. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    # YOLO expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(rgb_frame)

    # Draw bounding boxes
    frame = draw_boxes(results, frame)

    # Show frame
    cv2.imshow("Waste Detector - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
