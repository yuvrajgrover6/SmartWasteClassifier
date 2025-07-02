# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Class names
import cv2
from gui_components import load_model, transform_frame, predict_class

# Load model and class names
model, device, class_names = load_model()
transform = transform_frame()

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convert frame for model input
    input_tensor = transform(frame).unsqueeze(0).to(device)

    # Predict
    predicted_label = predict_class(model, input_tensor, class_names)

    # Draw fixed bounding box (centered)
    h, w, _ = frame.shape
    box_size = 400
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, predicted_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show
    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
