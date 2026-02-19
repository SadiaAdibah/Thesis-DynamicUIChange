from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import cv2
import numpy as np

# Load the model
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'c' to capture an image and detect emotion, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show live webcam feed
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Convert BGR to RGB and to PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        # Run the model
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        print(f"Predicted Emotion: {emotions[pred]}, Confidence: {probs[0][pred].item():.2f}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
