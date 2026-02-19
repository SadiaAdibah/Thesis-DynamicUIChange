from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import cv2
import numpy as np

# Load your existing model
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to colors (BGR format for OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 128, 0),     # Dark Green
    'Fear': (128, 0, 128),      # Purple
    'Happy': (0, 255, 255),     # Yellow
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (0, 165, 255),  # Orange
    'Neutral': (200, 200, 200)  # Gray
}

# Load image (or later, you can replace this with webcam capture)
image_path = 'happy.jpeg'
image = Image.open(image_path).convert("RGB")

# Run your existing prediction
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

emotion_label = emotions[pred]
confidence = probs[0][pred].item()
print(f"Predicted Emotion: {emotion_label}, Confidence: {confidence:.2f}")

# Convert PIL image to OpenCV format
frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Change the background color based on emotion
color = emotion_colors[emotion_label]
overlay = np.full(frame.shape, color, dtype=np.uint8)
blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

# Add text with emotion and confidence
cv2.putText(blended, f"{emotion_label} ({confidence*100:.1f}%)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show the image with UI change
cv2.imshow("Emotion Detection", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
