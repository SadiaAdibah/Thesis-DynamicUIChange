from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained(
    'abhilash88/face-emotion-detection'
)
model = ViTForImageClassification.from_pretrained(
    'abhilash88/face-emotion-detection'
)

image = Image.open('resources/sad.jpeg').convert("RGB")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Predicted Emotion:", emotions[pred])
print("Confidence:", probs[0][pred].item())