import gradio as gr
import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load emotion model
processor = ViTImageProcessor.from_pretrained("abhilash88/face-emotion-detection")
model = ViTForImageClassification.from_pretrained("abhilash88/face-emotion-detection")

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

emotion_colors = {
    "Happy": "#FFE066",
    "Sad": "#6FA8DC",
    "Angry": "#FF6B6B",
    "Fear": "#B39DDB",
    "Surprise": "#FFD1DC",
    "Neutral": "#E0E0E0",
    "Disgust": "#8BC34A"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_emotion(frame):

    if frame is None:
        return None, "No image", "#FFFFFF"

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return frame, "No face detected", "#FFFFFF"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    inputs = processor(pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    emotion = emotions[pred]
    confidence = probs[0][pred].item()

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, f"{emotion} ({confidence:.2f})", emotion_colors[emotion]


def update_ui(frame):
    img, text, color = detect_emotion(frame)

    css = f"""
    <style>
    .gradio-container {{
        background-color: {color} !important;
    }}
    </style>
    """

    return img, text, gr.HTML(css)


iface = gr.Interface(
    fn=update_ui,
    inputs=gr.Image(sources=["webcam"]),
    outputs=[
        gr.Image(label="Detected Face"),
        gr.Textbox(label="Emotion"),
        gr.HTML()
    ],
    live=True,
    title="🎭 Emotion Detector"
)

iface.launch()
