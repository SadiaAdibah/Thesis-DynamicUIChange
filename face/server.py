import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)  # 🔥 FIXES CORS ERROR

# ======================
# LOAD MODEL
# ======================
print("Loading model...")

processor = ViTImageProcessor.from_pretrained(
    "abhilash88/face-emotion-detection"
)
model = ViTForImageClassification.from_pretrained(
    "abhilash88/face-emotion-detection"
)

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Model loaded!")

# ======================
# DETECTION ROUTE
# ======================
@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"emotion": "No image received"})

    file = request.files["image"]

    img = Image.open(file.stream).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"emotion": "No face detected"})

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    inputs = processor(pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    emotion = emotions[pred]

    return jsonify({"emotion": emotion})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
