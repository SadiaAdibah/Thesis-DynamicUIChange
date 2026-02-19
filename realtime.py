import sys
import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from collections import deque
import time

# =========================
# LOAD MODEL
# =========================
print("Loading emotion model...")
processor = ViTImageProcessor.from_pretrained("abhilash88/face-emotion-detection")
model = ViTForImageClassification.from_pretrained("abhilash88/face-emotion-detection")
print("Model loaded!")

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

emotion_styles = {
    "Happy": "#FFE066",
    "Sad": "#6FA8DC",
    "Angry": "#FF6B6B",
    "Fear": "#B39DDB",
    "Surprise": "#FFD1DC",
    "Neutral": "#E0E0E0",
    "Disgust": "#8BC34A"
}

emotion_modes = {
    "Happy": "Joy Mode 😊",
    "Sad": "Calm Mode 🌧️",
    "Angry": "Cooling Mode 🔥",
    "Fear": "Safe Mode 😨",
    "Surprise": "Wow Mode 😲",
    "Neutral": "Normal Mode 😐",
    "Disgust": "Recovery Mode 🤢"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# APP
# =========================
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Adaptive App")
        self.resize(1000, 700)

        # ===== CAMERA LABEL =====
        self.camera_label = QLabel()
        
        # ===== INFO PANEL =====
        self.info_label = QLabel("Emotion: Neutral")
        self.info_label.setFont(QFont("Arial", 20))
        
        self.mode_label = QLabel("Mode: Normal")
        self.mode_label.setFont(QFont("Arial", 16))

        # ===== LAYOUT =====
        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.mode_label)
        self.setLayout(layout)

        # ===== CAMERA =====
        self.cap = cv2.VideoCapture(0)

        # ===== TIMER =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 🔥 SMOOTHING
        self.emotion_buffer = deque(maxlen=12)
        self.last_ui_update = time.time()
        self.current_emotion = "Neutral"

    def get_stable_emotion(self):
        if len(self.emotion_buffer) == 0:
            return "Neutral"
        return max(set(self.emotion_buffer), key=self.emotion_buffer.count)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        detected_emotion = "Neutral"
        confidence = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            inputs = processor(pil_img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()

            detected_emotion = emotions[pred]
            confidence = probs[0][pred].item()

            if confidence > 0.6:
                self.emotion_buffer.append(detected_emotion)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ===== STABLE EMOTION =====
        stable_emotion = self.get_stable_emotion()

        # ===== COOLDOWN =====
        if stable_emotion != self.current_emotion:
            if time.time() - self.last_ui_update > 2:
                self.current_emotion = stable_emotion
                self.last_ui_update = time.time()

        # ===== APPLY UI THEME =====
        color = emotion_styles[self.current_emotion]
        self.setStyleSheet(f"background-color: {color};")

        # ===== UPDATE TEXT PANEL =====
        self.info_label.setText(
            f"Emotion: {self.current_emotion}  | Confidence: {confidence:.2f}"
        )
        self.mode_label.setText(
            f"Mode: {emotion_modes[self.current_emotion]}"
        )

        # ===== SHOW CAMERA =====
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
