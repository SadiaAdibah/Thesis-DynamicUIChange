from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pydub import AudioSegment
import tempfile

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===========================
# Load emotion model
# ===========================
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Map voice model labels to face model labels
label_map = {
    "angry": "Angry",
    "calm": "Neutral",
    "disgust": "Disgust",
    "fearful": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprised": "Surprise",
}

print("Available emotion labels:", model.config.id2label)

# ===========================
# Routes
# ===========================

@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "index.html")


@app.route("/predict-emotion", methods=["POST"])
def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]

    try:
        audio_bytes = file.read()
        extension = None
        if file.filename and '.' in file.filename:
            extension = file.filename.rsplit('.', 1)[-1].lower()

        with tempfile.NamedTemporaryFile(suffix=f".{extension or 'webm'}") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            try:
                if extension:
                    audio = AudioSegment.from_file(tmp.name, format=extension)
                else:
                    audio = AudioSegment.from_file(tmp.name)
            except Exception:
                audio = AudioSegment.from_file(tmp.name)

            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            waveform, sr = librosa.load(wav_io, sr=16000, mono=True)

        inputs = feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()
        raw_emotion = model.config.id2label[predicted_id]
        emotion = label_map.get(raw_emotion, "Neutral")

        return jsonify({"emotion": emotion, "confidence": round(confidence, 3)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
