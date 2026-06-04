from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import logging
import os
import subprocess
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import tempfile

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)
app.logger.setLevel(logging.INFO)

VALID_AUDIO_FORMATS = {"wav", "webm", "mp3", "ogg", "flac", "m4a", "aac"}

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
# Audio decoding helper
# ===========================

def audio_bytes_to_waveform(audio_bytes: bytes, hint_ext=None):
    """
    Convert raw audio bytes to (waveform, sr=16000) using ffmpeg subprocess.

    Writes to a real temp file so ffmpeg can seek — avoids the
    'cache:pipe:0 / EBML header parsing failed' error that pydub triggers
    when it pipes webm bytes into ffmpeg (webm needs seekable input).
    """
    ext = hint_ext if hint_ext and hint_ext in VALID_AUDIO_FORMATS else "webm"

    in_tmp = out_tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes)
            in_tmp = f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_tmp = f.name

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_tmp,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_tmp],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {stderr[-800:]}")

        waveform, sr = librosa.load(out_tmp, sr=16000, mono=True)
        return waveform, sr

    finally:
        for path in [in_tmp, out_tmp]:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


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

        # Determine format hint from filename extension
        extension = None
        if file.filename and "." in file.filename:
            extension = file.filename.rsplit(".", 1)[-1].lower()
            if extension not in VALID_AUDIO_FORMATS:
                extension = None

        # Fallback: try content_type
        if not extension and file.content_type:
            ct_sub = file.content_type.split("/")[-1].split(";")[0].lower()
            if ct_sub in VALID_AUDIO_FORMATS:
                extension = ct_sub

        app.logger.info(
            "Received audio: filename=%s content_type=%s ext=%s size=%d",
            getattr(file, "filename", None),
            file.content_type,
            extension,
            len(audio_bytes),
        )

        if len(audio_bytes) < 1000:
            return jsonify({"error": "Audio too short"}), 400

        waveform, sr = audio_bytes_to_waveform(audio_bytes, extension)

        if len(waveform) < sr * 0.3:
            return jsonify({"error": "Audio clip too short for analysis"}), 400

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
        error_text = str(e)
        if any(pattern in error_text for pattern in [
            "Invalid data found",
            "Error opening input",
            "detected only with low score",
            "ffmpeg exited",
            "ffmpeg returned error code"
        ]):
            app.logger.warning("Audio decode failed for incoming voice data")
            app.logger.exception(e)
            return jsonify({"error": f"Decoding failed: {error_text}"}), 400

        app.logger.exception("Voice prediction failed")
        return jsonify({"error": error_text}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)