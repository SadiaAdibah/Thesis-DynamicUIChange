from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import numpy as np
import pandas as pd

try:
    from keystrokes.features import FEATURE_COLS
except ImportError:
    from features import FEATURE_COLS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, None, None, 1.0
    bundle = joblib.load(path)
    return (
        bundle.get("model"),
        bundle.get("label_encoder"),
        bundle.get("scaler"),
        bundle.get("neutral_penalty", 1.0),
    )


model, label_encoder, scaler, neutral_penalty = load_model()

if model is None:
    print("WARNING: model.joblib not found. Run train.py before making predictions.")


@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model, label_encoder, scaler

    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON payload received."}), 400

    missing = [col for col in FEATURE_COLS if col not in data]
    if missing:
        return jsonify({"error": f"Missing feature columns: {missing}"}), 400

    X = pd.DataFrame([{col: float(data[col]) for col in FEATURE_COLS}]).values

    if scaler is not None:
        X = scaler.transform(X)

    probs = model.predict_proba(X)[0]
    raw_prob_dict = {
        label_encoder.classes_[i]: round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    adjusted = np.copy(probs)
    if "Neutral" in label_encoder.classes_:
        neutral_col = list(label_encoder.classes_).index("Neutral")
        adjusted[neutral_col] *= neutral_penalty
    if adjusted.sum() > 0:
        adjusted /= adjusted.sum()

    pred_idx = int(np.argmax(adjusted))
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(adjusted))

    adjusted_prob_dict = {
        label_encoder.classes_[i]: round(float(adjusted[i]), 4)
        for i in range(len(adjusted))
    }

    return jsonify({
        "emotion": label,
        "confidence": round(confidence, 3),
        "neutral_penalty": float(neutral_penalty),
        "probabilities": raw_prob_dict,
        "probabilities_adjusted": adjusted_prob_dict,
    })


@app.route("/reload", methods=["POST"])
def reload_model():
    """Hot-reload the model after retraining without restarting the server."""
    global model, label_encoder, scaler
    model, label_encoder, scaler = load_model()
    if model is None:
        return jsonify({"error": "model.joblib not found."}), 404
    return jsonify({"status": "Model reloaded successfully."})


@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check — also confirms whether the model is loaded."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": label_encoder.classes_.tolist() if label_encoder is not None else [],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000)