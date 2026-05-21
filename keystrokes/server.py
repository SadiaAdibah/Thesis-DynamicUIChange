from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import numpy as np

from features import FEATURE_COLS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, None, None
    bundle = joblib.load(path)
    return bundle.get("model"), bundle.get("label_encoder"), bundle.get("scaler")


model, label_encoder, scaler = load_model()


@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model, label_encoder
    if model is None:
        return jsonify({"error": "Model not found. Train model first using train.py"}), 500

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON payload received"}), 400

    missing = [col for col in FEATURE_COLS if col not in data]
    if missing:
        return jsonify({"error": f"Missing feature columns: {missing}"}), 400

    import pandas as pd

    df_row = pd.DataFrame([{col: float(data[col]) for col in FEATURE_COLS}])
    X = df_row.values

    if scaler is not None:
        X = scaler.transform(X)

    probs = model.predict_proba(X)[0]
    pred_enc = int(model.predict(X)[0])
    label = label_encoder.inverse_transform([pred_enc])[0]
    confidence = float(np.max(probs))

    return jsonify({"emotion": label, "confidence": round(confidence, 3)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000)
