from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
from collections import defaultdict
import time
import math
from datetime import datetime

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===========================
# SERVICE ENDPOINTS
# ===========================
FACE_SERVICE_URL = "http://127.0.0.1:5000"
VOICE_SERVICE_URL = "http://127.0.0.1:8000"
KEYSTROKE_SERVICE_URL = "http://127.0.0.1:9000"

# ===========================
# WEIGHTS & CONFIG
# ===========================

# Source reliability weights (prior, will be dynamically modulated)
SOURCE_BASE_WEIGHTS = {
    "face":      0.50,   # most spatially reliable
    "keystroke": 0.30,   # behavioural, consistent
    "voice":     0.20,   # noisiest signal
}

# Recency half-life in seconds: readings lose half their weight after this many seconds.
# Rule of thumb: set to ~60% of WINDOW_DURATION so oldest readings still carry ~35% weight.
#   WINDOW_DURATION=5   → RECENCY_HALF_LIFE=3
#   WINDOW_DURATION=120 → RECENCY_HALF_LIFE=72
#   WINDOW_DURATION=180 → RECENCY_HALF_LIFE=108
RECENCY_HALF_LIFE = 3.0   # scaled to current 5s window

# Bayesian prior weight: how strongly the previous aggregated state pulls the new one
# 0 = ignore history, 1 = fully trust history
PRIOR_STRENGTH = 0.25

# Emotions and their semantic similarity matrix (for soft distance scoring)
EMOTIONS = ["happy", "focused", "neutral", "stressed", "sad"]

# Pairwise similarity (0=opposite, 1=same).
# Encodes that "focused" is close to "neutral", "stressed" is closer to "sad" than "happy", etc.
EMOTION_SIMILARITY = {
    ("happy",    "happy"):    1.00,
    ("happy",    "focused"):  0.55,
    ("happy",    "neutral"):  0.40,
    ("happy",    "stressed"): 0.10,
    ("happy",    "sad"):      0.05,

    ("focused",  "focused"):  1.00,
    ("focused",  "neutral"):  0.65,
    ("focused",  "happy"):    0.55,
    ("focused",  "stressed"): 0.30,
    ("focused",  "sad"):      0.15,

    ("neutral",  "neutral"):  1.00,
    ("neutral",  "focused"):  0.65,
    ("neutral",  "happy"):    0.40,
    ("neutral",  "stressed"): 0.35,
    ("neutral",  "sad"):      0.30,

    ("stressed", "stressed"): 1.00,
    ("stressed", "sad"):      0.45,
    ("stressed", "neutral"):  0.35,
    ("stressed", "focused"):  0.30,
    ("stressed", "happy"):    0.10,

    ("sad",      "sad"):      1.00,
    ("sad",      "stressed"): 0.45,
    ("sad",      "neutral"):  0.30,
    ("sad",      "focused"):  0.15,
    ("sad",      "happy"):    0.05,
}

def emotion_similarity(a, b):
    """Return similarity score in [0,1] between two emotion labels."""
    a, b = a.lower(), b.lower()
    return EMOTION_SIMILARITY.get((a, b), EMOTION_SIMILARITY.get((b, a), 0.0))


# Raw labels from face/voice/keystroke services → canonical 5-emotion vocabulary.
# Any label containing these substrings maps to the canonical emotion.
_STRESS_TRIGGERS = {
    "stressed":  ["stress", "angry", "anger", "fear", "disgust", "contempt",
                  "furious", "frustrated", "panic", "anxious", "anxiety",
                  "irritat", "hostile", "rage", "tense", "tension"],
    "sad":       ["sad", "depress", "grief", "sorrow", "unhappy", "miserable",
                  "disappoint", "lonely", "hopeless"],
    "happy":     ["happy", "joy", "excited", "surprise", "content", "elat",
                  "cheerful", "delight", "amused", "pleasure"],
    "focused":   ["focused", "concentrat", "engaged", "attentive", "calm",
                  "determin", "confident"],
    "neutral":   ["neutral", "normal", "rest"],
}

def normalize_emotion(raw: str) -> str:
    """
    Map any raw emotion label from an upstream service into the canonical
    5-emotion vocabulary: happy | focused | neutral | stressed | sad.

    Matching is substring-based and case-insensitive, with 'stressed' checked
    first so angry/fear/disgust are caught before the neutral fallback.
    Order: stressed → sad → happy → focused → neutral.
    """
    lower = raw.strip().lower()
    # Check in priority order (stressed first — most important for UX)
    for canonical in ("stressed", "sad", "happy", "focused", "neutral"):
        for trigger in _STRESS_TRIGGERS[canonical]:
            if trigger in lower:
                return canonical
    # Unknown label → neutral (safe default)
    return "neutral"


WINDOW_DURATION = 5   # seconds — keep in sync with solve.html WINDOW_DURATION
# ↑ when scaling to 2-3 min windows, also update RECENCY_HALF_LIFE above

# ===========================
# STATE
# ===========================

emotion_results = {
    "face": None,
    "voice": None,
    "keystroke": None
}

# Each entry: {"emotion": str, "confidence": float, "timestamp": float}
emotion_window = {
    "face":      [],
    "voice":     [],
    "keystroke": [],
}

window_start_time = time.time()

# Bayesian carry-over: tracks distribution from the previous window
# {"happy": 0.1, "focused": 0.6, ...}
prior_distribution = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}  # start uniform


# ===========================
# HELPER FUNCTIONS
# ===========================

def reset_window():
    """Reset the emotion window for a new cycle. Carry over prior from last result."""
    global window_start_time, emotion_window
    window_start_time = time.time()
    emotion_window = {"face": [], "voice": [], "keystroke": []}


def get_window_progress():
    elapsed = time.time() - window_start_time
    return min(elapsed / WINDOW_DURATION, 1.0)


def add_emotion_to_window(source, emotion, confidence=0.8):
    """Add a timestamped, confidence-weighted reading to the window.
    Raw upstream labels (angry, fear, disgust, etc.) are normalized first.
    """
    if source in emotion_window:
        emotion_window[source].append({
            "emotion": normalize_emotion(emotion),
            "confidence": float(confidence),
            "timestamp": time.time(),
        })


def recency_weight(timestamp):
    """
    Exponential decay weight based on age of reading.
    w = 2^(-(age / half_life))
    Fresh reading → 1.0, reading RECENCY_HALF_LIFE seconds old → 0.5.
    """
    age = time.time() - timestamp
    return math.pow(2.0, -age / RECENCY_HALF_LIFE)


def build_source_distribution(readings):
    """
    For a list of readings from ONE source, produce a probability distribution
    over EMOTIONS using confidence-weighted, recency-decayed soft voting.

    Each reading votes for its emotion proportional to:
        vote_weight = recency_weight(ts) * confidence

    The vote is spread softly across all emotions using the similarity matrix,
    so a "stressed" reading also contributes a small amount to "sad".

    Returns: dict {emotion: unnormalized_score}
    """
    scores = defaultdict(float)
    total_weight = 0.0

    for r in readings:
        rw = recency_weight(r["timestamp"]) * r["confidence"]
        total_weight += rw
        for emotion in EMOTIONS:
            sim = emotion_similarity(r["emotion"], emotion)
            scores[emotion] += rw * sim

    if total_weight == 0:
        return None  # no data

    # Normalize to a probability distribution
    total_score = sum(scores.values())
    if total_score == 0:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    return {e: scores[e] / total_score for e in EMOTIONS}


def calculate_weighted_emotion():
    """
    Multi-source Bayesian fusion:

    1. Build a per-source soft distribution using recency-decayed, confidence-
       weighted similarity voting.
    2. Fuse sources with their base weights (renormalized for available sources).
    3. Apply a Bayesian prior from the previous window (controlled by PRIOR_STRENGTH).
    4. Pick the argmax emotion and compute a confidence metric (1 − entropy).

    Returns: (emotion_str, confidence_float, full_distribution_dict)
    """
    global prior_distribution

    fused = defaultdict(float)
    available_weight = 0.0

    for source, readings in emotion_window.items():
        dist = build_source_distribution(readings)
        if dist is None:
            continue  # source has no data this window
        w = SOURCE_BASE_WEIGHTS[source]
        for e, p in dist.items():
            fused[e] += w * p
        available_weight += w

    if available_weight == 0:
        # No data at all → return flat prior
        return "neutral", 0.0, {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}

    # Normalize fused distribution over available sources
    for e in EMOTIONS:
        fused[e] /= available_weight

    # ── Bayesian blending with prior ──────────────────────────────────────────
    # posterior ∝ (1 - PRIOR_STRENGTH) * fused + PRIOR_STRENGTH * prior
    blended = {}
    for e in EMOTIONS:
        blended[e] = (1.0 - PRIOR_STRENGTH) * fused[e] + PRIOR_STRENGTH * prior_distribution[e]

    # Re-normalize
    total = sum(blended.values())
    for e in EMOTIONS:
        blended[e] /= total

    # ── Update prior for next window ──────────────────────────────────────────
    prior_distribution = dict(blended)

    # ── Pick winner ──────────────────────────────────────────────────────────
    best_emotion = max(blended, key=blended.get)
    best_prob = blended[best_emotion]

    # Confidence via normalized entropy:  conf = 1 - H/H_max
    H_max = math.log(len(EMOTIONS))
    H = -sum(p * math.log(p + 1e-9) for p in blended.values())
    confidence = max(0.0, 1.0 - H / H_max)

    return best_emotion, round(confidence, 4), dict(blended)


def top_contributing_sources(distribution):
    """
    Return which sources contributed most to the current window, for diagnostics.
    """
    source_info = {}
    for source, readings in emotion_window.items():
        if readings:
            dist = build_source_distribution(readings)
            if dist:
                dominant = max(dist, key=dist.get)
                source_info[source] = {
                    "count": len(readings),
                    "dominant_emotion": dominant,
                    "dominant_prob": round(dist[dominant], 3),
                }
    return source_info


# ===========================
# SERVE STATIC FILES
# ===========================

@app.route("/", methods=["GET"])
def serve_index():
    try:
        return send_from_directory("static", "index.html")
    except Exception:
        return send_from_directory('.', 'index.html')


@app.route("/solve.html", methods=["GET"])
def serve_solve():
    try:
        return send_from_directory("static", "solve.html")
    except Exception:
        return send_from_directory('.', 'solve.html')


# ===========================
# FACE DETECTION ENDPOINT
# ===========================

@app.route("/api/detect-face", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({"error": "No image received", "source": "face"}), 400
    try:
        file = request.files["image"]
        response = requests.post(f"{FACE_SERVICE_URL}/detect", files={"image": file})
        result = response.json()
        emotion = result.get("emotion", "neutral")
        confidence = result.get("confidence", 0.85)

        emotion_results["face"] = emotion
        add_emotion_to_window("face", emotion, confidence)

        return jsonify({
            "emotion": emotion,
            "source": "face",
            "confidence": confidence,
            "window_progress": get_window_progress(),
        })
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Face service not running on http://127.0.0.1:5000", "source": "face"}), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "face"}), 500


# ===========================
# VOICE EMOTION ENDPOINT
# ===========================

@app.route("/api/detect-voice", methods=["POST"])
def detect_voice():
    if "file" not in request.files:
        return jsonify({"error": "No audio file received", "source": "voice"}), 400
    try:
        file = request.files["file"]
        response = requests.post(f"{VOICE_SERVICE_URL}/predict-emotion", files={"file": file})
        result = response.json()

        if not response.ok:
            error_text = result.get("error", "Voice service returned an error")
            if any(pattern in error_text for pattern in [
                "Decoding failed",
                "ffmpeg returned error code",
                "Invalid data found",
                "Error opening input",
                "detected only with low score",
                "ffmpeg exited"
            ]):
                return ('', 204)
            return jsonify({"error": error_text, "source": "voice"}), response.status_code

        if result.get("emotion") is None:
            if result.get("error") and any(pattern in result["error"] for pattern in [
                "Decoding failed",
                "ffmpeg returned error code",
                "Invalid data found",
                "Error opening input",
                "detected only with low score",
                "ffmpeg exited"
            ]):
                return ('', 204)
            return jsonify({"error": result.get("error", "No emotion"), "source": "voice"}), 502

        emotion = result.get("emotion")
        confidence = result.get("confidence", 0.75)
        emotion_results["voice"] = emotion
        add_emotion_to_window("voice", emotion, confidence)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "source": "voice",
            "window_progress": get_window_progress(),
        })
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Voice service not running on http://127.0.0.1:8000", "source": "voice"}), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "voice"}), 500


# ===========================
# KEYSTROKE EMOTION ENDPOINT
# ===========================

@app.route("/api/detect-keystroke", methods=["POST"])
def detect_keystroke():
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No JSON payload received", "source": "keystroke"}), 400

        response = requests.post(f"{KEYSTROKE_SERVICE_URL}/predict", json=data)
        result = response.json()
        emotion = result.get("emotion", "neutral")
        confidence = result.get("confidence", 0.75)
        emotion_results["keystroke"] = emotion
        add_emotion_to_window("keystroke", emotion, confidence)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "source": "keystroke",
            "window_progress": get_window_progress(),
        })
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Keystroke service not running on http://127.0.0.1:9000", "source": "keystroke"}), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "keystroke"}), 500


# ===========================
# COMBINED EMOTION ENDPOINT
# ===========================

@app.route("/api/combined-emotion", methods=["GET"])
def get_combined_emotion():
    """
    Returns the Bayesian-fused emotion with:
    - Full probability distribution over all emotions
    - Entropy-based confidence score
    - Per-source breakdown for debugging
    """
    emotion, confidence, distribution = calculate_weighted_emotion()
    progress = get_window_progress()
    elapsed = time.time() - window_start_time

    sorted_distribution = sorted(distribution.items(), key=lambda x: -x[1])

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "distribution": {e: round(p, 4) for e, p in distribution.items()},
        "ranked": [{"emotion": e, "probability": round(p, 4)} for e, p in sorted_distribution],
        "window_progress": progress,
        "window_elapsed": int(elapsed),
        "window_duration": WINDOW_DURATION,
        "sources": top_contributing_sources(distribution),
        "message": f"Bayesian fusion complete ({progress:.0%} of window)"
    })


@app.route("/api/reset-emotion-window", methods=["POST"])
def reset_emotion_window_endpoint():
    reset_window()
    return jsonify({
        "status": "window_reset",
        "message": f"New {WINDOW_DURATION}-second emotion analysis window started"
    })


@app.route("/api/window-status", methods=["GET"])
def get_window_status():
    progress = get_window_progress()
    elapsed = time.time() - window_start_time
    remaining = max(0, WINDOW_DURATION - elapsed)
    return jsonify({
        "window_progress": progress,
        "window_elapsed": int(elapsed),
        "window_remaining": int(remaining),
        "window_duration": WINDOW_DURATION,
        "sources_count": {
            "face":      len(emotion_window["face"]),
            "keystroke": len(emotion_window["keystroke"]),
            "voice":     len(emotion_window["voice"]),
        }
    })


# ===========================
# STATUS ENDPOINT
# ===========================

@app.route("/api/status", methods=["GET"])
def get_status():
    services_status = {}
    for name, url in [("face", FACE_SERVICE_URL), ("voice", VOICE_SERVICE_URL), ("keystroke", KEYSTROKE_SERVICE_URL)]:
        try:
            requests.get(f"{url}/", timeout=2)
            services_status[name] = "running"
        except:
            services_status[name] = "unavailable"

    return jsonify({
        "services": services_status,
        "emotions": emotion_results,
        "integration_port": 7000
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "integrated-emotion-detection"})


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED EMOTION DETECTION SERVER (Bayesian Fusion)")
    print("=" * 60)
    print(f"\n✓ Running on http://127.0.0.1:7000")
    print(f"✓ Emotion window duration: {WINDOW_DURATION}s")
    print(f"✓ Recency half-life: {RECENCY_HALF_LIFE}s")
    print(f"✓ Prior carry-over strength: {PRIOR_STRENGTH*100:.0f}%")
    print("\nAggregation method: Bayesian multi-source fusion")
    print("  1. Per-source: recency-decayed × confidence × soft similarity voting")
    print("  2. Cross-source: reliability-weighted fusion")
    print("  3. Temporal: Bayesian prior from previous window")
    print("  4. Confidence: normalized entropy of posterior distribution")
    print("\nSource base weights:")
    for src, w in SOURCE_BASE_WEIGHTS.items():
        print(f"  {src:12s}: {w*100:.0f}%")
    print("\n" + "=" * 60)
    app.run(debug=True, host="127.0.0.1", port=7000)