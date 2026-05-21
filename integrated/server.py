from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
from collections import defaultdict
import time
from datetime import datetime

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===========================
# SERVICE ENDPOINTS
# ===========================
FACE_SERVICE_URL = "http://127.0.0.1:5000"
VOICE_SERVICE_URL = "http://127.0.0.1:8000"
KEYSTROKE_SERVICE_URL = "http://127.0.0.1:9000"

# Emotion weights (face is most reliable, then keystroke, then voice)
EMOTION_WEIGHTS = {
    "face": 0.5,        # 50% - Face is most accurate
    "keystroke": 0.3,   # 30% - Keystroke second
    "voice": 0.2        # 20% - Voice less reliable
}

# Emotion to numeric score mapping (for weighted averaging)
EMOTION_SCORES = {
    "happy": 4,
    "focused": 3,
    "neutral": 2,
    "sad": 1,
    "stressed": 0
}

# Store emotion results from each source with timestamp
emotion_results = {
    "face": None,
    "voice": None,
    "keystroke": None
}

# Store emotion window data - collects emotions over 30 seconds
emotion_window = {
    "face": [],
    "voice": [],
    "keystroke": []
}

window_start_time = time.time()
WINDOW_DURATION = 30  # seconds

# ===========================
# HELPER FUNCTIONS
# ===========================

def reset_window():
    """Reset the emotion window for a new 30-second cycle"""
    global window_start_time, emotion_window
    window_start_time = time.time()
    emotion_window = {
        "face": [],
        "voice": [],
        "keystroke": []
    }

def get_window_progress():
    """Get the progress of the current window (0.0 to 1.0)"""
    elapsed = time.time() - window_start_time
    progress = min(elapsed / WINDOW_DURATION, 1.0)
    return progress

def add_emotion_to_window(source, emotion):
    """Add an emotion to the current window"""
    if source in emotion_window:
        emotion_window[source].append({
            "emotion": emotion,
            "timestamp": time.time()
        })

def calculate_weighted_emotion():
    """
    Calculate the final emotion based on weighted average of all sources.
    Uses emotions collected over the entire 30-second window.
    Returns the emotion string and confidence score.
    """
    weighted_scores = {}
    valid_weights = {}
    
    for source, emotions in emotion_window.items():
        if emotions:
            # Calculate average emotion score for this source
            scores = [EMOTION_SCORES.get(e["emotion"].lower(), 2) for e in emotions]
            avg_score = sum(scores) / len(scores)
            weight = EMOTION_WEIGHTS[source]
            weighted_scores[source] = avg_score * weight
            valid_weights[source] = weight
    
    # If no data collected, return neutral
    if not weighted_scores:
        return "neutral", 0.0
    
    # Normalize weights
    total_weight = sum(valid_weights.values())
    final_score = sum(weighted_scores.values()) / total_weight
    
    # Map score back to emotion
    closest_emotion = "neutral"
    closest_diff = float('inf')
    
    for emotion, score in EMOTION_SCORES.items():
        diff = abs(score - final_score)
        if diff < closest_diff:
            closest_diff = diff
            closest_emotion = emotion
    
    # Calculate confidence based on agreement rate
    all_emotions = []
    for emotions in emotion_window.values():
        all_emotions.extend([e["emotion"] for e in emotions])
    
    if all_emotions:
        # Higher confidence if emotions are consistent
        emotion_counts = defaultdict(int)
        for e in all_emotions:
            emotion_counts[e] += 1
        max_count = max(emotion_counts.values())
        confidence = max_count / len(all_emotions)
    else:
        confidence = 0.0
    
    return closest_emotion, confidence

# ===========================
# SERVE STATIC FILES
# ===========================

@app.route("/", methods=["GET"])
def serve_index():
    """Serve the main index.html"""
    try:
        return send_from_directory("static", "index.html")
    except Exception:
        # Fallback to serve index.html from the current package directory
        return send_from_directory('.', 'index.html')


@app.route("/solve.html", methods=["GET"])
def serve_solve():
    """Serve the solve.html"""
    try:
        return send_from_directory("static", "solve.html")
    except Exception:
        # Fallback to serve solve.html from the current package directory
        return send_from_directory('.', 'solve.html')


# ===========================
# FACE DETECTION ENDPOINT
# ===========================

@app.route("/api/detect-face", methods=["POST"])
def detect_face():
    """
    Receive webcam frame, send to face detection service
    Expected: multipart/form-data with 'image' field
    """
    if "image" not in request.files:
        return jsonify({"error": "No image received", "source": "face"}), 400

    try:
        file = request.files["image"]
        
        # Forward to face service
        response = requests.post(
            f"{FACE_SERVICE_URL}/detect",
            files={"image": file}
        )
        
        result = response.json()
        emotion = result.get("emotion")
        
        # Store in current window and also keep latest
        emotion_results["face"] = emotion
        add_emotion_to_window("face", emotion)
        
        return jsonify({
            "emotion": emotion,
            "source": "face",
            "confidence": 0.85,  # Face detection doesn't provide confidence
            "window_progress": get_window_progress()
        })
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Face service not running on http://127.0.0.1:5000",
            "source": "face"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "face"}), 500


# ===========================
# VOICE EMOTION ENDPOINT
# ===========================

@app.route("/api/detect-voice", methods=["POST"])
def detect_voice():
    """
    Receive audio file, send to voice emotion service
    Expected: multipart/form-data with 'file' field (webm audio)
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file received", "source": "voice"}), 400

    try:
        file = request.files["file"]
        
        # Forward to voice service
        response = requests.post(
            f"{VOICE_SERVICE_URL}/predict-emotion",
            files={"file": file}
        )
        
        result = response.json()
        if not response.ok:
            error_text = result.get("error", "Voice service returned an error")
            if "Decoding failed" in error_text or "ffmpeg returned error code" in error_text:
                return ('', 204)
            return jsonify({
                "error": error_text,
                "source": "voice"
            }), response.status_code

        if result.get("emotion") is None:
            if result.get("error") and ("Decoding failed" in result["error"] or "ffmpeg returned error code" in result["error"]):
                return ('', 204)
            return jsonify({
                "error": result.get("error", "Voice service returned no emotion"),
                "source": "voice"
            }), 502

        emotion_results["voice"] = result.get("emotion")
        emotion = result.get("emotion")
        add_emotion_to_window("voice", emotion)
        
        return jsonify({
            "emotion": emotion,
            "confidence": result.get("confidence", 0.8),
            "source": "voice",
            "window_progress": get_window_progress()
        })
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Voice service not running on http://127.0.0.1:8000",
            "source": "voice"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "voice"}), 500


# ===========================
# KEYSTROKE EMOTION ENDPOINT
# ===========================

@app.route("/api/detect-keystroke", methods=["POST"])
def detect_keystroke():
    """
    Receive keystroke features, send to keystroke emotion service
    Expected: JSON with keystroke timing features
    """
    try:
        data = request.json
        
        if data is None:
            return jsonify({
                "error": "No JSON payload received",
                "source": "keystroke"
            }), 400
        
        # Forward to keystroke service
        response = requests.post(
            f"{KEYSTROKE_SERVICE_URL}/predict",
            json=data
        )
        
        result = response.json()
        emotion_results["keystroke"] = result.get("emotion")
        emotion = result.get("emotion")
        add_emotion_to_window("keystroke", emotion)
        
        return jsonify({
            "emotion": emotion,
            "confidence": result.get("confidence", 0.8),
            "source": "keystroke",
            "window_progress": get_window_progress()
        })
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Keystroke service not running on http://127.0.0.1:9000",
            "source": "keystroke"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e), "source": "keystroke"}), 500


# ===========================
# COMBINED EMOTION ENDPOINT
# ===========================

@app.route("/api/combined-emotion", methods=["GET"])
def get_combined_emotion():
    """
    Get the combined emotion based on weighted average from all three sources
    over the current 30-second window.
    
    Weights:
    - Face: 50% (most reliable)
    - Keystroke: 30%
    - Voice: 20%
    """
    emotion, confidence = calculate_weighted_emotion()
    progress = get_window_progress()
    elapsed = time.time() - window_start_time
    
    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "window_progress": progress,
        "window_elapsed": int(elapsed),
        "window_duration": WINDOW_DURATION,
        "sources": {
            "face": len(emotion_window["face"]),
            "keystroke": len(emotion_window["keystroke"]),
            "voice": len(emotion_window["voice"])
        },
        "message": f"Combined emotion from collected data (window {progress:.0%} complete)"
    })


@app.route("/api/reset-emotion-window", methods=["POST"])
def reset_emotion_window_endpoint():
    """
    Reset the 30-second emotion aggregation window.
    Called when the user wants to start a new analysis cycle.
    """
    reset_window()
    return jsonify({
        "status": "window_reset",
        "message": "New 30-second emotion analysis window started"
    })


@app.route("/api/window-status", methods=["GET"])
def get_window_status():
    """
    Get current window status without updating emotions
    """
    progress = get_window_progress()
    elapsed = time.time() - window_start_time
    remaining = max(0, WINDOW_DURATION - elapsed)
    
    return jsonify({
        "window_progress": progress,
        "window_elapsed": int(elapsed),
        "window_remaining": int(remaining),
        "window_duration": WINDOW_DURATION,
        "sources_count": {
            "face": len(emotion_window["face"]),
            "keystroke": len(emotion_window["keystroke"]),
            "voice": len(emotion_window["voice"])
        }
    })


# ===========================
# STATUS ENDPOINT
# ===========================

@app.route("/api/status", methods=["GET"])
def get_status():
    """
    Get the status of all services and current emotion state
    """
    services_status = {}
    
    # Check face service
    try:
        requests.get(f"{FACE_SERVICE_URL}/", timeout=2)
        services_status["face"] = "running"
    except:
        services_status["face"] = "unavailable"
    
    # Check voice service
    try:
        requests.get(f"{VOICE_SERVICE_URL}/", timeout=2)
        services_status["voice"] = "running"
    except:
        services_status["voice"] = "unavailable"
    
    # Check keystroke service
    try:
        requests.get(f"{KEYSTROKE_SERVICE_URL}/", timeout=2)
        services_status["keystroke"] = "running"
    except:
        services_status["keystroke"] = "unavailable"
    
    return jsonify({
        "services": services_status,
        "emotions": emotion_results,
        "integration_port": 7000
    })


# ===========================
# HEALTH CHECK
# ===========================

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "service": "integrated-emotion-detection"})


# ===========================
# ERROR HANDLERS
# ===========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED EMOTION DETECTION SERVER")
    print("=" * 60)
    print("\n✓ Serving static files from /integrated/static/")
    print("✓ Running on http://127.0.0.1:7000")
    print(f"✓ Emotion window duration: {WINDOW_DURATION} seconds")
    print("\nEmotion Weights:")
    print(f"  - Face Detection:   {EMOTION_WEIGHTS['face']*100:.0f}%")
    print(f"  - Keystroke Analysis: {EMOTION_WEIGHTS['keystroke']*100:.0f}%")
    print(f"  - Voice Detection:  {EMOTION_WEIGHTS['voice']*100:.0f}%")
    print("\nExpected Backend Services:")
    print(f"  - Face Detection:   {FACE_SERVICE_URL}")
    print(f"  - Voice Emotion:    {VOICE_SERVICE_URL}")
    print(f"  - Keystroke Emotion: {KEYSTROKE_SERVICE_URL}")
    print("\nAPI Endpoints:")
    print("  POST /api/detect-face           - Send webcam frame")
    print("  POST /api/detect-voice          - Send audio file")
    print("  POST /api/detect-keystroke      - Send keystroke features")
    print("  GET  /api/combined-emotion      - Get weighted emotion result")
    print("  POST /api/reset-emotion-window  - Start new 30-second window")
    print("  GET  /api/window-status         - Get current window status")
    print("  GET  /api/status                - Get service status")
    print("  GET  /health                    - Health check")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host="127.0.0.1", port=7000)
