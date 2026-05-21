# Integrated Emotion Detection Server

This folder contains the integrated server that combines all three emotion detection pipelines:
- **Face Detection** (webcam)
- **Voice Emotion** (audio recording)  
- **Keystroke Dynamics** (typing pattern analysis)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATED SERVER                        │
│                    (port 7000)                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │  Face Detector  │  │ Voice Emotion   │  │ Keystroke    ││
│  │  (port 5000)    │  │ (port 8000)     │  │ (port 9000)  ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┤
│  │  HTML Interface (index.html, solve.html)               │
│  └─────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Emotion Detection

#### 1. Face Detection
**POST** `/api/detect-face`

Send a webcam frame to detect facial emotion.

**Request:**
```
Content-Type: multipart/form-data
Body: image (binary)
```

**Response:**
```json
{
  "emotion": "Happy",
  "source": "face",
  "confidence": 0.85
}
```

---

#### 2. Voice Emotion
**POST** `/api/detect-voice`

Send an audio file to detect emotion from voice.

**Request:**
```
Content-Type: multipart/form-data
Body: file (webm audio)
```

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.92,
  "source": "voice"
}
```

---

#### 3. Keystroke Emotion
**POST** `/api/detect-keystroke`

Send keystroke timing features to detect emotion from typing patterns.

**Request:**
```json
{
  "hold_time": 0.15,
  "prev_hold_time": 0.18,
  "down_to_down": 0.35,
  "up_to_down": 0.02,
  "up_to_up": 0.37,
  "down_to_up_prev": 0.19,
  "down_to_down2": 0.68,
  "down_to_up2": 0.54
}
```

**Response:**
```json
{
  "emotion": "Focused",
  "confidence": 0.78,
  "source": "keystroke"
}
```

---

#### 4. Combined Emotion
**GET** `/api/combined-emotion`

Get the combined emotion result using majority voting from all available sources.

**Response:**
```json
{
  "emotion": "Happy",
  "sources": {
    "face": "Happy",
    "voice": "Happy",
    "keystroke": "Neutral"
  },
  "message": "Combined emotion from 3 source(s)"
}
```

---

### System Status

#### Service Status
**GET** `/api/status`

Get the health status of all backend services.

**Response:**
```json
{
  "services": {
    "face": "running",
    "voice": "running",
    "keystroke": "running"
  },
  "emotions": {
    "face": "Happy",
    "voice": "Happy",
    "keystroke": "Focused"
  },
  "integration_port": 7000
}
```

---

#### Health Check
**GET** `/health`

Simple health check for the integrated server.

**Response:**
```json
{
  "status": "healthy",
  "service": "integrated-emotion-detection"
}
```

## Setup Instructions

### Prerequisites
- All three backend services must be running:
  - Face Detection Server (port 5000)
  - Voice Emotion Server (port 8000)
  - Keystroke Server (port 9000)

### Start Individual Services

In separate terminals, from the thesis-code root directory:

**Terminal 1: Face Detection**
```bash
source venv/bin/activate
cd face
python server.py
```

**Terminal 2: Voice Emotion**
```bash
source venv/bin/activate
cd voice
python server.py
```

**Terminal 3: Keystroke Emotion**
```bash
source venv/bin/activate
cd keystrokes
python server.py
```

**Terminal 4: Integrated Server**
```bash
source venv/bin/activate
cd integrated
python server.py
```

### Access the Interface

Open your browser and navigate to:
```
http://127.0.0.1:7000
```

## File Structure

```
integrated/
├── server.py              # Flask server (this file)
├── README.md              # Documentation
└── static/
    ├── index.html         # Landing page with emotion controls
    └── solve.html         # Problem-solving interface
```

## Testing the Integration

### 1. Check Service Status
```bash
curl http://127.0.0.1:7000/health
curl http://127.0.0.1:7000/api/status
```

### 2. Test Face Detection
```bash
# Requires a webcam frame as image
curl -X POST -F "image=@test_face.jpg" http://127.0.0.1:7000/api/detect-face
```

### 3. Test Voice Detection
```bash
# Requires a webm audio file
curl -X POST -F "file=@test_audio.webm" http://127.0.0.1:7000/api/detect-voice
```

### 4. Test Keystroke Detection
```bash
curl -X POST http://127.0.0.1:7000/api/detect-keystroke \
  -H "Content-Type: application/json" \
  -d '{
    "hold_time": 0.15,
    "prev_hold_time": 0.18,
    "down_to_down": 0.35,
    "up_to_down": 0.02,
    "up_to_up": 0.37,
    "down_to_up_prev": 0.19,
    "down_to_down2": 0.68,
    "down_to_up2": 0.54
  }'
```

## Integration Points

### Frontend Integration

The HTML files should make AJAX requests to these endpoints:

```javascript
// Face emotion detection
fetch('/api/detect-face', {
    method: 'POST',
    body: formData  // FormData with image
})
.then(res => res.json())
.then(data => console.log(data.emotion))

// Voice emotion detection
fetch('/api/detect-voice', {
    method: 'POST',
    body: formData  // FormData with audio file
})
.then(res => res.json())
.then(data => console.log(data.emotion))

// Keystroke emotion detection
fetch('/api/detect-keystroke', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features)
})
.then(res => res.json())
.then(data => console.log(data.emotion))

// Get combined emotion
fetch('/api/combined-emotion')
.then(res => res.json())
.then(data => console.log(data.emotion))
```

## Emotion Labels

Both face and voice services use these standardized emotion labels:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

## Combining Emotions

The server uses **majority voting** to combine emotions from multiple sources:
- If 2 or more sources agree on an emotion, that emotion is selected
- If all sources disagree, the most recent source's emotion is used

## Environment Variables (Optional)

You can customize service URLs by setting environment variables:
```bash
export FACE_SERVICE_URL=http://127.0.0.1:5000
export VOICE_SERVICE_URL=http://127.0.0.1:8000
export KEYSTROKE_SERVICE_URL=http://127.0.0.1:9000
```

## Troubleshooting

### "Service not running" errors
- Ensure all three backend services are running on their respective ports
- Check that ports 5000, 8000, 9000 are not blocked
- Verify CORS is enabled on all services

### CORS issues
- All services have CORS enabled with `flask_cors`
- The integrated server also has CORS enabled
- No additional configuration needed

### Connection timeouts
- Check that services are actually running: `curl http://127.0.0.1:5000/health`
- Verify network connectivity
- Check firewall settings

## Performance Notes

- Face detection is fastest (~100-200ms per frame)
- Voice detection depends on audio length (~500-2000ms)
- Keystroke detection is very fast (~50-100ms)
- Combined emotion includes latency from all active sources

## Next Steps

1. Update `index.html` and `solve.html` to add real emotion detection UI
2. Capture webcam frames and send to `/api/detect-face`
3. Record audio and send to `/api/detect-voice`
4. Collect keystroke events and send to `/api/detect-keystroke`
5. Subscribe to `/api/combined-emotion` for real-time emotion state

See the individual service READMEs for more details:
- `../face/README.md` (if exists)
- `../voice/README.md` (if exists)
- `../keystrokes/README.md`
