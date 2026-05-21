# Integrated Emotion Detection System - Setup Guide

This guide will help you set up and run the complete integrated emotion detection system with face, voice, and keystroke emotion detection.

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│         INTEGRATED EMOTION DETECTION SYSTEM              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Frontend (index.html, solve.html)       [Port 7000]    │
│           ↓                                             │
│  Integrated API Server (server.py)       [Port 7000]    │
│           ↓                                             │
│  ┌──────────┬──────────┬──────────────┐                │
│  │          │          │              │                │
│  ↓          ↓          ↓              ↓                │
│ Face      Voice     Keystroke      Status             │
│ Server    Server     Server         Checks            │
│ 5000      8000       9000                           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- Webcam (for face detection)
- Microphone (for voice emotion detection)
- Virtual environment (venv)

## Step 1: Prepare the Virtual Environment

From the thesis-code root directory:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Step 2: Install Dependencies

Install dependencies for all services:

```bash
# Install face detection dependencies
cd face
pip install flask flask-cors torch torchvision transformers pillow opencv-python
cd ..

# Install voice emotion dependencies
cd voice
pip install flask flask-cors torch librosa transformers pydub
cd ..

# Install keystroke emotion dependencies
cd keystrokes
pip install -r requirements.txt
cd ..

# Install integrated server dependencies
cd integrated
pip install -r requirements.txt
cd ..
```

Alternatively, install everything at once:

```bash
pip install flask flask-cors torch torchvision transformers librosa pydub pillow opencv-python joblib scikit-learn pandas numpy requests
```

## Step 3: Train the Keystroke Model (If Not Already Done)

The keystroke emotion detection requires a trained model. Check if it exists:

```bash
ls keystrokes/model.joblib
```

If not found, train it:

```bash
cd keystrokes
python train.py
cd ..
```

This will create `keystrokes/model.joblib` with the trained model.

## Step 4: Start All Services

### Option A: Automatic Startup (Recommended)

Run all services with a single command:

**On Linux/macOS:**
```bash
cd integrated
python start_all_services.py
```

**On Windows:**
```bash
cd integrated
python start_all_services.py
```

### Option B: Manual Startup (One Service Per Terminal)

Open 4 separate terminal windows, all from the thesis-code root directory:

**Terminal 1 - Face Detection (Port 5000):**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd face
python server.py
```

Output should show:
```
 * Running on http://127.0.0.1:5000
```

**Terminal 2 - Voice Emotion (Port 8000):**
```bash
source venv/bin/activate
cd voice
python server.py
```

Output should show:
```
 * Running on http://127.0.0.1:8000
```

**Terminal 3 - Keystroke Emotion (Port 9000):**
```bash
source venv/bin/activate
cd keystrokes
python server.py
```

Output should show:
```
 * Running on http://127.0.0.1:9000
```

**Terminal 4 - Integrated Server (Port 7000):**
```bash
source venv/bin/activate
cd integrated
python server.py
```

Output should show:
```
 * Running on http://127.0.0.1:7000
```

## Step 5: Access the Application

Once all services are running, open your browser and navigate to:

```
http://127.0.0.1:7000
```

You should see the Neetcode Emotion Demo interface.

## Using the Application

### 1. Landing Page (index.html)

- View problem list with difficulty levels
- Navigate to problem-solving interface
- See information about the emotion detection system

### 2. Problem Solving Interface (solve.html)

**Face Emotion Detection:**
- Click "Start Camera" to enable webcam
- Click "Capture" to take a frame
- The system will detect emotion from your facial expression

**Voice Emotion Detection:**
- Click "Start" to begin recording
- Speak naturally
- Click "Stop" to end recording
- The system analyzes emotion from your voice

**Keystroke Emotion Detection:**
- Type in the code editor
- Click "Analyze Typing" to detect emotion
- The system analyzes keystroke patterns

**Combined Emotion:**
- The system uses majority voting to combine emotions from all three sources
- The UI theme changes based on the detected emotion

### Emotion-Based UI Changes

The interface adapts based on detected emotion:

- **Focused** (Blue/Green): Sharp, clear interface
- **Happy** (Yellow/Green): Warm, friendly colors
- **Sad** (Blue/Purple): Cool, contemplative colors
- **Stressed** (Red/Orange): Alert, intense colors

## Testing Services

### Check if All Services are Running

```bash
# Check integrated server
curl http://127.0.0.1:7000/health

# Check service status
curl http://127.0.0.1:7000/api/status

# Check individual services
curl http://127.0.0.1:5000  # Face
curl http://127.0.0.1:8000  # Voice
curl http://127.0.0.1:9000  # Keystroke
```

### Test Face Detection API

```bash
# Save a test image as test_face.jpg, then run:
curl -X POST -F "image=@test_face.jpg" http://127.0.0.1:7000/api/detect-face
```

### Test Voice Detection API

```bash
# Save a test audio as test_audio.webm, then run:
curl -X POST -F "file=@test_audio.webm" http://127.0.0.1:7000/api/detect-voice
```

### Test Keystroke API

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

## Troubleshooting

### Issue: "Port already in use"

If you get a "port already in use" error:

```bash
# On Linux/macOS, find and kill the process:
lsof -i :7000  # Shows process using port 7000
kill -9 <PID>

# On Windows:
netstat -ano | findstr :7000
taskkill /PID <PID> /F
```

### Issue: "Module not found" errors

Install missing dependencies:

```bash
source venv/bin/activate
pip install -r integrated/requirements.txt
pip install -r keystrokes/requirements.txt
```

### Issue: Keystroke model not found

Train the model:

```bash
source venv/bin/activate
cd keystrokes
python train.py
cd ..
```

### Issue: "Connection refused" or "Service not running"

Ensure all four services are running on their respective ports:
- Face: 127.0.0.1:5000
- Voice: 127.0.0.1:8000
- Keystroke: 127.0.0.1:9000
- Integrated: 127.0.0.1:7000

### Issue: Camera/Microphone not working

- Check browser permissions for camera/microphone
- Try using HTTPS (for production, not localhost)
- Ensure devices are properly connected
- Check for conflicting applications

## API Reference

For detailed API documentation, see [README.md](README.md)

## File Structure

```
thesis-code/
├── face/
│   ├── server.py           # Face detection server
│   └── requirements.txt
├── voice/
│   ├── server.py           # Voice emotion server
│   └── requirements.txt
├── keystrokes/
│   ├── server.py           # Keystroke emotion server
│   ├── train.py            # Model training
│   ├── model.joblib        # Trained model
│   └── requirements.txt
├── integrated/             # ← YOU ARE HERE
│   ├── server.py           # Main integrated server
│   ├── README.md           # API documentation
│   ├── SETUP.md            # This file
│   ├── requirements.txt
│   ├── start_all_services.py
│   ├── start_all_services.sh
│   └── static/
│       ├── index.html      # Landing page
│       └── solve.html      # Problem solver
└── venv/                   # Virtual environment
```

## Next Steps

1. **Customize the UI**: Modify `index.html` and `solve.html` to add your own styling
2. **Integrate with your app**: Use the API endpoints to add emotion detection to other applications
3. **Deploy**: Host on a server for remote access (requires HTTPS and proper CORS setup)
4. **Improve accuracy**: Collect more training data and retrain models

## Support

For issues or questions:
1. Check the server logs in terminals running the services
2. Review the [README.md](README.md) for API documentation
3. Check individual service directories for READMEs
4. Verify all dependencies are installed with `pip list`

## Performance Notes

- **Face detection**: ~100-200ms per frame
- **Voice detection**: Depends on audio length (500-2000ms typical)
- **Keystroke detection**: ~50-100ms
- **Network overhead**: ~50-100ms per request

For best experience:
- Use a modern browser (Chrome, Firefox, Edge)
- Ensure stable internet connection
- Close unnecessary background applications
- Use good lighting for face detection
- Speak clearly for voice detection

---

**Happy coding! 🚀**
