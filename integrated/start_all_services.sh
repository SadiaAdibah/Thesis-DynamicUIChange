#!/bin/bash

# ============================================================================
# INTEGRATED EMOTION DETECTION - STARTUP SCRIPT
# ============================================================================
# This script starts all required services for the integrated emotion detection
# 
# Services started:
#   - Face Detection Server      (port 5000)
#   - Voice Emotion Server       (port 8000)
#   - Keystroke Emotion Server   (port 9000)
#   - Integrated Server          (port 7000)
# ============================================================================

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
VENV_PATH="$WORKSPACE_DIR/venv"

echo "============================================================================"
echo "INTEGRATED EMOTION DETECTION SYSTEM - STARTUP"
echo "============================================================================"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo "Virtual Env: $VENV_PATH"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Please run: python3 -m venv $VENV_PATH"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo ""

# Check if all required packages are installed for integrated server
echo "Checking dependencies..."
python -c "import flask, requests" 2>/dev/null || {
    echo "❌ Missing dependencies. Please run:"
    echo "   pip install -r requirements.txt"
    exit 1
}
echo "✓ All dependencies installed"
echo ""

# Create a temporary script to run all services
TEMP_SCRIPT=$(mktemp)

cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash

WORKSPACE_DIR="$1"
VENV_PATH="$WORKSPACE_DIR/venv"

source "$VENV_PATH/bin/activate"

# Function to handle cleanup
cleanup() {
    echo ""
    echo "============================================================================"
    echo "Shutting down services..."
    echo "============================================================================"
    kill %1 %2 %3 %4 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

cd "$WORKSPACE_DIR"

echo "============================================================================"
echo "Starting all services (Press Ctrl+C to stop all)..."
echo "============================================================================"
echo ""

# Start all services in background
echo "[1/4] Starting Face Detection Server (port 5000)..."
cd "$WORKSPACE_DIR/face" && python server.py > /tmp/face_server.log 2>&1 &
FACE_PID=$!

echo "[2/4] Starting Voice Emotion Server (port 8000)..."
cd "$WORKSPACE_DIR/voice" && python server.py > /tmp/voice_server.log 2>&1 &
VOICE_PID=$!

echo "[3/4] Starting Keystroke Emotion Server (port 9000)..."
cd "$WORKSPACE_DIR/keystrokes" && python server.py > /tmp/keystroke_server.log 2>&1 &
KEYSTROKE_PID=$!

echo "[4/4] Starting Integrated Server (port 7000)..."
cd "$WORKSPACE_DIR/integrated" && python server.py > /tmp/integrated_server.log 2>&1 &
INTEGRATED_PID=$!

echo ""
echo "============================================================================"
echo "Service Status"
echo "============================================================================"
echo "Face Detection:    http://127.0.0.1:5000     (PID: $FACE_PID)"
echo "Voice Emotion:     http://127.0.0.1:8000     (PID: $VOICE_PID)"
echo "Keystroke Emotion: http://127.0.0.1:9000     (PID: $KEYSTROKE_PID)"
echo "Integrated Server: http://127.0.0.1:7000     (PID: $INTEGRATED_PID)"
echo ""
echo "============================================================================"
echo "Open your browser and go to: http://127.0.0.1:7000"
echo "============================================================================"
echo ""
echo "Log files:"
echo "  Face:       /tmp/face_server.log"
echo "  Voice:      /tmp/voice_server.log"
echo "  Keystroke:  /tmp/keystroke_server.log"
echo "  Integrated: /tmp/integrated_server.log"
echo ""

# Wait for all background processes
wait
EOF

chmod +x "$TEMP_SCRIPT"

# Run the service startup script
"$TEMP_SCRIPT" "$WORKSPACE_DIR"
EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_SCRIPT"

exit $EXIT_CODE
