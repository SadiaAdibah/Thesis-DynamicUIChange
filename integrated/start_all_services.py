#!/usr/bin/env python
"""
Start all services for the integrated emotion detection system.

This script will start:
  - Face Detection Server (port 5000)
  - Voice Emotion Server (port 8000)
  - Keystroke Emotion Server (port 9000)
  - Integrated Server (port 7000)
"""

import os
import sys
import subprocess
import time
import signal
import platform
from pathlib import Path

# Get workspace directory
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_DIR = SCRIPT_DIR.parent
VENV_PATH = WORKSPACE_DIR / "venv"

# Service configurations
SERVICES = [
    {
        "name": "Face Detection",
        "port": 5000,
        "dir": WORKSPACE_DIR / "face",
        "cmd": "python server.py"
    },
    {
        "name": "Voice Emotion",
        "port": 8000,
        "dir": WORKSPACE_DIR / "voice",
        "cmd": "python server.py"
    },
    {
        "name": "Keystroke Emotion",
        "port": 9000,
        "dir": WORKSPACE_DIR / "keystrokes",
        "cmd": "python server.py"
    },
    {
        "name": "Integrated Server",
        "port": 7000,
        "dir": WORKSPACE_DIR / "integrated",
        "cmd": "python server.py"
    }
]

processes = []


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text):
    """Print a formatted section."""
    print(f"\n{text}")
    print("-" * len(text))


def check_venv():
    """Check if virtual environment exists."""
    if not VENV_PATH.exists():
        print(f"❌ Virtual environment not found at: {VENV_PATH}")
        print(f"Please create it with: python -m venv {VENV_PATH}")
        sys.exit(1)
    print(f"✓ Virtual environment found at: {VENV_PATH}")


def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    try:
        import flask
        import requests
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


def start_service(service):
    """Start a single service."""
    service_name = service["name"]
    service_port = service["port"]
    service_dir = service["dir"]
    service_cmd = service["cmd"]

    if not service_dir.exists():
        print(f"❌ Service directory not found: {service_dir}")
        return None

    try:
        # Set up environment with activated venv
        env = os.environ.copy()
        if platform.system() == "Windows":
            env["PATH"] = str(VENV_PATH / "Scripts") + os.pathsep + env["PATH"]
        else:
            env["VIRTUAL_ENV"] = str(VENV_PATH)
            env["PATH"] = str(VENV_PATH / "bin") + os.pathsep + env["PATH"]

        # Use full path to python executable
        python_exe = str(VENV_PATH / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))
        cmd_with_python = f"{python_exe} {service_cmd.replace('python ', '')}"

        # Start the service
        process = subprocess.Popen(
            cmd_with_python,
            shell=True,
            cwd=str(service_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=None if platform.system() == "Windows" else os.setsid
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None


def monitor_services():
    """Monitor and display output from services."""
    import select
    while True:
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

def stop_services():
    """Stop all services."""
    print("\n\nShutting down services...")
    for process in processes:
        if process:
            try:
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except Exception as e:
                print(f"Error stopping process: {e}")


def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    stop_services()
    print("\n✓ All services stopped")
    sys.exit(0)


def main():
    """Main function."""
    signal.signal(signal.SIGINT, signal_handler)

    print_header("INTEGRATED EMOTION DETECTION SYSTEM")

    print(f"Workspace:     {WORKSPACE_DIR}")
    print(f"Virtual Env:   {VENV_PATH}")

    # Validation
    check_venv()
    check_dependencies()

    # Start services
    print_section("Starting Services")

    for i, service in enumerate(SERVICES, 1):
        print(f"\n[{i}/{len(SERVICES)}] Starting {service['name']} (port {service['port']})...")
        process = start_service(service)
        if process:
            processes.append(process)
            print(f"   ✓ Process started (PID: {process.pid})")
        else:
            print(f"   ✗ Failed to start")

    # Wait a moment for servers to boot
    print("\nWaiting for servers to boot up...")
    time.sleep(3)

    # Print status
    print_section("Service Status")

    status_lines = []
    running_count = 0
    for i, service in enumerate(SERVICES):
        process = processes[i] if i < len(processes) else None
        if process and process.poll() is None:
            status = "✓ Running"
            running_count += 1
        else:
            status = "✗ Failed"
        status_lines.append(
            f"{service['name']:.<30} http://127.0.0.1:{service['port']:<5} {status}"
        )

    for line in status_lines:
        print(line)

    print_section("Access the Application")

    print(f"\nOpen your browser and navigate to:")
    print(f"    http://127.0.0.1:7000")
    print(f"\nAll {running_count} service(s) are running!")
    print(f"\nPress Ctrl+C to stop all services.")

    print_section("Troubleshooting")

    print("""
If services are not starting:
1. Check that ports 5000, 7000, 8000, 9000 are available
2. Ensure the virtual environment is properly configured
3. Check that all required packages are installed

If models are not found:
1. Ensure you've run the keystrokes/train.py to generate the model
2. Check that model files exist in their respective directories
    """)

    # Keep the script running
    try:
        while True:
            time.sleep(1)
            # Check if any process has died unexpectedly
            for i, process in enumerate(processes):
                if process and process.poll() is not None:
                    print(f"\n⚠ {SERVICES[i]['name']} has stopped unexpectedly!")
                    # Try to show stderr output
                    if hasattr(process, 'stderr') and process.stderr:
                        try:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                print(f"   Error output: {stderr_output.decode('utf-8', errors='ignore')}")
                        except:
                            pass
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
