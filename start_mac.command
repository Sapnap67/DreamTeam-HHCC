#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export YOLO_CONFIG_DIR="$SCRIPT_DIR/.runtime/ultralytics"
export MPLCONFIGDIR="$SCRIPT_DIR/.runtime/matplotlib"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"

pause_on_error() {
    echo
    read -r -p "Press Return to close this window..."
}

find_python() {
    for candidate in python3.11 python3.12 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1; then
                printf '%s' "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

if [ ! -x "$PYTHON_EXE" ]; then
    BOOTSTRAP_PYTHON="$(find_python || true)"
    if [ -z "$BOOTSTRAP_PYTHON" ]; then
        echo
        echo "A supported Python installation was not found."
        echo "Install Python 3.11 from https://www.python.org/downloads/macos/"
        echo "Then run start_mac.command again."
        pause_on_error
        exit 1
    fi

    echo
    echo "[1/4] Creating a private Python environment inside this project..."
    if ! "$BOOTSTRAP_PYTHON" -m venv .venv; then
        echo "Could not create the local Python environment."
        pause_on_error
        exit 1
    fi
fi

echo
echo "[2/4] Checking Python packages..."
if ! "$PYTHON_EXE" -c "import flask, ultralytics, cv2, numpy, mediapipe" >/dev/null 2>&1; then
    echo "Installing required packages. The first setup can take several minutes..."
    if ! "$PYTHON_EXE" -m pip install --upgrade pip; then
        echo "pip upgrade failed. Check the internet connection."
        pause_on_error
        exit 1
    fi
    if ! "$PYTHON_EXE" -m pip install -r requirements.txt; then
        echo "Package installation failed. Check the messages above."
        pause_on_error
        exit 1
    fi
else
    echo "Required packages are ready."
fi

echo
echo "[3/4] Checking AI model files..."
if [ ! -s "yolo11n.pt" ]; then
    echo "Downloading the official YOLO11 nano model..."
    if ! "$PYTHON_EXE" download_yolo_model.py; then
        echo "YOLO model download failed. Check the internet connection."
        pause_on_error
        exit 1
    fi
else
    echo "YOLO model is ready."
fi

if [ ! -s "models/pose_landmarker_lite.task" ]; then
    echo "Downloading the optional MediaPipe pose model..."
    if ! "$PYTHON_EXE" download_pose_model.py; then
        echo "MediaPipe model download failed. YOLO warnings will still work."
    fi
else
    echo "MediaPipe pose model is ready."
fi

echo
echo "[4/4] Starting BlindSpot Guardian..."
echo "Keep this Terminal window open. Press Control+C here to stop."
if [ -z "${BLINDSPOT_NO_BROWSER:-}" ]; then
    (sleep 3; open "http://127.0.0.1:5000") >/dev/null 2>&1 &
fi

"$PYTHON_EXE" app.py
APP_STATUS=$?
if [ "$APP_STATUS" -ne 0 ]; then
    echo
    echo "BlindSpot Guardian stopped because of an application error."
    echo "Read the error above or take a screenshot for debugging."
    pause_on_error
fi
exit "$APP_STATUS"
