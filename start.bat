@echo off
setlocal
cd /d "%~dp0"
title BlindSpot Guardian

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if exist "%PYTHON_EXE%" goto environment_ready

set "EXISTING_YOLO_PYTHON=%USERPROFILE%\Documents\Codex\yolo-env\Scripts\python.exe"
if exist "%EXISTING_YOLO_PYTHON%" (
    set "PYTHON_EXE=%EXISTING_YOLO_PYTHON%"
    goto environment_ready
)

where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, enable "Add Python to PATH".
    pause
    exit /b 1
)

echo First-time setup: creating the local Python environment...
py -3 -m venv .venv
if errorlevel 1 goto setup_failed

echo Installing BlindSpot Guardian dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

:environment_ready
"%PYTHON_EXE%" -c "import cv2, flask, mediapipe, ultralytics" >nul 2>nul
if errorlevel 1 (
    echo Installing or updating BlindSpot Guardian dependencies...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
)

if not exist "yolo11n.pt" (
    echo The YOLO model file yolo11n.pt is missing from this folder.
    pause
    exit /b 1
)

if not exist "models\pose_landmarker_lite.task" (
    echo Downloading the MediaPipe pose model...
    if not exist "models" mkdir "models"
    powershell.exe -NoProfile -Command "Invoke-WebRequest -UseBasicParsing 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' -OutFile 'models\pose_landmarker_lite.task'"
    if errorlevel 1 (
        echo Pose model download failed. YOLO warnings will still work without pose cues.
    )
)

set "YOLO_CONFIG_DIR=%CD%\.runtime\ultralytics"
set "MPLCONFIGDIR=%CD%\.runtime\matplotlib"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

echo Starting BlindSpot Guardian...
echo Keep this window open. Press Ctrl+C here to stop the application.
if not defined BLINDSPOT_NO_BROWSER start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:5000'"
"%PYTHON_EXE%" app.py

if errorlevel 1 (
    echo.
    echo BlindSpot Guardian stopped because of an error.
    pause
)
exit /b

:setup_failed
echo.
echo Setup failed. Check your internet connection and the messages above.
pause
exit /b 1

