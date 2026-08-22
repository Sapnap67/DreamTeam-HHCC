@echo off
setlocal
cd /d "%~dp0"
title BlindSpot Guardian
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "YOLO_CONFIG_DIR=%CD%\.runtime\ultralytics"
set "MPLCONFIGDIR=%CD%\.runtime\matplotlib"

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if exist "%PYTHON_EXE%" goto environment_ready

call :find_python
if not defined BOOTSTRAP_PYTHON goto python_missing

echo.
echo [1/4] Creating a private Python environment inside this project...
%BOOTSTRAP_PYTHON% -m venv .venv
if errorlevel 1 goto setup_failed
if not exist "%PYTHON_EXE%" goto setup_failed

:environment_ready
echo.
echo [2/4] Checking Python packages...
"%PYTHON_EXE%" -c "import flask, ultralytics, cv2, numpy, mediapipe" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages. The first setup can take several minutes...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
) else (
    echo Required packages are ready.
)

echo.
echo [3/4] Checking AI model files...
if not exist "yolo11n.pt" (
    echo Downloading the official YOLO11 nano model...
    "%PYTHON_EXE%" download_yolo_model.py
    if errorlevel 1 goto model_failed
) else (
    echo YOLO model is ready.
)

if not exist "models\pose_landmarker_lite.task" (
    echo Downloading the optional MediaPipe pose model...
    "%PYTHON_EXE%" download_pose_model.py
    if errorlevel 1 (
        echo MediaPipe model download failed. YOLO warnings will still work.
    )
) else (
    echo MediaPipe pose model is ready.
)

echo.
echo [4/4] Starting BlindSpot Guardian...
echo Keep this window open. Press Ctrl+C here to stop the application.
if not defined BLINDSPOT_NO_BROWSER start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:5000'"
"%PYTHON_EXE%" app.py
if errorlevel 1 goto app_failed
exit /b 0

:find_python
set "BOOTSTRAP_PYTHON="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.11"
    if defined BOOTSTRAP_PYTHON exit /b 0

    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.12"
    if defined BOOTSTRAP_PYTHON exit /b 0

    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.10"
    if defined BOOTSTRAP_PYTHON exit /b 0
)

where python >nul 2>nul
if errorlevel 1 exit /b 0
python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
if not errorlevel 1 set "BOOTSTRAP_PYTHON=python"
exit /b 0

:python_missing
echo.
echo A supported Python installation was not found.
echo Install 64-bit Python 3.11 from https://www.python.org/downloads/
echo During installation, enable "Add python.exe to PATH".
echo Then double-click start.bat again.
pause
exit /b 1

:model_failed
echo.
echo YOLO model download failed.
echo Check the internet connection, then double-click start.bat again.
pause
exit /b 1

:setup_failed
echo.
echo Setup failed while creating the environment or installing packages.
echo Check the internet connection and the error messages above.
echo If needed, delete only the .venv folder and run start.bat again.
pause
exit /b 1

:app_failed
echo.
echo BlindSpot Guardian stopped because of an application error.
echo Read the error above or take a screenshot for debugging.
pause
exit /b 1
