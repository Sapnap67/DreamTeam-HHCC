@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

"%PYTHON_EXE%" download_pose_model.py
if errorlevel 1 (
  echo.
  echo Download failed. Check the internet connection and Python installation.
  pause
  exit /b 1
)

echo.
echo Pose model is ready.
pause
