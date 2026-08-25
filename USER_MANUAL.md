# BlindSpot Guardian User Manual

## Fastest Windows setup

1. Download [`BlindSpotGuardian-Windows-x64.zip`](https://github.com/Sapnap67/DreamTeam-HHCC/releases/latest/download/BlindSpotGuardian-Windows-x64.zip).
2. Right-click the ZIP and choose **Extract All**. Do not run the program from inside the ZIP.
3. Open the extracted `BlindSpotGuardian-Windows-x64` folder.
4. Double-click `BlindSpotGuardian.exe`.
5. Keep the program running while using the local browser page that opens.

The portable release already contains Python, YOLO, PyTorch, OpenCV, MediaPipe, the model files, and the web interface. The receiving Windows 10/11 x64 computer does not need Python, pip, Git, PowerShell, Codex, or administrator access.

## Normal use

1. Choose the intended zone mode and blind side.
2. Upload a local traffic video.
3. Start analysis and observe the real YOLO boxes, track IDs, measured processing information, warning state, and evidence panel.
4. Use the sound test controls before a demonstration if sound is required.
5. Export or review the session timeline when analysis is complete.

Closing only the browser tab does not stop the local server. Double-clicking `BlindSpotGuardian.exe` again reopens the existing session instead of starting a duplicate server.

## Stopping the portable application

Close the `BlindSpotGuardian.exe` application window or end BlindSpot Guardian from Task Manager if it does not exit normally. Runtime files and logs are stored under `%LOCALAPPDATA%\BlindSpotGuardian`.

## Source-code startup

For development, clone the repository and double-click `start.bat`. The first run creates a project-local `.venv`, installs the pinned dependencies, validates Torchvision NMS, and starts the local application at <http://127.0.0.1:5000>.

## Important meanings

- **MONITORING:** No supported active conflict is currently selected.
- **VEHICLE TRACKED:** A relevant vehicle is detected, without enough supported evidence for a warning.
- **CAUTION:** Multi-frame motion provides limited evidence of a possible conflict.
- **DANGER:** Stronger converging-motion or short image-space time-to-collision evidence is present.
- **Confidence:** YOLO detection confidence, not collision probability.
- **TTC evidence:** An image-space prototype heuristic, not an exact physical collision forecast.

## Troubleshooting

- **Windows warns about the executable:** The prototype is not code-signed. Use only the official team release and ask an administrator before bypassing an organization-managed security policy.
- **Program appears slow on first start:** Windows may be scanning the bundled runtime and the AI libraries may take time to load.
- **Browser does not open:** Wait briefly, then open <http://127.0.0.1:5000>. If that port was unavailable, check the application log under `%LOCALAPPDATA%\BlindSpotGuardian\logs`.
- **Video does not load:** Try an MP4 encoded with a common H.264-compatible codec and avoid unusually large files.
- **No MediaPipe observations:** Pose support is optional. YOLO detection and warning logic continue to work without it.
- **Sound is missing:** Check browser audio permission, system volume, and the in-app hardware signal test.
- **Another session is already running:** Double-clicking the executable should reopen it. If the old process is frozen, close it in Task Manager and try again.

## Privacy and safety

Processing runs locally. Uploaded working copies are temporary, and private input/output folders are excluded from Git. Do not upload private traffic footage to the public repository.

BlindSpot Guardian is an HHCC prototype. It is not certified for road deployment and must not be used as the sole method of collision prevention.
