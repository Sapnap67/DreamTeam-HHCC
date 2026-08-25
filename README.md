# BlindSpot Guardian

BlindSpot Guardian is a 36-hour HHCC 2026 prototype for exploring truck and road-user proximity in traffic video. It applies real YOLO detections and an image-space motion heuristic to show an understandable potential-collision-risk warning.

> **Prototype only — not for road use.** This is a prototype image-space proximity and motion heuristic, not accurate crash prediction or a certified safety system.

## Features

- Real YOLO detection of `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`
- Bounding boxes, confidence, inference time, and processing FPS
- Optional MediaPipe Pose Landmarker Lite observations for one selected real person crop, evaluated every third processed frame
- Separate caution chime and two-pulse danger alert, each emitted once per active risk episode
- Short smoothed motion histories using genuine YOLO IDs or local frame-to-frame association
- Multi-frame `MONITORING`, `VEHICLE TRACKED`, `CAUTION`, and `DANGER` warning states
- Local processing with temporary upload removal

## Setup

### Portable Windows executable

For the separate Windows 10/11 64-bit distribution:

1. Download [`BlindSpotGuardian-Windows-x64.zip`](https://github.com/Sapnap67/DreamTeam-HHCC/releases/latest/download/BlindSpotGuardian-Windows-x64.zip) from the latest GitHub Release.
2. Right-click it and choose **Extract All**. Do not run the program from inside the ZIP.
3. Open the extracted `BlindSpotGuardian-Windows-x64` folder and double-click `BlindSpotGuardian.exe`.
4. Closing the browser tab does not stop the local analysis server. Double-click `BlindSpotGuardian.exe` again to reopen the existing session; it will not start a duplicate server.

Python, pip, Git, PowerShell, and administrator access are not required on the recipient computer. The first start may take longer while Windows checks the bundled executable and loads the local AI runtimes. A modern 64-bit CPU and several gigabytes of available RAM are recommended; GPU acceleration is not required by this CPU-configured prototype.

The one-folder distribution is intentionally much larger than the source ZIP because it includes Python, PyTorch, YOLO, OpenCV, MediaPipe, and their DLLs. The current extracted build is approximately 1.2 GB; the compressed download is smaller. See `THIRD_PARTY_NOTICES.md` for model and dependency licensing. Compatibility targets Windows 10/11 x64 but is not guaranteed for every computer or security policy. The executable is not code-signed, so organization-managed Windows Application Control or antivirus policies may warn about or block it until an administrator approves the build.

The portable executable is an additional distribution. The source launch methods below remain supported and are not replaced.

### Packaged macOS application

The prepared macOS packaging creates a third, separate distribution and does not replace the source version or Windows executable. A valid app must be built on a real Mac for its native architecture:

- Apple Silicon: `BlindSpotGuardian-macOS-arm64.zip`
- Intel: `BlindSpotGuardian-macOS-x86_64.zip`

Recipient instructions after a real Mac build:

1. Download the ZIP matching the Mac architecture.
2. Extract it completely.
3. On first launch, right-click `BlindSpotGuardian.app` and select **Open**.
4. Keep the application running while using the browser interface.

The app bundles Python and its required dependencies, opens the browser automatically, and stores runtime data under `~/Library/Application Support/BlindSpotGuardian`. Logs are stored under `~/Library/Logs/BlindSpotGuardian`. No Terminal commands, Python installation, pip, Git, or Codex are required on the recipient Mac.

The current configuration uses ad-hoc signing unless a Developer ID identity is explicitly supplied. It is not notarized by default, so Gatekeeper may show a warning. A macOS artifact is not currently included because it must still be built and tested on a real Mac. See `packaging/macos/README_MAC_BUILD.md`.

Python 3.10 or newer is recommended.

### Easy Windows startup

Double-click `start.bat`. Its first run creates an isolated project-local `.venv`, installs the compatible versions in `requirements.txt`, validates Torchvision NMS, and then starts the application. It does not reuse a shared YOLO environment, so dependency changes in another project cannot break BlindSpot Guardian. It also opens <http://127.0.0.1:5000> automatically.

Keep the command window open while using BlindSpot Guardian. Press `Ctrl+C` in that window to stop it.

To enable the optional pedestrian-observation card, double-click `download_pose_model.bat` once. It downloads the official MediaPipe Pose Landmarker Lite task file over HTTPS into the ignored local `models` folder. If MediaPipe or the model is missing, the interface says it is unavailable and the YOLO warning system keeps working.

### Windows (PowerShell)

```powershell
cd "path\to\blindspot-guardian"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS (Terminal)

```bash
cd "/path/to/blindspot-guardian"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Press `Ctrl+C` to stop. The included `yolo11n.pt` is used by default. If the model is elsewhere, set `YOLO_MODEL_PATH` to its full path before running the app.

## Demo

1. Choose an MP4, MOV, AVI, MKV, or M4V traffic video.
2. Select the truck blind side.
3. Select **Start processing**.
4. Watch the real detections, measured performance, warning state, and reason.
5. Select **Stop / reset** before another video.

Only one video is processed at a time. Uploads are temporary; processed video is not saved.

## Risk heuristic

Every tracked car, motorcycle, bus, and truck is compared with every tracked person and bicycle. The highest-risk supported pair controls the warning. The internal heuristic considers smoothed bottom-center motion, relative direction, decreasing image-space distance, projected path convergence, short image-space time-to-collision where reliable, adaptive safety-margin overlap, confidence, and multi-frame persistence. Internal margins are never drawn.

`CAUTION` requires two consecutive supported frames and six clear frames to reset. `DANGER` requires four consecutive supported frames and eight clear frames to reset. Persistence is tied to the selected pair, so evidence from unrelated pairs is not combined.

## Limitations

- No steering, turn-signal, driver-intent, crash, turn, or calibrated trajectory prediction
- No real-world distance or speed measurement
- Optional pose cues are conservative observations only; pedestrian awareness, intent, attention, and emotion cannot be inferred
- Quality depends on lighting, occlusion, perspective, and video quality
- Image-space behavior varies with camera motion and perspective
- Uploaded video is supported; a live roadside feed is not yet implemented

## Responsible AI usage

AI assisted with documentation, code generation, debugging, and explanation. The team owns the concept and decisions and must review, understand, test, and meaningfully modify all assisted work. See [AI_USAGE.md](AI_USAGE.md).

## Team

Dream Team — HHCC 2026 Prototype Development Track  
Theme: **AI Reshaping the Automotive Industry**
