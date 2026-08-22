# BlindSpot Guardian

BlindSpot Guardian is a 36-hour HHCC 2026 prototype for exploring vehicle and vulnerable-road-user proximity in traffic video, with particular attention to large-vehicle blind spots. It applies real YOLO detections and an image-space motion heuristic to show an understandable potential-collision-risk warning.

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

Use **64-bit Python 3.11** for the most predictable Windows compatibility. Model files are downloaded on the first setup and ignored by Git.

### Easy Windows startup

1. Download the repository as a ZIP and extract it, or clone it with Git.
2. Install [Python 3.11](https://www.python.org/downloads/) if necessary. During installation, enable **Add python.exe to PATH**.
3. Double-click `start.bat`.

The portable launcher uses a private `.venv` inside the project and does not depend on environments or paths from the original developer's computer. On its first run it will:

- select Python 3.11, 3.12, or 3.10;
- create the local environment;
- install the packages in `requirements.txt`;
- download the required official `yolo11n.pt` model;
- attempt to download the optional MediaPipe Pose Landmarker model;
- start Flask and open <http://127.0.0.1:5000>.

The first setup requires internet access and can take several minutes. Keep the command window open. Later starts reuse the environment and downloaded models and should be much faster. Press `Ctrl+C` in the command window to stop the server.

If setup fails, read the final message rather than closing the window. Check the internet connection and confirm that Python is a supported 64-bit version. If the local environment became incomplete, delete **only** the project's `.venv` folder and double-click `start.bat` again.

### Manual Windows startup

```powershell
cd "path\to\DreamTeam-HHCC"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_yolo_model.py
python download_pose_model.py
python app.py
```

The MediaPipe model is optional. If its download fails, the real YOLO detection and warning system can still run.

### Easy macOS startup

1. Download the repository ZIP and extract it, or clone it with Git.
2. Install [Python 3.11 for macOS](https://www.python.org/downloads/macos/) if necessary.
3. Open Terminal, type `cd ` followed by a space, drag the extracted project folder into Terminal, and press Return.
4. Run:

```bash
chmod +x start_mac.command
./start_mac.command
```

The first command is normally needed only once. Later, `start_mac.command` can be opened again from Terminal or, depending on macOS security settings, by double-clicking it in Finder.

The macOS launcher provides the same portable setup as `start.bat`: it creates a project-local `.venv`, installs missing packages, downloads the required YOLO model, attempts the optional MediaPipe model, starts Flask, and opens <http://127.0.0.1:5000>.

The first setup requires internet access and can take several minutes. Keep Terminal open while using the application and press `Control+C` to stop it. If macOS blocks the launcher, right-click `start_mac.command`, choose **Open**, and confirm once.

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

