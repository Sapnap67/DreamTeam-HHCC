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

Use **Python 3.11** for the most predictable Windows and macOS dependency compatibility.

The pretrained model files are **not stored in this repository**. They are downloaded from their official providers and ignored by Git:

- `yolo11n.pt` is required for YOLO detection.
- `models/pose_landmarker_lite.task` is optional. Without it, YOLO warnings still work and the pedestrian-observation card reports that MediaPipe is unavailable.

### Windows (PowerShell)

```powershell
cd "path\to\DreamTeam-HHCC"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python download_pose_model.py

python app.py
```

Open <http://127.0.0.1:5000>. Keep PowerShell open and press `Ctrl+C` to stop.

After this first-time setup, Windows users can normally double-click `start.bat`. If `yolo11n.pt` is missing, the launcher intentionally stops instead of pretending that detection is available. Run the YOLO download command above, then start it again.

The MediaPipe model is optional. Windows users may run `download_pose_model.bat` instead of the Python downloader.

### macOS (Terminal)

Install Python 3.11 first if `python3.11` is unavailable, then run:

```bash
cd "/path/to/DreamTeam-HHCC"

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python download_pose_model.py

python app.py
```

Open <http://127.0.0.1:5000>. Keep Terminal open and press `Control+C` to stop.

The application uses CPU inference for portability. The first setup requires internet access to install dependencies and download models; download everything before the roadshow so the demonstration can run offline.

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

