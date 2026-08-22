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

Python 3.10 or newer is recommended.

### Easy Windows startup

Double-click `start.bat`. On this computer it reuses the existing YOLO Python environment. On a new Windows computer, its first run creates `.venv`, installs `requirements.txt`, and then starts the application. It also opens <http://127.0.0.1:5000> automatically.

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

Open <http://127.0.0.1:5000>. Press `Ctrl+C` to stop. Download `yolo11n.pt` through Ultralytics or set `YOLO_MODEL_PATH` to an existing model file before starting. Model weights are intentionally excluded from Git.

## Cloud demo deployment

The repository includes a Docker deployment and a Render Blueprint for a small public demonstration service. The container downloads the YOLO 11 nano weights and the official MediaPipe Pose Landmarker Lite model during its build.

1. Push this branch to GitHub.
2. In Render, create a new Blueprint from this repository's `render.yaml`.
3. Wait for `/api/health` to report that both model files are available.
4. Put the resulting `https://...onrender.com` URL into the companion website's backend configuration.

The free service is suitable only for a short hackathon demonstration: it can sleep when idle, processing is CPU-only, uploads are temporary, and only one video can run at a time. The default cloud upload limit is 100 MB. Do not use this prototype for road-safety decisions.

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


