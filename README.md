# BlindSpot Guardian

BlindSpot Guardian is an HHCC 2026 prototype for safer heavy-vehicle turns at pedestrian crossings. A roadside camera analyzes traffic video for buses, trucks, and vulnerable road users, then drives a bilingual pedestrian warning signal when an image-space risk heuristic detects sustained proximity or converging motion.

> **Prototype only — not for road use.** This is not certified collision prediction or traffic-control technology.

## What the prototype demonstrates

- Real YOLO inference on uploaded traffic video
- Detection of `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`
- Selection and tracking of a primary bus or truck
- Image-space proximity and short motion-history analysis
- A bilingual pedestrian signal with `MONITORING`, `CAUTION`, and `DANGER` states
- Explainable evidence showing why the current state was selected
- Multi-frame confirmation and clearing delays to reduce alert flicker
- Bounding boxes, confidence values, inference time, and processing FPS
- A clearly marked signal-test mode that does not alter AI results
- Local processing with temporary upload removal

Cars remain visible as detections but do not trigger the pedestrian warning. Only people, bicycles, and motorcycles are treated as vulnerable road users.

## How it works

```text
Uploaded video → YOLO detections → short object histories
              → heavy-vehicle/road-user risk evidence
              → warning hysteresis → pedestrian warning signal
```

The bottom-center of each bounding box is used as an approximate road-contact point. The app compares image-space distance, monitored side, lower safety-margin overlap, decreasing separation, and converging motion. It does not measure real-world distance or speed.

## Project structure

```text
blindspot-guardian/
├── app.py
├── requirements.txt
├── start.bat
├── static/
│   └── styles.css
└── templates/
    └── index.html
```

Test videos, output files, virtual environments, runtime caches, and YOLO weights are intentionally excluded from GitHub.

## Setup

You need Python 3.10 or newer.

### Windows: easiest method

Place `yolo11n.pt` in the project folder, then double-click `start.bat`.

### Windows: PowerShell

```powershell
cd "path\to\blindspot-guardian"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python app.py
```

### macOS: Terminal

```bash
cd "/path/to/blindspot-guardian"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Press `Ctrl+C` to stop the server.

If the model is elsewhere, set `YOLO_MODEL_PATH` to its full path before starting.

## Using the demo

1. Choose an MP4, MOV, AVI, MKV, or M4V traffic video.
2. Select the monitored turn direction.
3. Select **Start processing**.
4. Watch the real detections, evidence panel, performance, and pedestrian signal.
5. Use the advanced signal test only to demonstrate the warning hardware UI; it is visibly labeled as not being AI output.
6. Select **Stop / reset** before loading another video.

Only one video is processed at a time. Uploaded files are temporary and processed video is not saved.

## Warning states

- **MONITORING — NO WARNING:** no supported heavy-vehicle/road-user risk evidence; this is not permission to cross
- **CAUTION — HEAVY VEHICLE NEAR CROSSING:** a vulnerable road user remains near the monitored side of a tracked bus or truck
- **DANGER — BLIND-SPOT COLLISION RISK:** stronger proximity or converging-motion evidence persists for multiple frames

The displayed evidence comes from the backend calculation. No crash probability is claimed.

## Current limitations

- Image-space distance is not real-world distance.
- The prototype does not reliably predict steering, turn signals, speed, driver intention, or future trajectories.
- A pretrained model may classify an e-bike as a bicycle or motorcycle.
- Detection and tracking quality depend on lighting, occlusion, perspective, and video quality.
- Thresholds require testing and calibration for a specific stationary roadside camera.
- Uploaded video is supported; a live roadside camera feed is not yet implemented.

## Responsible AI usage

AI tools assisted with documentation, code generation, debugging, and explanation. The team owns the concept and engineering decisions and must review, understand, test, and meaningfully modify all assisted work. See [AI_USAGE.md](AI_USAGE.md).

## Team

Dream Team — HHCC 2026 Prototype Development Track  
Theme: **AI Reshaping the Automotive Industry**
