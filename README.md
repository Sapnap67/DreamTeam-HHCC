# BlindSpot Guardian

BlindSpot Guardian is a 36-hour HHCC 2026 prototype for safer truck turns at pedestrian crossings. A camera mounted near a traffic light analyzes video for trucks and nearby vulnerable road users, then displays a warning when their detected positions enter a possible blind-spot conflict area.

> **Prototype only — not for road use.** The current warning is based on detection and geometric zone overlap. It is not a certified collision-prediction or traffic-control system.

## What the prototype demonstrates

- Real YOLO inference on uploaded traffic video
- Detection of `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`
- Bounding boxes, confidence values, inference time, and processing FPS
- A **Fixed Intersection Camera** mode with configurable scene polygons
- A **Moving-Camera Demo** mode whose zones follow the primary detected truck
- A bilingual pedestrian warning signal with `SAFE`, `CAUTION`, and `DANGER` states
- Explainable evidence showing why the current warning state was selected
- Short multi-frame confirmation and clearing delays to reduce alert flicker
- Local processing: uploaded videos are removed after processing or reset

## How it works

```text
Uploaded video → YOLO object detection → road-contact points
              → truck/road-user zone checks → warning state → live browser display
```

The app uses the bottom-center of each bounding box as an approximate road-contact point. In fixed-camera mode, the points are checked against polygons in `zones.json`. In moving-camera mode, the polygons are generated relative to a smoothed primary-truck box for demonstration footage.

## Project structure

```text
blindspot-guardian/
├── app.py
├── requirements.txt
├── zones.json
├── static/
│   └── styles.css
└── templates/
    └── index.html
```

The app creates temporary `input/` and `output/` folders when it runs. Test videos, generated output, virtual environments, and YOLO weight files are intentionally excluded from GitHub.

## Setup

You need Python 3.10 or newer and the `yolo11n.pt` model file.

### Windows (PowerShell)

```powershell
cd "path\to\blindspot-guardian"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python app.py
```

### macOS (Terminal)

```bash
cd "/path/to/blindspot-guardian"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Press `Ctrl+C` in PowerShell or Terminal to stop the server.

If the model is stored somewhere else, set `YOLO_MODEL_PATH` before starting:

```powershell
# Windows PowerShell
$env:YOLO_MODEL_PATH = "C:\path\to\yolo11n.pt"
python app.py
```

```bash
# macOS
export YOLO_MODEL_PATH="/path/to/yolo11n.pt"
python app.py
```

## Using the demo

1. Choose an MP4, MOV, AVI, MKV, or M4V traffic video.
2. Select the zone mode and truck-turn side that fit the footage.
3. Select **Start processing**.
4. Watch the detections, zones, measured performance, and warning state.
5. Select **Stop / reset** before loading another video.

Only one video is processed at a time. Uploaded files are temporary and the current version does not save a processed video.

## Zone configuration

For a stationary intersection camera, edit `zones.json`. Every point is `[x, y]`, normalized from `0.0` to `1.0` from the top-left of the frame:

- `TRUCK_TURN_ZONE`: where a turning truck must be present
- `ROAD_USER_APPROACH_ZONE`: where a person or two-wheeler is approaching
- `CONFLICT_ZONE`: the higher-risk overlap area

The current warning uses an image-space proximity and motion heuristic around the tracked heavy vehicle. It does not display safety polygons or claim calibrated collision prediction.

## Current limitations

- It does not predict the truck's actual steering, turn signal, speed, or future path.
- It does not estimate attention, pose, emotion, or head direction.
- A pretrained model may classify an e-bike as a bicycle or motorcycle.
- Detection quality depends on lighting, occlusion, perspective, and video quality.
- The default fixed zones are generic and must be calibrated for a specific camera.
- Dynamic zones are demonstration geometry, not validated blind-spot boundaries.
- The prototype currently accepts uploaded video rather than a live roadside camera feed.

## Responsible AI usage

AI tools assisted with documentation, code generation, debugging, and explanation. The team owns the concept and engineering decisions and must review, understand, test, and meaningfully modify all assisted work. See [AI_USAGE.md](AI_USAGE.md) for the declaration.

## Team

Dream Team — HHCC 2026 Prototype Development Track  
Theme: **AI Reshaping the Automotive Industry**
