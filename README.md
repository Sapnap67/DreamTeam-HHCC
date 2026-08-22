# BlindSpot Guardian

BlindSpot Guardian is a local Flask prototype that combines real YOLO detections with MediaPipe Pose landmarks on uploaded video. It displays actual detections for `person`, `bicycle`, `motorcycle`, and `truck`, plus conservative, observable pedestrian cues.

> **PROTOTYPE — NOT FOR ROAD USE.** The warning is a transparent zone-overlap rule, not verified collision prediction or production-ready safety technology.

## Start

Open PowerShell in this project folder and run:

```powershell
.\start.ps1
```

Then open <http://127.0.0.1:5000>.

If port 5000 is already in use, choose another local port before launching:

```powershell
$env:APP_PORT = "5001"
.\start.ps1
```

Then open <http://127.0.0.1:5001>.

The launcher creates a local Python 3.11 or 3.12 `.venv`, installs the Python packages, and downloads two small model files on first use:

- YOLO: `models/yolo11n.pt`
- MediaPipe Pose Landmarker Lite: `models/pose_landmarker_lite.task`

It requires an internet connection only for those first downloads. If PowerShell blocks scripts, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

To use different model files without copying them into `models/`, set `YOLO_MODEL_PATH` and/or `MEDIAPIPE_POSE_MODEL_PATH` before starting the app.

Python 3.11 or 3.12 is required. Python 3.13 and 3.14 are not supported by the selected MediaPipe package on Windows.

## Use

1. Choose an MP4, MOV, AVI, MKV, or M4V video.
2. Select **Start processing**.
3. Watch the MJPEG stream, real detections, and measured performance.
4. Select **Stop / reset** to stop the worker and remove the temporary upload.

Only one processing job can run at a time. Uploaded files are placed temporarily in `input/` and removed when processing ends or is stopped. `output/` is reserved for future explicit exports; the current app streams frames without saving a processed copy.

## Scene context detection

The pretrained YOLO model also displays real detections for cars, buses, traffic lights, and stop signs, alongside people, bicycles, motorcycles, and trucks. Cars and buses provide scene context but do not replace the truck-specific blind-zone rule. A detected traffic light or stop sign only means the object is visible; this prototype does **not** read signal colour, infer right-of-way, or decide whether it is safe to cross.

## Zone configuration

The interface provides two zone modes.

### Fixed Intersection Camera

This is the intended deployment mode for a camera mounted on a stationary traffic-light pole. On the first frame of a fixed-camera clip, the app runs a Cityscapes semantic-segmentation model to highlight **road** and **sidewalk** pixels and propose editable draft zones at their boundary. The model downloads once on first use, so the computer needs internet access for that first analysis.

The draft is not a safety decision and is never enabled automatically. Review it, choose **Use draft zones**, then adjust any corners in the **Fixed-camera calibration** panel and choose **Save fixed zones**. The app saves approved geometry to `zones.json` and uses it immediately. Each of the three zones needs at least three points. Recalibrate whenever the camera is moved or its view changes.

### Moving-Camera Demo

This demonstration mode is the default for uploaded dashcam footage. It never creates zones until a real truck is detected. The largest truck is primary when genuine YOLO track IDs are unavailable. If real tracking is available, the previous primary ID is retained while visible. Each primary-truck box coordinate is smoothed with:

`smoothed = 0.25 * detected + 0.75 * previous`

All dynamic geometry uses named truck-width and truck-height scale constants in `app.py`:

- `TRUCK_TURN_ZONE` expands the smoothed box horizontally and vertically.
- `CONFLICT_ZONE` is a close trapezoid from the selected truck side to `1.10 × truck width` outward, with its vertical corners scaled by truck height.
- `ROAD_USER_APPROACH_ZONE` starts farther out and reaches `2.35 × truck width`, with a taller truck-height-scaled vertical span.
- Left-side mode mirrors the two trapezoids by changing the horizontal direction multiplier from `+1` to `-1`.
- Every point is clipped to the video frame and then normalized for drawing and point-in-polygon checks.

The last real truck-anchored geometry is held for at most five missing frames and then hidden. Reset Zone Tracking clears the smoothed box, selected ID, and retained geometry.

### Warning states

- Fixed mode: `MONITORING`, `TRUCK PRESENT`, `CAUTION`, and `DANGER` use the configured intersection polygons.
- Moving mode: `MONITORING`, `TRUCK TRACKED`, `CAUTION`, and `DANGER` use real truck-relative geometry.
- `DANGER` requires three consecutive matching frames and clears after five safe frames.

The bottom-center of each bounding box is used as the approximate road-contact point. Danger clears after five non-danger frames.

## MediaPipe pedestrian cues

MediaPipe Pose Landmarker runs once per frame and estimates up to four poses. The app matches the nearest visible torso landmarks to the YOLO person selected as most relevant: a person in the conflict zone first, then one in the approach zone, then the largest visible person.

- **Likely walking / likely standing:** short-term hip movement, normalized by the person's YOLO box height. Several frames are required before a label appears.
- **Likely crouching:** a medium-confidence heuristic using visible hip, knee, and ankle geometry.
- **Head vs. truck:** a low-confidence 2D nose-to-ear offset compared with the truck's horizontal direction. It is displayed only as an orientation proxy.
- **Noticed truck?:** always `CANNOT BE INFERRED`. Body pose cannot establish awareness, eye contact, comprehension, or intent.
- **Crossing advisory:** `DO NOT CROSS`, `WAIT — CONFLICT POSSIBLE`, `WAIT — TRUCK PRESENT`, or `CHECK SIGNAL AND TRAFFIC`, derived from the existing zone state. It never reports `SAFE TO CROSS`.

The warning zone has priority over pose cues. For example, a person oriented toward the truck still receives `DO NOT CROSS` when the conflict zone is active.

## Model behavior and limitations

- Model: project-local `models/yolo11n.pt` by default; override with `YOLO_MODEL_PATH`
- Inference: CPU, `imgsz=640`, confidence threshold `0.35`
- Supported displayed classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- A pretrained model may categorize an e-bike as bicycle or motorcycle; it does not provide a separate e-bike class here.
- MediaPipe pose and coarse head-orientation cues are heuristic observations, not attention, emotion, awareness, or intent recognition.
- No verified turn-intention or trajectory prediction is performed.
- Moving-camera zones are demonstration geometry, not calibrated intersection zones.
- Genuine YOLO track IDs are used only when the optional tracking dependency is installed; otherwise the largest truck is selected without displaying an ID.
- Default zones are generic and must be calibrated for the actual camera view.
- Detection quality depends on lighting, occlusion, perspective, video quality, and the pretrained model.
- Activity labels need several consecutive frames and are less reliable without stable YOLO track IDs.
- Webcam input is intentionally deferred until uploaded-video mode is proven reliable.

