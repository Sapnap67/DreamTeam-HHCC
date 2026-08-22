# BlindSpot Guardian Change Log

This file records user-visible and safety-logic changes to the project. Update it whenever the application, detection behavior, zone geometry, warning rules, interface, dependencies, or documentation changes.

## 2026-08-22 — Heavy-Vehicle Warning Tuning and Signal Test

### Changed

- Added real YOLO COCO `bus` detection (`class_id = 5`) and allowed either a bus or truck to become the tracked primary heavy vehicle while preserving existing truck behavior.
- Renamed primary-selection logic to describe heavy vehicles and made the tracked-object status identify whether the selected detection is a bus or truck.
- Tuned caution to require a heavy vehicle, a vulnerable road user on the monitored side, and image-space contact-point distance within `1.40 ×` the detected heavy-vehicle width for two consecutive processed frames.
- Kept danger evidence-based: caution must continue and decreasing separation, lower safety-margin overlap, extreme proximity within `0.60 ×` heavy-vehicle width, or converging image-space paths must persist for four processed frames.
- Updated pedestrian-facing caution and danger wording for heavy vehicles in English and Chinese.

### Explainability

- Expanded `WHY THIS STATE?` with the real caution-distance and safety-margin-overlap values used by the backend.
- Added live `X / 2` caution and `X / 4` danger persistence values from backend hysteresis counters.
- Added state-aware green, amber, and red evidence highlighting while retaining gray inactive evidence.

### Hardware Test

- Added a collapsed `Advanced / hardware test` control that cycles the warning device through Safe, Caution, and Danger for five seconds.
- Clearly labels every simulated state as `SIGNAL TEST — NOT AI OUTPUT`, leaves detections untouched, and returns automatically to the current backend state.
- Stop/reset cancels an active signal test immediately.

### Validation

- Processed both original Desktop traffic clips with real local YOLO inference; the second source produced bus tracking, pedestrian caution, and sustained danger while car-only frames produced no warning.
- In the 29.57-second source `597c87ba7da9a558ff93383a67213301.mp4`, the full-frame validation observed Safe at `00:00.00`, first Caution at `00:15.87`, first Danger at `00:19.20`, and a return to the safe presentation (`VEHICLE TRACKED`) at `00:21.30`.
- Verified the five-second signal test, automatic return to live state, mute control, stop/reset cancellation, and an empty browser warning/error console on the updated local page.
- Confirmed no visible safety polygons or hitboxes were reintroduced.

## 2026-08-22 — Pedestrian Warning Signal Interface

### Changed

- Reframed the application as an `AI CAMERA ANALYSIS` and `PEDESTRIAN WARNING SIGNAL` two-panel roadside prototype.
- Added synchronized safe, caution, and danger signal-device presentations with English and Chinese pedestrian-facing messages.
- Added a one-shot danger-transition tone, an accessible mute control, and reduced-motion behavior that suppresses flashing and pulsing.
- Moved monitored turn direction and risk-tracking reset into a compact `Advanced` section.
- Renamed the primary object presentation to `Tracked heavy vehicle` and retained real measured FPS and inference time beside the detection stream.

### Added

- Added `WHY THIS STATE?` evidence indicators for the actual backend conditions: heavy vehicle, vulnerable road user, monitored side, decreasing distance, path convergence, and four-frame sustained risk.
- Added actual evidence fields to `/api/status`, including the evaluated vulnerable-road-user class and motion-history readiness.

### Risk Focus

- Kept cars in real YOLO detection boxes, confidence values, and the object list.
- Excluded cars from pedestrian-facing caution and danger evaluation; only person, bicycle, and motorcycle detections can activate those states.
- Preserved the existing image-space motion heuristic thresholds and multi-frame warning hysteresis.

### Limitations

- Added the explicit statement: `Image-space proximity and motion heuristic — not calibrated collision prediction.`

## 2026-08-22 — Internal Image-Space Collision-Risk Heuristic

### Changed

- Removed all visible `CONFLICT`, `APPROACH`, fixed-zone, dynamic-zone, connector, contact-point, and road-calibration drawing from processed video.
- Replaced polygon-overlap warnings with an internal heuristic that compares the selected real truck with real person, bicycle, motorcycle, and car detections.
- Added short smoothed bottom-center histories for every detection, using genuine YOLO track IDs when available and class/IoU/center-distance association otherwise.
- Added image-space motion, decreasing-distance, projected convergence, blind-side context, lower-truck safety-margin overlap, persistence, and detection-confidence evidence.
- Added four honest states: `MONITORING`, `VEHICLE TRACKED`, `CAUTION — ROAD USER NEAR TRUCK`, and `DANGER — POTENTIAL COLLISION RISK`.
- Added understandable warning reasons and multi-frame entry/clear hysteresis to reduce flicker.

### Removed

- Removed zone mode, zone visibility, road calibration, and all other dead zone-related controls and browser state.
- Removed unused polygon generation, loading, point-in-polygon warning checks, and drawing code from the application.

### Preserved

- Real YOLO video detection and confidence values, including the COCO `car` class.
- Upload, start, stop/reset, primary-truck selection, blind-side selection, genuine track-ID display, measured FPS/inference time, and temporary-file cleanup.
- Compatibility with stationary-intersection and moving-camera footage; the heuristic remains explicitly image-space only.

### Limitations

- This is a prototype image-space proximity and motion heuristic for potential collision risk.
- It does not predict crashes or turns, recognize driver intent, measure real-world distance or speed, provide calibrated trajectory forecasts, or claim production-road readiness.

## 2026-08-22 — Perspective-Aware Road-Surface Geometry

### Changed

- Reworked only the Moving-Camera Demo hitbox geometry so `CONFLICT` and `APPROACH` begin at the smoothed primary truck's bottom tire/road-contact line.
- Changed both zones into perspective trapezoids whose successive shared boundaries become narrower and receive a small pull toward a configurable vanishing point.
- Clipped displayed dynamic polygons to a configurable road-surface wedge and lower-frame margin.
- Kept `CONFLICT` adjacent to the selected truck side and made `APPROACH` begin at the exact outer boundary of `CONFLICT` without meaningful overlap.
- Preserved the existing dynamic polygon objects as the exact polygons used by bottom-center road-user warning checks.

### Added

- Added compact Vanishing X, Vanishing Y, Road horizon, Road left, and Road right calibration controls.
- Added `Reset road calibration` with defaults selected for the supplied real traffic video.
- Stored road-calibration values in browser local storage and synchronized them through the existing settings endpoint.
- Added the honest label `Perspective-aware road-surface demo geometry` without claiming 3D reconstruction or calibrated homography.

### Preserved

- Fixed Intersection Mode and `zones.json` behavior.
- Real YOLO detection, primary-truck selection, exponential smoothing, temporary lost-truck retention, reset behavior, and warning hysteresis.
- Deployment, dependencies, upload handling, and measured processing information.

### Validation

- Tested with the supplied real traffic video in both blind-side directions.
- Confirmed dynamic zones followed and resized with the smoothed truck box and stayed below the truck road-contact line within the configured road wedge.
- Confirmed displayed `CONFLICT` and `APPROACH` polygons shared a zero-area boundary and were the same coordinates used by warning checks.
- Confirmed no dynamic zones appeared without a detected or temporarily retained truck.
- Confirmed Fixed Intersection Mode continued to return the exact `zones.json` polygons.
- Confirmed no Python or browser-console errors occurred.

## 2026-08-22 — Real Car Detection

### Added

- Added the pretrained YOLO COCO `car` class (`class_id = 2`) to real video inference.
- Added car bounding boxes, confidence values, contact points, detected-object listings, and a distinct display color.

### Preserved

- Cars do not select or replace the primary truck.
- Cars do not generate truck-relative zones.
- Cars do not activate `CAUTION` or `DANGER`; those rules still use only real person, bicycle, and motorcycle contact points.
- Existing truck tracking, zone geometry, uploads, measurements, and Fixed Intersection Mode remain unchanged.

### Validation

- Confirmed the YOLO inference class filter includes `car` alongside the existing four classes.
- Confirmed warning road-user filters remain limited to person, bicycle, and motorcycle.
- Tested against the supplied real traffic video and confirmed real car detections appear with YOLO confidence values.

## 2026-08-22 — Easier Portable Startup

### Added

- Added `start.bat` for double-click startup on Windows.
- Added automatic first-run creation of a project-local `.venv` and installation from `requirements.txt` when no compatible existing environment is available.
- Added automatic browser opening after the local Flask server begins starting.
- Included `yolo11n.pt` in the project folder so recipients do not need the original computer's model path.

### Changed

- Updated setup documentation with the double-click workflow and portable model location.

### Validation

- Confirmed the launcher resolves the current YOLO environment when available and otherwise follows the local `.venv` setup path.
- Confirmed `app.py` resolves the bundled model relative to the project directory.

## 2026-08-22 — Dynamic Safety Envelope Refinement

### Changed

- Changed only the Moving-Camera Demo zone geometry and drawing behavior.
- Hid `TRUCK_TURN_ZONE` in Moving-Camera Demo because the real YOLO truck box already identifies the truck.
- Replaced the three visually overlapping dynamic quadrilaterals with two adjoining polygons:
  - Red `CONFLICT` zone extending 0.55 smoothed truck widths from a 0.02-width side gap.
  - Amber `APPROACH` zone extending another 0.75 smoothed truck widths from the conflict boundary.
- Made the two polygons share a boundary without substantial overlap.
- Set conflict fill opacity to 14%, approach fill opacity to 9%, and outlines to 2 pixels.
- Shortened dynamic labels to `CONFLICT` and `APPROACH`.
- Removed the connector line between the truck and conflict zone.
- Preserved coordinate clipping, smoothing, primary-truck selection, lost-truck handling, warning hysteresis, and left/right mirroring.

### Validation

- Tested using the supplied real traffic video.
- Confirmed real YOLO boxes, confidence values, measured FPS, and measured inference time remained operational.
- Confirmed dynamic zones followed and resized with the smoothed primary-truck box.
- Confirmed left/right mirroring and zero polygon-area overlap.
- Confirmed Fixed Intersection Mode polygons still exactly matched `zones.json`.
- Confirmed no Python traceback or browser console warning/error occurred.

## 2026-08-22 — Two Zone Modes and Truck-Relative Demo

### Added

- Added `FIXED INTERSECTION CAMERA` and `MOVING-CAMERA DEMO` modes.
- Added zone-mode, blind-side, zone-visibility, and zone-tracking-reset controls.
- Added primary-truck status, real confidence, real track ID when available, current mode, and warning-state displays.
- Added truck-relative dynamic zones generated from the selected real truck detection.
- Added exponential box smoothing with `alpha = 0.25`.
- Added five-processed-frame temporary zone retention after truck loss, followed by zone removal.
- Added three-frame danger activation and five-safe-frame danger clearing.

### Changed

- Kept the previously selected real truck track ID when available; otherwise selected the largest real truck detection.
- Used `TRUCK TRACKED`, `CAUTION`, and `DANGER — TRUCK BLIND ZONE` wording in Moving-Camera Demo.
- Preserved the original normalized `zones.json` polygons and existing warning rules in Fixed Intersection Mode.
- Avoided unsupported speed, direction, turning-intent, trajectory-prediction, and risk-percentage claims.

### Validation

- Tested repeated processing, mode switching, blind-side switching, visibility control, and tracking reset.
- Confirmed zones were never generated without a real truck detection.
- Confirmed the application fell back to largest-truck selection when real YOLO tracking IDs were unavailable.
- Confirmed browser requests completed successfully without Python errors.

