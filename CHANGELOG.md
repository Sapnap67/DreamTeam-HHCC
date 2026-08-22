# BlindSpot Guardian Change Log

This file records user-visible and safety-logic changes to the project. Update it whenever the application, detection behavior, zone geometry, warning rules, interface, dependencies, or documentation changes.

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

