# AI-assisted development note

This prototype was developed with AI coding assistance. The team should review, run, and be able to explain every part before presenting it.

## What the code does

1. `app.py` runs YOLO on each uploaded-video frame and keeps only person, bicycle, motorcycle, and truck detections.
2. The primary truck anchors demo conflict polygons, or `zones.json` supplies fixed-camera polygons.
3. `behavior.py` runs MediaPipe Pose Landmarker and matches the nearest pose to the most relevant YOLO person.
4. A short pose history produces conservative `LIKELY WALKING`, `LIKELY STANDING`, or `LIKELY CROUCHING` labels.
5. Nose position relative to the ears provides a low-confidence head-orientation proxy toward or away from the detected truck.
6. `crossing_advisory()` uses only the zone warning state. Pose and head direction cannot cancel a warning.

## Claims the team must not make

- The system cannot know whether a pedestrian noticed or understood the truck.
- Head orientation is not eye gaze or attention detection.
- The system cannot guarantee that crossing is safe.
- The zones are not calibrated for a real intersection unless the team measures and validates them.
- This prototype is not ready for road deployment.

## Suggested judge explanation

“YOLO tells us which road users are present. MediaPipe gives us observable pose cues such as likely movement and coarse head orientation. Those cues help explain the scene, but our warning decision remains conservative and comes from truck/person conflict zones. We intentionally do not claim to read awareness or certify safety.”

