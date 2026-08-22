# AI-assisted development note

This prototype was developed with AI coding assistance. The team should review, run, and be able to explain every part before presenting it.

## What the code does

1. `app.py` runs YOLO on each uploaded-video frame and keeps relevant road-user, vehicle, and traffic-control detections.
2. `scene.py` proposes road and sidewalk polygons. A person can correct and save any number of surface polygons; they are informational and never drive warnings.
3. The primary truck anchors demo conflict polygons, or reviewed `zones.json` geometry supplies fixed-camera polygons.
4. `behavior.py` runs MediaPipe Pose Landmarker and matches the nearest pose to the most relevant YOLO person.
5. A short pose history produces conservative `LIKELY WALKING`, `LIKELY STANDING`, or `LIKELY CROUCHING` labels.
6. Nose position relative to the ears provides a low-confidence head-orientation proxy toward or away from the detected truck.
7. `crossing_advisory()` uses only the zone warning state. Pose, head direction, and automatic surface suggestions cannot cancel or create a warning.

## Claims the team must not make

- The system cannot know whether a pedestrian noticed or understood the truck.
- Head orientation is not eye gaze or attention detection.
- The system cannot guarantee that crossing is safe.
- The zones are not calibrated for a real intersection unless the team measures and validates them.
- This prototype is not ready for road deployment.

## Suggested judge explanation

“YOLO tells us which road users are present. MediaPipe gives us observable pose cues such as likely movement and coarse head orientation. Those cues help explain the scene, but our warning decision remains conservative and comes from truck/person conflict zones. We intentionally do not claim to read awareness or certify safety.”


