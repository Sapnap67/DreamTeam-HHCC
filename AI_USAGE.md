# AI Usage Declaration

BlindSpot Guardian uses two local computer-vision components:

- **Ultralytics YOLO 11n** performs the real object detections, labels, confidence values, and available tracking IDs shown in the camera panel. The existing image-space proximity and motion heuristic is the sole authority for warning states and safety timeline events.
- **MediaPipe Pose Landmarker Lite** optionally produces conservative, display-only observations for one selected real YOLO person crop. It may show likely walking, likely standing, uncertain motion, or a low-confidence head-orientation proxy. It never changes a warning, risk calculation, event, or detection.

The system does not infer pedestrian awareness, intent, real-world speed or distance, driver intention, collision probability, or a calibrated trajectory. If MediaPipe or its model is unavailable, YOLO processing and the warning heuristic continue normally.

No cloud AI service, camera access, remote inference, or uploaded-data service is used by the application. Video processing occurs on the computer running the Flask server.

