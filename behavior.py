from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POSE_VISIBILITY_THRESHOLD = 0.45
POSE_CROP_PADDING_RATIO = 0.15
POSE_MAX_INPUT_SIDE = 384
POSE_HISTORY_LENGTH = 6
WALKING_MOTION_THRESHOLD = 0.075
STANDING_MOTION_THRESHOLD = 0.035
HEAD_PROXY_DEADBAND = 0.055


def unavailable_observation(message: str = "MediaPipe pose module unavailable") -> dict[str, Any]:
    return {
        "available": False,
        "pose_detected": False,
        "status": "Unavailable",
        "message": message,
        "activity": "POSE NOT AVAILABLE",
        "activity_confidence": "UNKNOWN",
        "orientation": "HEAD ORIENTATION UNKNOWN",
        "orientation_confidence": "UNKNOWN",
        "awareness": "CANNOT BE INFERRED",
        "person_track_key": None,
        "mediapipe_ms": 0.0,
    }


class PoseBehaviorAnalyzer:
    """Optional, display-only MediaPipe cues; never a risk-state authority."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.landmarker: Any | None = None
        self.available = False
        self.message = "MediaPipe pose module has not been initialized"
        self.history: dict[str, deque[tuple[float, float, float]]] = {}
        self.last_timestamp_ms = -1

    def start(self) -> None:
        if self.landmarker is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not self.model_path.is_file():
                raise FileNotFoundError(f"Pose model not found: {self.model_path.name}")
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
            self.available = True
            self.message = "MediaPipe Pose Landmarker Lite available"
        except Exception as exc:
            self.available = False
            self.landmarker = None
            self.message = f"MediaPipe unavailable: {exc}"

    def reset(self) -> None:
        self.history.clear()
        self.last_timestamp_ms = -1

    def close(self) -> None:
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        self.landmarker = None
        self.available = False

    @staticmethod
    def classify_activity(samples: list[tuple[float, float, float]]) -> tuple[str, str]:
        if len(samples) < 4:
            return "MOTION UNCERTAIN", "LOW"
        movements = []
        for previous, current in zip(samples, samples[1:]):
            px, py, ph = previous
            cx, cy, ch = current
            scale = max((ph + ch) * 0.5, 1e-6)
            movements.append(float(np.hypot(cx - px, cy - py)) / scale)
        median_motion = float(np.median(movements))
        if median_motion >= WALKING_MOTION_THRESHOLD:
            return "LIKELY WALKING", "MEDIUM"
        if median_motion <= STANDING_MOTION_THRESHOLD:
            return "LIKELY STANDING", "MEDIUM"
        return "MOTION UNCERTAIN", "LOW"

    @staticmethod
    def classify_orientation(nose_x: float, left_ear_x: float, right_ear_x: float, vehicle_direction: float) -> tuple[str, str]:
        ear_midpoint = (left_ear_x + right_ear_x) * 0.5
        nose_offset = nose_x - ear_midpoint
        if abs(nose_offset) < HEAD_PROXY_DEADBAND or vehicle_direction == 0:
            return "HEAD ORIENTATION UNKNOWN", "UNKNOWN"
        toward = np.sign(nose_offset) == np.sign(vehicle_direction)
        return (
            "ORIENTED TOWARD VEHICLE — LOW-CONFIDENCE PROXY" if toward
            else "ORIENTED AWAY FROM VEHICLE — LOW-CONFIDENCE PROXY",
            "LOW",
        )

    def analyze(
        self,
        frame: np.ndarray,
        person: dict[str, Any],
        heavy_vehicle: dict[str, Any] | None,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self.start()
        if not self.available or self.landmarker is None:
            result = unavailable_observation(self.message)
            result["person_track_key"] = person.get("motion_track_key")
            return result
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = [float(value) for value in person["box"]]
            padding_x = (x2 - x1) * POSE_CROP_PADDING_RATIO
            padding_y = (y2 - y1) * POSE_CROP_PADDING_RATIO
            left = max(0, int(x1 - padding_x))
            top = max(0, int(y1 - padding_y))
            right = min(width, int(x2 + padding_x))
            bottom = min(height, int(y2 + padding_y))
            if right - left < 24 or bottom - top < 40:
                raise ValueError("Selected person crop is too small for a reliable pose")
            crop = frame[top:bottom, left:right]
            scale = min(1.0, POSE_MAX_INPUT_SIDE / max(crop.shape[:2]))
            if scale < 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            timestamp_ms = max(int(timestamp_ms), self.last_timestamp_ms + 1)
            self.last_timestamp_ms = timestamp_ms
            pose_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            if not pose_result.pose_landmarks:
                result = unavailable_observation("No reliable pose found in the selected person crop")
                result.update({"available": True, "status": "No reliable pose", "person_track_key": person.get("motion_track_key")})
                result["mediapipe_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                return result

            landmarks = pose_result.pose_landmarks[0]
            required = [landmarks[index] for index in (0, 7, 8, 23, 24)]
            if any(getattr(point, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD for point in required):
                raise ValueError("Key pose landmarks are not sufficiently visible")

            left_hip, right_hip = landmarks[23], landmarks[24]
            hip_x = (left_hip.x + right_hip.x) * 0.5
            hip_y = (left_hip.y + right_hip.y) * 0.5
            ys = [point.y for point in landmarks if getattr(point, "visibility", 1.0) >= POSE_VISIBILITY_THRESHOLD]
            body_height = max(max(ys) - min(ys), 0.05)
            person_key = str(person.get("motion_track_key") or "selected-person")
            samples = self.history.setdefault(person_key, deque(maxlen=POSE_HISTORY_LENGTH))
            samples.append((hip_x, hip_y, body_height))
            activity, activity_confidence = self.classify_activity(list(samples))

            orientation = "HEAD ORIENTATION UNKNOWN"
            orientation_confidence = "UNKNOWN"
            if heavy_vehicle is not None:
                person_center = (x1 + x2) * 0.5
                hx1, _, hx2, _ = [float(value) for value in heavy_vehicle["box"]]
                vehicle_direction = np.sign((hx1 + hx2) * 0.5 - person_center)
                orientation, orientation_confidence = self.classify_orientation(
                    landmarks[0].x, landmarks[7].x, landmarks[8].x, vehicle_direction
                )
            return {
                "available": True,
                "pose_detected": True,
                "status": "Pose cue available",
                "message": "Display-only MediaPipe observation; not used by the warning heuristic",
                "activity": activity,
                "activity_confidence": activity_confidence,
                "orientation": orientation,
                "orientation_confidence": orientation_confidence,
                "awareness": "CANNOT BE INFERRED",
                "person_track_key": person_key,
                "mediapipe_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }
        except Exception as exc:
            result = unavailable_observation(f"Pose cue unavailable: {exc}")
            result.update({"available": self.available, "status": "No reliable pose", "person_track_key": person.get("motion_track_key")})
            result["mediapipe_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            return result

