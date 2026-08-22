from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def empty_analysis(status: str = "NO PERSON DETECTED") -> dict[str, Any]:
    return {
        "status": status,
        "activity": "UNKNOWN",
        "activity_confidence": "LOW",
        "head_orientation": "UNKNOWN",
        "head_orientation_confidence": "LOW",
        "awareness": "CANNOT BE INFERRED",
        "pose_detected": False,
        "mediapipe_ms": 0.0,
    }


class PoseBehaviorAnalyzer:
    """Extract observable pose cues without claiming attention or intent."""

    def __init__(self, model_path: Path, max_poses: int = 4) -> None:
        self.model_path = model_path
        self.max_poses = max_poses
        self.landmarker: Any | None = None
        self.mp: Any | None = None
        self.unavailable_reason: str | None = None
        self.history: dict[str, deque[tuple[int, float, float, float]]] = {}

    def start(self) -> bool:
        if self.landmarker is not None:
            return True
        if not self.model_path.is_file():
            self.unavailable_reason = "POSE MODEL NOT INSTALLED"
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=self.max_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.mp = mp
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.history.clear()
            self.unavailable_reason = None
            return True
        except (ImportError, RuntimeError, ValueError) as exc:
            self.unavailable_reason = f"MEDIAPIPE UNAVAILABLE: {exc}"
            return False

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        self.landmarker = None
        self.mp = None
        self.history.clear()

    @staticmethod
    def _visible(landmark: Any, threshold: float = 0.45) -> bool:
        return (
            float(getattr(landmark, "visibility", 1.0) or 0.0) >= threshold
            and float(getattr(landmark, "presence", 1.0) or 0.0) >= threshold
        )

    @classmethod
    def _midpoint(cls, landmarks: list[Any], first: int, second: int) -> tuple[float, float] | None:
        a, b = landmarks[first], landmarks[second]
        if not cls._visible(a) or not cls._visible(b):
            return None
        return ((float(a.x) + float(b.x)) / 2.0, (float(a.y) + float(b.y)) / 2.0)

    @staticmethod
    def _angle(a: Any, b: Any, c: Any) -> float:
        first = np.asarray([float(a.x) - float(b.x), float(a.y) - float(b.y)])
        second = np.asarray([float(c.x) - float(b.x), float(c.y) - float(b.y)])
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-8:
            return 180.0
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _person_key(person: dict[str, Any]) -> str:
        if person.get("motion_track_key"):
            return str(person["motion_track_key"])
        x1, y1, x2, y2 = person["box"]
        return f"cell-{round(((x1 + x2) / 2) / 100)}-{round(((y1 + y2) / 2) / 100)}"

    @staticmethod
    def _pose_center(landmarks: list[Any]) -> tuple[float, float] | None:
        candidates = [landmarks[index] for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)]
        visible = [point for point in candidates if PoseBehaviorAnalyzer._visible(point)]
        if len(visible) < 2:
            return None
        return (
            sum(float(point.x) for point in visible) / len(visible),
            sum(float(point.y) for point in visible) / len(visible),
        )

    @staticmethod
    def _inside_box(center: tuple[float, float], person: dict[str, Any], width: int, height: int) -> bool:
        x1, y1, x2, y2 = person["box"]
        x, y = center[0] * width, center[1] * height
        margin_x = max((x2 - x1) * 0.12, 8.0)
        margin_y = max((y2 - y1) * 0.08, 8.0)
        return x1 - margin_x <= x <= x2 + margin_x and y1 - margin_y <= y <= y2 + margin_y

    def _classify_activity(
        self,
        landmarks: list[Any],
        person: dict[str, Any],
        timestamp_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[str, str]:
        hip = self._midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
        shoulder = self._midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        if hip is None or shoulder is None:
            return "UNKNOWN", "LOW"

        body_height = max(float(person["box"][3] - person["box"][1]), 1.0)
        samples = self.history.setdefault(self._person_key(person), deque(maxlen=18))
        samples.append((timestamp_ms, hip[0], hip[1], body_height))
        while samples and timestamp_ms - samples[0][0] > 900:
            samples.popleft()

        legs_visible = all(
            self._visible(landmarks[index])
            for index in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)
        )
        if legs_visible:
            left_knee = self._angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
            right_knee = self._angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
            torso_height = max(hip[1] - shoulder[1], 0.001)
            knee_y = (float(landmarks[LEFT_KNEE].y) + float(landmarks[RIGHT_KNEE].y)) / 2.0
            if min(left_knee, right_knee) < 105.0 and knee_y - hip[1] < torso_height * 0.8:
                return "LIKELY CROUCHING", "MEDIUM"

        if len(samples) >= 4 and samples[-1][0] - samples[0][0] >= 250:
            _, old_x, old_y, old_height = samples[0]
            displacement_px = math.hypot(
                (hip[0] - old_x) * frame_width,
                (hip[1] - old_y) * frame_height,
            )
            normalized_motion = displacement_px / max((body_height + old_height) / 2.0, 1.0)
            if normalized_motion >= 0.075:
                return "LIKELY WALKING", "MEDIUM"
            if normalized_motion <= 0.035:
                return "LIKELY STANDING", "MEDIUM"
        return "MOTION UNCERTAIN", "LOW"

    def _head_orientation(
        self,
        landmarks: list[Any],
        person: dict[str, Any],
        truck: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if truck is None:
            return "UNKNOWN — NO HEAVY VEHICLE", "LOW"
        nose = landmarks[NOSE]
        ear_mid = self._midpoint(landmarks, LEFT_EAR, RIGHT_EAR)
        if not self._visible(nose, 0.5) or ear_mid is None:
            return "UNKNOWN", "LOW"

        x1, _y1, x2, _y2 = person["box"]
        truck_x1, _truck_y1, truck_x2, _truck_y2 = truck["box"]
        truck_direction = math.copysign(1.0, ((truck_x1 + truck_x2) / 2.0) - ((x1 + x2) / 2.0))
        nose_offset = float(nose.x) - ear_mid[0]
        if abs(nose_offset) < 0.018:
            return "UNKNOWN / FORWARD", "LOW"
        if math.copysign(1.0, nose_offset) == truck_direction:
            return "ORIENTED TOWARD VEHICLE (PROXY)", "LOW"
        return "ORIENTED AWAY FROM VEHICLE (PROXY)", "LOW"

    @staticmethod
    def _select_person(people: list[dict[str, Any]], preferred_class: str | None) -> dict[str, Any]:
        candidates = [person for person in people if preferred_class == "person"] or people
        return max(candidates, key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]))

    def analyze(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        heavy_vehicle: dict[str, Any] | None,
        timestamp_ms: int,
        preferred_class: str | None = None,
    ) -> dict[str, Any]:
        people = [item for item in detections if item["class"] == "person"]
        if not people:
            return empty_analysis()
        if not self.start() or self.landmarker is None or self.mp is None:
            return empty_analysis(self.unavailable_reason or "MEDIAPIPE NOT READY")

        start = time.perf_counter()
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        primary = self._select_person(people, preferred_class)
        primary["is_primary_person"] = True

        height, width = frame.shape[:2]
        center_x = ((primary["box"][0] + primary["box"][2]) / 2.0) / width
        center_y = ((primary["box"][1] + primary["box"][3]) / 2.0) / height
        matches: list[tuple[float, list[Any]]] = []
        for pose in result.pose_landmarks:
            center = self._pose_center(pose)
            if center is None or not self._inside_box(center, primary, width, height):
                continue
            matches.append((math.hypot(center[0] - center_x, center[1] - center_y), pose))

        if not matches:
            analysis = empty_analysis("PERSON FOUND — POSE NOT FOUND")
            analysis["mediapipe_ms"] = round(elapsed_ms, 1)
            primary["behavior"] = analysis
            return analysis

        landmarks = min(matches, key=lambda pair: pair[0])[1]
        activity, activity_confidence = self._classify_activity(
            landmarks, primary, timestamp_ms, width, height
        )
        orientation, orientation_confidence = self._head_orientation(landmarks, primary, heavy_vehicle)
        analysis = {
            "status": "POSE ANALYZED",
            "activity": activity,
            "activity_confidence": activity_confidence,
            "head_orientation": orientation,
            "head_orientation_confidence": orientation_confidence,
            "awareness": "CANNOT BE INFERRED",
            "pose_detected": True,
            "mediapipe_ms": round(elapsed_ms, 1),
        }
        primary["behavior"] = analysis
        return analysis
