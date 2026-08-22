from __future__ import annotations

import atexit
import importlib.util
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from behavior import PoseBehaviorAnalyzer, empty_analysis


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
ZONES_PATH = BASE_DIR / "zones.json"
MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", MODEL_DIR / "yolo11n.pt"))
POSE_MODEL_PATH = Path(os.environ.get("MEDIAPIPE_POSE_MODEL_PATH", MODEL_DIR / "pose_landmarker_lite.task"))
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
RELEVANT_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle", 7: "truck"}
ZONE_MODE_FIXED = "fixed"
ZONE_MODE_MOVING = "moving"
ZONE_MODE_LABELS = {
    ZONE_MODE_FIXED: "FIXED INTERSECTION CAMERA",
    ZONE_MODE_MOVING: "MOVING-CAMERA DEMO",
}
BLIND_SIDES = {"right", "left"}
SMOOTHING_ALPHA = 0.25
MAX_TRUCK_LOST_FRAMES = 5
YOLO_TRACKING_AVAILABLE = importlib.util.find_spec("lap") is not None

# Dynamic-zone dimensions are expressed as multiples of the smoothed truck size.
TRUCK_ZONE_PAD_WIDTH = 0.18
TRUCK_ZONE_PAD_HEIGHT = 0.14
CONFLICT_NEAR_GAP_WIDTH = 0.06
CONFLICT_FAR_REACH_WIDTH = 1.10
CONFLICT_TOP_OFFSET_HEIGHT = 0.12
CONFLICT_FAR_TOP_OFFSET_HEIGHT = 0.28
CONFLICT_NEAR_BOTTOM_OFFSET_HEIGHT = 0.24
CONFLICT_FAR_BOTTOM_OFFSET_HEIGHT = 0.62
APPROACH_NEAR_REACH_WIDTH = 0.72
APPROACH_FAR_REACH_WIDTH = 2.35
APPROACH_TOP_OFFSET_HEIGHT = -0.18
APPROACH_FAR_TOP_OFFSET_HEIGHT = 0.02
APPROACH_NEAR_BOTTOM_OFFSET_HEIGHT = 0.72
APPROACH_FAR_BOTTOM_OFFSET_HEIGHT = 1.08
CLASS_COLORS = {
    "person": (86, 220, 255),
    "bicycle": (102, 236, 161),
    "motorcycle": (107, 179, 255),
    "truck": (255, 167, 70),
}
ZONE_COLORS = {
    "TRUCK_TURN_ZONE": (255, 225, 70),
    "ROAD_USER_APPROACH_ZONE": (45, 178, 255),
    "CONFLICT_ZONE": (55, 55, 255),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_zones() -> dict[str, list[list[float]]]:
    with ZONES_PATH.open("r", encoding="utf-8") as handle:
        zones = json.load(handle)
    required = {"TRUCK_TURN_ZONE", "ROAD_USER_APPROACH_ZONE", "CONFLICT_ZONE"}
    if set(zones) != required:
        raise ValueError(f"zones.json must contain exactly: {', '.join(sorted(required))}")
    for name, points in zones.items():
        if len(points) < 3 or any(len(point) != 2 for point in points):
            raise ValueError(f"{name} must contain at least three [x, y] points")
        if any(not 0 <= value <= 1 for point in points for value in point):
            raise ValueError(f"{name} coordinates must be normalized between 0 and 1")
    return zones


def blank_frame(message: str = "Upload a video to begin real YOLO processing") -> bytes:
    canvas = np.full((720, 1280, 3), (13, 19, 25), dtype=np.uint8)
    cv2.rectangle(canvas, (38, 38), (1242, 682), (35, 47, 57), 2)
    cv2.putText(canvas, "BLINDSPOT GUARDIAN", (78, 302), cv2.FONT_HERSHEY_DUPLEX, 1.2, (90, 222, 184), 2)
    cv2.putText(canvas, message, (78, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (184, 195, 202), 2)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else b""


class DetectionEngine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.model: YOLO | None = None
        self.pose_analyzer = PoseBehaviorAnalyzer(POSE_MODEL_PATH)
        self.source_path: Path | None = None
        self.frame_number = 0
        self.latest_jpeg = blank_frame()
        self.zone_mode = ZONE_MODE_MOVING
        self.blind_side = "right"
        self.zones_visible = True
        self.reset_tracking_requested = False
        self.warning_reset_requested = False
        self.selected_truck_id: int | None = None
        self.smoothed_truck_box: np.ndarray | None = None
        self.last_dynamic_zones: dict[str, list[list[float]]] | None = None
        self.truck_lost_frames = 0
        self.status: dict[str, Any] = self._initial_status()

    @staticmethod
    def _initial_status() -> dict[str, Any]:
        return {
            "running": False,
            "state": "MONITORING",
            "action": "Upload a video and start processing.",
            "fps": 0.0,
            "inference_ms": 0.0,
            "detections": [],
            "frame_index": 0,
            "error": None,
            "source_name": None,
            "zones_active": False,
            "zone_polygons": None,
            "primary_truck": {
                "status": "NOT DETECTED",
                "confidence": None,
                "track_id": None,
                "lost_frames": 0,
            },
            "pedestrian_analysis": empty_analysis("MONITORING", "WAITING FOR VIDEO"),
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            snapshot = dict(self.status)
            snapshot.update(
                {
                    "zone_mode": self.zone_mode,
                    "zone_mode_label": ZONE_MODE_LABELS[self.zone_mode],
                    "blind_side": self.blind_side,
                    "zones_visible": self.zones_visible,
                    "tracking_available": YOLO_TRACKING_AVAILABLE,
                }
            )
            return snapshot

    def update_settings(self, payload: dict[str, Any]) -> tuple[bool, str]:
        with self.lock:
            mode = payload.get("zone_mode", self.zone_mode)
            blind_side = payload.get("blind_side", self.blind_side)
            zones_visible = payload.get("zones_visible", self.zones_visible)
            if mode not in ZONE_MODE_LABELS:
                return False, "Unknown zone mode."
            if blind_side not in BLIND_SIDES:
                return False, "Blind side must be right or left."
            if not isinstance(zones_visible, bool):
                return False, "Zone visibility must be true or false."
            if mode != self.zone_mode:
                self.warning_reset_requested = True
            self.zone_mode = mode
            self.blind_side = blind_side
            self.zones_visible = zones_visible
            if payload.get("reset_tracking") is True:
                self.reset_tracking_requested = True
                self.warning_reset_requested = True
            return True, "Zone settings updated."

    def _reset_primary_truck(self) -> None:
        self.selected_truck_id = None
        self.smoothed_truck_box = None
        self.last_dynamic_zones = None
        self.truck_lost_frames = 0

    def _reset_yolo_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def start(self, source_path: Path, source_name: str) -> tuple[bool, str]:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return False, "A video is already being processed. Stop it before starting another."
            if not MODEL_PATH.is_file():
                return False, (
                    f"YOLO model not found: {MODEL_PATH}. "
                    "Run start.ps1 once to install dependencies and download yolo11n.pt, "
                    "or set YOLO_MODEL_PATH to an existing model file."
                )
            if not POSE_MODEL_PATH.is_file():
                return False, (
                    f"MediaPipe pose model not found: {POSE_MODEL_PATH}. "
                    "Run start.ps1 once to download pose_landmarker_lite.task, "
                    "or set MEDIAPIPE_POSE_MODEL_PATH to an existing model file."
                )
            self.stop_event.clear()
            self.source_path = source_path
            self.frame_number = 0
            self._reset_primary_truck()
            self._reset_yolo_tracker()
            self.status = self._initial_status()
            self.status.update({"running": True, "source_name": source_name, "action": "Loading YOLO model..."})
            self.worker = threading.Thread(target=self._process_video, name="yolo-video-worker", daemon=True)
            self.worker.start()
            return True, "Processing started."

    def stop(self, reset: bool = True) -> None:
        self.stop_event.set()
        worker = self.worker
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        with self.lock:
            if reset:
                self._reset_primary_truck()
                self.status = self._initial_status()
                self.latest_jpeg = blank_frame()
                self.frame_number += 1
                self.frame_ready.notify_all()

    def _load_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(str(MODEL_PATH))
        return self.model

    def _process_video(self) -> None:
        capture: cv2.VideoCapture | None = None
        danger_frames = 0
        safe_frames = 0
        visible_state = "MONITORING"
        previous_frame_time: float | None = None
        fps_ema = 0.0
        processed_frame_index = 0
        try:
            fixed_zones = load_zones()
            model = self._load_model()
            self.pose_analyzer.start()
            capture = cv2.VideoCapture(str(self.source_path))
            if not capture.isOpened():
                raise ValueError("OpenCV could not open the uploaded video.")

            source_fps = capture.get(cv2.CAP_PROP_FPS)
            source_delay = 1.0 / source_fps if source_fps and source_fps > 0 else 0.0

            while not self.stop_event.is_set():
                cycle_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                processed_frame_index += 1

                inference_start = time.perf_counter()
                inference_options = {
                    "source": frame,
                    "classes": list(RELEVANT_CLASSES),
                    "conf": 0.35,
                    "imgsz": 640,
                    "device": "cpu",
                    "verbose": False,
                }
                if YOLO_TRACKING_AVAILABLE:
                    results = model.track(**inference_options, persist=True, tracker="bytetrack.yaml")
                else:
                    results = model.predict(**inference_options)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                detections = self._extract_detections(results[0], frame.shape[1], frame.shape[0])
                with self.lock:
                    if self.reset_tracking_requested:
                        self._reset_primary_truck()
                        self._reset_yolo_tracker()
                        self.reset_tracking_requested = False
                    if self.warning_reset_requested:
                        danger_frames = 0
                        safe_frames = 0
                        visible_state = "MONITORING"
                        self.warning_reset_requested = False
                    zone_mode = self.zone_mode
                    blind_side = self.blind_side
                    zones_visible = self.zones_visible

                primary_truck, dynamic_zones, primary_status = self._update_primary_truck(
                    detections, frame.shape[1], frame.shape[0], blind_side
                )
                active_zones = fixed_zones if zone_mode == ZONE_MODE_FIXED else dynamic_zones
                raw_state = self._raw_warning_state(detections, active_zones, zone_mode, primary_truck)

                if raw_state == "DANGER":
                    danger_frames += 1
                    safe_frames = 0
                    if danger_frames >= 3:
                        visible_state = "DANGER"
                else:
                    danger_frames = 0
                    if visible_state == "DANGER":
                        safe_frames += 1
                        if safe_frames >= 5:
                            visible_state = raw_state
                            safe_frames = 0
                    else:
                        visible_state = raw_state

                pose_timestamp_ms = round(processed_frame_index * 1000.0 / source_fps) if source_fps and source_fps > 0 else processed_frame_index * 33
                try:
                    pedestrian_analysis = self.pose_analyzer.analyze(
                        frame,
                        detections,
                        primary_truck,
                        active_zones,
                        visible_state,
                        pose_timestamp_ms,
                        self._inside,
                    )
                except Exception as exc:
                    pedestrian_analysis = empty_analysis(visible_state, "MEDIAPIPE ANALYSIS ERROR")
                    pedestrian_analysis["error"] = str(exc)

                annotated = self._draw_frame(
                    frame,
                    detections,
                    active_zones,
                    visible_state,
                    zone_mode,
                    primary_truck,
                    zones_visible,
                )
                encode_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 84])
                if not encode_ok:
                    continue

                now = time.perf_counter()
                if previous_frame_time is not None:
                    instant_fps = 1.0 / max(now - previous_frame_time, 0.0001)
                    fps_ema = instant_fps if fps_ema == 0 else (0.82 * fps_ema + 0.18 * instant_fps)
                previous_frame_time = now

                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.frame_number += 1
                    self.status.update(
                        {
                            "running": True,
                            "state": visible_state,
                            "action": self._action_for(visible_state, zone_mode),
                            "fps": round(fps_ema, 1),
                            "inference_ms": round(inference_ms, 1),
                            "detections": detections,
                            "frame_index": self.frame_number,
                            "error": None,
                            "primary_truck": primary_status,
                            "zones_active": bool(zones_visible and active_zones),
                            "zone_polygons": active_zones,
                            "pedestrian_analysis": pedestrian_analysis,
                        }
                    )
                    self.frame_ready.notify_all()

                remaining = source_delay - (time.perf_counter() - cycle_start)
                if remaining > 0:
                    self.stop_event.wait(remaining)
        except Exception as exc:
            with self.lock:
                self.status.update({"running": False, "error": str(exc), "action": "Processing stopped because of an error."})
                self.latest_jpeg = blank_frame("Processing error. Check the status panel.")
                self.frame_number += 1
                self.frame_ready.notify_all()
        finally:
            self.pose_analyzer.close()
            if capture is not None:
                capture.release()
            with self.lock:
                self.status["running"] = False
                if not self.status.get("error") and not self.stop_event.is_set():
                    self.status["action"] = "Video complete. Upload another video or reset."
                self.frame_ready.notify_all()
            source = self.source_path
            self.source_path = None
            if source:
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _extract_detections(result: Any, width: int, height: int) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
        for index, (box, confidence, class_id) in enumerate(zip(xyxy, confidences, classes)):
            if class_id not in RELEVANT_CLASSES:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            item: dict[str, Any] = {
                "class": RELEVANT_CLASSES[class_id],
                "confidence": round(float(confidence), 3),
                "box": [round(x1), round(y1), round(x2), round(y2)],
                "contact": [round(((x1 + x2) / 2) / width, 4), round(y2 / height, 4)],
            }
            if ids is not None:
                item["track_id"] = int(ids[index])
            detections.append(item)
        return detections

    @staticmethod
    def _box_area(item: dict[str, Any]) -> float:
        x1, y1, x2, y2 = item["box"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _update_primary_truck(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        blind_side: str,
    ) -> tuple[dict[str, Any] | None, dict[str, list[list[float]]] | None, dict[str, Any]]:
        trucks = [item for item in detections if item["class"] == "truck"]
        for item in trucks:
            item["is_primary"] = False

        primary: dict[str, Any] | None = None
        if self.selected_truck_id is not None:
            primary = next((item for item in trucks if item.get("track_id") == self.selected_truck_id), None)
        if primary is None and trucks:
            primary = max(trucks, key=self._box_area)

        if primary is None:
            self.truck_lost_frames += 1
            if self.last_dynamic_zones is not None and self.truck_lost_frames <= MAX_TRUCK_LOST_FRAMES:
                return (
                    None,
                    self.last_dynamic_zones,
                    {
                        "status": f"TEMPORARILY LOST ({self.truck_lost_frames}/{MAX_TRUCK_LOST_FRAMES})",
                        "confidence": None,
                        "track_id": self.selected_truck_id,
                        "lost_frames": self.truck_lost_frames,
                    },
                )
            self.last_dynamic_zones = None
            self.smoothed_truck_box = None
            self.selected_truck_id = None
            return None, None, {"status": "NOT DETECTED", "confidence": None, "track_id": None, "lost_frames": self.truck_lost_frames}

        primary["is_primary"] = True
        current_track_id = primary.get("track_id")
        track_changed = (
            current_track_id is not None
            and self.selected_truck_id is not None
            and current_track_id != self.selected_truck_id
        )
        detected_box = np.asarray(primary["box"], dtype=np.float32)
        if self.smoothed_truck_box is None or track_changed:
            self.smoothed_truck_box = detected_box
        else:
            self.smoothed_truck_box = (
                SMOOTHING_ALPHA * detected_box + (1.0 - SMOOTHING_ALPHA) * self.smoothed_truck_box
            )
        self.selected_truck_id = current_track_id
        self.truck_lost_frames = 0
        self.last_dynamic_zones = self._dynamic_zones(
            self.smoothed_truck_box, frame_width, frame_height, blind_side
        )
        return (
            primary,
            self.last_dynamic_zones,
            {
                "status": "TRACKED",
                "confidence": primary["confidence"],
                "track_id": current_track_id,
                "lost_frames": 0,
                "smoothed_box": [round(float(value), 1) for value in self.smoothed_truck_box],
            },
        )

    @staticmethod
    def _dynamic_zones(
        smoothed_box: np.ndarray,
        frame_width: int,
        frame_height: int,
        blind_side: str,
    ) -> dict[str, list[list[float]]]:
        x1, y1, x2, y2 = [float(value) for value in smoothed_box]
        truck_width = max(x2 - x1, 1.0)
        truck_height = max(y2 - y1, 1.0)
        direction = 1.0 if blind_side == "right" else -1.0
        side_edge = x2 if direction > 0 else x1

        def normalized(x: float, y: float) -> list[float]:
            clipped_x = min(max(x, 0.0), float(frame_width - 1))
            clipped_y = min(max(y, 0.0), float(frame_height - 1))
            return [clipped_x / frame_width, clipped_y / frame_height]

        truck_zone = [
            normalized(x1 - TRUCK_ZONE_PAD_WIDTH * truck_width, y1 - TRUCK_ZONE_PAD_HEIGHT * truck_height),
            normalized(x2 + TRUCK_ZONE_PAD_WIDTH * truck_width, y1 - TRUCK_ZONE_PAD_HEIGHT * truck_height),
            normalized(x2 + TRUCK_ZONE_PAD_WIDTH * truck_width, y2 + TRUCK_ZONE_PAD_HEIGHT * truck_height),
            normalized(x1 - TRUCK_ZONE_PAD_WIDTH * truck_width, y2 + TRUCK_ZONE_PAD_HEIGHT * truck_height),
        ]
        conflict_zone = [
            normalized(side_edge + direction * CONFLICT_NEAR_GAP_WIDTH * truck_width, y1 + CONFLICT_TOP_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * CONFLICT_FAR_REACH_WIDTH * truck_width, y1 + CONFLICT_FAR_TOP_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * CONFLICT_FAR_REACH_WIDTH * truck_width, y2 + CONFLICT_FAR_BOTTOM_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * CONFLICT_NEAR_GAP_WIDTH * truck_width, y2 + CONFLICT_NEAR_BOTTOM_OFFSET_HEIGHT * truck_height),
        ]
        approach_zone = [
            normalized(side_edge + direction * APPROACH_NEAR_REACH_WIDTH * truck_width, y1 + APPROACH_TOP_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * APPROACH_FAR_REACH_WIDTH * truck_width, y1 + APPROACH_FAR_TOP_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * APPROACH_FAR_REACH_WIDTH * truck_width, y2 + APPROACH_FAR_BOTTOM_OFFSET_HEIGHT * truck_height),
            normalized(side_edge + direction * APPROACH_NEAR_REACH_WIDTH * truck_width, y2 + APPROACH_NEAR_BOTTOM_OFFSET_HEIGHT * truck_height),
        ]
        return {
            "TRUCK_TURN_ZONE": truck_zone,
            "ROAD_USER_APPROACH_ZONE": approach_zone,
            "CONFLICT_ZONE": conflict_zone,
        }

    @staticmethod
    def _inside(point: list[float], polygon: list[list[float]]) -> bool:
        contour = np.asarray(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    def _raw_warning_state(
        self,
        detections: list[dict[str, Any]],
        zones: dict[str, list[list[float]]] | None,
        zone_mode: str,
        primary_truck: dict[str, Any] | None,
    ) -> str:
        if zones is None:
            return "MONITORING"
        if zone_mode == ZONE_MODE_MOVING:
            if primary_truck is None:
                return "MONITORING"
            road_users = [item for item in detections if item["class"] in {"person", "bicycle", "motorcycle"}]
            if any(self._inside(item["contact"], zones["CONFLICT_ZONE"]) for item in road_users):
                return "DANGER"
            if any(self._inside(item["contact"], zones["ROAD_USER_APPROACH_ZONE"]) for item in road_users):
                return "CAUTION"
            return "TRUCK TRACKED"

        truck_present = any(
            item["class"] == "truck" and self._inside(item["contact"], zones["TRUCK_TURN_ZONE"])
            for item in detections
        )
        if not truck_present:
            return "MONITORING"
        road_users = [item for item in detections if item["class"] in {"person", "bicycle", "motorcycle"}]
        if any(self._inside(item["contact"], zones["CONFLICT_ZONE"]) for item in road_users):
            return "DANGER"
        if any(self._inside(item["contact"], zones["ROAD_USER_APPROACH_ZONE"]) for item in road_users):
            return "CAUTION"
        return "TRUCK PRESENT"

    @staticmethod
    def _action_for(state: str, zone_mode: str) -> str:
        actions = {
            "MONITORING": "Continue monitoring the configured zones.",
            "TRUCK PRESENT": "Truck contact point is inside the turn zone.",
            "TRUCK TRACKED": "A real truck detection is anchoring the demonstration zones.",
            "CAUTION": "Road user detected in the approach zone. Use caution.",
            "DANGER": "DO NOT ENTER THE BLIND ZONE",
        }
        if state == "MONITORING" and zone_mode == ZONE_MODE_MOVING:
            return "Waiting for a real truck detection before generating zones."
        return actions[state]

    @staticmethod
    def _pixel_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
        return np.asarray([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)

    def _draw_frame(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        zones: dict[str, list[list[float]]] | None,
        state: str,
        zone_mode: str,
        primary_truck: dict[str, Any] | None,
        zones_visible: bool,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        if zones_visible and zones is not None:
            overlay = frame.copy()
            polygons: dict[str, np.ndarray] = {}
            for name, points in zones.items():
                polygon = self._pixel_polygon(points, width, height)
                polygons[name] = polygon
                cv2.fillPoly(overlay, [polygon], ZONE_COLORS[name])
            zone_opacity = 0.20 if state == "DANGER" and self.frame_number % 2 == 0 else 0.12
            cv2.addWeighted(overlay, zone_opacity, frame, 1.0 - zone_opacity, 0, frame)
            for name, polygon in polygons.items():
                color = ZONE_COLORS[name]
                cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
                anchor = tuple(polygon[0])
                cv2.putText(
                    frame,
                    name,
                    (int(anchor[0]) + 6, max(18, int(anchor[1]) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                )

            if zone_mode == ZONE_MODE_MOVING and primary_truck is not None:
                truck_box = self.smoothed_truck_box
                conflict_polygon = polygons.get("CONFLICT_ZONE")
                if truck_box is not None and conflict_polygon is not None:
                    truck_center = (round(float((truck_box[0] + truck_box[2]) / 2)), round(float((truck_box[1] + truck_box[3]) / 2)))
                    conflict_center = tuple(np.mean(conflict_polygon, axis=0).astype(int))
                    cv2.line(frame, truck_center, conflict_center, ZONE_COLORS["CONFLICT_ZONE"], 2, cv2.LINE_AA)

        for item in detections:
            x1, y1, x2, y2 = [round(value) for value in item["box"]]
            color = CLASS_COLORS[item["class"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{item['class']} {item['confidence']:.2f}"
            if item.get("is_primary"):
                label += " PRIMARY"
                if item.get("track_id") is not None:
                    label += f" ID {item['track_id']}"
            behavior = item.get("behavior")
            behavior_label = ""
            if behavior:
                behavior_label = f"{behavior['activity']} | {behavior['head_orientation']}"
            label_width = max(120, len(label) * 10, len(behavior_label) * 7)
            label_height = 44 if behavior_label else 24
            cv2.rectangle(frame, (x1, max(0, y1 - label_height)), (x1 + label_width, y1), color, -1)
            label_y = max(16, y1 - (25 if behavior_label else 6))
            cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 14, 18), 2)
            if behavior_label:
                cv2.putText(frame, behavior_label, (x1 + 5, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (8, 14, 18), 1)
            contact_x = round(item["contact"][0] * width)
            contact_y = round(item["contact"][1] * height)
            cv2.circle(frame, (contact_x, contact_y), 5, color, -1)

        if state == "DANGER":
            border = 16 if self.frame_number % 2 == 0 else 7
            cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (30, 35, 255), border)
            warning = "DANGER - TURNING TRUCK" if zone_mode == ZONE_MODE_FIXED else "DANGER - TRUCK BLIND ZONE"
            cv2.putText(frame, warning, (32, 54), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        return frame

    def stream(self):
        last_frame = -1
        while True:
            with self.frame_ready:
                self.frame_ready.wait_for(lambda: self.frame_number != last_frame, timeout=1.0)
                frame = self.latest_jpeg
                last_frame = self.frame_number
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


engine = DetectionEngine()
atexit.register(engine.stop, False)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(engine.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    return jsonify(engine.snapshot())


@app.post("/api/settings")
def api_settings():
    payload = request.get_json(silent=True) or {}
    updated, message = engine.update_settings(payload)
    if not updated:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True, "message": message, "settings": engine.snapshot()})


@app.post("/api/start")
def api_start():
    if engine.snapshot()["running"]:
        return jsonify({"ok": False, "error": "A processing job is already running."}), 409
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Choose a video file first."}), 400
    original_name = secure_filename(upload.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": "Unsupported video type. Use MP4, MOV, AVI, MKV, or M4V."}), 400
    temporary_path = INPUT_DIR / f"{uuid.uuid4().hex}{extension}"
    upload.save(temporary_path)
    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": "The uploaded video is empty."}), 400
    started, message = engine.start(temporary_path, original_name)
    if not started:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": message}), 409
    return jsonify({"ok": True, "message": message})


@app.post("/api/stop")
def api_stop():
    engine.stop(reset=True)
    return jsonify({"ok": True})


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"ok": False, "error": "Upload exceeds the 500 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=APP_PORT, threaded=True, debug=False)


