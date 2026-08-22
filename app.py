from __future__ import annotations

import atexit
import importlib.util
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from behavior import PoseBehaviorAnalyzer, unavailable_observation


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", BASE_DIR / "yolo11n.pt"))
POSE_MODEL_PATH = Path(os.environ.get("POSE_MODEL_PATH", BASE_DIR / "models" / "pose_landmarker_lite.task"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
RELEVANT_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BLIND_SIDES = {"right", "left"}
SMOOTHING_ALPHA = 0.25
MAX_TRUCK_LOST_FRAMES = 5
YOLO_TRACKING_AVAILABLE = importlib.util.find_spec("lap") is not None
POSE_ANALYSIS_INTERVAL_FRAMES = 3
POSE_RESULT_RETENTION_FRAMES = 5

# Image-space motion tracking and risk configuration.
POSITION_SMOOTHING_ALPHA = 0.35
TRACK_HISTORY_LENGTH = 8
MIN_MOTION_HISTORY = 4
STALE_TRACK_FRAMES = 12
FALLBACK_MIN_IOU = 0.08
FALLBACK_MAX_CENTER_DISTANCE = 0.10
TRUCK_LOWER_START = 0.38
TRUCK_FORWARD_MARGIN_WIDTH = 0.30
TRUCK_BLIND_MARGIN_WIDTH = 0.55
TRUCK_REAR_MARGIN_WIDTH = 0.12
TRUCK_MARGIN_DEPTH_HEIGHT = 0.35
ROAD_USER_MARGIN_RATIO = 0.08
MIN_CLOSE_DISTANCE = 0.07
CLOSE_DISTANCE_TRUCK_WIDTH = 0.85
CAUTION_DISTANCE_HEAVY_WIDTH = 1.40
EXTREME_DISTANCE_HEAVY_WIDTH = 0.60
DISTANCE_CLOSING_EPSILON = 0.0025
PATH_LOOKAHEAD_FRAMES = 8.0
PATH_CONVERGENCE_GAIN = 0.012
DANGER_REQUIRED_FRAMES = 4
DANGER_CLEAR_FRAMES = 6
CAUTION_REQUIRED_FRAMES = 2
CAUTION_CLEAR_FRAMES = 3
CLASS_COLORS = {
    "person": (86, 220, 255),
    "bicycle": (102, 236, 161),
    "car": (214, 132, 255),
    "motorcycle": (107, 179, 255),
    "truck": (255, 167, 70),
    "bus": (255, 151, 55),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


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
        self.source_path: Path | None = None
        self.frame_number = 0
        self.latest_jpeg = blank_frame()
        self.blind_side = "right"
        self.reset_tracking_requested = False
        self.warning_reset_requested = False
        self.selected_truck_id: int | None = None
        self.selected_truck_key: str | None = None
        self.smoothed_truck_box: np.ndarray | None = None
        self.truck_lost_frames = 0
        self.motion_tracks: dict[str, dict[str, Any]] = {}
        self.next_fallback_track_id = 1
        self.timeline_events: list[dict[str, Any]] = []
        self.observed_heavy_vehicle_keys: set[str] = set()
        self.observed_vulnerable_road_user_keys: set[str] = set()
        self.session_event_counter = 0
        self.session_started_at: str | None = None
        self.last_primary_class: str | None = None
        self.last_primary_confidence: float | None = None
        self.pose_analyzer = PoseBehaviorAnalyzer(POSE_MODEL_PATH)
        self.latest_pose_observation = unavailable_observation()
        self.last_pose_frame = -POSE_RESULT_RETENTION_FRAMES - 1
        self.last_pose_person_key: str | None = None
        self.risk_episode_id = 0
        self.active_risk_episode = False
        self.caution_sound_emitted = False
        self.danger_sound_emitted = False
        self.sound_event_counter = 0
        self.latest_sound_event: dict[str, Any] | None = None
        self.status: dict[str, Any] = self._initial_status()

    @staticmethod
    def _initial_status() -> dict[str, Any]:
        return {
            "running": False,
            "state": "MONITORING",
            "state_label": "MONITORING",
            "action": "Upload a video and start processing.",
            "fps": 0.0,
            "inference_ms": 0.0,
            "detections": [],
            "frame_index": 0,
            "error": None,
            "source_name": None,
            "primary_truck": {
                "status": "NOT DETECTED",
                "confidence": None,
                "track_id": None,
                "lost_frames": 0,
            },
            "evidence": DetectionEngine._empty_evidence(),
            "timeline": [],
            "session_summary": DetectionEngine._empty_session_summary(),
            "session_started_at": None,
            "pedestrian_observation": unavailable_observation(),
            "sound_event": None,
        }

    @staticmethod
    def _empty_session_summary() -> dict[str, int]:
        return {
            "heavy_vehicles_observed": 0,
            "vulnerable_road_users_observed": 0,
            "caution_events": 0,
            "danger_events": 0,
        }

    @staticmethod
    def _empty_evidence() -> dict[str, Any]:
        return {
            "heavy_vehicle_detected": False,
            "vulnerable_road_user_detected": False,
            "road_user_on_monitored_side": False,
            "road_user_within_caution_distance": False,
            "distance_decreasing": False,
            "motion_paths_converging": False,
            "expanded_margin_overlap": False,
            "motion_history_ready": False,
            "risk_sustained_frames": 0,
            "risk_sustained": False,
            "caution_persistence_frames": 0,
            "caution_required_frames": CAUTION_REQUIRED_FRAMES,
            "danger_persistence_frames": 0,
            "danger_required_frames": DANGER_REQUIRED_FRAMES,
            "road_user_class": None,
            "road_user_track_key": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            snapshot = dict(self.status)
            snapshot["timeline"] = [dict(event) for event in self.timeline_events]
            snapshot["session_summary"] = self._session_summary()
            snapshot["session_started_at"] = self.session_started_at
            snapshot.update(
                {
                    "blind_side": self.blind_side,
                    "tracking_available": YOLO_TRACKING_AVAILABLE,
                }
            )
            return snapshot

    def _reset_session_events(self, started: bool = False) -> None:
        self.timeline_events.clear()
        self.observed_heavy_vehicle_keys.clear()
        self.observed_vulnerable_road_user_keys.clear()
        self.session_event_counter = 0
        self.session_started_at = datetime.now(timezone.utc).isoformat() if started else None
        self.last_primary_class = None
        self.last_primary_confidence = None
        self._reset_sound_events()

    def _reset_sound_events(self) -> None:
        self.risk_episode_id = 0
        self.active_risk_episode = False
        self.caution_sound_emitted = False
        self.danger_sound_emitted = False
        self.latest_sound_event = None

    def _update_sound_event(self, previous_state: str, state: str) -> None:
        risky_states = {"CAUTION", "DANGER"}
        if state in risky_states and not self.active_risk_episode:
            self.risk_episode_id += 1
            self.active_risk_episode = True
            self.caution_sound_emitted = False
            self.danger_sound_emitted = False
        cue: str | None = None
        if state == "CAUTION" and not self.caution_sound_emitted:
            cue = "CAUTION_CHIME"
            self.caution_sound_emitted = True
        elif state == "DANGER" and not self.danger_sound_emitted:
            cue = "DANGER_TWO_PULSE"
            self.danger_sound_emitted = True
        if cue is not None:
            self.sound_event_counter += 1
            self.latest_sound_event = {
                "id": self.sound_event_counter,
                "episode_id": self.risk_episode_id,
                "cue": cue,
            }
        if previous_state in risky_states and state not in risky_states:
            self.active_risk_episode = False

    def _session_summary(self) -> dict[str, int]:
        return {
            "heavy_vehicles_observed": len(self.observed_heavy_vehicle_keys),
            "vulnerable_road_users_observed": len(self.observed_vulnerable_road_user_keys),
            "caution_events": sum(event["state"] == "CAUTION" for event in self.timeline_events),
            "danger_events": sum(event["state"] == "DANGER" for event in self.timeline_events),
        }

    def _update_session_observations(self, detections: list[dict[str, Any]]) -> None:
        for item in detections:
            key = item.get("motion_track_key")
            if not key:
                continue
            if item["class"] in {"truck", "bus"}:
                self.observed_heavy_vehicle_keys.add(key)
            elif item["class"] in {"person", "bicycle", "motorcycle"}:
                self.observed_vulnerable_road_user_keys.add(key)

    def _record_transition_event(
        self,
        previous_state: str,
        state: str,
        timestamp_seconds: float,
        reason: str,
        primary: dict[str, Any] | None,
        evidence: dict[str, Any],
    ) -> None:
        safe_states = {"MONITORING", "VEHICLE TRACKED"}
        meaningful = (
            previous_state in safe_states and state == "CAUTION"
            or previous_state == "CAUTION" and state == "DANGER"
            or previous_state == "DANGER" and state == "CAUTION"
            or previous_state in {"CAUTION", "DANGER"} and state in safe_states
        )
        if not meaningful:
            return
        if primary is not None:
            self.last_primary_class = primary["class"]
            self.last_primary_confidence = float(primary["confidence"])
        active_evidence = [
            key for key, value in evidence.items()
            if isinstance(value, bool) and value and key not in {"risk_sustained", "motion_history_ready"}
        ]
        self.session_event_counter += 1
        self.timeline_events.append(
            {
                "id": f"E{self.session_event_counter:04d}",
                "timestamp_seconds": round(max(0.0, timestamp_seconds), 3),
                "state": state,
                "reason": reason,
                "heavy_vehicle_class": primary["class"] if primary is not None else self.last_primary_class,
                "heavy_vehicle_confidence": (
                    round(float(primary["confidence"]), 3) if primary is not None else self.last_primary_confidence
                ),
                "road_user_class": evidence.get("road_user_class"),
                "active_evidence": active_evidence,
                "caution_persistence": int(evidence.get("caution_persistence_frames", 0)),
                "danger_persistence": int(evidence.get("danger_persistence_frames", 0)),
            }
        )

    def update_settings(self, payload: dict[str, Any]) -> tuple[bool, str]:
        with self.lock:
            blind_side = payload.get("blind_side", self.blind_side)
            if blind_side not in BLIND_SIDES:
                return False, "Blind side must be right or left."
            if blind_side != self.blind_side:
                self.warning_reset_requested = True
            self.blind_side = blind_side
            if payload.get("reset_tracking") is True:
                self.reset_tracking_requested = True
                self.warning_reset_requested = True
            return True, "Risk-tracking settings updated."

    def _reset_primary_truck(self) -> None:
        self.selected_truck_id = None
        self.selected_truck_key = None
        self.smoothed_truck_box = None
        self.truck_lost_frames = 0
        self.motion_tracks.clear()
        self.next_fallback_track_id = 1
        self.pose_analyzer.reset()
        self.latest_pose_observation = unavailable_observation()
        self.last_pose_frame = -POSE_RESULT_RETENTION_FRAMES - 1
        self.last_pose_person_key = None

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
                return False, f"YOLO model not found: {MODEL_PATH}"
            self.stop_event.clear()
            self.source_path = source_path
            self.frame_number = 0
            self._reset_primary_truck()
            self._reset_session_events(started=True)
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
                self._reset_session_events(started=False)
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
        caution_frames = 0
        safe_frames = 0
        visible_state = "MONITORING"
        visible_reason = "No meaningful risk evidence"
        previous_frame_time: float | None = None
        fps_ema = 0.0
        try:
            model = self._load_model()
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
                source_timestamp_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

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
                        caution_frames = 0
                        safe_frames = 0
                        visible_state = "MONITORING"
                        visible_reason = "Risk tracking reset"
                        self._reset_sound_events()
                        self.warning_reset_requested = False
                    blind_side = self.blind_side

                self._update_motion_tracks(detections, frame.shape[1], frame.shape[0])
                self._update_session_observations(detections)
                primary_truck, primary_status = self._update_primary_heavy_vehicle(detections)
                if primary_truck is not None:
                    self.last_primary_class = primary_truck["class"]
                    self.last_primary_confidence = float(primary_truck["confidence"])
                raw_state, raw_reason, raw_evidence = self._evaluate_risk(
                    detections, primary_truck, frame.shape[1], frame.shape[0], blind_side
                )

                previous_visible_state = visible_state
                if raw_state == "DANGER":
                    danger_frames += 1
                    caution_frames += 1
                    safe_frames = 0
                    if danger_frames >= DANGER_REQUIRED_FRAMES:
                        visible_state = "DANGER"
                        visible_reason = raw_reason
                    elif visible_state != "DANGER" and caution_frames >= CAUTION_REQUIRED_FRAMES:
                        visible_state = "CAUTION"
                        visible_reason = raw_reason
                elif raw_state == "CAUTION":
                    danger_frames = 0
                    caution_frames += 1
                    if visible_state == "DANGER":
                        safe_frames += 1
                        if safe_frames >= DANGER_CLEAR_FRAMES:
                            visible_state = "CAUTION"
                            visible_reason = raw_reason
                            safe_frames = 0
                    elif caution_frames >= CAUTION_REQUIRED_FRAMES:
                        visible_state = "CAUTION"
                        visible_reason = raw_reason
                else:
                    danger_frames = 0
                    caution_frames = 0
                    if visible_state in {"DANGER", "CAUTION"}:
                        safe_frames += 1
                        clear_frames = DANGER_CLEAR_FRAMES if visible_state == "DANGER" else CAUTION_CLEAR_FRAMES
                        if safe_frames >= clear_frames:
                            visible_state = raw_state
                            visible_reason = raw_reason
                            safe_frames = 0
                    else:
                        visible_state = raw_state
                        visible_reason = raw_reason

                raw_evidence["risk_sustained_frames"] = danger_frames
                raw_evidence["risk_sustained"] = visible_state == "DANGER"
                raw_evidence["caution_persistence_frames"] = min(caution_frames, CAUTION_REQUIRED_FRAMES)
                raw_evidence["danger_persistence_frames"] = min(danger_frames, DANGER_REQUIRED_FRAMES)
                self._record_transition_event(
                    previous_visible_state,
                    visible_state,
                    source_timestamp_seconds,
                    visible_reason,
                    primary_truck,
                    raw_evidence,
                )
                self._update_sound_event(previous_visible_state, visible_state)
                selected_person = self._select_pose_person(detections, primary_truck, raw_evidence)
                pose_observation = self._update_pose_observation(
                    frame, selected_person, primary_truck, source_timestamp_seconds
                )

                annotated = self._draw_frame(
                    frame,
                    detections,
                    visible_state,
                    primary_truck,
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
                            "state_label": self._state_label(visible_state),
                            "action": visible_reason,
                            "fps": round(fps_ema, 1),
                            "inference_ms": round(inference_ms, 1),
                            "detections": detections,
                            "frame_index": self.frame_number,
                            "error": None,
                            "primary_truck": primary_status,
                            "evidence": raw_evidence,
                            "timeline": [dict(event) for event in self.timeline_events],
                            "session_summary": self._session_summary(),
                            "session_started_at": self.session_started_at,
                            "pedestrian_observation": pose_observation,
                            "sound_event": dict(self.latest_sound_event) if self.latest_sound_event else None,
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

    @staticmethod
    def _box_iou(first: list[float], second: list[float]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center(box: list[float], width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.asarray([((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height], dtype=np.float32)

    def _update_motion_tracks(self, detections: list[dict[str, Any]], width: int, height: int) -> None:
        used_fallback_keys: set[str] = set()
        for item in detections:
            track_id = item.get("track_id")
            if track_id is not None:
                key = f"yolo:{item['class']}:{track_id}"
            else:
                center = self._center(item["box"], width, height)
                candidates: list[tuple[float, str]] = []
                for candidate_key, track in self.motion_tracks.items():
                    if not candidate_key.startswith("local:") or candidate_key in used_fallback_keys:
                        continue
                    if track["class"] != item["class"] or self.frame_number - track["last_frame"] > STALE_TRACK_FRAMES:
                        continue
                    iou = self._box_iou(item["box"], track["box"])
                    distance = float(np.linalg.norm(center - track["center"]))
                    if iou >= FALLBACK_MIN_IOU or distance <= FALLBACK_MAX_CENTER_DISTANCE:
                        candidates.append((2.0 * iou - distance, candidate_key))
                if candidates:
                    key = max(candidates)[1]
                else:
                    key = f"local:{self.next_fallback_track_id}"
                    self.next_fallback_track_id += 1
                used_fallback_keys.add(key)

            contact = np.asarray(item["contact"], dtype=np.float32)
            center = self._center(item["box"], width, height)
            track = self.motion_tracks.get(key)
            if track is None:
                track = {
                    "class": item["class"],
                    "box": list(item["box"]),
                    "center": center,
                    "smoothed_contact": contact,
                    "history": deque(maxlen=TRACK_HISTORY_LENGTH),
                    "last_frame": self.frame_number,
                }
                self.motion_tracks[key] = track
            else:
                track["smoothed_contact"] = (
                    POSITION_SMOOTHING_ALPHA * contact
                    + (1.0 - POSITION_SMOOTHING_ALPHA) * track["smoothed_contact"]
                )
                track["center"] = center
                track["box"] = list(item["box"])
                track["last_frame"] = self.frame_number
            track["history"].append((self.frame_number, track["smoothed_contact"].copy()))
            item["motion_track_key"] = key
            item["smoothed_contact"] = [float(value) for value in track["smoothed_contact"]]

        stale_keys = [
            key for key, track in self.motion_tracks.items()
            if self.frame_number - track["last_frame"] > STALE_TRACK_FRAMES
        ]
        for key in stale_keys:
            self.motion_tracks.pop(key, None)
            if key == self.selected_truck_key:
                self.selected_truck_key = None

    def _motion(self, item: dict[str, Any]) -> tuple[np.ndarray, int]:
        track = self.motion_tracks.get(item.get("motion_track_key", ""))
        if track is None or len(track["history"]) < MIN_MOTION_HISTORY:
            return np.zeros(2, dtype=np.float32), 0
        oldest_frame, oldest = track["history"][0]
        newest_frame, newest = track["history"][-1]
        span = max(newest_frame - oldest_frame, 1)
        return (newest - oldest) / span, len(track["history"])

    def _update_primary_heavy_vehicle(
        self,
        detections: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        heavy_vehicles = [item for item in detections if item["class"] in {"truck", "bus"}]
        for item in heavy_vehicles:
            item["is_primary"] = False

        primary: dict[str, Any] | None = None
        if self.selected_truck_key is not None:
            primary = next((item for item in heavy_vehicles if item.get("motion_track_key") == self.selected_truck_key), None)
        if primary is None and heavy_vehicles:
            primary = max(heavy_vehicles, key=self._box_area)

        if primary is None:
            self.truck_lost_frames += 1
            if self.smoothed_truck_box is not None and self.truck_lost_frames <= MAX_TRUCK_LOST_FRAMES:
                return None, {
                    "status": f"TEMPORARILY LOST ({self.truck_lost_frames}/{MAX_TRUCK_LOST_FRAMES})",
                    "confidence": None,
                    "track_id": self.selected_truck_id,
                    "lost_frames": self.truck_lost_frames,
                }
            self.smoothed_truck_box = None
            self.selected_truck_id = None
            self.selected_truck_key = None
            return None, {"status": "NOT DETECTED", "confidence": None, "track_id": None, "lost_frames": self.truck_lost_frames}

        primary["is_primary"] = True
        current_track_id = primary.get("track_id")
        current_track_key = primary.get("motion_track_key")
        track_changed = self.selected_truck_key is not None and current_track_key != self.selected_truck_key
        detected_box = np.asarray(primary["box"], dtype=np.float32)
        if self.smoothed_truck_box is None or track_changed:
            self.smoothed_truck_box = detected_box
        else:
            self.smoothed_truck_box = (
                SMOOTHING_ALPHA * detected_box + (1.0 - SMOOTHING_ALPHA) * self.smoothed_truck_box
            )
        self.selected_truck_id = current_track_id
        self.selected_truck_key = current_track_key
        self.truck_lost_frames = 0
        return (
            primary,
            {
                "status": f"TRACKED {primary['class'].upper()}",
                "confidence": primary["confidence"],
                "track_id": current_track_id,
                "lost_frames": 0,
                "smoothed_box": [round(float(value), 1) for value in self.smoothed_truck_box],
            },
        )

    def _evaluate_risk(
        self,
        detections: list[dict[str, Any]],
        primary_truck: dict[str, Any] | None,
        width: int,
        height: int,
        blind_side: str,
    ) -> tuple[str, str, dict[str, Any]]:
        evidence = self._empty_evidence()
        if primary_truck is None or self.smoothed_truck_box is None:
            return "MONITORING", "Waiting for a real heavy-vehicle detection", evidence

        evidence["heavy_vehicle_detected"] = True

        truck_motion, truck_history = self._motion(primary_truck)
        truck_contact = np.asarray(primary_truck.get("smoothed_contact", primary_truck["contact"]), dtype=np.float32)
        tx1, ty1, tx2, ty2 = [float(value) for value in self.smoothed_truck_box]
        truck_width = max(tx2 - tx1, 1.0)
        truck_height = max(ty2 - ty1, 1.0)
        if blind_side == "right":
            margin_x1 = tx1 - TRUCK_REAR_MARGIN_WIDTH * truck_width
            margin_x2 = tx2 + TRUCK_BLIND_MARGIN_WIDTH * truck_width
        else:
            margin_x1 = tx1 - TRUCK_BLIND_MARGIN_WIDTH * truck_width
            margin_x2 = tx2 + TRUCK_REAR_MARGIN_WIDTH * truck_width
        margin_y1 = ty1 + TRUCK_LOWER_START * truck_height
        margin_y2 = ty2 + TRUCK_MARGIN_DEPTH_HEIGHT * truck_height

        best_state = "VEHICLE TRACKED"
        best_reason = f"{primary_truck['class'].capitalize()} tracked; no vulnerable road user nearby"
        road_users = [
            item for item in detections
            if item is not primary_truck and item["class"] in {"person", "bicycle", "motorcycle"}
        ]
        evidence["vulnerable_road_user_detected"] = bool(road_users)
        closest_distance = float("inf")
        for item in road_users:
            road_motion, road_history = self._motion(item)
            road_contact = np.asarray(item.get("smoothed_contact", item["contact"]), dtype=np.float32)
            relative = road_contact - truck_contact
            distance = float(np.linalg.norm(relative))
            close_limit = max(MIN_CLOSE_DISTANCE, CLOSE_DISTANCE_TRUCK_WIDTH * truck_width / width)
            close = distance <= close_limit
            caution_limit = CAUTION_DISTANCE_HEAVY_WIDTH * truck_width / width
            within_caution_distance = distance <= caution_limit
            extremely_close = distance <= EXTREME_DISTANCE_HEAVY_WIDTH * truck_width / width

            rx1, ry1, rx2, ry2 = [float(value) for value in item["box"]]
            road_margin_x = ROAD_USER_MARGIN_RATIO * max(rx2 - rx1, 1.0)
            road_margin_y = ROAD_USER_MARGIN_RATIO * max(ry2 - ry1, 1.0)
            expanded_overlap = not (
                rx2 + road_margin_x < margin_x1 or rx1 - road_margin_x > margin_x2
                or ry2 + road_margin_y < margin_y1 or ry1 - road_margin_y > margin_y2
            )
            on_selected_side = (
                road_contact[0] >= tx2 / width if blind_side == "right" else road_contact[0] <= tx1 / width
            )
            beside_or_ahead = on_selected_side and (ty1 + 0.20 * truck_height) / height <= road_contact[1] <= margin_y2 / height

            enough_history = truck_history >= MIN_MOTION_HISTORY and road_history >= MIN_MOTION_HISTORY
            item_evidence = {
                **evidence,
                "road_user_on_monitored_side": bool(on_selected_side),
                "road_user_within_caution_distance": bool(within_caution_distance),
                "expanded_margin_overlap": bool(expanded_overlap),
                "motion_history_ready": bool(enough_history),
                "road_user_class": item["class"],
                "road_user_track_key": item.get("motion_track_key"),
            }
            if distance < closest_distance:
                closest_distance = distance
                evidence = item_evidence
            decreasing = False
            paths_converging = False
            if enough_history:
                relative_motion = road_motion - truck_motion
                closing_rate = -float(np.dot(relative, relative_motion)) / max(distance, 1e-6)
                decreasing = closing_rate > DISTANCE_CLOSING_EPSILON
                projected_relative = relative + relative_motion * PATH_LOOKAHEAD_FRAMES
                projected_distance = float(np.linalg.norm(projected_relative))
                paths_converging = projected_distance + PATH_CONVERGENCE_GAIN < distance
            confidence_supported = min(float(primary_truck["confidence"]), float(item["confidence"])) >= 0.40
            item_evidence["distance_decreasing"] = bool(decreasing)
            item_evidence["motion_paths_converging"] = bool(paths_converging)
            label = item["class"].capitalize()

            caution_evidence = on_selected_side and within_caution_distance and confidence_supported
            stronger_evidence = decreasing or expanded_overlap or paths_converging or extremely_close
            danger_evidence = caution_evidence and stronger_evidence
            if danger_evidence:
                if paths_converging:
                    return "DANGER", f"{label} and heavy-vehicle paths converging", item_evidence
                if expanded_overlap:
                    return "DANGER", f"{label} overlapping the heavy vehicle's safety margin", item_evidence
                if decreasing:
                    return "DANGER", f"{label} closing on the heavy vehicle", item_evidence
                return "DANGER", f"{label} remaining extremely close to the heavy vehicle", item_evidence
            if caution_evidence and best_state != "CAUTION":
                best_state = "CAUTION"
                evidence = item_evidence
                best_reason = (
                    f"{label} near the heavy vehicle on the monitored side"
                    if on_selected_side and decreasing
                    else f"{label} within caution distance on the monitored side"
                )
        return best_state, best_reason, evidence

    def _select_pose_person(
        self,
        detections: list[dict[str, Any]],
        primary_heavy_vehicle: dict[str, Any] | None,
        evidence: dict[str, Any],
    ) -> dict[str, Any] | None:
        people = [item for item in detections if item["class"] == "person"]
        if not people:
            return None
        risk_key = evidence.get("road_user_track_key") if evidence.get("road_user_class") == "person" else None
        if risk_key:
            matched = next((item for item in people if item.get("motion_track_key") == risk_key), None)
            if matched is not None:
                return matched
        if primary_heavy_vehicle is not None:
            heavy_contact = np.asarray(
                primary_heavy_vehicle.get("smoothed_contact", primary_heavy_vehicle["contact"]), dtype=np.float32
            )
            return min(
                people,
                key=lambda item: float(
                    np.linalg.norm(
                        np.asarray(item.get("smoothed_contact", item["contact"]), dtype=np.float32) - heavy_contact
                    )
                ),
            )
        return max(people, key=self._box_area)

    def _update_pose_observation(
        self,
        frame: np.ndarray,
        selected_person: dict[str, Any] | None,
        primary_heavy_vehicle: dict[str, Any] | None,
        timestamp_seconds: float,
    ) -> dict[str, Any]:
        if selected_person is None:
            self.latest_pose_observation = unavailable_observation("No real person detection selected for pose analysis")
            self.last_pose_person_key = None
            return self.latest_pose_observation
        person_key = str(selected_person.get("motion_track_key") or "selected-person")
        selection_changed = person_key != self.last_pose_person_key
        analysis_due = selection_changed or self.frame_number % POSE_ANALYSIS_INTERVAL_FRAMES == 0
        if analysis_due:
            self.latest_pose_observation = self.pose_analyzer.analyze(
                frame,
                selected_person,
                primary_heavy_vehicle,
                round(timestamp_seconds * 1000),
            )
            self.last_pose_frame = self.frame_number
            self.last_pose_person_key = person_key
        elif self.frame_number - self.last_pose_frame > POSE_RESULT_RETENTION_FRAMES:
            self.latest_pose_observation = unavailable_observation("Pose observation expired; waiting for the next sample")
        return dict(self.latest_pose_observation)

    @staticmethod
    def _state_label(state: str) -> str:
        return {
            "MONITORING": "MONITORING",
            "VEHICLE TRACKED": "VEHICLE TRACKED",
            "CAUTION": "CAUTION — HEAVY VEHICLE NEAR CROSSING",
            "DANGER": "STOP — BLIND-SPOT COLLISION RISK",
        }[state]

    def _draw_frame(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        state: str,
        primary_truck: dict[str, Any] | None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        for item in detections:
            x1, y1, x2, y2 = [round(value) for value in item["box"]]
            color = CLASS_COLORS[item["class"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{item['class']} {item['confidence']:.2f}"
            if item.get("is_primary"):
                label += " PRIMARY"
                if item.get("track_id") is not None:
                    label += f" ID {item['track_id']}"
            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + max(120, len(label) * 10), y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 14, 18), 2)

        if state == "DANGER":
            border = 16 if self.frame_number % 2 == 0 else 7
            cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (30, 35, 255), border)
            cv2.putText(frame, "DANGER - POTENTIAL COLLISION RISK", (32, 54), cv2.FONT_HERSHEY_DUPLEX, 0.92, (255, 255, 255), 2)
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


@app.get("/api/session-report")
def api_session_report():
    snapshot = engine.snapshot()
    summary = snapshot["session_summary"]
    report = {
        "project_name": "BlindSpot Guardian",
        "source_filename": snapshot.get("source_name"),
        "processing_date_time": snapshot.get("session_started_at"),
        "total_caution_events": summary["caution_events"],
        "total_danger_events": summary["danger_events"],
        "timeline_events": snapshot["timeline"],
        "disclaimer": "Prototype image-space observations, not verified crash predictions.",
    }
    return Response(
        json.dumps(report, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=blindspot-guardian-session-report.json"},
    )


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
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)

