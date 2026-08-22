from __future__ import annotations

import json
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import INPUT_DIR, DetectionEngine  # noqa: E402


def main() -> None:
    source = Path(sys.argv[1])
    temporary = INPUT_DIR / f"validation-{uuid.uuid4().hex}{source.suffix}"
    shutil.copy2(source, temporary)
    engine = DetectionEngine()
    started_at = time.perf_counter()
    started, message = engine.start(temporary, source.name)
    if not started:
        raise RuntimeError(message)
    pose_statuses: Counter[str] = Counter()
    pose_samples = 0
    fps_samples: list[float] = []
    inference_samples: list[float] = []
    mediapipe_samples: list[float] = []
    while True:
        snapshot = engine.snapshot()
        observation = snapshot.get("pedestrian_observation") or {}
        if observation.get("person_track_key"):
            pose_samples += 1
            pose_statuses[observation.get("activity", "UNKNOWN")] += 1
            mediapipe_samples.append(float(observation.get("mediapipe_ms", 0)))
        if snapshot.get("fps", 0):
            fps_samples.append(float(snapshot["fps"]))
        if snapshot.get("inference_ms", 0):
            inference_samples.append(float(snapshot["inference_ms"]))
        if not snapshot["running"]:
            break
        time.sleep(0.05)
    if engine.worker is not None:
        engine.worker.join(timeout=5.0)
    temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started_at
    final = engine.snapshot()
    output = {
        "elapsed_seconds": round(elapsed, 2),
        "frames": final["frame_index"],
        "throughput_fps": round(final["frame_index"] / max(elapsed, 0.001), 2),
        "mean_displayed_fps": round(sum(fps_samples) / len(fps_samples), 2) if fps_samples else 0,
        "mean_yolo_ms": round(sum(inference_samples) / len(inference_samples), 2) if inference_samples else 0,
        "mean_mediapipe_ms": round(sum(mediapipe_samples) / len(mediapipe_samples), 2) if mediapipe_samples else 0,
        "pose_poll_samples": pose_samples,
        "pose_activity_observations": dict(pose_statuses),
        "timeline": [{"timestamp": event["timestamp_seconds"], "state": event["state"]} for event in final["timeline"]],
        "sound_events_emitted": engine.sound_event_counter,
        "error": final.get("error"),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

