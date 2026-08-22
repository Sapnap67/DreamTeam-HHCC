"""Download the official Ultralytics YOLO11 nano weights for local use."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"


def main() -> None:
    """Download into the project folder and verify that the file exists."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        print(f"YOLO model already available: {MODEL_PATH}")
        return

    os.chdir(BASE_DIR)
    print("Downloading official YOLO11 nano weights...")
    YOLO("yolo11n.pt")

    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(f"YOLO download did not create {MODEL_PATH}")

    print(f"YOLO model ready: {MODEL_PATH}")


if __name__ == "__main__":
    main()
