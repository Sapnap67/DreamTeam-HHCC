from pathlib import Path
from urllib.request import Request, urlopen


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
TARGET = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024


def main() -> None:
    TARGET.parent.mkdir(exist_ok=True)
    temporary = TARGET.with_suffix(".task.download")
    request = Request(MODEL_URL, headers={"User-Agent": "BlindSpot-Guardian/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Pose model download exceeded the expected maximum size")
                output.write(chunk)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError("Downloaded pose model is unexpectedly small")
        temporary.replace(TARGET)
        print(f"MediaPipe Pose Landmarker Lite saved to {TARGET}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

