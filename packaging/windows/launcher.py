from __future__ import annotations

import argparse
import ctypes
import logging
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


APP_NAME = "BlindSpotGuardian"
MUTEX_NAME = "Local\\BlindSpotGuardian-Server-v1"
PREFERRED_PORT = 5000
PORT_SCAN_COUNT = 40
STARTUP_TIMEOUT_SECONDS = 45
EXISTING_INSTANCE_TIMEOUT_SECONDS = 15
PORT_FILE_NAME = "server-port.txt"


def resolve_resource_root(
    frozen: bool | None = None,
    meipass: str | None = None,
    launcher_file: str | None = None,
) -> Path:
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if frozen:
        root = meipass or getattr(sys, "_MEIPASS", None)
        if not root:
            raise RuntimeError("Packaged resource directory is unavailable.")
        return Path(root).resolve()
    return Path(launcher_file or __file__).resolve().parents[2]


def resolve_runtime_root(local_appdata: str | None = None) -> Path:
    base = local_appdata or os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base).resolve() / APP_NAME


def show_error(title: str, message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)


def acquire_single_instance() -> Any:
    if os.name != "nt":
        return object()
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        raise OSError("Windows could not create the application instance lock.")
    if ctypes.windll.kernel32.GetLastError() == 183:
        ctypes.windll.kernel32.CloseHandle(handle)
        return None
    return handle


def existing_instance_url(runtime_root: Path) -> str | None:
    """Return the live packaged-server URL recorded by the first instance."""
    port_file = runtime_root / PORT_FILE_NAME
    try:
        port = int(port_file.read_text(encoding="utf-8").strip())
        if not 1 <= port <= 65535:
            return None
        status_url = f"http://127.0.0.1:{port}/api/status"
        with urllib.request.urlopen(status_url, timeout=2) as response:
            if response.status == 200:
                return f"http://127.0.0.1:{port}/"
    except (OSError, ValueError):
        return None
    return None


def wait_for_existing_instance(runtime_root: Path) -> str | None:
    """Allow a first instance that is still starting time to publish its port."""
    deadline = time.monotonic() + EXISTING_INSTANCE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        url = existing_instance_url(runtime_root)
        if url:
            return url
        time.sleep(0.25)
    return None


def choose_port(preferred: int = PREFERRED_PORT) -> int:
    for port in range(preferred, preferred + PORT_SCAN_COUNT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available localhost port was found.")


def wait_for_health(url: str, timeout: float = STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Local server did not become ready: {last_error}")


def configure_logging(runtime_root: Path) -> Path:
    log_dir = runtime_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "blindspot-guardian.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return log_path


def load_application(resource_root: Path, runtime_root: Path, disable_mediapipe: bool = False):
    # Direct source launches begin with packaging/windows on sys.path. Add the
    # project resource root without changing app.py or its normal launch paths.
    resource_path = str(resource_root)
    if resource_path not in sys.path:
        sys.path.insert(0, resource_path)
    import app as source_app

    input_dir = runtime_root / "input"
    output_dir = runtime_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_app.INPUT_DIR = input_dir
    source_app.OUTPUT_DIR = output_dir
    source_app.MODEL_PATH = resource_root / "yolo11n.pt"
    source_app.POSE_MODEL_PATH = (
        runtime_root / "disabled-pose-model.task"
        if disable_mediapipe
        else resource_root / "models" / "pose_landmarker_lite.task"
    )
    source_app.engine.pose_analyzer.model_path = source_app.POSE_MODEL_PATH
    source_app.app.template_folder = str(resource_root / "templates")
    source_app.app.static_folder = str(resource_root / "static")
    source_app.app.config["MAX_CONTENT_LENGTH"] = source_app.MAX_UPLOAD_BYTES
    return source_app


def verify_video(source_app, video_path: Path, runtime_root: Path) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Verification video not found: {video_path}")
    temporary = runtime_root / "input" / f"package-verification{video_path.suffix.lower()}"
    shutil.copy2(video_path, temporary)
    try:
        started, message = source_app.engine.start(temporary, video_path.name)
        if not started:
            raise RuntimeError(message)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            snapshot = source_app.engine.snapshot()
            if snapshot.get("error"):
                raise RuntimeError(snapshot["error"])
            if snapshot.get("frame_index", 0) >= 3:
                return
            if not snapshot.get("running"):
                raise RuntimeError("Packaged inference stopped before producing three frames.")
            time.sleep(0.1)
        raise RuntimeError("Packaged YOLO inference timed out.")
    finally:
        source_app.engine.stop(reset=True)
        worker = getattr(source_app.engine, "worker", None)
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=30.0)
        # OpenCV can retain the file handle briefly while the worker finishes
        # releasing VideoCapture on Windows. Retry instead of turning a passed
        # inference check into a false packaging failure.
        for attempt in range(20):
            try:
                temporary.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 19:
                    logging.warning("Could not remove verification video after retries: %s", temporary)
                    break
                time.sleep(0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--disable-mediapipe", action="store_true")
    parser.add_argument("--verify-video", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mutex_handle = None
    active_port = None
    runtime_root = resolve_runtime_root()
    log_path = configure_logging(runtime_root)
    try:
        mutex_handle = acquire_single_instance()
        if mutex_handle is None:
            existing_url = wait_for_existing_instance(runtime_root)
            if not existing_url:
                raise RuntimeError(
                    "BlindSpot Guardian is running, but its browser address could not be reached. "
                    "End BlindSpotGuardian.exe in Task Manager and try again."
                )
            logging.info("Reopening existing BlindSpot Guardian instance at %s", existing_url)
            webbrowser.open(existing_url, new=2)
            return 0
        resource_root = resolve_resource_root()
        source_app = load_application(resource_root, runtime_root, args.disable_mediapipe)
        if not source_app.MODEL_PATH.is_file():
            raise FileNotFoundError("The bundled YOLO model yolo11n.pt is missing.")
        if args.verify_video:
            verify_video(source_app, args.verify_video, runtime_root)
            logging.info("Packaged YOLO video verification completed successfully.")
            return 0

        from werkzeug.serving import make_server

        port = choose_port()
        active_port = port
        server = make_server("127.0.0.1", port, source_app.app, threaded=True)
        server_thread = threading.Thread(target=server.serve_forever, name="blindspot-local-server", daemon=True)
        server_thread.start()
        status_url = f"http://127.0.0.1:{port}/api/status"
        wait_for_health(status_url)
        (runtime_root / PORT_FILE_NAME).write_text(str(port), encoding="utf-8")
        logging.info("BlindSpot Guardian ready at %s", status_url)
        if args.self_test:
            server.shutdown()
            return 0
        webbrowser.open(f"http://127.0.0.1:{port}/", new=2)
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            server.shutdown()
        finally:
            source_app.engine.stop(reset=True)
        return 0
    except Exception as exc:
        logging.exception("Startup failed")
        show_error("BlindSpot Guardian could not start", f"{exc}\n\nDiagnostic log:\n{log_path}")
        return 1
    finally:
        if active_port is not None:
            port_file = runtime_root / PORT_FILE_NAME
            try:
                if port_file.read_text(encoding="utf-8").strip() == str(active_port):
                    port_file.unlink(missing_ok=True)
            except OSError:
                pass
        if mutex_handle is not None and os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(mutex_handle)


if __name__ == "__main__":
    raise SystemExit(main())
