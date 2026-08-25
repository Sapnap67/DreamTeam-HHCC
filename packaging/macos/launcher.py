from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

APP_NAME = "BlindSpotGuardian"
PREFERRED_PORT = 5000
PORT_SCAN_COUNT = 40
STARTUP_TIMEOUT_SECONDS = 45
EXISTING_INSTANCE_TIMEOUT_SECONDS = 15
PORT_FILE_NAME = "server-port.txt"
LOCK_FILE_NAME = "server.lock"


def resolve_resource_root(frozen=None, meipass=None, launcher_file=None):
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if frozen:
        root = meipass or getattr(sys, "_MEIPASS", None)
        if not root:
            raise RuntimeError("Packaged resource directory is unavailable.")
        return Path(root).resolve()
    return Path(launcher_file or __file__).resolve().parents[2]


def resolve_runtime_paths(home=None):
    home_path = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()
    return (
        home_path / "Library" / "Application Support" / APP_NAME,
        home_path / "Library" / "Logs" / APP_NAME,
    )


def resolve_pose_model(resource_root: Path, runtime_root: Path, disabled=False):
    bundled = resource_root / "models" / "pose_landmarker_lite.task"
    return bundled if not disabled and bundled.is_file() else runtime_root / "unavailable-pose-model.task"


def show_error(message: str) -> None:
    if sys.platform == "darwin":
        clean_message = " ".join(message.splitlines()).replace("\\", "\\\\").replace('"', '\\"')
        script = f'display alert "BlindSpot Guardian could not start" message "{clean_message}" as critical'
        try:
            subprocess.run(["osascript", "-e", script], check=False, timeout=15)
            return
        except Exception:
            pass
    print(f"BlindSpot Guardian could not start: {message}", file=sys.stderr)


def acquire_single_instance(lock_path: Path, flock_module=None):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    if flock_module is None:
        import fcntl as flock_module
    try:
        flock_module.flock(handle.fileno(), flock_module.LOCK_EX | flock_module.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def release_single_instance(handle, flock_module=None):
    if handle is None:
        return
    try:
        if flock_module is None:
            import fcntl as flock_module
        flock_module.flock(handle.fileno(), flock_module.LOCK_UN)
    except Exception:
        pass
    handle.close()


def existing_instance_url(runtime_root: Path):
    try:
        port = int((runtime_root / PORT_FILE_NAME).read_text(encoding="utf-8").strip())
        if not 1 <= port <= 65535:
            return None
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as response:
            return f"http://127.0.0.1:{port}/" if response.status == 200 else None
    except (OSError, ValueError):
        return None


def wait_for_existing_instance(runtime_root: Path):
    deadline = time.monotonic() + EXISTING_INSTANCE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        url = existing_instance_url(runtime_root)
        if url:
            return url
        time.sleep(0.25)
    return None


def choose_port(preferred=PREFERRED_PORT):
    for port in range(preferred, preferred + PORT_SCAN_COUNT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available localhost port was found.")


def wait_for_health(url: str, timeout=STARTUP_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Local server did not become ready: {last_error}")


def configure_logging(log_root: Path):
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "blindspot-guardian.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return log_path


def load_application(resource_root: Path, runtime_root: Path, disable_mediapipe=False):
    if str(resource_root) not in sys.path:
        sys.path.insert(0, str(resource_root))
    import app as source_app

    source_app.INPUT_DIR = runtime_root / "input"
    source_app.OUTPUT_DIR = runtime_root / "output"
    source_app.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_app.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_app.MODEL_PATH = resource_root / "yolo11n.pt"
    source_app.POSE_MODEL_PATH = resolve_pose_model(resource_root, runtime_root, disable_mediapipe)
    source_app.engine.pose_analyzer.model_path = source_app.POSE_MODEL_PATH
    source_app.app.template_folder = str(resource_root / "templates")
    source_app.app.static_folder = str(resource_root / "static")
    source_app.app.config["MAX_CONTENT_LENGTH"] = source_app.MAX_UPLOAD_BYTES
    return source_app


def verify_video(source_app, video_path: Path, runtime_root: Path):
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
        temporary.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--disable-mediapipe", action="store_true")
    parser.add_argument("--verify-video", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    lock_handle = None
    active_port = None
    server = None
    source_app = None
    runtime_root, log_root = resolve_runtime_paths()
    log_path = log_root / "blindspot-guardian.log"
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        log_path = configure_logging(log_root)
        lock_handle = acquire_single_instance(runtime_root / LOCK_FILE_NAME)
        if lock_handle is None:
            existing_url = wait_for_existing_instance(runtime_root)
            if not existing_url:
                raise RuntimeError("BlindSpot Guardian is running, but it cannot be reached. Quit it from the Dock and try again.")
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

        active_port = choose_port()
        server = make_server("127.0.0.1", active_port, source_app.app, threaded=True)
        server_thread = threading.Thread(target=server.serve_forever, name="blindspot-local-server", daemon=True)
        server_thread.start()
        status_url = f"http://127.0.0.1:{active_port}/api/status"
        wait_for_health(status_url)
        (runtime_root / PORT_FILE_NAME).write_text(str(active_port), encoding="utf-8")
        logging.info("BlindSpot Guardian ready at %s", status_url)
        if args.self_test:
            server.shutdown()
            return 0

        stopping = threading.Event()
        signal.signal(signal.SIGTERM, lambda _signum, _frame: stopping.set())
        signal.signal(signal.SIGINT, lambda _signum, _frame: stopping.set())
        webbrowser.open(f"http://127.0.0.1:{active_port}/", new=2)
        while server_thread.is_alive() and not stopping.wait(0.5):
            pass
        server.shutdown()
        return 0
    except Exception as exc:
        logging.exception("Startup failed")
        show_error(f"{exc}\n\nDiagnostic log:\n{log_path}")
        return 1
    finally:
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        if source_app is not None:
            source_app.engine.stop(reset=True)
        if active_port is not None:
            port_file = runtime_root / PORT_FILE_NAME
            try:
                if port_file.read_text(encoding="utf-8").strip() == str(active_port):
                    port_file.unlink(missing_ok=True)
            except OSError:
                pass
        release_single_instance(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
