import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launcher


class FakeFlock:
    LOCK_EX = 1
    LOCK_NB = 2
    LOCK_UN = 4

    def __init__(self, blocked=False):
        self.blocked = blocked

    def flock(self, _descriptor, operation):
        if self.blocked and operation == self.LOCK_EX | self.LOCK_NB:
            raise BlockingIOError("locked")


class MacLauncherTests(unittest.TestCase):
    def test_frozen_resources_use_meipass(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(launcher.resolve_resource_root(True, directory), Path(directory).resolve())

    def test_normal_resources_use_project_root(self):
        fake = Path("/project/packaging/macos/launcher.py")
        self.assertEqual(launcher.resolve_resource_root(False, launcher_file=str(fake)), fake.resolve().parents[2])

    def test_runtime_paths_use_macos_user_library(self):
        support, logs = launcher.resolve_runtime_paths("/Users/Test")
        expected_home = Path("/Users/Test").resolve()
        self.assertEqual(support, expected_home / "Library" / "Application Support" / "BlindSpotGuardian")
        self.assertEqual(logs, expected_home / "Library" / "Logs" / "BlindSpotGuardian")

    def test_port_selection_skips_busy_preferred_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            preferred = occupied.getsockname()[1]
            selected = launcher.choose_port(preferred)
            self.assertNotEqual(selected, preferred)
            self.assertLess(selected, preferred + launcher.PORT_SCAN_COUNT)

    def test_duplicate_instance_is_rejected_by_file_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            handle = launcher.acquire_single_instance(Path(directory) / "server.lock", FakeFlock())
            self.assertIsNotNone(handle)
            launcher.release_single_instance(handle, FakeFlock())
            self.assertIsNone(launcher.acquire_single_instance(Path(directory) / "server.lock", FakeFlock(blocked=True)))

    def test_missing_optional_pose_model_uses_unavailable_path(self):
        with tempfile.TemporaryDirectory() as resources, tempfile.TemporaryDirectory() as runtime:
            selected = launcher.resolve_pose_model(Path(resources), Path(runtime))
            self.assertEqual(selected, Path(runtime) / "unavailable-pose-model.task")

    def test_bundled_pose_model_is_selected(self):
        with tempfile.TemporaryDirectory() as resources, tempfile.TemporaryDirectory() as runtime:
            model = Path(resources) / "models" / "pose_landmarker_lite.task"
            model.parent.mkdir()
            model.touch()
            self.assertEqual(launcher.resolve_pose_model(Path(resources), Path(runtime)), model)

    def test_browser_readiness_uses_real_health_response(self):
        response = mock.MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        with mock.patch.object(launcher.urllib.request, "urlopen", return_value=response) as urlopen:
            launcher.wait_for_health("http://127.0.0.1:5000/api/status", timeout=1)
            urlopen.assert_called_once_with("http://127.0.0.1:5000/api/status", timeout=2)


if __name__ == "__main__":
    unittest.main()
