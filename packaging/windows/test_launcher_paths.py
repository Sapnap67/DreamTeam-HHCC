import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

import launcher
from launcher import existing_instance_url, load_application, resolve_resource_root, resolve_runtime_root


class LauncherPathTests(unittest.TestCase):
    def test_frozen_resources_use_meipass(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_resource_root(True, directory), Path(directory).resolve())

    def test_source_resources_use_project_root(self):
        fake_launcher = Path("C:/example/project/packaging/windows/launcher.py")
        self.assertEqual(resolve_resource_root(False, launcher_file=str(fake_launcher)), fake_launcher.resolve().parents[2])

    def test_runtime_data_uses_local_appdata(self):
        self.assertEqual(
            resolve_runtime_root("C:/Users/Test/AppData/Local"),
            Path("C:/Users/Test/AppData/Local/BlindSpotGuardian").resolve(),
        )

    def test_source_application_root_is_added_to_import_path(self):
        with tempfile.TemporaryDirectory() as resource_directory, tempfile.TemporaryDirectory() as runtime_directory:
            resource_root = Path(resource_directory).resolve()
            fake_app = mock.MagicMock()
            fake_app.MAX_UPLOAD_BYTES = 123
            with mock.patch.dict(sys.modules, {"app": fake_app}):
                with mock.patch.object(sys, "path", [entry for entry in sys.path if entry != str(resource_root)]):
                    load_application(resource_root, Path(runtime_directory), disable_mediapipe=True)
                    self.assertEqual(sys.path[0], str(resource_root))

    def test_existing_instance_url_uses_recorded_live_port(self):
        with tempfile.TemporaryDirectory() as runtime_directory:
            runtime_root = Path(runtime_directory)
            (runtime_root / launcher.PORT_FILE_NAME).write_text("5007", encoding="utf-8")
            response = mock.MagicMock()
            response.status = 200
            response.__enter__.return_value = response
            with mock.patch.object(launcher.urllib.request, "urlopen", return_value=response) as urlopen:
                self.assertEqual(existing_instance_url(runtime_root), "http://127.0.0.1:5007/")
                urlopen.assert_called_once_with("http://127.0.0.1:5007/api/status", timeout=2)

    def test_existing_instance_url_rejects_stale_or_invalid_port_files(self):
        with tempfile.TemporaryDirectory() as runtime_directory:
            runtime_root = Path(runtime_directory)
            self.assertIsNone(existing_instance_url(runtime_root))
            (runtime_root / launcher.PORT_FILE_NAME).write_text("not-a-port", encoding="utf-8")
            self.assertIsNone(existing_instance_url(runtime_root))


if __name__ == "__main__":
    unittest.main()
