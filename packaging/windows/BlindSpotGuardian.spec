# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]
launcher = PROJECT_ROOT / "packaging" / "windows" / "launcher.py"

datas = [
    (str(PROJECT_ROOT / "templates"), "templates"),
    (str(PROJECT_ROOT / "static"), "static"),
    (str(PROJECT_ROOT / "yolo11n.pt"), "."),
    (str(PROJECT_ROOT / "zones.json"), "."),
    (str(PROJECT_ROOT / "packaging" / "windows" / "placeholders" / "input.keep"), "input"),
    (str(PROJECT_ROOT / "packaging" / "windows" / "placeholders" / "output.keep"), "output"),
]
pose_model = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
if pose_model.is_file():
    datas.append((str(pose_model), "models"))

hiddenimports = ["app", "behavior", "lap", "werkzeug.serving"]
binaries = collect_dynamic_libs("torch")
for package in ("ultralytics", "mediapipe"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
hiddenimports += collect_submodules("flask")

a = Analysis(
    [str(launcher)],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BlindSpotGuardian",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BlindSpotGuardian-Windows-x64",
)
