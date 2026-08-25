# -*- mode: python ; coding: utf-8 -*-
import platform
import plistlib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]
MACOS_ROOT = PROJECT_ROOT / "packaging" / "macos"
TARGET_ARCH = platform.machine()
if TARGET_ARCH not in {"arm64", "x86_64"}:
    raise SystemExit(f"Unsupported macOS build architecture: {TARGET_ARCH}")

datas = [
    (str(PROJECT_ROOT / "templates"), "templates"),
    (str(PROJECT_ROOT / "static"), "static"),
    (str(PROJECT_ROOT / "yolo11n.pt"), "."),
    (str(PROJECT_ROOT / "zones.json"), "."),
    (str(PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(MACOS_ROOT / "placeholders" / "input.keep"), "input"),
    (str(MACOS_ROOT / "placeholders" / "output.keep"), "output"),
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

with (MACOS_ROOT / "Info.plist").open("rb") as plist_file:
    info_plist = plistlib.load(plist_file)
info_plist["LSArchitecturePriority"] = [TARGET_ARCH]

a = Analysis(
    [str(MACOS_ROOT / "launcher.py")],
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
    target_arch=TARGET_ARCH,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BlindSpotGuardian",
)
app = BUNDLE(
    coll,
    name="BlindSpotGuardian.app",
    bundle_identifier="io.hhcc.blindspotguardian",
    version="1.0.0",
    info_plist=info_plist,
)
