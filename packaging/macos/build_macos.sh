#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="$PROJECT_ROOT/build/macos"
DIST_ROOT="$PROJECT_ROOT/dist/BlindSpotGuardian-macOS"
APP_PATH="$DIST_ROOT/BlindSpotGuardian.app"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
TEST_VIDEO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_COMMAND="$2"; shift 2 ;;
    --test-video) TEST_VIDEO="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This package must be built on macOS. No .app was created." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" && "$ARCH" != "x86_64" ]]; then
  echo "Unsupported Mac architecture: $ARCH" >&2
  exit 1
fi

ZIP_PATH="$PROJECT_ROOT/dist/BlindSpotGuardian-macOS-$ARCH.zip"
VENV="$BUILD_ROOT/venv"
PYTHON="$VENV/bin/python"
cd "$PROJECT_ROOT"
mkdir -p "$BUILD_ROOT" "$DIST_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  "$PYTHON_COMMAND" -m venv "$VENV"
fi
"$PYTHON" -m pip install --upgrade "pip==26.0.1"
"$PYTHON" -m pip install -r requirements.txt "PyInstaller==6.16.0" "lap==0.5.13"

export MPLCONFIGDIR="$BUILD_ROOT/matplotlib-cache"
export YOLO_CONFIG_DIR="$BUILD_ROOT/ultralytics-cache"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" "$SCRIPT_DIR/test_launcher.py" -v
plutil -lint "$SCRIPT_DIR/Info.plist"

rm -rf "$APP_PATH"
rm -f "$ZIP_PATH"
"$PYTHON" -m PyInstaller --noconfirm --clean \
  --distpath "$DIST_ROOT" \
  --workpath "$BUILD_ROOT/pyinstaller" \
  "$SCRIPT_DIR/BlindSpotGuardian-macOS.spec"

EXECUTABLE="$APP_PATH/Contents/MacOS/BlindSpotGuardian"
[[ -x "$EXECUTABLE" ]] || { echo "Packaged executable is missing." >&2; exit 1; }
BUILT_ARCHS="$(lipo -archs "$EXECUTABLE")"
[[ "$BUILT_ARCHS" == "$ARCH" ]] || { echo "Unexpected executable architecture: $BUILT_ARCHS" >&2; exit 1; }

SIGN_IDENTITY="${MACOS_CODESIGN_IDENTITY:--}"
if [[ "$SIGN_IDENTITY" == "-" ]]; then
  codesign --force --deep --sign - "$APP_PATH"
  SIGNING_DESCRIPTION="ad-hoc signed; not notarized"
else
  codesign --force --deep --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_PATH"
  SIGNING_DESCRIPTION="Developer ID signed with $SIGN_IDENTITY"
fi
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
"$EXECUTABLE" --self-test
"$EXECUTABLE" --self-test --disable-mediapipe

if [[ -n "$TEST_VIDEO" ]]; then
  "$EXECUTABLE" --verify-video "$TEST_VIDEO"
fi

ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"
if [[ -n "${MACOS_NOTARY_PROFILE:-}" ]]; then
  [[ "$SIGN_IDENTITY" != "-" ]] || { echo "Notarization requires a Developer ID signature." >&2; exit 1; }
  xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP_PATH"
  rm -f "$ZIP_PATH"
  ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"
  SIGNING_DESCRIPTION="$SIGNING_DESCRIPTION; notarized and stapled"
fi

echo "macOS: $(sw_vers -productVersion)"
echo "Architecture: $ARCH"
echo "Python: $($PYTHON --version 2>&1)"
echo "PyInstaller: $($PYTHON -m PyInstaller --version)"
echo "Signing: $SIGNING_DESCRIPTION"
echo "App: $APP_PATH"
echo "ZIP: $ZIP_PATH"
du -sh "$APP_PATH" "$ZIP_PATH"
shasum -a 256 "$ZIP_PATH"
