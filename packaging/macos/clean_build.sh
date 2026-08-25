#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
rm -rf "$PROJECT_ROOT/build/macos"
rm -rf "$PROJECT_ROOT/dist/BlindSpotGuardian-macOS"
rm -f "$PROJECT_ROOT/dist/BlindSpotGuardian-macOS-arm64.zip"
rm -f "$PROJECT_ROOT/dist/BlindSpotGuardian-macOS-x86_64.zip"
echo "Removed only generated macOS packaging output."
