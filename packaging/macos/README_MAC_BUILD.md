# BlindSpot Guardian macOS Build

The macOS application is an additional distribution. It does not replace the source launchers or the separate Windows executable distribution.

## Build host

Build on a real Mac using native Python for that Mac. The script detects `arm64` or `x86_64` and refuses to build on Windows or an unsupported architecture. It does not produce or claim a universal2 app.

```bash
chmod +x packaging/macos/build_macos.sh packaging/macos/clean_build.sh
./packaging/macos/build_macos.sh --test-video "/absolute/path/to/test-video.mp4"
```

Set `PYTHON_COMMAND` or pass `--python /path/to/python3` if needed. The build creates an isolated environment only under `build/macos/`, runs the existing tests and macOS launcher tests, builds the app, checks its architecture, signs it, runs packaged startup checks, optionally runs real packaged YOLO inference, and creates one architecture-specific ZIP:

- `dist/BlindSpotGuardian-macOS/BlindSpotGuardian.app`
- `dist/BlindSpotGuardian-macOS-arm64.zip`, or
- `dist/BlindSpotGuardian-macOS-x86_64.zip`

The required YOLO model and the optional MediaPipe model are bundled when present. The app never installs Python or packages on the recipient's Mac. Runtime uploads and state use `~/Library/Application Support/BlindSpotGuardian`; logs use `~/Library/Logs/BlindSpotGuardian`.

## Signing and notarization

Without configuration, the script applies an ad-hoc signature. That is not Apple notarization. Recipients may need to right-click the app and choose **Open** on first launch.

For Developer ID signing, set `MACOS_CODESIGN_IDENTITY` to a certificate name already installed in the build Mac's keychain. For optional notarization, also store credentials in a Keychain profile and set `MACOS_NOTARY_PROFILE` to that profile name. Credentials are never written to the repository.

## Required clean-Mac validation

Before release, test the resulting architecture-specific ZIP in a clean macOS account or VM without Python, Git, Codex, a project virtual environment, or separately downloaded models. Confirm double-click startup, browser opening, duplicate-instance handling, upload cleanup, optional MediaPipe fallback, one real video upload, and real YOLO inference. Record the Mac model, macOS version, architecture, versions, ZIP size, checksum, signing status, and results in the release notes.

This configuration was prepared and syntax-tested on Windows. No valid macOS `.app` or macOS ZIP can be produced or claimed until the real-Mac build and tests above are completed.
