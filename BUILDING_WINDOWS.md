# Building the Separate Windows Distribution

The Windows executable is an additional distribution. It does not replace `app.py`, `start.bat`, a macOS launcher, or any existing portable ZIP.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.12 available to the builder
- Internet access for the isolated build environment's first dependency installation
- Several gigabytes of free disk space

## Build

From PowerShell in the project directory:

```powershell
.\packaging\windows\build_windows.ps1 -TestVideo "C:\path\to\traffic-video.mp4"
```

The script creates its isolated environment and temporary output under `build/`. The finished folder and ZIP are written under `dist/`:

```text
dist/
  BlindSpotGuardian-Windows-x64/
    BlindSpotGuardian.exe
    _internal/
    README.md
    THIRD_PARTY_NOTICES.md
    BUILDING_WINDOWS.md
  BlindSpotGuardian-Windows-x64.zip
```

The build verifies normal startup, startup without the optional pose model, repeated startup, operation from a path containing spaces, and—when `-TestVideo` is supplied—actual packaged YOLO inference.

## Clean generated output

```powershell
.\packaging\windows\clean_build.ps1
```

Generated executables, build environments, and ZIPs are ignored by Git. Clean-machine or Windows Sandbox verification is still required before public distribution.

If Windows reports that an application-control policy blocked the generated executable, test the unchanged ZIP in an approved Windows Sandbox or clean VM. Do not disable an organization-managed security policy merely to run the prototype. A production distribution should be signed with a trusted code-signing certificate.
