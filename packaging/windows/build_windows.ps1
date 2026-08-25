[CmdletBinding()]
param(
    [string]$PythonCommand = "py",
    [string]$TestVideo = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildRoot = Join-Path $ProjectRoot "build"
$BuildDrive = "B:"
$BuildDriveMapped = $false
$VenvRoot = "$BuildDrive\v"
$PythonExe = "$VenvRoot\Scripts\python.exe"
$DistRoot = Join-Path $ProjectRoot "dist"
$PackageFolder = Join-Path $DistRoot "BlindSpotGuardian-Windows-x64"
$ZipPath = Join-Path $DistRoot "BlindSpotGuardian-Windows-x64.zip"

Set-Location $ProjectRoot
function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}
trap {
    if ($BuildDriveMapped) {
        & subst.exe $BuildDrive /D | Out-Null
    }
    throw $_
}
foreach ($required in @("app.py", "behavior.py", "templates", "static", "yolo11n.pt", "requirements.txt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $required))) {
        throw "Required asset is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $DistRoot | Out-Null
if (Test-Path "$BuildDrive\") {
    throw "$BuildDrive is already in use. Remove that mapping or select another build drive."
}
& subst.exe $BuildDrive $BuildRoot
Assert-NativeSuccess "Temporary short-path mapping"
$BuildDriveMapped = $true
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating isolated Windows build environment..."
    if ($PythonCommand -eq "py") {
        & py -3.12 -m venv $VenvRoot
    } else {
        & $PythonCommand -m venv $VenvRoot
    }
    Assert-NativeSuccess "Virtual environment creation"
}

Write-Host "Installing pinned application and packaging dependencies..."
& $PythonExe -m pip install --upgrade "pip==26.0.1"
Assert-NativeSuccess "pip bootstrap"
& $PythonExe -m pip install -r requirements.txt "PyInstaller==6.16.0" "lap==0.5.13"
Assert-NativeSuccess "Dependency installation"

Write-Host "Checking Torchvision NMS compatibility..."
& $PythonExe -c "import torch, torchvision; torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([0.9]), 0.5)"
Assert-NativeSuccess "Torchvision NMS compatibility check"

Write-Host "Running automated tests..."
$env:MPLCONFIGDIR = Join-Path $BuildRoot "matplotlib-cache"
$env:YOLO_CONFIG_DIR = Join-Path $BuildRoot "ultralytics-cache"
& $PythonExe -m unittest discover -s tests -v
Assert-NativeSuccess "Application tests"
& $PythonExe (Join-Path $PSScriptRoot "test_launcher_paths.py") -v
Assert-NativeSuccess "Launcher path tests"

if (Test-Path -LiteralPath $PackageFolder) {
    Remove-Item -LiteralPath $PackageFolder -Recurse -Force
}
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Write-Host "Building the one-folder Windows executable..."
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $DistRoot `
    --workpath (Join-Path $BuildRoot "pyinstaller") `
    (Join-Path $PSScriptRoot "BlindSpotGuardian.spec")
Assert-NativeSuccess "PyInstaller build"

Copy-Item -LiteralPath "README.md", "THIRD_PARTY_NOTICES.md", "BUILDING_WINDOWS.md" -Destination $PackageFolder

Write-Host "Testing packaged localhost startup..."
& (Join-Path $PackageFolder "BlindSpotGuardian.exe") --self-test
Assert-NativeSuccess "Packaged startup test"
& (Join-Path $PackageFolder "BlindSpotGuardian.exe") --self-test --disable-mediapipe
Assert-NativeSuccess "Packaged MediaPipe fallback test"
& (Join-Path $PackageFolder "BlindSpotGuardian.exe") --self-test
Assert-NativeSuccess "Packaged repeated startup test"

$SpaceTestRoot = Join-Path $env:TEMP "BlindSpot Guardian Portable Test"
if (Test-Path -LiteralPath $SpaceTestRoot) {
    Remove-Item -LiteralPath $SpaceTestRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $SpaceTestRoot | Out-Null
Copy-Item -LiteralPath $PackageFolder -Destination $SpaceTestRoot -Recurse
& (Join-Path $SpaceTestRoot "BlindSpotGuardian-Windows-x64\BlindSpotGuardian.exe") --self-test
Assert-NativeSuccess "Packaged spaces-in-path test"
Remove-Item -LiteralPath $SpaceTestRoot -Recurse -Force

if ($TestVideo) {
    $ResolvedTestVideo = (Resolve-Path -LiteralPath $TestVideo).Path
    Write-Host "Testing packaged YOLO inference with $ResolvedTestVideo..."
    & (Join-Path $PackageFolder "BlindSpotGuardian.exe") --verify-video $ResolvedTestVideo
    Assert-NativeSuccess "Packaged YOLO inference test"
}

Write-Host "Creating portable ZIP..."
Compress-Archive -LiteralPath $PackageFolder -DestinationPath $ZipPath -CompressionLevel Optimal
$Hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
$SizeMb = [math]::Round((Get-Item -LiteralPath $ZipPath).Length / 1MB, 2)
Write-Host "ZIP: $ZipPath"
Write-Host "Size: $SizeMb MB"
Write-Host "SHA-256: $($Hash.Hash)"
& subst.exe $BuildDrive /D | Out-Null
$BuildDriveMapped = $false
