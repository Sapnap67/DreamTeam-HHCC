param(
    [switch]$SkipModelDownload
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$venvDir = Join-Path $projectDir '.venv'
$pythonExe = Join-Path $venvDir 'Scripts\python.exe'
$modelDir = Join-Path $projectDir 'models'
$modelPath = Join-Path $modelDir 'yolo11n.pt'
$poseModelPath = Join-Path $modelDir 'pose_landmarker_lite.task'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        $version = $null
        foreach ($candidate in @('3.12', '3.11')) {
            & $pythonLauncher.Source "-$candidate" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $version = $candidate
                break
            }
        }
        if ($null -eq $version) {
            Write-Host 'Install Python 3.11 or 3.12, then run this launcher again.'
            exit 1
        }
        & $pythonLauncher.Source "-$version" -m venv $venvDir
    } else {
        & python -c "import sys; assert (3, 11) <= sys.version_info < (3, 13), 'Install Python 3.11 or 3.12, then run this launcher again.'"
        if ($LASTEXITCODE -ne 0) { exit 1 }
        & python -m venv $venvDir
    }
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r (Join-Path $projectDir 'requirements.txt')

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
if (-not $SkipModelDownload -and -not (Test-Path -LiteralPath $modelPath)) {
    Invoke-WebRequest `
        -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' `
        -OutFile $modelPath
}

if (-not $SkipModelDownload -and -not (Test-Path -LiteralPath $poseModelPath)) {
    Invoke-WebRequest `
        -Uri 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' `
        -OutFile $poseModelPath
}

if (-not (Test-Path -LiteralPath $modelPath)) {
    Write-Host "Model missing. Put yolo11n.pt in $modelDir or run .\start.ps1 without -SkipModelDownload."
    exit 1
}

if (-not (Test-Path -LiteralPath $poseModelPath)) {
    Write-Host "MediaPipe model missing. Put pose_landmarker_lite.task in $modelDir or run .\start.ps1 without -SkipModelDownload."
    exit 1
}

& $pythonExe (Join-Path $projectDir 'app.py')

