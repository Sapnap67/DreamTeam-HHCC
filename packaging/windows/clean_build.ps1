[CmdletBinding(SupportsShouldProcess)]
param()

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Targets = @(
    (Join-Path $ProjectRoot "build"),
    (Join-Path $ProjectRoot "dist\BlindSpotGuardian-Windows-x64"),
    (Join-Path $ProjectRoot "dist\BlindSpotGuardian-Windows-x64.zip")
)

foreach ($Target in $Targets) {
    $FullTarget = [System.IO.Path]::GetFullPath($Target)
    if (-not $FullTarget.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the project: $FullTarget"
    }
    if ((Test-Path -LiteralPath $FullTarget) -and $PSCmdlet.ShouldProcess($FullTarget, "Remove generated packaging output")) {
        Remove-Item -LiteralPath $FullTarget -Recurse -Force
    }
}
