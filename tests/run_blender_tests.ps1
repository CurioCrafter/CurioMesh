param(
    [string]$BlenderExe = $env:BLENDER_EXE,
    [switch]$InstallBlender
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PortableRoot = Join-Path $RepoRoot ".tools"
$PortableExe = Join-Path $PortableRoot "blender-4.5.1-windows-x64\blender.exe"

if (-not $BlenderExe) {
    $candidates = @(
        $PortableExe,
        "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $BlenderExe = $candidate
            break
        }
    }
}

if (-not $BlenderExe -and $InstallBlender) {
    New-Item -ItemType Directory -Force -Path $PortableRoot | Out-Null
    $zip = Join-Path $PortableRoot "blender-4.5.1-windows-x64.zip"
    if (-not (Test-Path $zip)) {
        Invoke-WebRequest `
            -Uri "https://download.blender.org/release/Blender4.5/blender-4.5.1-windows-x64.zip" `
            -OutFile $zip
    }
    if (-not (Test-Path $PortableExe)) {
        Expand-Archive -Path $zip -DestinationPath $PortableRoot -Force
    }
    $BlenderExe = $PortableExe
}

if (-not $BlenderExe -or -not (Test-Path $BlenderExe)) {
    throw "Blender executable not found. Set BLENDER_EXE or run tests/run_blender_tests.ps1 -InstallBlender."
}

& $BlenderExe --background --factory-startup --python (Join-Path $RepoRoot "tests\blender_smoke.py")
exit $LASTEXITCODE
