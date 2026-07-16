param(
    [string]$BlenderExe = $env:BLENDER_EXE,
    [switch]$InstallBlender
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PortableRoot = Join-Path $RepoRoot ".tools"
$PortableVersion = "4.5.10"
$PortableFolder = "blender-$PortableVersion-windows-x64"
$PortableExe = Join-Path $PortableRoot "$PortableFolder\blender.exe"

if (-not $BlenderExe) {
    $RunningBlenderExe = Get-Process -Name blender -ErrorAction SilentlyContinue |
        Where-Object { $_.Path } |
        Select-Object -First 1 -ExpandProperty Path
    $candidates = @(
        $PortableExe,
        $RunningBlenderExe,
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
    $zip = Join-Path $PortableRoot "$PortableFolder.zip"
    $checksumFile = Join-Path $PortableRoot "blender-$PortableVersion.sha256"
    $releaseBase = "https://mirror.clarkson.edu/blender/release/Blender4.5"
    if (-not (Test-Path $zip)) {
        Invoke-WebRequest `
            -Uri "$releaseBase/$PortableFolder.zip" `
            -OutFile $zip
    }
    Invoke-WebRequest `
        -Uri "$releaseBase/blender-$PortableVersion.sha256" `
        -OutFile $checksumFile
    $expectedLine = Get-Content $checksumFile | Where-Object { $_ -match [regex]::Escape("$PortableFolder.zip") } | Select-Object -First 1
    if (-not $expectedLine) {
        throw "No SHA-256 entry found for $PortableFolder.zip"
    }
    $expectedHash = ($expectedLine -split "\s+")[0].ToLowerInvariant()
    $actualHash = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Blender archive checksum mismatch: expected $expectedHash, got $actualHash"
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
if ($LASTEXITCODE -ne 0) {
    throw "CurioMesh source Blender smoke failed with exit code $LASTEXITCODE"
}

$PythonExe = Get-Command python -ErrorAction Stop | Select-Object -ExpandProperty Source
$TempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$TempRoot = [IO.Path]::GetFullPath((Join-Path $TempBase "curiomesh-release-smoke-$([guid]::NewGuid().ToString('N'))"))
$TempPrefix = $TempBase.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) +
    [IO.Path]::DirectorySeparatorChar
if (-not $TempRoot.StartsWith($TempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use package-smoke directory outside the system temp root: $TempRoot"
}

$EnvironmentNames = @(
    "BLENDER_USER_CONFIG",
    "BLENDER_USER_DATAFILES",
    "BLENDER_USER_EXTENSIONS",
    "BLENDER_USER_SCRIPTS"
)
$PreviousEnvironment = @{}
foreach ($Name in $EnvironmentNames) {
    $PreviousEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}

try {
    $ReleaseRoot = Join-Path $TempRoot "release"
    New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
    & $PythonExe (Join-Path $RepoRoot "scripts\build_release.py") --output $ReleaseRoot
    if ($LASTEXITCODE -ne 0) {
        throw "CurioMesh release build failed with exit code $LASTEXITCODE"
    }

    $LegacyArchives = @(Get-ChildItem -LiteralPath $ReleaseRoot -Filter "*-addon.zip")
    $ExtensionArchives = @(Get-ChildItem -LiteralPath $ReleaseRoot -Filter "*-extension.zip")
    if ($LegacyArchives.Count -ne 1 -or $ExtensionArchives.Count -ne 1) {
        throw "Expected one legacy and one extension archive in $ReleaseRoot"
    }

    $PackageSmokeScript = Join-Path $RepoRoot "tests\blender_package_smoke.py"
    $Packages = @(
        @{ Layout = "legacy"; Archive = $LegacyArchives[0] },
        @{ Layout = "extension"; Archive = $ExtensionArchives[0] }
    )
    foreach ($Package in $Packages) {
        $LayoutRoot = Join-Path $TempRoot $Package.Layout
        $env:BLENDER_USER_CONFIG = Join-Path $LayoutRoot "config"
        $env:BLENDER_USER_DATAFILES = Join-Path $LayoutRoot "datafiles"
        $env:BLENDER_USER_EXTENSIONS = Join-Path $LayoutRoot "extensions"
        $env:BLENDER_USER_SCRIPTS = Join-Path $LayoutRoot "scripts"
        foreach ($Name in $EnvironmentNames) {
            New-Item -ItemType Directory -Force -Path ([Environment]::GetEnvironmentVariable($Name, "Process")) |
                Out-Null
        }

        & $BlenderExe --background --factory-startup --python $PackageSmokeScript -- `
            --layout $Package.Layout $Package.Archive.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "CurioMesh $($Package.Layout) installed-package smoke failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    foreach ($Name in $EnvironmentNames) {
        $PreviousValue = $PreviousEnvironment[$Name]
        if ($null -eq $PreviousValue) {
            Remove-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($Name, $PreviousValue, "Process")
        }
    }
    if (Test-Path -LiteralPath $TempRoot) {
        $ResolvedTempRoot = (Resolve-Path -LiteralPath $TempRoot).Path
        if (-not $ResolvedTempRoot.StartsWith($TempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove package-smoke directory outside the system temp root: $ResolvedTempRoot"
        }
        Remove-Item -LiteralPath $ResolvedTempRoot -Recurse -Force
    }
}

Write-Host "CurioMesh Blender release ladder passed (source, legacy archive, extension archive)."
