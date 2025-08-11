# Requires: Git, Visual Studio 2019+ with MSBuild (and vswhere)
#
# This script clones the AssemblyMC sources and builds them locally.
# No binaries are committed to this repository; everything happens in a
# temporary directory under the user's profile.

param(
    [string]$RepoUrl = "https://github.com/croningp/Paper-AssemblyTreeOfLife"
)

$ErrorActionPreference = "Stop"

# Create temporary working directory
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("AssemblyMC_" + [System.Guid]::NewGuid().ToString())
Write-Host "Cloning $RepoUrl to $tempDir"

& git clone $RepoUrl $tempDir | Out-Host

$slnDir = Join-Path $tempDir "AssemblyMC_cpp_sc"
$slnFiles = Get-ChildItem -Path $slnDir -Filter *.sln
if (-not $slnFiles) {
    Write-Error "No solution files found in $slnDir"
    exit 1
}

# Locate MSBuild using vswhere
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\\Installer\\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2019 or later."
}
$msbuildPath = & $vswhere -latest -requires Microsoft.Component.MSBuild -find MSBuild\\**\\Bin\\MSBuild.exe
if (-not $msbuildPath) {
    throw "MSBuild.exe not found. Ensure Visual Studio 2019+ is installed."
}
$msbuild = $msbuildPath[0]

foreach ($sln in $slnFiles) {
    Write-Host "Building $($sln.Name)"
    & $msbuild $sln.FullName /p:Configuration=Release /m | Out-Host
}

# Try to locate the resulting executable
$exe = Get-ChildItem -Path $slnDir -Recurse -Filter AssemblyMC.exe | Select-Object -First 1
if (-not $exe) {
    $exe = Get-ChildItem -Path $tempDir -Recurse -Filter *.exe | Where-Object { $_.Name -match "AssemblyMC" } | Select-Object -First 1
}

if ($exe) {
    Write-Host "AssemblyMC built: $($exe.FullName)"
    Write-Host "Set the ASSEMBLYMC_BIN environment variable to use it:"
    Write-Host "  setx ASSEMBLYMC_BIN \"$($exe.FullName)\""
} else {
    Write-Host "Build completed, but AssemblyMC.exe not found."
}
