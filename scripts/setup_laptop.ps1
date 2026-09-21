<#
.SYNOPSIS
    Sets up the development environment for the 5AX3D slicer on a Windows laptop.

.DESCRIPTION
    Creates (or reuses) the `atomizer` conda environment on Python 3.10, installs
    this repository in editable mode together with its dev dependencies, and then
    reports the three things every later task depends on:

        * the Python version in the environment,
        * the Blender version on PATH (the first pipeline stage shells out to it),
        * whether Taichi can reach a CUDA GPU, or falls back to CPU.

    Conda is used rather than `python -m venv` because Taichi requires Python
    >= 3.7, < 3.11 (see pyproject.toml) and conda can create a 3.10 interpreter
    regardless of which Python, if any, is installed system-wide.

.PARAMETER EnvName
    Name of the conda environment. Defaults to `atomizer`, matching README.md.

.EXAMPLE
    .\scripts\setup_laptop.ps1

.NOTES
    Prerequisite: Git, Blender and Miniconda installed and on PATH. The repo's
    install_git_blender_miniconda.ps1 does that; run it from an Administrator
    PowerShell, then close and reopen PowerShell before running this script.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "atomizer",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

function Write-Section($Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

# Run from the repository root regardless of where the script was invoked from.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repository root: $RepoRoot"

Write-Section "Checking conda"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-Host "conda was not found on PATH." -ForegroundColor Red
    Write-Host "Run install_git_blender_miniconda.ps1 from an Administrator PowerShell,"
    Write-Host "then CLOSE and REOPEN PowerShell and run this script again."
    exit 1
}
Write-Host "Found conda at: $($conda.Source)"

Write-Section "Creating the '$EnvName' environment (Python $PythonVersion)"
$existing = & conda env list | Select-String -Pattern "^\s*$([regex]::Escape($EnvName))\s"
if ($existing) {
    Write-Host "Environment '$EnvName' already exists. Reusing it." -ForegroundColor Yellow
}
else {
    & conda create --name $EnvName "python=$PythonVersion" --yes
    if ($LASTEXITCODE -ne 0) { throw "conda create failed with exit code $LASTEXITCODE" }
}

Write-Section "Installing the 'atom' package and dev dependencies"
# `conda run` avoids needing `conda activate` to work inside a script.
& conda run --name $EnvName --no-capture-output python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }

Write-Section "Environment report (paste this into the PR / chat)"

Write-Host "-- Python --"
& conda run --name $EnvName --no-capture-output python --version

Write-Host "-- Blender --"
$blender = Get-Command blender -ErrorAction SilentlyContinue
if ($blender) {
    & blender --version | Select-Object -First 1
}
else {
    Write-Host "blender NOT found on PATH." -ForegroundColor Red
    Write-Host "The first pipeline stage (tools/process_for_atomizer.py) shells out to"
    Write-Host "'blender -b -P ...', so the pipeline cannot run until this is fixed."
    Write-Host "Tested versions: Blender 4.4 and 4.5."
}

Write-Host "-- Taichi / GPU --"
$probe = @'
import taichi as ti

try:
    ti.init(arch=ti.cuda)
    arch = ti.lang.impl.current_cfg().arch
    print(f"taichi {ti.__version__}; requested cuda; got {arch}")
    print("CUDA AVAILABLE" if str(arch).endswith("cuda") else "CUDA NOT AVAILABLE (fell back)")
except Exception as exc:  # noqa: BLE001 - this is a diagnostic probe
    print(f"taichi cuda init raised: {type(exc).__name__}: {exc}")
    print("CUDA NOT AVAILABLE")
'@
& conda run --name $EnvName --no-capture-output python -c $probe

Write-Section "Running the unit test suite"
& conda run --name $EnvName --no-capture-output python -m pytest -m unit -q
$testExit = $LASTEXITCODE

Write-Section "Done"
Write-Host "To use the environment in a new PowerShell window:"
Write-Host "    conda activate $EnvName" -ForegroundColor Green
if ($testExit -ne 0) {
    Write-Host "Unit tests did NOT pass (exit code $testExit). Paste the output above." -ForegroundColor Red
    exit $testExit
}
Write-Host "Unit tests passed." -ForegroundColor Green
