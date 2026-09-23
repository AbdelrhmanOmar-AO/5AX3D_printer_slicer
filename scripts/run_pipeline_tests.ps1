<#
.SYNOPSIS
    Runs the `pipeline` tier of the test suite on the operator's laptop.

.DESCRIPTION
    These tests drive real Atomizer stages and need a CUDA GPU plus Blender on
    PATH, so they are skipped everywhere else. The full console output is saved
    under logs\ so it can be pasted into the pull request.

.PARAMETER EnvName
    Conda environment to run in. Defaults to `atomizer`.

.PARAMETER Benchmark
    Also run the `benchmark` tier (full part runs; can take a long time).

.EXAMPLE
    .\scripts\run_pipeline_tests.ps1
    .\scripts\run_pipeline_tests.ps1 -Benchmark
#>

[CmdletBinding()]
param(
    [string]$EnvName = "atomizer",
    [switch]$Benchmark
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$tier = if ($Benchmark) { "benchmark" } else { "pipeline" }
$LogFile = Join-Path $LogDir "$tier-$stamp.txt"

$pytestArgs = @("-m", "pytest", "-v")
if ($Benchmark) {
    $pytestArgs += @("--run-pipeline", "--run-benchmark", "-m", "pipeline or benchmark")
}
else {
    $pytestArgs += @("--run-pipeline", "-m", "pipeline")
}

Write-Host "Running: python $($pytestArgs -join ' ')" -ForegroundColor Cyan
Write-Host "Log file: $LogFile"

& conda run --name $EnvName --no-capture-output python @pytestArgs 2>&1 |
    Tee-Object -FilePath $LogFile
$testExit = $LASTEXITCODE

Write-Host ""
Write-Host "Saved log to $LogFile" -ForegroundColor Green
if ($testExit -ne 0) {
    Write-Host "Tests FAILED (exit code $testExit)." -ForegroundColor Red
}
exit $testExit
