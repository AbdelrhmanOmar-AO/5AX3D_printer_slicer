<#
.SYNOPSIS
    Runs the P0.8 overhang baseline matrix and writes the summary table.

.DESCRIPTION
    Every part at every max_slope, through the stock pipeline, measured with
    tools/overhang_report.py. These are the "before" numbers the overhang-aware
    field is judged against in P2.5.

    A part that fails does not stop the matrix: the failure is recorded and the
    run continues, so one bad combination does not cost a whole night.

    Estimated total time by size, from tests/golden/baseline.md:
        xs   1.6 h     s    7.2 h     m   24.4 h     l   57.8 h

.PARAMETER Sizes
    Which generated sizes to run: xs, s, m, l. Default just s. Give several to
    sweep them, which turns the runtime column of the summary into a scaling
    curve rather than a single point. Estimated totals per size:
        xs   1.6 h     s    7.2 h     m   24.4 h     l   57.8 h

.PARAMETER Slopes
    max_slope values in degrees. Default 7, 15, 30. 30 is the reference
    machine's bed limit.

.PARAMETER Parts
    Part names without the size suffix. Default is the six ramps, the T-shape
    and the twin domes: the 8 parts that make up the 24-run matrix.

.PARAMETER EnvName
    Conda environment. Default `atomizer`.

.EXAMPLE
    .\scripts\run_baseline_matrix.ps1
    .\scripts\run_baseline_matrix.ps1 -Sizes xs
    .\scripts\run_baseline_matrix.ps1 -Sizes xs,s -Slopes 7,30
#>

[CmdletBinding()]
param(
    [ValidateSet("xs", "s", "m", "l")][string[]]$Sizes = @("s"),
    [double[]]$Slopes = @(7, 15, 30),
    [string[]]$Parts = @("ramp45", "ramp50", "ramp60", "ramp70", "ramp80", "ramp90",
                         "tshape", "twin_domes"),
    [string]$EnvName = "atomizer"
)

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir "baseline-matrix-$(Get-Date -Format 'yyyy-MM-dd_HHmmss').txt"

$total = $Parts.Count * $Slopes.Count * $Sizes.Count
$index = 0
$failures = @()
$startedAll = Get-Date

Write-Host "Baseline matrix: $($Parts.Count) parts x $($Slopes.Count) slopes x $($Sizes.Count) size(s) = $total runs" -ForegroundColor Cyan
Write-Host "Sizes: $($Sizes -join ', ')  Slopes: $($Slopes -join ', ') deg"
Write-Host "Log: $LogFile"
Write-Host "Leave this running. A failed run is recorded and the matrix continues." -ForegroundColor Yellow

foreach ($size in $Sizes) {
  foreach ($slope in $Slopes) {
    foreach ($part in $Parts) {
        $index++
        $solid = "${part}_${size}"
        $paramPath = "data/param/$solid.json"

        if (-not (Test-Path $paramPath)) {
            Write-Host "[$index/$total] SKIP $solid - no $paramPath" -ForegroundColor Yellow
            $failures += "$solid @ $slope (missing parameter file)"
            continue
        }

        $elapsedSoFar = (Get-Date) - $startedAll
        Write-Host ""
        Write-Host "[$index/$total] $solid at max_slope $slope deg (elapsed $($elapsedSoFar.ToString('hh\:mm\:ss')))" -ForegroundColor Cyan

        $started = Get-Date
        & conda run --name $EnvName --no-capture-output python tools/overhang_report.py `
            $paramPath --max-slope $slope 2>&1 | Tee-Object -FilePath $LogFile -Append
        $exit = $LASTEXITCODE
        $took = (Get-Date) - $started

        if ($exit -ne 0) {
            Write-Host "  FAILED (exit $exit) after $($took.ToString('hh\:mm\:ss'))" -ForegroundColor Red
            $failures += "$solid @ $slope (exit $exit)"
        }
        else {
            Write-Host "  done in $($took.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        }
    }
  }
}

Write-Host ""
Write-Host "=== Writing the summary ===" -ForegroundColor Cyan
& conda run --name $EnvName --no-capture-output python tools/overhang_report.py --summarize |
    Tee-Object -FilePath $LogFile -Append

$totalElapsed = (Get-Date) - $startedAll
Write-Host ""
Write-Host "Matrix finished in $($totalElapsed.ToString('hh\:mm\:ss'))." -ForegroundColor Green
if ($failures.Count -gt 0) {
    Write-Host "$($failures.Count) of $total runs failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}
Write-Host "Summary: reports\baseline_overhang.md"
Write-Host "Commit reports\ and paste the summary table." -ForegroundColor Cyan
