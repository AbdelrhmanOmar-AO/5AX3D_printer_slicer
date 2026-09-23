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

.PARAMETER Resume
    Skip combinations that already have a result from the current metrics.
    Every run writes its report the moment it finishes, so an interrupted
    matrix keeps everything done so far; -Resume continues from there rather
    than repeating it.

    Check what is outstanding first:
        python tools/overhang_report.py --status xs s

.EXAMPLE
    .\scripts\run_baseline_matrix.ps1
    .\scripts\run_baseline_matrix.ps1 -Sizes xs
    .\scripts\run_baseline_matrix.ps1 -Sizes xs,s -Slopes 7,30
    .\scripts\run_baseline_matrix.ps1 -Sizes xs,s -Resume   # after an interruption
#>

[CmdletBinding()]
param(
    [ValidateSet("xs", "s", "m", "l")][string[]]$Sizes = @("s"),
    [double[]]$Slopes = @(7, 15, 30),
    [string[]]$Parts = @("ramp45", "ramp50", "ramp60", "ramp70", "ramp80", "ramp90",
                         "tshape", "twin_domes"),
    [string]$EnvName = "atomizer",

    # Skip combinations that already have a result from the current metric
    # definition. Use this to continue an interrupted matrix: each run writes
    # its report as it finishes, so nothing completed is lost.
    [switch]$Resume
)

$ErrorActionPreference = "Continue"

# tqdm draws its progress bars with Unicode block characters, which render as
# mojibake on the default Windows console code page and pollute the log file.
# PowerShell also surfaces anything a program writes to stderr as a
# NativeCommandError record even when nothing failed; progress bars go to
# stderr, so that noise is expected and harmless.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
}
catch {
    Write-Host "Could not switch the console to UTF-8; progress bars may look garbled." -ForegroundColor Yellow
}

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
Write-Host "Each run's report is written as it finishes, so an interruption loses at most one run." -ForegroundColor Yellow

# What is already done under the current metric definition?
$statusText = & conda run --name $EnvName --no-capture-output python tools/overhang_report.py `
    --status @Sizes 2>&1 | Out-String
$completeLine = ($statusText -split "`n" | Select-String "complete \(metrics")
if ($completeLine) {
    Write-Host ""
    Write-Host ("Existing results: " + $completeLine.ToString().Trim()) -ForegroundColor Cyan
    if (-not $Resume) {
        Write-Host "Running everything. Pass -Resume to skip what is already done." -ForegroundColor Yellow
    }
}

$skipped = 0

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

        if ($Resume) {
            $check = & conda run --name $EnvName --no-capture-output python -c @"
import json, sys
from pathlib import Path
p = Path('reports/baseline_overhang/${solid}_ms$($slope -replace '\.0$','').json')
try:
    print('done' if json.loads(p.read_text())['metrics_version'] == 2 else 'stale')
except Exception:
    print('missing')
"@ 2>$null
            if ($check -match 'done') {
                $skipped++
                Write-Host "[$index/$total] SKIP $solid at $slope deg - already done" -ForegroundColor DarkGray
                continue
            }
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
if ($skipped -gt 0) {
    Write-Host "$skipped run(s) were skipped as already complete." -ForegroundColor DarkGray
}
if ($failures.Count -gt 0) {
    Write-Host "$($failures.Count) of $total runs failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}
Write-Host "Summary: reports\baseline_overhang.md"
Write-Host "Progress log: reports\matrix_progress.csv" -ForegroundColor Cyan
Write-Host "Commit reports\ and paste the summary table." -ForegroundColor Cyan
