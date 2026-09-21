<#
.SYNOPSIS
    Finds Miniconda and Blender on this machine and puts them on your PATH.

.DESCRIPTION
    Neither the Miniconda nor the Blender installer adds itself to PATH by
    default, so after installing them `conda` and `blender` are still "not
    recognized" in PowerShell. This script locates both, appends them to your
    *user* PATH (no Administrator needed, and nothing changes for other users),
    and runs `conda init powershell` so that `conda activate` works.

    It is safe to run more than once: entries already on PATH are left alone.

.EXAMPLE
    .\scripts\fix_tool_paths.ps1

.NOTES
    Close and reopen PowerShell afterwards. Both the PATH change and
    `conda init` only take effect in newly-started shells.

    If this script itself will not run ("running scripts is disabled on this
    system"), run this once first:
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Write-Section($Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Add-ToUserPath {
    <#
      Appends a directory to the user PATH if it is not already there.
      Uses exact element comparison rather than -like, because PowerShell
      wildcard matching would misread a path containing [ ] * or ?.
    #>
    param([Parameter(Mandatory)][string]$Directory)

    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    if ($current) {
        $entries = @($current -split ';' | Where-Object { $_ -ne '' })
    }

    if ($entries -contains $Directory) {
        Write-Host "  already on PATH: $Directory" -ForegroundColor Yellow
        return
    }

    $entries += $Directory
    [Environment]::SetEnvironmentVariable("Path", ($entries -join ';'), "User")
    Write-Host "  added to PATH:   $Directory" -ForegroundColor Green
}

function Find-CondaRoot {
    # Typical install locations first; winget's Anaconda.Miniconda3 package puts
    # it under ProgramData when elevated, under the user profile otherwise.
    $candidates = @(
        "$env:ProgramData\miniconda3",
        "$env:ProgramData\Miniconda3",
        "$env:ProgramData\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\Miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "$env:LOCALAPPDATA\Continuum\miniconda3"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "Scripts\conda.exe")) {
            return $candidate
        }
    }

    # Fall back to a bounded search of the usual roots.
    Write-Host "  not in the usual places; searching (this takes a moment)..."
    foreach ($base in @($env:ProgramData, $env:USERPROFILE, $env:LOCALAPPDATA)) {
        if (-not (Test-Path $base)) { continue }
        $hit = Get-ChildItem -Path $base -Filter "conda.exe" -Recurse -Depth 4 `
                             -ErrorAction SilentlyContinue |
               Where-Object { $_.DirectoryName -match '\\Scripts$' } |
               Select-Object -First 1
        if ($hit) { return (Split-Path -Parent $hit.DirectoryName) }
    }
    return $null
}

function Find-BlenderDir {
    $roots = @(
        "$env:ProgramFiles\Blender Foundation",
        "${env:ProgramFiles(x86)}\Blender Foundation",
        "$env:LOCALAPPDATA\Programs\Blender Foundation",
        "$env:USERPROFILE\Apps"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        # Newest version first, so a machine with several Blenders picks the latest.
        $hit = Get-ChildItem -Path $root -Filter "blender.exe" -Recurse `
                             -ErrorAction SilentlyContinue |
               Sort-Object FullName -Descending |
               Select-Object -First 1
        if ($hit) { return $hit.DirectoryName }
    }
    return $null
}

$failures = @()

Write-Section "Looking for Miniconda"
$condaRoot = Find-CondaRoot
if ($condaRoot) {
    Write-Host "  found: $condaRoot"
    # `condabin` is the minimal directory that exposes the `conda` command.
    Add-ToUserPath (Join-Path $condaRoot "condabin")

    Write-Section "Running 'conda init powershell'"
    # Sets up the PowerShell profile hook so `conda activate <env>` works.
    & (Join-Path $condaRoot "Scripts\conda.exe") init powershell
}
else {
    Write-Host "  NOT FOUND." -ForegroundColor Red
    Write-Host "  Install it from an Administrator PowerShell:"
    Write-Host "    winget install --id Anaconda.Miniconda3 -e --source winget --accept-package-agreements"
    $failures += "conda"
}

Write-Section "Looking for Blender"
$blenderDir = Find-BlenderDir
if ($blenderDir) {
    Write-Host "  found: $blenderDir"
    Add-ToUserPath $blenderDir
}
else {
    Write-Host "  NOT FOUND." -ForegroundColor Red
    Write-Host "  Install it from an Administrator PowerShell:"
    Write-Host "    winget install --id BlenderFoundation.Blender -e --source winget --accept-package-agreements"
    $failures += "blender"
}

# Refresh this window's PATH so the verification below sees the new entries.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

Write-Section "Verifying"
foreach ($tool in @("git", "conda", "blender")) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host ("  {0,-8} OK   {1}" -f $tool, $command.Source) -ForegroundColor Green
    }
    else {
        Write-Host ("  {0,-8} MISSING" -f $tool) -ForegroundColor Red
    }
}

Write-Section "Next step"
if ($failures.Count -gt 0) {
    Write-Host "Install the missing tools above, then run this script again." -ForegroundColor Red
    exit 1
}
Write-Host "CLOSE and REOPEN PowerShell, then run:" -ForegroundColor Green
Write-Host "    .\scripts\setup_laptop.ps1"
