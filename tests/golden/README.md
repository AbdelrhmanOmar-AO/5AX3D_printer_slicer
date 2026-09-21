# Golden baseline

This folder holds a frozen record of what the slicer produces **today**, before
any of our changes. Every later change is compared against it to prove we did
not break or silently alter existing behaviour.

Status: **captured** on 2026-09-21. See [`baseline.md`](baseline.md) for the
commit, environment, stage timings and the determinism result.

The pipeline proved deterministic, so comparisons use the G-code SHA-256
directly. Re-check at any time with:

```powershell
pytest --run-pipeline tests/test_golden.py -v
```

The instructions below are kept for re-capturing the baseline (for example on a
new machine, or if the baseline is deliberately moved to a later commit).

## Which commit is the baseline?

**This fork's `main` as it stands today** — not the original upstream Atomizer.

The fork already carries work we depend on (G-code generation for the triple-Z
printer, infill, the calibration-cube infill parameters). Those are part of the
"before" picture we want to preserve, so the baseline is taken here rather than
at `xavierchermain/atomizer`.

Record the exact commit in `baseline.md` (below) when capturing.

## What goes in this folder

| File | What it is |
|---|---|
| `calibration_cube.stats.json` | Output of `tools/gcode_stats.py` on the baseline G-code |
| `calibration_cube.toolpath.npz` | The final toolpath (`..._platform.npz`), if under 5 MB |
| `baseline.md` | Commit hash, environment, timings, determinism result |
| `../test_golden.py` | The regression test that re-runs and compares |

## How to capture it (operator, on the laptop)

Run each block in PowerShell, from the repository root, with the `atomizer`
conda environment active (`conda activate atomizer`).

### 1. Confirm you are on the baseline commit

```powershell
git switch main
git pull
git rev-parse HEAD        # write this hash down for baseline.md
git status                # must say "nothing to commit, working tree clean"
```

### 2. Run the pipeline on the calibration cube

The calibration cube is 20 x 20 x 21 mm with 136 triangles, so this is the
fastest meaningful part in the repository.

```powershell
python tools/atomize.py data/param/calibration_cube.json 2>&1 | Tee-Object -FilePath run1.txt
```

Note how long it takes. If it fails, stop and paste `run1.txt` — everything
downstream depends on this working.

### 3. Capture the statistics

```powershell
python tools/gcode_stats.py data/gcode/calibration_cube.gcode -o tests/golden/calibration_cube.stats.json
# `_platform.npz` is the last toolpath stage before G-code (tools/atomize.py).
Copy-Item data/toolpath/calibration_cube_platform.npz tests/golden/calibration_cube.toolpath.npz
# Skip this copy if the file is larger than 5 MB; note its size instead.
```

### 4. Determinism check — run it a second time

This tells us whether comparisons can use an exact hash or need tolerances.

```powershell
python tools/atomize.py data/param/calibration_cube.json 2>&1 | Tee-Object -FilePath run2.txt
python tools/gcode_stats.py data/gcode/calibration_cube.gcode -o run2.stats.json
Compare-Object (Get-Content tests/golden/calibration_cube.stats.json) (Get-Content run2.stats.json)
```

If `Compare-Object` prints nothing, the pipeline is deterministic and golden
comparisons can use the SHA-256 directly. If it prints differences, note which
fields differ and by how much — comparisons will use tolerances instead
(`rtol=1e-5` on floats, exact on counts).

### 5. Record the environment

```powershell
python -c "import taichi as ti; print('taichi', ti.__version__)"
python -c "import taichi as ti; ti.init(arch=ti.cuda); print(ti.lang.impl.current_cfg().arch)"
blender --version
Get-Content data/log/calibration_cube.log
```

The log file lists the per-stage commands and their runtimes.

### 6. Report back

Paste into the pull request or the chat:

- the commit hash from step 1,
- how long each run took,
- the result of the `Compare-Object` in step 4,
- the versions from step 5,
- and commit the files in this folder.

## Cleanup

`run1.txt`, `run2.txt` and `run2.stats.json` are scratch files — delete them
once the numbers are recorded. They are covered by `.gitignore`.
