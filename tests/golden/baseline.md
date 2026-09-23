# Golden baseline record

Captured for build plan task P0.2. This is the frozen "before" state: every
later change must be shown not to alter it, except where a task says otherwise.

## Baseline commit

```
e7b71ea89046281ccff89ec8b14429ef92418a6d   (main)
```

This fork's `main`, **not** upstream `xavierchermain/atomizer`. The fork already
carries the triple-Z G-code generation and the infill work, and those are part
of the behaviour being preserved.

Captured on 2026-09-21.

## Environment

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.10.21 (conda env `atomizer`, created from conda-forge) |
| Taichi | 1.7.4, llvm 15.0.1, commit b4b956fd, win |
| Taichi backend | **The stock mix**, i.e. each stage's own default: the eight `gpu` stages on CUDA (`ti.init(arch=ti.cuda)` reports `Arch.cuda`), the five `cpu` stages — including `order_atoms` — on the CPU. `ATOM_TI_ARCH` was **unset**. |
| CUDA driver | 13.0 |
| Blender | 5.2.1 LTS (hash 9e2066aef7ef) |

## Determinism

**The pipeline is deterministic on a fixed backend mix.** Two consecutive runs
of the calibration cube produced statistics records that compare equal with no
differences (`Compare-Object` returned nothing).

Golden comparisons may therefore use the **SHA-256 of the G-code directly**;
tolerance-based float comparison is not needed — *as long as the backends are
the ones in the Environment table above.*

### It is not deterministic across backends

Re-running the whole pipeline with `ATOM_TI_ARCH=cpu` (2026-09-23, 19 min 54 s,
the same laptop) produced a **different toolpath**, not merely different
rounding:

| Statistic | This baseline | `ATOM_TI_ARCH=cpu` |
|---|---:|---:|
| `G1` moves | 47 837 | 49 004 (+2.44 %) |
| Deposition runs (retracts) | 530 | 527 |
| Fan toggles (`M106`) | 50 | 30 |
| Total extrusion | 3231.197 mm | 3223.138 mm |
| SHA-256 | `d9fb441c…` | `bc2ac5bb…` |

`ATOM_TI_ARCH=cpu` moves the iterative field solvers off CUDA and leaves
`order_atoms` where it already was; the solvers' last-bit differences pass
through a threshold in `extract_explicit_atoms` and are then amplified by the
greedy planner. The reasoning and the full table are in
`docs/plan_corrections.md` **3.9**, along with what it means for running the
matrix on a CPU-only machine.

So `tests/test_golden.py` splits the check in two:

* `ATOM_TI_ARCH` unset → the exact SHA-256 comparison (this baseline);
* `ATOM_TI_ARCH` set → invariants plus a drift tolerance
  (`test_forced_backend_stays_within_tolerance`), because a tolerance wide
  enough to accept the table above would not catch a real regression.

**If the golden ever has to be recaptured on a different machine, recapture it
on that machine.** The hash is not portable.

## Part

`data/param/calibration_cube.json` — the stock parameters, unmodified:

| | |
|---|---|
| Mesh | `calibration_cube.stl`, 20 x 20 x 21 mm, 136 triangles |
| Deposition width | 0.90 mm |
| Layer height | 0.45 mm |
| Cell sides length | 0.169 mm |
| Max slope | 7.0 degrees |
| Infill | on (gyroid) |
| SDF grid | 119 x 119 x 125 cells |
| Atom count | 31 630 |

## Stage timings

From `data/log/calibration_cube.log`. Only the stages that log a duration are
listed; the remainder (Blender remesh, OBJ to BPN, BPN to SDF, infill, smooth,
tesselate, add platform, G-code) are not individually timed and together are a
small fraction of the total.

| Stage | Time (s) | Share of logged time |
|---|---:|---:|
| Direction field | 7.2 | 1.9 % |
| Implicit layers | 8.8 | 2.3 % |
| Tangents | 12.1 | 3.2 % |
| Atom alignment | 23.1 | 6.0 % |
| Explicit atom extraction | 0.9 | 0.2 % |
| **Order atoms (toolpath planner)** | **331.6** | **86.4 %** |
| Total (logged) | 383.7 | |

Memory: atom aligner 0.082 GB, atom extractor 0.099 GB.

### Why this matters for planning

`order_atoms` dominates, and its cost scales with the atom count, which scales
with part volume. The calibration cube is 8 400 mm^3 and yields 31 630 atoms
(~3.8 atoms/mm^3). Scaling that to larger benchmark parts, and assuming
`order_atoms` is no worse than linear in atom count:

| Part size | Volume | Atoms (est.) | `order_atoms` (est.) |
|---|---:|---:|---:|
| 20 x 20 x 21 (this cube) | 8 400 mm^3 | 31 630 | 5.5 min |
| 100 x 45 x 20 | 90 000 mm^3 | ~340 000 | ~1 hour |
| 100 x 100 x 100 | 1 000 000 mm^3 | ~3 800 000 | ~11 hours |

The P0.8 baseline matrix is 24 runs. At one hour per run that is a day of
compute; at eleven it is not practical. Benchmark part dimensions (P0.7) need to
be chosen with this in mind — see the note in that task.

## Files in this folder

| File | What it is |
|---|---|
| `calibration_cube.stats.json` | `tools/gcode_stats.py` output for the baseline G-code |
| `calibration_cube.toolpath.npz` | `data/toolpath/calibration_cube_platform.npz`, the last toolpath stage before G-code |

## Regression history

| Date | Change verified | Result |
|---|---|---|
| 2026-09-21 | P0.5, machine constants moved into profiles (first vendored edit) | Pass, 3 tests in 457 s. G-code byte-identical. |
| 2026-09-23 | P0.4 exit criterion: whole pipeline under `ATOM_TI_ARCH=cpu` | **Hash differs by design** — see "It is not deterministic across backends" above and correction 3.9. Run took 1194 s. |

## Re-checking it

```powershell
pytest --run-pipeline tests/test_golden.py -v
```

See `tests/test_golden.py`. Capture instructions are in `README.md` in this
folder.
