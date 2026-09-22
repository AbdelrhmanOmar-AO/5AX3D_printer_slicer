# Session handoff

Everything a fresh session needs to pick this work up. Written to be read cold,
with no prior conversation.

**Keep this file updated as the work progresses.**

Last updated: 2026-09-21. **Phase P0 is complete**; 344 unit tests pass.

---

## 1. What this project is

A fork of [Atomizer](https://github.com/xavierchermain/atomizer) (Chermain et
al., SGP 2025) being extended for a 5-axis FFF printer with a bed on three
independent Z lead screws.

The goal, from `SLICER_BUILD_PLAN.md` v3: make Atomizer's tool-orientation field
**overhang-aware**, so the bed tilts toward overhangs and they print without
supports. Stock Atomizer already tilts the bed continuously, point by point; its
field simply ignores overhangs, constraining only the first layer and
low-curvature top surfaces.

The governing relationship: **printable overhang is about 45 degrees plus the
usable tilt**. At the reference machine's 30-degree limit that is ~75 degrees. A
fully horizontal overhang (90 degrees) needs at least 45 degrees of tilt. That
is a mechanical question (gate M2), not a software one.

Read `SLICER_BUILD_PLAN.md` for the task breakdown, then
`docs/plan_corrections.md` for everything in it that is wrong or has been
deliberately departed from. **Read the corrections file before trusting the plan
on any specific fact.**

---

## 2. Who does what

- **Claude Code** writes code and tests, and can run only CPU unit tests. No GPU,
  no Blender, and the `atom` package cannot even be installed in the session
  container (Python 3.11 there; `pyproject.toml` pins `< 3.11`). Run tests with
  `PYTHONPATH=src python3 -m pytest`.
- **The operator** (the user) reviews, and runs anything needing the GPU laptop
  or the printer. They are a mechanical engineer, **new to git and to coding** —
  explain git concepts in plain language and give exact PowerShell commands.
  Their stated preference: *never make assumptions, always ask when unsure.*

---

## 3. The operator's machine

| | |
|---|---|
| OS | Windows 11 (10.0.26200), PowerShell 5.1 |
| Username | `Abdo Yasser` — **has a space**, so quote paths: `cd "$HOME\5AX3D_printer_slicer"` |
| Python | 3.10.21, conda environment `atomizer`, created from **conda-forge** |
| Taichi | 1.7.4 |
| GPU | CUDA 13.0 driver, `ti.init(arch=ti.cuda)` reports `Arch.cuda` — working |
| Blender | 5.2.1 LTS, on PATH |
| Repo | `C:\Users\Abdo Yasser\5AX3D_printer_slicer` |

Setup pain already solved, all documented in `README.md`:

1. PowerShell blocks scripts by default → `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
2. Neither Miniconda nor Blender adds itself to PATH → `scripts\fix_tool_paths.ps1`.
3. conda's default channels demand Terms of Service acceptance → conda-forge.

---

## 4. Repository conventions

- **Branch:** everything on `claude/new-session-l8g46d`. `main` is untouched and
  holds the golden baseline commit. Never push elsewhere without asking.
- **Commits:** one per build-plan task, subject line `P0.7: <what>`. Explain
  *why*, and record anything surprising that was discovered.
- **Tests:** three tiers, in `pytest.ini` and `tests/conftest.py`.

  | Command | Runs | Time |
  |---|---|---|
  | `pytest` | unit only (default; unmarked tests count as `unit`) | seconds |
  | `pytest --run-pipeline` | + real pipeline on the calibration cube | ~7 min |
  | `pytest --run-benchmark` | + the full matrix | hours |

- **CI:** `.github/workflows/unit-tests.yml`, Ubuntu + Python 3.10, `pytest -m unit`.
- A module that defines Taichi kernels **must not** use
  `from __future__ import annotations`. See `docs/plan_corrections.md` 3.2.

---

## 5. Where the work stands

**Phase P0 is complete.** Every foundation task is built, tested and verified
on the operator's laptop. What remains in P0 is compute time, not code.

| Task | Status |
|---|---|
| P0.1 Dev environment and packaging | **Done**, verified on the laptop |
| P0.2 Golden baseline | **Done**. Deterministic; captured, committed and passing |
| P0.3 Test harness and CI | **Done**, verified on the laptop |
| P0.4 Taichi arch helper (`ti_env`) | **Done**. All 23 `ti.init` calls route through `ATOM_TI_ARCH` |
| P0.5 Machine profiles | **Done**. Golden test passed after it: G-code byte-identical |
| P0.6 Tilt contracts and conventions | **Done**. `atom.tilt`, `atom.contracts`, `docs/conventions.md` |
| P0.7 Benchmark meshes | **Done**. 36 meshes, 9 parts x 4 sizes |
| P0.8 Overhang metrics | **Tooling done.** The 24-run matrix has not been run |

344 unit tests pass; 6 skipped (the `pipeline` and `benchmark` tiers).

### Verification status

Both laptop verifications have passed:

| Check | Result |
|---|---|
| `pytest --run-pipeline tests/test_golden.py` | 3 passed in 457 s, after the P0.5 vendored edit |
| `python tools/atomize.py data/param/ramp60_xs.json` | ran normally; the generated meshes are pipeline-valid |

The golden baseline is committed (`tests/golden/calibration_cube.stats.json`,
`calibration_cube.toolpath.npz`), so it no longer lives on one machine.

**Re-run the golden test after every vendored edit.** About 7.5 minutes.

### The immediate next step

```powershell
.\scripts\run_baseline_matrix.ps1                # size s, ~7.2 h, 24 runs
.\scripts\run_baseline_matrix.ps1 -Sizes xs      # ~1.6 h, proves the flow first
.\scripts\run_baseline_matrix.ps1 -Sizes xs,s    # ~8.8 h, adds the scaling curve
```

Note that `-Sizes` is what turns the summary's runtime table from a single
point into a scaling curve. The overhang baseline itself needs only one size;
the sizes exist to measure how pipeline cost grows with part volume, which
nobody has published for Atomizer.

This produces `reports/baseline_overhang.md`: the "before" numbers the whole
contribution is measured against, and what gate D0 is waiting on. Everything
needed is built and tested; only the compute time remains.

Expect stock Atomizer to fail much of the ramp family. That is the finding, not
a bug: the field constrains only the first layer and low-curvature top
surfaces, so it has no reason to lean into an overhang. On the calibration cube
it used 5.53 degrees of a 7 degree budget with no overhang to aim at.

### After that

1. **P1** — kinematics test suite, templated header/footer, infill parameters,
   and the G-code validator that P3, P4, P5 and P7 all use. Lane A, unblocked.
2. **P2** — the overhang-aware field. The core contribution and the only
   research-shaped phase. Depends on P0.6 (done), P0.7 (done) and P0.8's
   numbers.
3. **P4** — motion safety. Lane C, needs only P1.4 and P0.6, so it can run
   beside P2.

Note that P1, P2 and P4 are meant to run in parallel, but there is one operator
doing every laptop run. The plan's advice is at most two Claude Code sessions
at a time.

## 6. What has been built

### New modules (`src/atom/`)

| Module | Purpose |
|---|---|
| `machine_profile.py` | Loads machine constants from `config/machines/*.json`, selected by `ATOM_MACHINE` (default `reference`). Frozen dataclass, validated, warns loudly on a PLACEHOLDER profile. |
| `benchmark_meshes.py` | Parametric box / ramp / T-shape / twin-dome generators. Self-contained geometry (ear-clipping triangulation, height-field solid) so trimesh's optional extras are not needed. |
| `overhang_metrics.py` | The three measurements the contribution is judged by: effective overhang angle, unsupported deposition, maximum tilt used. numpy + scipy only. |
| `tilt.py` | Tilt angles, rotations and limits. Pure numpy. `rotate_toward` is the operation P2.2 performs. |
| `contracts.py` | `MachineToolpath`: a toolpath plus the machine state it implies. Runs the IK over every point and marks failures rather than aborting, which is what P3.3, P4 and P5.4 are built on. Verified against the golden toolpath: 46 773 points, 0 unreachable, and the Z/U/V maxima match the written G-code exactly. |
| `ti_env.py` | One switch for the Taichi backend across every stage. |

### New tools (`tools/`)

| Tool | Purpose |
|---|---|
| `gcode_stats.py` | G-code to a stable JSON record (hash, counts, axis ranges, extrusion). The golden comparison. |
| `make_benchmarks.py` | Generates the benchmark meshes and parameter files; prints estimated runtime before writing. |
| `overhang_report.py` | Runs a part at a chosen `max_slope`, measures it, writes a JSON report. `--summarize` builds the comparison table. |

### Configuration and scripts

- `config/machines/reference.json` — upstream values, `status: verified`.
- `config/machines/ours.json` — **every number a placeholder**, `status: PLACEHOLDER`, gates M1/M2.
- `scripts/setup_laptop.ps1`, `scripts/fix_tool_paths.ps1`, `scripts/run_pipeline_tests.ps1`,
  `scripts/run_baseline_matrix.ps1` (drives the whole P0.8 matrix).

### Key documents

- `tests/golden/baseline.md` — baseline commit, environment, stage timings, determinism.
- `tests/golden/README.md` — how to re-capture the baseline.
- `docs/plan_corrections.md` — **read this before trusting the build plan.** Its
  section 6 is a five-line index of what a later task is most likely to get
  wrong.
- `docs/conventions.md` — frames, what `normal` means, the three screws, how the
  IK signals failure, and a worked numeric example. Derived from the code by
  running it.

---

## 7. Numbers worth remembering

From the calibration cube (20 x 20 x 21 mm, 31 630 atoms), in
`tests/golden/baseline.md`:

- `order_atoms` is **86 %** of runtime (331.6 s of 383.7 s logged) and runs on
  the **CPU**.
- About **3.8 atoms per mm^3** at 0.9 mm deposition width, and cost scales with
  volume.

Estimated cost of the 24-run P0.8 matrix:

| Size | Principal dimension | Full matrix |
|---|---:|---:|
| `xs` | 30 mm | 1.6 h |
| `s` | 50 mm | 7.2 h |
| `m` | 75 mm | 24.4 h |
| `l` | 100 mm | 57.8 h |

P2.5 repeats the matrix with the overhang-aware field, so each figure lands
twice. Treat them as a floor: ordering is a nearest-neighbour problem and may be
worse than linear.

---

## 7a. The `order_atoms` question, settled

`order_atoms` is 86 % of pipeline runtime and runs on the CPU. Moving it to the
GPU was tried via `ATOM_TI_ARCH=cuda` and **is not a win**: the run was
abandoned after far exceeding the 457 s the same test takes on the CPU. That is
"not faster", not a measured factor, and some of the excess is one-off kernel
compilation on a new backend.

Why the structure argues against it:

* The ordering loop is **sequential** — one atom appended per iteration
  (31 630 for the calibration cube), each choice depending on every earlier
  one. No backend parallelises that.
* The work inside each iteration is already Taichi kernels
  (`src/atom/toolpath3.py` has 27) and the BVH is Taichi too, so both would run
  on the GPU unchanged.
* But every iteration **synchronises to the host**:
  `compute_cost_and_find_best_next(...).to_numpy()` returns to Python, which
  branches on the result. Many tiny kernels punctuated by host round-trips is
  the pattern GPUs handle worst.

There was also a correctness risk that never came into play: GPU reductions and
atomics have no deterministic ordering, so `find_best_next` could break ties
differently and produce a different toolpath. The golden test would have caught
it.

**Still untested, and cheaper:** `tools/order_atoms.py` passes
`kernel_profiler=True`, which is not free. Removing it might speed up the CPU
path with no determinism risk at all.

Practical consequence: long runs are CPU-bound. A machine with faster or more
CPU cores helps; a better GPU does not. Sustained load is within a laptop's
design spec — expect loud fans, a hot chassis and thermal throttling that makes
the runtime estimates optimistic, not damage. Keep it plugged in and stop it
sleeping (`powercfg /change standby-timeout-ac 0`).

## 7b. The P0.8 baseline result (2026-09-22)

48 runs, 8 parts x 3 slopes x 2 sizes, 19.9 hours wall clock, **none failed**.
`reports/baseline_overhang.md` and the per-run JSON are committed.

### The headline finding

**Stock Atomizer does not merely ignore overhangs; it tilts away from them, and
a larger tilt budget makes that worse.** Geometric angle against the effective
angle achieved, size `s`:

| Part | max_slope 7° | 15° | 30° | Tilt used at 30° |
|---|---|---|---|---|
| `ramp45` | 45° | 45° | **50°** | 6.3° |
| `ramp50` | 50° | 50° | **65°** | 17.1° |
| `ramp60` | 60° | 64° | **90°** | 29.7° |
| `ramp70` | 70° | 79° | **90°** | 20.6° |
| `ramp80` | 83° | 90° | 90° | 14.6° |
| `ramp90` | 90° | 90° | 90° | 11.9° |

The damage tracks the tilt almost exactly one-for-one: `ramp60` at 30° uses
29.7° of tilt and its overhang worsens by 30°, turning a 60° slope into a
horizontal ceiling. That is the same `theta_eff = theta_geo ± tilt` relation
P0.6 verified, with the wrong sign, and it is the clearest possible statement
of what P2 exists to fix: the same tilt aimed correctly would turn 60° into
30°.

`ramp90` and `tshape` are already horizontal and cannot get worse. At
`max_slope 7` the field barely tilts at all (0.6–1.3°) on most parts, so the
harm only appears once there is a budget to misuse.

No part passed the thresholds, at any tilt setting.

### Caveat on the unsupported column

Those runs predate the bed-contact fix (`plan_corrections.md` 4.5), so their
unsupported figures are inflated by roughly three percentage points. The
verdicts do not change — every part fails on the effective angle alone — but
the absolute percentages should not be quoted until the runs are re-scored.

Re-scoring needs archived toolpaths, which were added *after* this matrix. So
either accept the caveat, or re-run. Runs from now on archive to
`reports/toolpaths/` and `--reanalyse` re-scores them in seconds.

### Measured runtime scaling

`order_atoms` grows as **points^1.5**, not linearly (mean exponent 1.50 over 24
part/slope pairs, range 1.39–1.61). This does not appear to be published for
Atomizer, so it is a result in its own right.

| Size | Volume vs `s` | Per run | 24-run matrix |
|---|---|---|---|
| `s` (50 mm) | 1x | 42 min | 17 h (measured) |
| `m` (75 mm) | 3.4x | ~4.4 h | ~105 h (4 days) |
| `l` (100 mm) | 8.0x | ~16 h | ~384 h (16 days) |

**`l` is out of reach** for a full matrix and `m` is a four-day commitment.
Size `s` is the practical ceiling for anything run repeatedly; reserve `l` for
individual showcase parts and the printed benchmarks.

This is also why the first matrix took 19.9 h against an 8.8 h estimate: that
estimate assumed linear scaling.

## 8. Gates (blocked on other people)

| Gate | Question | Status |
|---|---|---|
| **M2** | Tilt budget: max tilt on both axes at once, cone or box, can the design exceed 30 degrees, does reachable tilt vary with position? | **Most important.** Deferred until the mechanical design is settled. Determines whether horizontal overhangs are possible at all. |
| M1 | Final bed geometry | Deferred; needed at P6.1 |
| M3 | Clearance envelope | Deferred; needed at P6.2 |
| E1 | Firmware (RRF or Klipper) and the three screws' axis letters | Deferred. The current header is RepRapFirmware-only (`G32`, `M98 P"..."`). |
| HW | Printer assembled | Not yet |
| D0 | Tilt budget and benchmark geometry | Needs P0.8 results plus P2.1's bound |
| D1 | `max_overhang_deg`, overhang-vs-ceiling priority | Placeholder 45 degrees |
| D2 | Plan B trigger (fall back to v2.1 poses) | Week ~5 decision |
| D3 | Max tilt rate and acceleration | Tuned on hardware |

The operator has been told M2 is worth raising with the mechanical team early,
since the design is still open and the 45-degree threshold is a hardware fact no
slicer can work around.
