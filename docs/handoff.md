# Session handoff

Everything a fresh session needs to pick this work up. Written to be read cold,
with no prior conversation.

**Keep this file updated as the work progresses.**

Last updated: 2026-09-21.

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

| Task | Status |
|---|---|
| P0.1 Dev environment and packaging | **Done**, verified on the laptop |
| P0.2 Golden baseline | **Done**; pipeline is deterministic, baseline captured |
| P0.3 Test harness and CI | **Done**, verified on the laptop |
| P0.4 Taichi arch helper (`ti_env`) | **Done**. All 23 `ti.init` calls route through `ATOM_TI_ARCH` |
| P0.5 Machine profiles | **Done and verified** — golden test passed on the laptop, G-code byte-identical |
| P0.6 Tilt contracts and conventions | **Not started** — ★ this unblocks lanes B and C |
| P0.7 Benchmark meshes | **Done**, 36 meshes committed |
| P0.8 Overhang metrics | **Core done**; `tools/overhang_report.py` and the 24-run matrix remain |

153 unit tests pass.

### Verification status

The golden baseline is committed (`tests/golden/calibration_cube.stats.json`,
`calibration_cube.toolpath.npz`) and the golden test passes on the laptop as
of 2026-09-21:
`pytest --run-pipeline tests/test_golden.py -v` -> 3 passed in 457 s. P0.5 was
the first edit to a vendored Atomizer file, and the G-code it produces is
byte-identical to the baseline. The approach of editing vendored files behind
the golden net is therefore proven, not merely assumed.

**Re-run that test after every subsequent vendored edit.** It takes about
7.5 minutes.

### Suggested next steps

1. **P0.6** — tilt contracts and `docs/conventions.md`. Marked ★ in the plan
   because lanes B and C both wait on it. Note correction 1.1: the tilt
   contract must handle the spherical `(N, 2)` form.
2. Finish **P0.8** — `tools/overhang_report.py`, then the 24-run matrix, which
   produces the "before" numbers the whole contribution is measured against.
3. **Open experiment: can `order_atoms` use the GPU?** It is 86 % of runtime
   and runs on the CPU. `ATOM_TI_ARCH=cuda` now makes this a one-command test
   (see "The order_atoms question" below). Worth settling before the P0.8
   matrix, since it could change the cost of everything downstream.

---

## 6. What has been built

### New modules (`src/atom/`)

| Module | Purpose |
|---|---|
| `machine_profile.py` | Loads machine constants from `config/machines/*.json`, selected by `ATOM_MACHINE` (default `reference`). Frozen dataclass, validated, warns loudly on a PLACEHOLDER profile. |
| `benchmark_meshes.py` | Parametric box / ramp / T-shape / twin-dome generators. Self-contained geometry (ear-clipping triangulation, height-field solid) so trimesh's optional extras are not needed. |
| `overhang_metrics.py` | The three measurements the contribution is judged by: effective overhang angle, unsupported deposition, maximum tilt used. numpy + scipy only. |

### New tools (`tools/`)

| Tool | Purpose |
|---|---|
| `gcode_stats.py` | G-code to a stable JSON record (hash, counts, axis ranges, extrusion). The golden comparison. |
| `make_benchmarks.py` | Generates the benchmark meshes and parameter files; prints estimated runtime before writing. |

### Configuration and scripts

- `config/machines/reference.json` — upstream values, `status: verified`.
- `config/machines/ours.json` — **every number a placeholder**, `status: PLACEHOLDER`, gates M1/M2.
- `scripts/setup_laptop.ps1`, `scripts/fix_tool_paths.ps1`, `scripts/run_pipeline_tests.ps1`.

### Key documents

- `tests/golden/baseline.md` — baseline commit, environment, stage timings, determinism.
- `tests/golden/README.md` — how to re-capture the baseline.
- `docs/plan_corrections.md` — **read this before trusting the build plan.**

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

## 7a. The `order_atoms` question

`order_atoms` is 86 % of pipeline runtime and initialises Taichi on the CPU.
Moving it to the GPU is now a one-command experiment via `ATOM_TI_ARCH`, but
the structure argues against a large win:

* The ordering loop is **inherently sequential** — one atom appended per
  iteration (31 630 of them for the calibration cube), each choice depending on
  every earlier one. No backend parallelises that.
* The work **inside** each iteration is already Taichi kernels
  (`src/atom/toolpath3.py` has 27), and the BVH is Taichi too, so both are
  backend-agnostic and would run on the GPU unchanged.
* But every iteration **synchronises back to the host**:
  `compute_cost_and_find_best_next(...).to_numpy()` returns to Python, which
  then branches on the result. On a GPU that is ~31 630 forced pipeline
  flushes.

So the per-iteration work would get faster while per-iteration overhead gets
worse. At 10.5 ms per iteration on the CPU there is room to win, but it is an
empirical question.

**Risk to watch:** GPU reductions and atomics do not have a deterministic
ordering, so `find_best_next` could break ties differently and produce a
different toolpath. The golden test detects this immediately. If it fails on
GPU, the backend is not a free switch and the CPU default must stand.

The experiment, on the laptop:

```powershell
$env:ATOM_TI_ARCH = "cuda"
pytest --run-pipeline tests/test_golden.py -v   # correctness, ~7.5 min on CPU
Remove-Item Env:\ATOM_TI_ARCH
```

Compare the reported `order_atoms` time in `data/log/calibration_cube.log`
against the 331.6 s baseline. Note `kernel_profiler=True` is set on that stage
and is not free; worth measuring with it off too.

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
