# Session handoff

Everything a fresh session needs to pick this work up. Written to be read cold,
with no prior conversation.

**Keep this file updated as the work progresses.**

Last updated: 2026-09-23. **Phase P0 is complete** — all 48 matrix runs are in
at metrics v2 — and **P5.4** (toolpath viewer, bed-motion animation, Qt window)
is built. Two sessions worked in parallel and have now been merged; see
section 7c.

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

**Every measured runtime in this document was taken on this laptop.** Per
correction 3.9 that is not a detail: a run on another machine is a different
computation, so anything compared against the committed baseline has to run
here.

| | |
|---|---|
| CPU | **AMD Ryzen 5 5600H** (Zen 3, 2021), 6 cores / 12 threads, 3.3 GHz base, 4.2 GHz boost |
| RAM | Ample; the pipeline's peak measured usage is 0.18 GB, so RAM has never been the constraint |
| OS | Windows 11 (10.0.26200), PowerShell 5.1 |
| Username | `Abdo Yasser` — **has a space**, so quote paths: `cd "$HOME\5AX3D_printer_slicer"` |
| Python | 3.10.21, conda environment `atomizer`, created from **conda-forge** |
| Taichi | 1.7.4 |
| GPU | CUDA 13.0 driver, `ti.init(arch=ti.cuda)` reports `Arch.cuda` — working |
| Blender | 5.2.1 LTS, on PATH |
| Repo | `C:\Users\Abdo Yasser\5AX3D_printer_slicer` |

### Two university machines were assessed (2026-09-24)

Both were offered as somewhere to move the long runs. **One is slower and was
rejected; the other is worth taking, and changes how the matrix should be run.**

| | Laptop | Lab PC A | **Lab PC B** |
|---|---|---|---|
| CPU | Ryzen 5 5600H (Zen 3, 2021) | Xeon Silver 4112 (Skylake-SP, 2017) | **2 x Xeon Gold 6254** (Cascade Lake, 2019) |
| Cores / threads | 6 / 12 | 4 / 8 | **36 / 72** (two sockets) |
| Base / boost | 3.3 / **4.2 GHz** | 2.6 / 3.0 GHz | 3.1 / 4.0 GHz |
| RAM | Ample | 128 GB | 64 GB |
| GPU | The CUDA GPU the golden was captured on | Quadro P600, 2 GB VRAM | **RTX A6000, 48 GB VRAM** |
| Verdict | The reference machine | **Rejected** | **Take it** — see below |

Everything below is reasoned from clock and architecture. **Nothing on either
lab machine has been measured yet.**

#### Why single-core speed is what matters

`order_atoms` is 86 % of runtime and is a sequential 31 630-iteration loop
(section 7a). Core count does nothing for it; clock and per-clock throughput do.
On that path the laptop's 4.2 GHz Zen 3 beats both lab machines, so **neither
makes one run faster.** RAM is irrelevant either way: peak measured usage across
the whole pipeline is 0.18 GB.

#### Lab PC A: rejected

Roughly **1.7x slower** per core than the laptop (3.0 GHz against 4.2, on an
architecture four years older), so the 19.9-hour matrix would take about 34
hours. Its 128 GB buys nothing, and the P600's 2 GB of real VRAM is tighter than
what the baseline was captured on. The reported "65.8 GB" for that card is
Windows adding shared system memory to its 2 GB.

#### Lab PC B: take it, for throughput rather than speed

Per run it is roughly a wash with the laptop. What it offers is **36 physical
cores against a matrix of 48 independent runs.**

The partition is set by a constraint this tool already documents: every run of a
given part writes the same `data/` paths whatever its `max_slope`
(`tools/overhang_report.py` docstring), so **two concurrent runs of one part
would silently overwrite each other's intermediates.** That is the same class of
failure as correction 4.7 and must not be got wrong.

But the matrix is 8 parts x 2 sizes = **16 distinct `solid_name`s**, each with 3
slopes. So:

> **Run the 16 part+size combinations concurrently, each doing its 3 slopes in
> sequence.** No two workers ever share a `data/` filename.

16 workers on 36 cores is about 2 cores each, which is close to all the
sequential loop can use. Estimated **75 to 90 minutes** for the full matrix
against the laptop's 19.9 hours — roughly **13x** on wall-clock, from a change
to the runner script and none to the slicer.

#### Why that matters more than the speed

It **dissolves correction 3.9.** Re-baselining on a new machine costs 34 hours
on Lab PC A, which is fatal; on Lab PC B it costs about 90 minutes. So the stock
baseline can be re-run there, the whole comparison stays on one consistent
machine as 3.9 requires, and every P2.5 iteration becomes a 90-minute loop
instead of a 20-hour one.

It may also return **`l`-size parts** to gate D0's options: about 384 hours
serial becomes plausibly 24 to 36 hours. The A6000's 48 GB also gives the CUDA
field stages far more headroom than the laptop for large parts.

#### Two hazards the parallel runner must handle

1. **Taichi's offline kernel cache.** A single interrupted run already produced
   `Lock C:/taichi_cache/ticache/ticache.lock failed`. Sixteen processes sharing
   that directory will contend. Each worker needs its own cache directory —
   **no code change required**, the path is relocatable per process by
   environment variable, verified on Taichi 1.7.4:

   ```powershell
   $env:TI_OFFLINE_CACHE_FILE_PATH = "$env:LOCALAPPDATA\ticache"
   ```

   `ti.init(offline_cache_file_path=...)` works too, but the environment
   variable is what a worker wrapper should use, since it needs no edit to any
   of the 13 stages.

   **But a per-worker cache starts empty, and an empty cache is expensive.**
   The lab machine's first ever run (2026-09-24, cold cache) spent 49.9 s on
   direction computation against the laptop's 7.1 s, 56.6 s on implicit layers
   against 8.8 s, and 151.9 s on atom alignment against 26.6 s — the GPU
   stages, which compile the most kernels, on a far better GPU. Sixteen
   independent caches would pay that sixteen times: perhaps 80 minutes of pure
   compilation on a job estimated at 2 to 4 hours.

   So the runner must **warm one cache with a single run, then copy that
   directory to each worker before starting.** Copying a few hundred megabytes
   sixteen times costs seconds. Do not simply point each worker at an empty
   directory. Measured on the lab machine: a cold run's GPU stages cost 49.9 /
   56.6 / 45.7 / 151.9 s against 8.7 / 10.5 / 16.0 / 30.7 s warm.
2. **`reports/matrix_progress.csv`.** Sixteen processes appending to the
   safeguard log added after the laptop restarted mid-run. Needs a lock, or
   per-worker files merged at the end.

Both fail quietly, so both need tests.

#### Lab PC B: no usable `C:\` home, but local scratch works

The operator's home directory there is a network share (`Z:` and `U:`, one
volume, 627 GB free). **`%TEMP%` is nevertheless on local `C:`, writable, with
931 GB free** (`C:\Users\CA7387~1.USE\AppData\Local\Temp`); `D:` reports 0 GB
and is unusable.

That matters because the pipeline writes large `.npz` files at every one of 13
stages, and running it over SMB would put I/O on the critical path and make the
concurrency estimate worthless. So:

| What | Where |
|---|---|
| Repo and all `data/` I/O | `%USERPROFILE%\5AX3D_printer_slicer` (local C:) |
| Conda | Already installed all-users at `C:\ProgramData\Anaconda3`; **no admin**, so `conda init` is refused — put its `shell\condabin\conda-hook.ps1` in your own `$PROFILE` instead, and point `envs_dirs`/`pkgs_dirs` at `%USERPROFILE%\.conda` |
| Blender / Git | Portable: the Blender **`.zip`** and **PortableGit**, extracted under `%USERPROFILE%`. The `.msi` and the standard Git installer both demand admin; the portable forms need none |
| Taichi kernel cache | `%LOCALAPPDATA%\ticache`, via `TI_OFFLINE_CACHE_FILE_PATH` |
| Durable copy of `reports/` | The `Z:` share, and GitHub |

Nothing has to be copied off the laptop: the repository is **35 MB** and carries
all 36 benchmark STLs, the 48 JSON reports and the golden baseline. Only
`reports/toolpaths/` (~220 MB, gitignored) is bulky, and the lab machine will
generate its own.

**Its graphics context cannot run the viewer tests.** `pytest` there aborted at
the last test with `Windows fatal exception: access violation` inside
pyvista's `renderer.close`, which kills the process and took the other 450
tests with it. Set `ATOM_SKIP_DISPLAY_TESTS=1` on that machine; `tests/_display.py`
holds the gate and `tests/test_display_gate.py` pins it. Nothing in the slicing
pipeline touches those tests. Leave the variable unset on the laptop, which can
run them and found correction 4.13 by doing so.

**Unverified:** the profile path is an 8.3 short name ending `.USE`, which can
indicate a **temporary profile wiped at logout**. If it is, conda and Blender
would have to be reinstalled every session. Test by writing a file to
`%USERPROFILE%`, logging out and back in, and reading it. If it does not
survive, ask IT for a persistent local folder.

#### Order of work, decided with the operator

The runner is **not built yet, by choice**: measure the machine before writing
code against an estimate.

1. Get access, and ask IT the questions in the list below.
2. Set it up with the existing `scripts\setup_laptop.ps1` and
   `scripts\fix_tool_paths.ps1`.
3. Run the 12-minute known-answer check (below). It confirms the environment
   *and* measures real per-run speed.
4. Then build `-Parallel N` with measured numbers in hand.

```powershell
python tools/overhang_report.py data/param/ramp45_xs.json --max-slope 7
```

Expect about 0.2 % unsupported and `printable: True`. The wall-clock against the
laptop's ~7 minutes is the number that decides whether the estimates above hold.

#### Result of that check on Lab PC B (2026-09-24)

**It passed, identically to the laptop**: 0.24 % unsupported, `printable: True`,
worst effective overhang 45.0 degrees, max tilt used 0.61 degrees. The machine
computes correct results, on Blender 5.2.1 LTS (the same version as the laptop)
and Taichi on `Arch.cuda` against the A6000.

Timing, against the committed laptop run of the same part and slope
(`reports/baseline_overhang/ramp45_xs_ms7.json`):

| Stage | Laptop | Lab PC B (cold cache) |
|---|---:|---:|
| Direction computation | 7.1 s | 49.9 s |
| Implicit layers | 8.8 s | 56.6 s |
| Tangent computation | 13.5 s | 45.7 s |
| Atoms alignment | 26.6 s | 151.9 s |
| Extracting atoms | 0.9 s | 3.1 s |
| **Toolpath planner (CPU)** | **230.7 s** | **354.8 s** |

Those cold figures are **not** the machine's speed. The CPU stage was 1.54x
slower, roughly what the two processors predict, but the GPU stages were 3.4x to
7x slower *on a far better GPU*, which is not physically sensible. It was
first-run kernel compilation into an empty cache. A warm-cache re-run confirmed
it:

| Stage | Laptop | Lab PC B, **warm** | Ratio |
|---|---:|---:|---:|
| Direction computation | 7.1 s | 8.7 s | 1.23x |
| Implicit layers | 8.8 s | 10.5 s | 1.19x |
| Tangent computation | 13.5 s | 16.0 s | 1.19x |
| Atoms alignment | 26.6 s | 30.7 s | 1.15x |
| Extracting atoms | 0.9 s | 0.9 s | 1.00x |
| Toolpath planner | 230.7 s | 282.0 s | 1.22x |
| **Whole pipeline** | **341.4 s** | **414.4 s** | **1.21x** |

**The lab machine is 1.21x slower per run**, uniformly across stages. Even the
planner carried compilation in the cold run (354.8 s against 282.0 warm), and
the ordering rate went from 70.9 to 89.2 steps/s. Wall-clock was 6.96 minutes
against the laptop's ~7.

Two lessons, both cheap to forget: **a cold cache inflates the GPU stages five
to sevenfold**, so never benchmark a fresh machine on its first run; and when a
ratio contradicts the hardware, the measurement is wrong, not the hardware.

#### What that means for the matrix

Projected from the 48 committed runtimes rather than estimated:

| | |
|---|---|
| Laptop, serial (recorded) | **21.8 h**, mean 27.2 min/run (observed wall-clock 19.9 h) |
| Lab PC B, serial (x 1.21) | 26.4 h |
| Lab PC B, 16 workers, one per part+size | **3.7 h** |
| Lab PC B, 16 **worker copies** + work queue | **~1.7 to 2 h** |

The 3.7 h is a critical path, not a throughput limit: `ramp80_s` and `ramp90_s`
each need 3.7 h for their three slopes while the fastest worker finishes in 21
minutes, because slopes of one part cannot run concurrently without overwriting
each other's `data/` files.

**That limit is removable, and cheaply.** The working tree is 35 MB, so each
worker can have its own copy — 16 copies is 560 MB against 931 GB free. Then
nothing is shared, all 48 runs are independent, and a work queue keeps every
worker busy: bounded by total work over 16, about 1.7 h, or 2 to 3 h with
contention. Against the laptop's 19.9 h that is roughly **10x**.

It is also the simpler code: no partitioning by part and size, no sequencing of
slopes, just N directories and a queue. **This is the design to build.**

**Do not commit anything under `reports/` from the lab machine** until reports
record which machine produced them (corrections, section 5). A run there
silently overwrites the committed cell for that part and slope, and one
lab-measured cell inside a laptop-measured baseline is exactly what 3.9 warns
against.

Setup pain already solved, all documented in `README.md`:

1. PowerShell blocks scripts by default → `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
2. Neither Miniconda nor Blender adds itself to PATH → `scripts\fix_tool_paths.ps1`.
3. conda's default channels demand Terms of Service acceptance → conda-forge.

---

## 4. Repository conventions

- **Branch:** P0 is on `claude/new-session-l8g46d`. `main` is untouched and
  holds the golden baseline commit. P5.4 is on `claude/wonderful-hypatia-iwzai1`,
  which starts from the P0 branch's last commit (`84bae2e`), so merging it
  brings P0 along. Never push elsewhere without asking.
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

**Phase P0 is complete**, baseline included. The next work is P2, the
overhang-aware field: the project's actual contribution.

| Task | Status |
|---|---|
| P0.1 Dev environment and packaging | **Done**, verified on the laptop |
| P0.2 Golden baseline | **Done**. Deterministic; captured, committed and passing |
| P0.3 Test harness and CI | **Done**, verified on the laptop |
| P0.4 Taichi arch helper (`ti_env`) | **Done**. All 23 `ti.init` calls route through `ATOM_TI_ARCH`. The CPU golden run — its last exit criterion — is answered below: the hash differs, by design |
| P0.5 Machine profiles | **Done**. Golden test passed after it: G-code byte-identical |
| P0.6 Tilt contracts and conventions | **Done**. `atom.tilt`, `atom.contracts`, `docs/conventions.md` |
| P0.7 Benchmark meshes | **Done**. 36 meshes, 9 parts x 4 sizes |
| P0.8 Overhang metrics | **Done.** 48 runs at metrics v2; see section 7b |
| P5.4a Toolpath viewer | **Built**, pulled forward at the operator's request (`plan_corrections.md` 2.11). Laptop check pending |
| P5.4b Bed-motion animation | **Built and checked on the laptop** (Play and smooth playback confirmed by the operator). Side-by-side view deferred until P2 |
| P5.4 UI | **Qt window built** (`tools/viewer_qt.py`): side panel, timeline, toggle switches, dropdown. Needs `conda install -c conda-forge pyside6 pyvistaqt` once; falls back to the classic window without it. Laptop check pending |

**450 unit tests pass; 13 skipped** (the `pipeline` and `benchmark` tiers, plus
the 7 viewer display tests, which skip where there is no display; under
`xvfb-run` with PySide6 and pyvistaqt installed those 7 run too). In the
session container the Qt tests need `pip install pyside6 pyvistaqt` and
`apt-get install libegl1 libxkbcommon-x11-0 libxcb-cursor0` (plus the other
`libxcb-*` libraries Qt lists). The Play fix
(`plan_corrections.md` 4.13) is Windows-specific and needs the laptop to
confirm it.

### Verification status

The laptop verifications have passed:

| Check | Result |
|---|---|
| `pytest --run-pipeline tests/test_golden.py` | 3 passed in 457 s, after the P0.5 vendored edit |
| `python tools/atomize.py data/param/ramp60_xs.json` | ran normally; the generated meshes are pipeline-valid |
| `ATOM_TI_ARCH=cpu pytest --run-pipeline tests/test_golden.py` | **Hash differed**, by design — see below |

The golden baseline is committed (`tests/golden/calibration_cube.stats.json`,
`calibration_cube.toolpath.npz`), so it no longer lives on one machine.

**Re-run the golden test after every vendored edit.** About 7.5 minutes.

#### The CPU golden run, and why it did not match (P0.4's last exit criterion)

P0.4 asked for the golden test with every stage forced onto the CPU. It ran on
2026-09-23 (19 min 54 s) and the SHA-256 **differed**. The difference is not
float noise: **+2.44 % G1 moves, three fewer deposition runs, twenty fewer fan
toggles, 8 mm less extrusion.** The CPU backend produced a different toolpath.

The reason, in one line: `ATOM_TI_ARCH=cpu` moves the *iterative field solvers*
off CUDA (`order_atoms` was always on the CPU); their last-bit differences pass
through a threshold in `extract_explicit_atoms`, and the greedy planner then
amplifies a few flipped atoms into a different path. Nothing is broken. Full
measurement and reasoning: `docs/plan_corrections.md` **3.9**;
`tests/golden/baseline.md` carries the same table.

**The three things a later session needs to take from it:**

1. The golden hash is valid **only on the stock backend mix**. `test_golden.py`
   now skips the hash when `ATOM_TI_ARCH` is set and checks invariants plus a
   6 % / 2 % drift tolerance instead.
2. **The 48-run baseline matrix is unaffected.** `run_baseline_matrix.ps1` never
   sets `ATOM_TI_ARCH`, so every run used the same mix as the golden capture.
3. **Runs from a CPU-only machine cannot be pooled with the laptop's.** This
   bears directly on the plan to use university CPU machines (section 3): a
   stock-vs-overhang-aware comparison must be measured on **one** machine, or
   the backend difference will be credited to the contribution. Always record
   the backend beside a reported number.

### The immediate next step

**P0 is finished, baseline included.** The next work is **P2**, the
overhang-aware orientation field — the project's contribution. Everything it
depends on is in place: the tilt contracts (P0.6), the conventions document, the
benchmark parts, validated metrics, and a baseline that says exactly what is
wrong with the current field.

Start with **P2.1** (document the orientation-field pipeline and the analytic
tilt bound), then **P2.2** (the overhang constraint itself). P2.5 re-runs this
same matrix with the flag on and compares, so keep the reports committed.

Two things to re-run after any change to the metrics or to a vendored file:

```powershell
python tools/overhang_report.py --reanalyse          # re-score, seconds
pytest --run-pipeline tests/test_golden.py           # ~7.5 min
```

**Before any long run, do one short one.** Two of the three metric flaws were
caught this way:

```powershell
python tools/overhang_report.py data/param/ramp45_xs.json --max-slope 7
```

`ramp45_xs` at `max_slope 7` should report about **0.2 % unsupported** and
**printable: True**. A 45-degree overhang is one any 3-axis printer manages, so
that is the case whose answer is known; if it moves, something has regressed.

**An interrupted matrix resumes.** Check what is outstanding, then continue:

```powershell
python tools/overhang_report.py --status xs s
.\scripts\run_baseline_matrix.ps1 -Sizes xs,s -Resume
```

**A metric change no longer costs a re-run.** Every run archives its toolpath to
`reports/toolpaths/` (gitignored, ~220 MB for a full matrix), and `--reanalyse`
re-scores every archived run in seconds. This recovered 39 of 48 runs after the
metrics changed mid-study.

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
| `overhang_metrics.py` | The three measurements the contribution is judged by: effective overhang angle, unsupported deposition, maximum tilt used. numpy + scipy only. Validated against a known case: a 45-degree overhang reads 0.24 % unsupported and printable. |
| `tilt.py` | Tilt angles, rotations and limits. Pure numpy. `rotate_toward` is the operation P2.2 performs. |
| `contracts.py` | `MachineToolpath`: a toolpath plus the machine state it implies. Runs the IK over every point and marks failures rather than aborting, which is what P3.3, P4 and P5.4 are built on. Verified against the golden toolpath: 46 773 points, 0 unreachable, and the Z/U/V maxima match the written G-code exactly. |
| `ti_env.py` | One switch for the Taichi backend across every stage. |
| `bed_motion.py` | The bed's rigid pose for a machine state, recovered from three `forward` calls; bed corners and ball joints from the profile. Behind the viewer's machine view. Found the 0.14-degree gap in 1.9. |
| `toolpath_view.py` | Everything the toolpath viewer shows that can be tested without a display: visible segments, the colour modes, the shell/infill guess, the unsupported mask (exactly the P0.8 metric). numpy + scipy only. |

### New tools (`tools/`)

| Tool | Purpose |
|---|---|
| `gcode_stats.py` | G-code to a stable JSON record (hash, counts, axis ranges, extrusion). The golden comparison. |
| `make_benchmarks.py` | Generates the benchmark meshes and parameter files; prints estimated runtime before writing. |
| `viewer_qt.py` | The Qt window around `visualize_5ax.Viewer`: side panel (colour dropdown, switches, Z clip, camera presets, current point, notes) and timeline (transport buttons, scrubber, speed). Playback on a Qt timer. |
| `visualize_5ax.py` | **Look at the output.** Opens the Qt window when available. A slicer-style 3D preview of a toolpath `.npz` or the G-code: print-order slider, Z clip, colour by tilt / tilt direction / bead size / feed / unsupported / shell-infill, STL overlay, nozzle cone, Play/Pause with adjustable speed, and a machine view where the bed tilts under a fixed nozzle. `--screenshot` saves a PNG. G-code is mapped back through the forward kinematics and paired with its `.npz` (matches within 0.0001 mm on the golden cube). |
| `overhang_report.py` | Runs a part at a chosen `max_slope`, measures it, writes a JSON report, and archives the toolpath. `--summarize` builds the comparison table; `--reanalyse` re-scores every archived run against the current metrics in seconds. |

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
- `reports/baseline_overhang.md` — the P0.8 comparison table and runtime
  scaling, regenerated by `--summarize`.

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

## 7b. The P0.8 baseline result (final, 2026-09-23)

48 runs, 8 parts x 3 slopes x 2 sizes, all at metrics v2.
`reports/baseline_overhang.md`, the per-run JSON and
`reports/matrix_progress.csv` are committed.

### Result 1: stock Atomizer's overhang limit is 45 degrees

Exactly the classic planar 3-axis limit. A 5-axis machine performing like a
3-axis one on overhangs.

| Part | ms 7° | ms 15° | ms 30° |
|---|---|---|---|
| `ramp45` | ✅ 45° / 0.2 % | ✅ 45° / 0.2 % | ❌ 51° / 3.1 % |
| `ramp50` | ❌ 50° / 4.6 % | ❌ 50° / 4.1 % | ❌ 67° / 8.4 % |
| `ramp60` | ❌ 60° / 4.7 % | ❌ 65° / 4.9 % | ❌ 90° / 26.8 % |
| `ramp70` | ❌ 70° / 9.8 % | ❌ 80° / 16.5 % | ❌ 90° / 26.9 % |
| `ramp80` | ❌ 83° / 17.8 % | ❌ 90° / 25.8 % | ❌ 90° / 24.9 % |
| `ramp90` | ❌ 90° / 24.7 % | ❌ 90° / 24.6 % | ❌ 90° / 25.5 % |
| `tshape` | ❌ 90° / 24.4 % | ❌ 90° / 24.4 % | ❌ 90° / 24.5 % |
| `twin_domes` | – no overhang | – no overhang | – no overhang |

Effective angle / unsupported deposition near overhangs, size `s`. Size `xs`
agrees within a couple of percentage points throughout, so the result does not
depend on part size.

### Result 2: more tilt budget makes overhangs worse, one degree for one

The worsening of the effective angle **equals the tilt used**:

```
theta_eff = theta_geo + tilt_used      (capped at 90 degrees)
```

Measured over the 18 runs that used more than 2 degrees of tilt and were not
already horizontal: mean absolute deviation **0.57 degrees**, correlation
**r = 0.992**.

So the field's tilt points almost exactly *away* from the overhang — not
randomly, systematically opposite. `ramp60` at `max_slope 30` spends 29.7
degrees of tilt turning a 60-degree slope into a horizontal ceiling.

The sharpest form of it: **`ramp45` passes at `max_slope` 7 and 15, and fails
at 30.** Raising the tilt budget turns the one printable overhang into an
unprintable one. Upstream's 5.5–7 degree defaults hide this; the harm only
appears once there is a budget to misuse.

### What this means for P2

The mechanism is not subtly wrong, it is sign-wrong, and the relationship is
exact. The same magnitude of tilt aimed correctly would turn `ramp60`'s
60 degrees into **30**, well inside the threshold. P2's job is to change where
the tilt points, not how much of it there is.

### Measured runtime scaling

`order_atoms` grows as **points^1.5** (mean exponent 1.50 over 24 part/slope
pairs, range 1.39–1.61). Not published for Atomizer as far as we can tell.

| Size | Volume vs `s` | Per run | 24-run matrix |
|---|---|---|---|
| `s` (50 mm) | 1x | 42 min | 17 h (measured) |
| `m` (75 mm) | 3.4x | ~4.4 h | ~105 h (4 days) |
| `l` (100 mm) | 8.0x | ~16 h | ~384 h (16 days) |

`l` is out of reach for a repeated matrix; `s` is the practical ceiling.

### How this run went, and what made it survivable

The matrix was interrupted twice — a spontaneous restart 12 hours in, then
PowerShell being closed — and lost about 40 minutes of work in total.

* Each run writes its report the moment it finishes, so completed work is never
  in flight.
* Each run archives its toolpath, so `--reanalyse` recovered **39 of 48 runs in
  seconds** after the metrics changed, rather than 13 hours of re-slicing.
* `metrics_version` stops a stale report from masquerading as current, and
  `-Resume` skips only genuinely current results.

Three flaws in the metric were found and fixed before this run
(`plan_corrections.md` 4.5, 4.6, 1.8); 3.7 records the validation.

## 7c. Two sessions ran in parallel, and have been merged

Worth knowing, because the branch history shows it and the numbering of the
corrections moved.

For about a day two Claude Code sessions worked on the same branch family: one
finished **P0.8** (the 48-run matrix, the resume machinery, metrics v2) and one
built **P5.4** (`tools/visualize_5ax.py`, `tools/viewer_qt.py`,
`src/atom/toolpath_view.py`, `src/atom/bed_motion.py` and their tests). They
touched disjoint code; only the two living documents overlapped. Merged on
2026-09-23 in commit `b88d897`, on `claude/new-session-l8g46d`.

**One renumbering to be aware of.** Both sessions added a correction `4.12`.
The P0.8 session had already shifted 4.7 through 4.12 down by one when it
inserted a new 4.7; the viewer session's VTK-timer entry is therefore now
**4.13**, and its citations in `tools/viewer_qt.py` and in correction 2.11
follow it. Nothing else was renumbered, so any other reference either session
wrote still points where it did. If an older note cites "4.12" for the VTK
timer, it means 4.13.

**The lesson for the next parallel pair:** the code merged without a single
conflict, and the only friction was two people numbering a list. If two
sessions run again, have each append its corrections under a session-specific
heading and renumber once at merge time, rather than both editing the same
counter.

**Also relevant:** `pyproject.toml` gained an optional `gui` extra (PySide6,
pyvistaqt). Without it the viewer falls back to the classic pyvista window, and
CI installs neither, so `tests/test_viewer_qt.py` skips there.

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
