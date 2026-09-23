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
