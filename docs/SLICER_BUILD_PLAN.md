# 5-Axis Slicer — Detailed Build Plan (for Claude Code)

> **Version:** v3.3 · 2026-09-23 · **P0 closed; P5.4 built early; next phase is P1 + P2.** The 48-run baseline finished at metrics v2 and settled both questions v3.2 left open: stock Atomizer's support-free limit is **45°**, and its tilt points almost exactly away from overhangs (`θ_eff = θ_geo + tilt_used`, r = 0.992). The viewer (P5.4a/b + a Qt window) was built while the matrix ran. New corrections folded in: the kinematics realise a requested direction only to ~0.14°, and the golden hash is valid only on the backend mix that captured it. Source: `docs/plan_corrections.md`, `docs/handoff.md` and `reports/baseline_overhang.md` at commit `a003735d`. Changes in Appendix G.
> **Audience:** Claude Code, working in the team's fork of Atomizer on GitHub. A human teammate (the "operator") reviews each task's commit and runs anything that needs the GPU laptop or the printer.
> **Companion doc:** "5-Axis Slicer: Revised Build Plan" (Claude Doc) holds the high-level view: phase outcomes, inputs, dependencies and testing strategy. This file holds the task-level instructions.

## What "continuous reorientation" means here (read once)

Stock Atomizer **already** tilts the real bed continuously, point by point: every toolpath point has a tool orientation, and `kinematics3z.inverse(position, normal)` turns it into three lead-screw heights. Two things keep that tilt small and aimed at the wrong target today:

1. **The field only listens to top surfaces.** In `fff3.init_spherical_direction_field_from_sdf`, the only constraints are the first layer (forced up) and low-curvature *top* surfaces ("ceilings", within `CEIL_MAX_ANGLE`). The floor/overhang branch is commented out (`fff3.py` L92, L101), even though `FLOOR_MAX_ANGLE = 1°` is still defined and looks live on a grep (L23). Nothing reads it, so on an overhang the tilt is whatever the smoothing happens to propagate there.
2. **The tilt budget is a per-part setting.** `max_slope` in each param JSON (examples use 5.5°–30°), capped at 50° by the nozzle cone and by the bed's mechanical limit (30° on the reference machine).

v3 makes the field **overhang-aware** (tilt toward overhangs, ramped in early enough) and lets it use the **full reachable tilt**, which can vary with position. The main limit: no slicer can beat the hardware. With a 45° overhang rule, the steepest printable overhang ≈ 45° + usable tilt. At 30° that is ~75°. A fully horizontal overhang (90°) needs ≥ 45° of usable tilt at that spot. That is a mechanical question (gate M2), not a software one.

## 0. How to use this file (read first, Claude Code)

1. Work **one task at a time** (task IDs look like `P2.3`). Each task has: *Goal · Files · Steps · Tests · Done when*.
2. Before starting a task, check its **Depends on** line. If a dependency is not merged into `main`, stop and say so.
3. If a task touches a **gate** (`M*`, `E*`, `HW`, `D*` — see §3), **stop and ask the operator**. Never pick a value for a gated item yourself.
4. **Never invent machine numbers.** Bed geometry, clearances, temperatures and firmware commands come from the operator or from a file already in the repo. If they are missing, use the `reference` machine profile (upstream Atomizer values) and say that you did.
5. **Vendored Atomizer files** (everything that came from `xavierchermain/atomizer`) may only be edited where a task says so. Every such edit must keep the golden tests (`tests/golden/`) green.
6. If what you find in the code contradicts this plan, **stop and report** the contradiction with file/line references instead of working around it silently.
7. **Record every contradiction or deliberate deviation in `docs/plan_corrections.md`**, the living corrections log. Mark items that need a plan change as **PLAN EDIT**. The operator folds those into the next plan version.
8. Read §0.4 (standing hazards) before writing any Taichi kernel, G-code parser, machine constant or "is it committed?" check. Every item there was hit for real in P0.
9. **Known-answer check before any long run and before any number goes in a table.** Run `python tools/overhang_report.py data/param/ramp45_xs.json --max-slope 7` (~7 min). It must report about **0.24 % unsupported** and **printable: True**: a 45° overhang is one any 3-axis printer manages. If it moves, something regressed; stop and find out why before the long run. Also chase any figure that **does not move** when the thing it depends on changed. Both habits caught real metric flaws in P0.8 (plan_corrections 3.7, 4.6).
10. **Record the backend and machine beside every reported number**, and never pool runs from different backends or machines. Forcing the stages onto one backend changes the toolpath structurally, not just its last digits (plan_corrections 3.9): +2.4 % G1 moves, three fewer deposition runs on the golden cube. A stock run measured on one machine and an overhang-aware run on another would credit the difference to the contribution.
11. **Version any metric whose meaning changes**, separately from the file layout (`metrics_version`, plan_corrections 4.7), so a resumed or partially re-run matrix can tell current results from stale ones.

### 0.1 Branch and commit protocol (as adopted in P0)

- **One working branch** for the session (currently `claude/new-session-l8g46d`), **one commit per task**. `main` is not touched by Claude Code; the operator merges. This replaces v3's branch-per-task + PR-per-task rule, at the operator's choice: the session is restricted to one branch, and one review surface is easier to follow.
- A task may span several commits only if the task text says so.
- Commit subject: `P?.? <short title>`. The commit body (and the report to the operator) uses this template:

```markdown
## Task
P?.? — <title>

## What changed
- ...

## Tests
- Added: <test files / test names>
- Ran here (CPU): `pytest -m "unit"` → <N passed>
- **Laptop run needed:** yes/no
  - Command(s): `...`
  - Expected: <what the operator should see>, and roughly how long it takes (see §0.2 compute budget)

## Gates touched
- none | <gate id + what is still open>

## Deviations from plan
- none | <what and why> (also added to docs/plan_corrections.md)
```

### 0.2 Where tests run

| Marker | Runs where | Who runs it | What it covers |
|---|---|---|---|
| `unit` (default) | Anywhere, CPU only, each test < 2 s | Claude Code, CI | Pure Python/numpy logic, small Taichi kernels on `ti.cpu` |
| `pipeline` | Operator's laptop (RTX 3050, CUDA) | Operator, on request in the task report | Runs real Atomizer stages on small meshes (calibration cube, box) |
| `benchmark` | Operator's laptop | Operator, at phase exits | Full runs on T-shape / twin domes (may take long) |
| `hw` | The printer | Operator + team | Manual checklists in §P7/§P8, not pytest |

- `pipeline` and `benchmark` tests are **skipped unless** `pytest --run-pipeline` / `--run-benchmark` is passed (implemented in P0.3).
- Claude Code must never mark a phase complete based only on `unit` tests when the phase's exit criteria list a `pipeline` or `benchmark` test: it asks for a laptop run and waits for the result.
- The operator's laptop is Windows. Give laptop commands for **PowerShell**, run inside the conda env `atomizer` (e.g. `conda run --name atomizer --no-capture-output python ...`).

**Compute budget (measured in P0.8, `docs/handoff.md` §7b).** Pipeline runs are **CPU-bound**: `order_atoms` runs on `ti.cpu`, takes **86 %** of stage time on the calibration cube, and was not faster on CUDA. Its cost grows as **points^1.5**, not linearly (mean exponent 1.50, range 1.39–1.61 over 24 part/slope pairs), so estimates that assumed linear scaling were about 2× too low. The first baseline (48 runs: 8 parts × 3 slopes × sizes `xs` + `s`) took **19.9 h**. Plan laptop work in these tiers:

| Tier | What | Time | Use it for |
|---|---|---|---|
| Unit | CPU tests, tiny SDF grids | seconds | every commit |
| Re-score | `tools/overhang_report.py --reanalyse` on archived toolpaths (`reports/toolpaths/`, gitignored, ~220 MB per matrix) | seconds | **any change to a metric**: never re-run the pipeline for that |
| Known-answer | `ramp45_xs` at `max_slope 7` (§0 rule 9) | ~7 min | before every long run |
| Field-only | pipeline up to `extract_explicit_atoms`, skipping `order_atoms` (P2.0) | a fraction of a full run | iterating on the orientation field (P2) |
| Golden | `pytest --run-pipeline tests/test_golden.py` | ~7.5 min | after every vendored edit |
| Matrix `xs` | 24 runs at 30 mm, 6–10 min per run | **~2.8 h (measured)** | quick full comparisons |
| Matrix `s` | 24 runs at 50 mm, 30–88 min per run | **~17 h (measured)** | gate decisions (D0, D2) |
| Matrix `m` | 24 runs at 75 mm, ~4.4 h per run | ~105 h (4 days) | avoid; only if a gate truly needs it |
| Size `l` | 100 mm, ~16 h per run | ~384 h per matrix: **out of reach** | single showcase parts and printed benchmarks only |

Both matrix figures are measured (`reports/baseline_overhang.md`, runtime table). A matrix is resumable: `-Resume` skips only results at the current `metrics_version`, and each run appends to `reports/matrix_progress.csv`. Changing a metric costs `--reanalyse` (seconds), not a re-run. **Runs on another machine (e.g. university CPU machines) are a different computation** and belong in their own group (§0 rule 10). Always state the tier and expected time when asking for a laptop run.

### 0.3 Definition of Done (every task)

- Code + tests committed on the working branch (one commit per task) and reviewed by the operator.
- `pytest -m unit` green in CI.
- Any `pipeline` tests the task adds were run on the laptop and the result pasted into the report.
- Any contradiction or deviation found is in `docs/plan_corrections.md`.
- If the task touched a metric, the known-answer check (§0 rule 9) still passes, and archived runs were re-scored with `--reanalyse`, not re-run.
- Golden tests still green (from P0.2 onward).
- New public functions have docstrings stating units (mm, degrees vs radians) and coordinate frames.

### 0.4 Standing hazards (each one was hit in P0)

1. **No `from __future__ import annotations` in any module that defines a `@ti.kernel`** (or tests that import one). Taichi reads annotations as live objects and fails with `TaichiSyntaxError: Invalid type annotation`. Put a comment at the top of such modules saying why the import is absent. Hit twice in P0.
2. **G-code parsers must strip quoted strings first.** `M98 P"/macros/enable3Z.g"` otherwise yields a bogus `E3` word and corrupts extrusion totals.
3. **Machine constants keep their Python types.** `MAX_X_AXIS`, `MAX_Y_AXIS`, `MAX_Z_AXIS`, `BALL_TO_CORNER`, `NOZZLE_TO_GAUNTRY`, `DEPOSITON_FEEDRATE`, `TRAVEL_FEEDRATE`, `RETRACT_SPEED` are **ints**; Taichi compiles against the Python type. Write them without decimal points in profile JSON; a test asserts the types of all 21 constants.
4. **Upstream spellings are load-bearing:** `NOZZLE_TO_GAUNTRY`, `DEPOSITON_FEEDRATE`, `get_plaftorm_size`, `spherical_field_constrain_fisrt_layer_up`, and `kinematics3z.Toolpath._init_`. Never "fix" them.
5. **`tool_orientation` is `(N, 2)` spherical `[theta, phi]`**, not Cartesian: `theta` = polar angle from +Z (= the tilt), `phi` = azimuth from +X. Convert with `atom.overhang_metrics.spherical_to_cartesian` (mirrors `direction.spherical_to_cartesian`, `direction.py:309`).
6. **IK failure is `NaN` in the `offset` return value** of `kinematics3z.inverse` → `(x, y, z0, z1, z2, offset)`. A finite, non-zero `offset` is a required vertical clearance in mm (used to size the platform), **not** an error. Full rules in `docs/conventions.md` §5.
7. **Compare machine coordinates in one frame.** `toolpath_to_gcode` re-centres the toolpath on the bed, and tilt pivots about the ball joints, so re-centring changes screw heights **non-uniformly** (7–13 mm difference on the golden cube). Use `contracts.from_toolpath(..., center_on_bed=True)` / `contracts.bed_centering_offset` when comparing with written G-code.
8. **The header's purge lines extend G-code axis ranges** (fixed moves at `X0.1 Y20`/`Y200`, screws at `0.3 + Z_OFFSET`). Compare toolpath vs file with equality on maxima and an inequality on minima.
9. **`data/mesh/.gitignore` ignores `*.stl` and whitelists by name.** A new mesh is silently skipped by `git add`. Whitelist it; the guard tests catch misses.
10. **"Is it committed?" checks use `git ls-tree -r --name-only HEAD`**, not the working tree and not `git ls-files` (which lists the index).
11. **Taichi falls back to CPU silently** when a GPU arch is unavailable (logs `falling back to CPU`, reports `Arch.x64`, no exception). Any GPU check must inspect the resulting arch.
12. **The machine profile is fixed at import.** `atom.kinematics3z` bakes constants into Taichi closures on import; select the machine with `ATOM_MACHINE` *before* importing. `contracts.from_toolpath(tp, profile)` raises if `profile` differs from the imported one.
13. **New machine setup:** set `git config --global user.name/user.email` first. Without it, commits fail quietly and a later push reports "Everything up-to-date".
14. **Support is decided by the cone, not the search radius.** A deposition point is supported if earlier material lies inside the 65° half-angle cone (`SUPPORTING_REGION_CONE_ANGLE / 2`) about `−d`; the search radius (`SUPPORT_SEARCH_HEIGHTS = 2.5` × layer height) exists only to bound the neighbour search and must not bind before the cone (1.5 × height cut off at 48.2°, plan_corrections 1.8).
15. **Bed contact follows the part:** the bed supports points within one layer height of the part's **lowest deposition point**, not of z = 0. Atomizer's first layer is a conformal shell, not a plane (plan_corrections 4.5).
17. **`conda run --name X python -c` cannot take a multi-line script** ( `NotImplementedError: Support for scripts where arguments contain newlines` ). It has already caused two bugs. Use a script file, or a tool flag whose **exit code** carries the answer (`overhang_report.py --check-done PART SLOPE`), which also survives a `Tee-Object` pipeline (plan_corrections 3.1a).
18. **The golden SHA-256 holds only on the backend mix it was captured on.** `test_golden.py` skips the hash when `ATOM_TI_ARCH` is set and checks invariants with a 6 % / 2 % drift tolerance instead (plan_corrections 3.8, 3.9).
19. **Decode subprocess output as UTF-8 explicitly** (`encoding="utf-8", errors="replace"`, plus `PYTHONIOENCODING=utf-8` for the child). The Windows locale codec raised `UnicodeDecodeError` inside a reader thread and would have destroyed the diagnostic of a genuinely failing stage (plan_corrections 3.10).
16. **Sample surfaces densely:** subdivide mesh faces to ≤ one deposition width (`trimesh.remesh.subdivide_to_size`, in the tool, not in `overhang_metrics`) before searching near them. Face centroids alone covered ~12 % of a ramp's overhang (plan_corrections 4.6).

---

## 1. Facts about the codebase this plan relies on

Checked against `xavierchermain/atomizer` `main` on 2026-09-19 and corrected against the team fork during P0 (2026-09-21). Rows marked ✱ were changed in v3.1.

| Fact | Where | Why it matters |
|---|---|---|
| The pipeline is a chain of CLI scripts called by `tools/atomize.py` via shell commands: `process_for_atomizer.py` (Blender remesh) → `obj_to_bpn.py` → `bpn_to_sdf.py` → `sdf_to_isdf.py` (infill) → `compute_tool_orientations.py` → `sdf_df_to_layers.py` → `compute_tangents.py` → `align_atoms.py` → `extract_explicit_atoms.py` → `order_atoms.py` → `smooth_toolpath_point.py` → `tesselate_toolpath_orientations.py` → `add_platform.py` → `toolpath_to_gcode.py` (→ `ratrig_to_craftware.py`) | `tools/atomize.py` ~L150–185 | v1 plan missed the last four stages. v3 keeps the stock chain and only edits the orientation-field stage. |
| Machine constants are module-level globals at the top of `src/atom/kinematics3z.py` (`BALL_2DPOS_*`, `RAIL_ANGLE_*`, `BALL_Z`, `Z_OFFSET`, `MAX_TILT_ANGLE_DEG`, `MAX_*_AXIS`, `BALL_TO_CORNER`, `NOZZLE_TO_GAUNTRY`, feed/retract values) | `kinematics3z.py` L6–26 | Moved into a machine profile in P0.5. |
| `HEADER`/`FOOTER` are f-strings evaluated **at import time**, with temperatures hard-coded (`M190 S55`, `M104/M109 S210`) and RepRapFirmware-style commands (`G32`, `M98 P"/macros/enable3Z.g"`) | `kinematics3z.py` L28–62 | Firmware dialect is an open question (gate E1). If the team's firmware is Klipper, `M98 P"..."` does not exist there. |
| `KinematicTaichi.inverse(position, normal)` computes (z0, z1, z2) from a point **and its tool orientation**. The bed already tilts continuously, point by point, in stock Atomizer. | `kinematics3z.py` L141–345 | Continuous bed reorientation already exists end to end. v3 changes *where the tilt is aimed* (P2), not the kinematics. |
| `toolpath_to_gcode()` **re-centres the whole toolpath on the bed** (`center_toolpath`) and aborts the whole file if any point fails IK ("Fatal Error: collision found!") | `tools/toolpath_to_gcode.py` | Position-dependent reachability (P2.3) must be computed in the re-centred frame; IK failures need a diagnostic tool (P3.3). |
| ✱ `toolpath3.Toolpath` stores `point (N,3)`, `travel_type`, **`tool_orientation (N,2)` spherical `[theta, phi]`**, `width`, `height`, `point_count`, `platform_height`, saved as `.npz`-style dicts | `src/atom/toolpath3.py` (`Toolpath.allocate`), `direction.py:309` | `theta` *is* the tilt. Anything needing a Cartesian direction converts (hazard 5). Base of the `MachineToolpath` contract (P0.6). |
| `infill_period = 8` and `shell_thickness = 2` are hard-coded inside `sdf_to_isdf()` | `tools/sdf_to_isdf.py` L34–36 | Threaded through params in P1.3. |
| ✱ Stages call `ti.init` at import. GPU: `bpn_to_sdf`, `compute_tool_orientations`, `sdf_df_to_layers`, `compute_tangents`, `align_atoms`, `extract_explicit_atoms`, `add_platform`, `toolpath_to_gcode`. CPU: `obj_to_bpn`, `sdf_to_isdf`, `order_atoms` (with `kernel_profiler=True`), `smooth_toolpath_point`, `tesselate_toolpath_orientations`. Verified on Taichi 1.7.4: a GPU request with no GPU **falls back to CPU silently** | `tools/*.py`; `atom.ti_env` (P0.4) | `order_atoms` is 86 % of runtime and CPU-bound, so long runs are CPU-bound (§0.2). `ATOM_TI_ARCH` overrides the arch. |
| Sample inputs exist: `data/param/calibration_cube.json` + `data/mesh/calibration_cube.stl`, `dome.json`, etc. | `data/` | Calibration cube = golden baseline part. |
| Atomizer's tool-orientation field is initialised in `fff3.init_spherical_direction_field_from_sdf`: first layer (`z < layer_height`) forced up; low-curvature **top** surfaces within `CEIL_MAX_ANGLE` constrained to their normal; the floor/overhang branch is **commented out** (`is_floor` L92, `if is_ceiling or is_floor` L101), while `FLOOR_MAX_ANGLE = 1°` is still **defined and re-assigned at runtime but read by nothing** (L22–23, `compute_tool_orientations.py`). Then `SphericalMultigridAligner.align()` + 32 smoothing passes. Max tilt = per-part `max_slope` (examples 5.5°–30°), capped at 50° by the nozzle cone | `fff3.py` ~L64–100, L19–23; `tools/compute_tool_orientations.py` | Overhangs get no constraint today, which is why the tilt stays small there. P2 adds an overhang constraint (the core v3 work). |
| `toolpath_to_gcode.calculate_feedrate` scales feed by (distance in X,Y,Z,U,V,E) / (tool-tip distance) | `tools/toolpath_to_gcode.py` | Same as Open5x's `F′ = F·d/l`, so it already exists. Verified by a test in P3.4. |
| `tesselate_toolpath_orientations.py` subdivides segments so the orientation changes ≤ `degree_angle_max_diff` (1.0° default in `atomize.py`) | `tools/atomize.py` L80, L181 | Matches the triple-Z paper's ≤ 1° advice. Keep it on for large tilts. |
| `kinematics3z.Toolpath` defines `_init_` (single underscores), not `__init__` | `kinematics3z.py` L65 | Upstream quirk. Don't "fix" it (hazard 4). |
| ✱ `kinematics3z.inverse` returns `(x, y, z0, z1, z2, offset)`; **`offset` is NaN when unreachable**, otherwise a non-negative required vertical clearance (mm) that `get_plaftorm_size` maxes over the toolpath | `kinematics3z.py`; `docs/conventions.md` §5 | The only failure signal: no flag, no exception, and no indication of *which* check failed (matters for P3.3). |
| ✱ `forward()` imposes **no** X-then-Y tilt decomposition: it rotates about an axis ⟂ ball0→ball1 (matching z0−z1), then about that axis once rotated (matching z0−z2), then about +Z from the slot constraints | `kinematics3z.py` (`forward` closure) | `(tilt_a, tilt_b)` is a **reporting convention we define** (X then Y applied to +Z), not something the code constrains. |
| ✱ The pipeline is **deterministic** (two runs, identical statistics) | `tests/golden/baseline.md` | Golden comparisons use the G-code SHA-256 directly. |
| ✱ The golden baseline is this fork's `main` (`e7b71ea8…`), which already carries the triple-Z G-code generation and infill work | `tests/golden/baseline.md` | v3's "unmodified upstream" was wrong for this fork. |
| ✱ Blender 5.2.1 LTS runs `process_for_atomizer.py` unchanged (README lists 4.4/4.5) | P0 run | Blender is invoked once per pipeline run. |

---

## 2. Coordinate and naming conventions (fixed in P0.6, used everywhere; details in `docs/conventions.md`)

- Units: **mm** for length, **degrees** at every public API and in JSON, radians only inside functions.
- **Machine frame** `M`: X/Y as the CoreXY head moves, +Z up (away from the bed at zero tilt). Same frame `kinematics3z` uses. P0.6 writes down the exact definition from the code in `docs/conventions.md`.
- **Tool orientation**: stored per toolpath point as spherical `[theta, phi]` (`toolpath3.Toolpath.tool_orientation`, shape `(N,2)`); `theta` from +Z, `phi` from +X. The Cartesian unit vector `d = (cos φ sin θ, sin φ sin θ, cos θ)` is derived when needed. **Tilt** of a point = `theta`. **Bed state** = (z0, z1, z2) from `kinematics3z.inverse(position, d)`; `offset` NaN = unreachable.
- **Tilt parameters** `(tilt_a_deg, tilt_b_deg)` (reporting and the reachability map only): **defined** as rotation about machine X then machine Y, applied to +Z (`atom.tilt`, `docs/conventions.md`). This is our convention; `forward()` uses a ball-geometry decomposition of its own, and nothing in the code needs to match X-then-Y.
- **Overhang angles:** geometric `θ_geo` = angle between surface and vertical (0° wall, 90° flat ceiling); effective `θ_eff = 90° − arccos(n · (−d))` (defined in P0.8).
- **Benchmark part names:** meshes and param files are `<part>_<size>` (`ramp60_xs`, `tshape_s`, …) with sizes `xs/s/m/l` = 30/50/75/100 mm. A bare part name in this plan (`ramp60`) means "that part, at the size the task's tier calls for" (§0.2); name the size explicitly in commands and reports. Print sizes for P7/P8 are the operator's call.
- New modules live in `src/atom/` (package `atom`), new CLIs in `tools/`, tests in `tests/`. In the Claude Code container, `atom` can't be pip-installed (Python 3.11 there vs the `<3.11` pin), so run tests with `PYTHONPATH=src python3 -m pytest`.

---

## 3. Gates (stop-and-ask points)

| ID | Type | Question | Owner | Blocks |
|---|---|---|---|---|
| **M1** | External (mechanical) | Final bed geometry: ball-pivot XY positions, rail angles, ball Z, Z offset, axis travel, ball-to-corner, nozzle-to-gantry | Mechanical | P6.1 only |
| **M2** | External (mechanical) | **Tilt budget, now the most important mechanical input.** (a) Max tilt on both axes **at the same time**, and is the limit a cone or a box? (b) Can the design exceed 30°? A fully horizontal overhang needs ≥ 45° of usable tilt where it is printed. (c) Does the reachable tilt depend on position/height, as on Multipole's Archer ("limited when the extruder is close to the bed, sharp angles further out")? | Mechanical | P6.1; informs D0 |
| **M3** | External (mechanical) | 3D clearance envelope of gantry/hotend/frame (STEP or simple boxes) | Mechanical | P6.2 (P4 and P2.3 use a proxy meanwhile) |
| **E1** | External (electronics) | Firmware and how the three Z screws are exposed (Z/U/V?). What should enable/disable-3Z do? Evidence now points to **RepRapFirmware on a Duet**: Atomizer's header is RRF, Open5x runs RRF 3.1.1 on a Duet 2, Multipole ships Duet 3 + RRF. Klipper would need a new dialect. | Electronics | P1.5 final templates, P6.3 |
| **HW** | External | Printer assembled and wired | Mech + Elec | P7, P8 |
| **D0** | Team decision | **Tilt budget and benchmark geometry.** Inputs: P0.8 (**complete**: support-free limit 45°, `θ_eff = θ_geo + tilt_used`), P2.1's analytic bound (45° + usable tilt), and M2. Decide: the overhang angles the benchmarks target (e.g. give the T-shape crossbar an angled underside if usable tilt < 45°), and whether to ask mechanical for more tilt. Taken end of week ~2, re-checked after P2.5 and when M2 arrives. | Team | P2.5 targets, P5.3, P8 |
| **D1** | Team decision | `max_overhang_deg` (placeholder 45°) and priority when a cell is both a top surface and near an overhang | Team | P2.2 tuning |
| **D2** | Team decision | **Plan B trigger.** If by end of week ~5 the overhang-aware field does not pass P2.5 on `ramp60_s`/`ramp70_s` (within the tilt budget), choose (note: each full `s` matrix costs ~17 h, so book the D2 run at least two days before the decision): keep iterating, fall back to the v2.1 pose design (Appendix D), or accept stock Atomizer + reduced scope | Team | P3–P5 scope |
| **D3** | Team decision (tuned on HW) | Max tilt rate (deg per mm of travel, deg/s) and tilt acceleration, balancing adhesion/part swing against speed | Team | P7.4 tuning (code uses parameters with defaults) |

Rule: for D-gates, code takes the value as a **parameter** with a placeholder default, and the default is marked `# GATE D?: placeholder, team decision pending` in code and `"_gate": "D?"` in JSON examples.

---

## 4. Phase map and parallel lanes

```
Lane A (pipeline / kinematics)   P0 ✅ (P0.8 matrix run pending) ──► P1 ──► P3 (large-tilt pipeline) ──┐
                                  │                                                         │
Lane B (orientation field)        └──► P2 (overhang-aware continuous field) ────────────────┼──► P5 ──► P7 ──► P8
                                  │                                                         │     ▲
Lane C (motion safety + viz)      └──► P4 (after P1.4) ─────────────────────────────────────┘     │
                                                                                                  │
Gated side track                  P6 (our machine + firmware, whenever M1/M2/M3/E1 arrive, after P1) ┘
```

| Phase | One-line outcome | Depends on | Lane | Est. |
|---|---|---|---|---|
| P0 Foundation & contracts ✅ | Done 2026-09-21: env, golden baseline, harness + CI, machine profiles, tilt/toolpath contracts, 36 benchmark meshes, overhang metrics tooling. **Pending:** running the P0.8 matrix (compute only) | — | A | done (matrix: 1 night) |
| P1 Kinematics & G-code toolchain | Kinematics proven by tests; temps/infill as params; G-code validator; feed-rate compensation and tilt tessellation verified | P0 | A | 1 wk |
| P2 Overhang-aware continuous orientation field | Atomizer's tool-orientation field tilts toward overhangs, ramps in early enough, and uses the full reachable tilt; measured against the P0.8 baseline | P0.6–P0.8 | B | 3 wk |
| P3 Pipeline at large tilts | Layers, alignment, ordering, smoothing and IK all work with steep, fast-changing fields; motion rates within machine limits | P1, (P2 for final runs) | A | 1.5 wk |
| P4 Motion safety for continuous tilt | Every move (printing and travel) checked for bed/part vs gantry and nozzle vs printed-part collisions, in simulation | P1.4, P0.6 | C | 1.5 wk |
| P5 Integration | One command: STL → validated, safety-checked continuous 5-axis G-code + report + tilt animation; flag-off regression = golden. **P5.4 (viewer) already built** | P2, P3, P4 | all | ~0.5 wk left |
| P6 Our machine & firmware | Our geometry, tilt budget, clearances, firmware in place; suite re-run | P1 + M1/M2/M3/E1 | side | 0.5–1 wk once inputs exist |
| P7 Hardware bring-up | Dry runs, calibration, first continuous-tilt prints | P5, P6, HW | — | 2 wk |
| P8 Validation benchmarks | Thesis results: overhang-aware vs stock Atomizer vs planar, on ramps, T-shape, twin domes (+ infill) | P7 | — | 1 wk |

**Critical path:** P0 → P2 → P5 → P7 → P8 ≈ 9 weeks from P0 start; P0 is closed (2026-09-23) and P5.4 is already built, so the next work is P1 + P4 in parallel, with P2 starting once D0 is recorded. **Compute is now a scheduling constraint:** every full matrix at size `s` costs ~17 h (a full day) of laptop time, so P2 iterates in the field-only tier (P2.0) and only gate decisions use full matrices. P3 and P4 run beside P2: P3 starts with stock Atomizer at `max_slope = 30` (already steep fields) and switches to P2 fields as they land.
**Main risk:** P2 is research. It edits Atomizer's orientation-field internals and has no reference implementation. Gate D2 (week ~5) decides whether to continue, fall back to poses (Plan B), or cut scope.
**Real bottleneck:** one operator reviews every commit and does every laptop run. Run **at most two Claude Code sessions at a time**. Lane B needs the most laptop runs; pair it with Lane C (mostly CPU/synthetic).

---

## P0 — Foundation & contracts ✅ COMPLETE (2026-09-21)

**Status (2026-09-23): P0 is closed, P0.8 included.** All eight tasks built and verified on the operator's laptop; 451 unit tests pass, 13 skipped. The baseline matrix is complete at metrics v2 (48 of 48; `--reanalyse` recovered 39 of them from archived toolpaths after the metric fixes, so only 9 had to be re-run). P0.9a and P0.9b are closed (see below). **P5.4 (the viewer) was built early** while the matrix ran — see the P5 section. The next phase is **P1 and P4 in parallel, then P2 once D0 is taken**; D0 now has everything it needs. The golden test passes in ~457 s (G-code byte-identical after the P0.5 vendored edit), and its forced-backend companion is a separate, tolerance-based test (hazard 18).
**Where it lives:** GitHub `AbdelrhmanOmar-AO/5AX3D_printer_slicer` (renamed from `atomizer`; the old name redirects), branch `claude/new-session-l8g46d`. `docs/handoff.md` is the cold-start brief, `docs/plan_corrections.md` the corrections log, and `docs/conventions.md` covers frames, IK and failure semantics.

This section is now an **as-built record**. Later tasks should build on what exists here, not on the v3 task text (which is kept in `SLICER_BUILD_PLAN_v3.md` for history).

| Task | As built | Deviations from the v3 text (see `docs/plan_corrections.md`) |
|---|---|---|
| P0.1 Env & packaging | conda env `atomizer`, Python 3.10.21, created from **conda-forge** by `scripts/setup_laptop.ps1`; `scripts/fix_tool_paths.ps1` puts Miniconda/Blender on PATH; `requirements-dev.txt` incl. `scipy`, `trimesh>=4.0,<6.0` | conda instead of `venv` (Taichi needs Python < 3.11; conda provides 3.10); conda-forge because Anaconda's channels need ToS acceptance/licence; Blender 5.2.1 LTS verified |
| P0.2 Golden baseline | `tests/golden/calibration_cube.stats.json`, `calibration_cube.toolpath.npz` (the `_platform` toolpath), `baseline.md` (environment, stage timings, regression history), `tools/gcode_stats.py` (strips quoted strings) | Baseline is this fork's `main` `e7b71ea8…`, not upstream; pipeline is deterministic, so the SHA-256 is compared directly |
| P0.3 Harness & CI | `pytest.ini`, `tests/conftest.py` (`--run-pipeline`, `--run-benchmark`, `ti_cpu`), `.github/workflows/unit-tests.yml`, `scripts/run_pipeline_tests.ps1` | — |
| P0.4 `ti_env` | `atom.ti_env`; all 23 `ti.init` calls route through `ATOM_TI_ARCH` | Verified: a GPU request without a GPU falls back to CPU silently (hazard 11). `order_atoms` on CUDA is not faster |
| P0.5 Machine profiles | `atom.machine_profile`, `config/machines/reference.json` (verified), `ours.json` (all PLACEHOLDER, gates M1/M2) | Integer constants kept as ints (hazard 3) |
| P0.6 Contracts | `atom.tilt` (incl. `rotate_toward`), `atom.contracts.MachineToolpath` (IK over every point, NaN-`offset` points marked invalid, `center_on_bed`, `bed_centering_offset`), `docs/conventions.md` | Tool orientation is spherical `(N,2)`; the `(a, b)` tilt convention is **defined**, not confirmed; `from_toolpath` validates the profile rather than selecting it. Verified on the golden toolpath: 46 773 points, 0 unreachable, Z/U/V maxima match the G-code |
| P0.7 Meshes | `atom.benchmark_meshes`, `tools/make_benchmarks.py`; **9 parts × 4 sizes = 36 meshes**, named `<part>_<size>.stl` with `xs/s/m/l` = 30/50/75/100 mm principal dimension; depth 0.45×, height 0.60× length; T-shape `underside_angle_deg` default 90 (D0) | Four sizes instead of one ≤ 40 mm size, to measure how cost scales; non-cubic because `order_atoms` cost scales with volume. Parts: `box`, `ramp45…ramp90`, `tshape`, `twin_domes` |
| P0.8 Overhang metrics | `atom.overhang_metrics` (`effective_overhang_angles(face_normals, face_centres, face_areas, toolpath, …)`, `unsupported_deposition`, `max_tool_tilt`, `spherical_to_cartesian`), `tools/overhang_report.py` (+ `--summarize`), `scripts/run_baseline_matrix.ps1` | Takes mesh arrays, not a mesh object (numpy + scipy only); measures `<part>_smoothed.npz` (after ordering and smoothing, before tessellation and platform); matrix is **24 runs** per size (v3 said 18 in one place). **v3.2 metric definitions:** support radius 2.5 × height so the cone decides (1.8); bed contact anchored to the lowest deposition point (4.5); faces subdivided to ≤ one deposition width before sampling (4.6); validated on `ramp45_xs` at 7° = 0.24 % unsupported, printable (3.7). Every run archives its toolpath to `reports/toolpaths/`; `--reanalyse` re-scores them in seconds |

### P0.8 results (final, metrics v2, 2026-09-23) — the baseline P2 is measured against

Full table in `reports/baseline_overhang.md`; both sizes agree within a couple of points, so the result does not depend on part size. Cells below are worst effective overhang / unsupported near overhangs / tilt actually used, at size `s`.

| Part | max_slope 7° | 15° | 30° |
|---|---|---|---|
| `ramp45_s` | ✅ 45° / 0.2 % / 0.7° | ✅ 45° / 0.2 % / 0.8° | ❌ 51° / 3.1 % / 6.3° |
| `ramp50_s` | ❌ 50° / 4.6 % / 0.7° | ❌ 50° / 4.1 % / 0.8° | ❌ 67° / 8.4 % / 17.1° |
| `ramp60_s` | ❌ 60° / 4.7 % / 0.7° | ❌ 65° / 4.9 % / 4.8° | ❌ 90° / 26.8 % / 29.7° |
| `ramp70_s` | ❌ 70° / 9.8 % / 0.7° | ❌ 80° / 16.5 % / 9.7° | ❌ 90° / 26.9 % / 20.6° |
| `ramp80_s` | ❌ 83° / 17.8 % / 3.0° | ❌ 90° / 25.8 % / 10.2° | ❌ 90° / 24.9 % / 14.6° |
| `ramp90_s`, `tshape_s` | ❌ 90° (already horizontal) | ❌ 90° | ❌ 90° |
| `twin_domes_s` | – no overhang to measure | – | – |

**Two results, both for the thesis:**

1. **Stock Atomizer's support-free limit is 45°** — `ramp45` passes, `ramp50` fails — which is exactly the classic planar 3-axis limit. A 5-axis machine performing like a 3-axis one on overhangs.
2. **Raising the tilt budget makes overhangs worse, one degree for one:** `θ_eff = θ_geo + tilt_used` (capped at 90°). Over the 18 runs that used more than 2° of tilt and were not already horizontal: mean absolute deviation 0.57°, r = 0.992. The tilt points almost exactly **away** from the overhang — systematically, not randomly. Sharpest form: **`ramp45` passes at 7° and 15° and fails at 30°**; raising the budget turns the one printable overhang into an unprintable one. Upstream's 5.5–7° defaults hide it.

This sets up P2 precisely: the mechanism is **sign-wrong**, not subtly wrong, and the relation is exact, so the same magnitude of tilt aimed correctly turns `ramp60`'s 60° into 30°. P2's first target is therefore `ramp50` and `ramp60` at a 30° budget.

**P0.9a (verify the tilt direction) — closed.** The r = 0.992 fit over 18 runs, plus the viewer's machine view, confirm the direction; no sign or azimuth mix-up survives that relation. **P0.9b (no-overhang verdict) — closed:** reports carry `verdict.assessable`, the summary prints "– no overhang", and unassessable parts no longer count against the conclusion.

**Runtime, measured:** `order_atoms` dominates as expected; per-run times are in the report's runtime table (size `xs` 6–10 min, size `s` 30–88 min). The points^1.5 scaling from v3.2 stands.

### Next action

1. **Take gate D0 to the team.** Everything it needs exists: the baseline above, the tilt bound (45° + usable tilt), and the open question M2. Decide the benchmark geometry (the T-shape underside angle) and whether to ask mechanical for more than 30°.
2. **Start P1 (Lane A) and P4 (Lane C).** Both are unblocked. P1.1 now also has to check the physical bed pose (see the task), and P1.6 (turn off the `order_atoms` profiler) is worth doing before the next long run.
3. **P2 begins once D0 is recorded.** Its target is set by the baseline: `ramp50` and `ramp60` at a 30° budget, judged on the same parts, thresholds and machine.

---

## P1 — Kinematics & G-code toolchain (machine-agnostic)

**Outcome:** kinematics proven correct by tests, temperatures and infill exposed as parameters, and a G-code validator that every later phase uses as its safety net. All against the `reference` profile; our own numbers come in P6.
**Depends on:** P0. **Lane:** A. **Parallel inside:** P1.1, P1.2/P1.3 and P1.4 are independent.

### P1.1 Kinematics test suite
- **Files:** `tests/test_kinematics3z.py` (**new**; there is no placeholder in this fork).
- **Steps / tests (`unit`, `ti_cpu` fixture):**
  1. **FK→IK round trip:** a seeded random grid of 200 (x, y, z0, z1, z2) inside the profile limits whose `forward()` result is valid → `inverse()` recovers (z0, z1, z2) within 1e-3 mm.
  2. **IK→FK round trip:** points and normals with total tilt ≤ `MAX_TILT_ANGLE_DEG − 1°` → FK gives back position (1e-3 mm) and normal (1e-4 rad).
  3. **Zero tilt:** normal = +Z → z0 = z1 = z2 (to tolerance).
  4. **Limits:** tilt of `MAX + 2°`, X beyond `MAX_X_AXIS`, a nozzle-bed collision case → `offset` is NaN (already established in P0; `docs/conventions.md` §5). Also assert the opposite: a reachable point that needs clearance returns a finite, **positive** `offset` and is **not** treated as a failure.
  5. **Physical pose check (new in v3.3, plan_corrections 1.9):** the IK maps a requested build direction to the bed normal by flipping x and y, and `forward` flips them back, so the round trip in step 2 agrees with itself **by construction** and cannot detect the real error. Build the bed's pose with `atom.bed_motion` (three `forward` probes) and require `R @ d` to be within a stated tolerance of +Z. Measured gaps: 0.0003° at 5.5° of tilt, 0.08° at 25°, 0.14° at 30°; `tests/test_bed_motion.py` pins it below 0.2°. This is far under the 1° tessellation step, so it does not matter for printing — but it must be measured, not assumed, and the kinematics maths is left unchanged.
  6. **Monotonicity:** increasing tilt toward a fixed azimuth changes (z0, z1, z2) monotonically in the expected direction (sign check). Build tool directions from spherical `[theta, phi]` as the toolpath stores them (hazard 5), and use `atom.tilt` only for reporting (a, b), since `forward()` has no X-then-Y decomposition.
- **Done when:** all green in CI. If a round trip fails, **report it; don't change the kinematics maths.** Note in the test module that a forced backend (`ATOM_TI_ARCH`) changes results structurally (hazard 18), so kinematics tolerances are stated per backend.

### P1.2 Templated header/footer + temperatures
- **Files:** `src/atom/gcode_templates.py` (new), `src/atom/kinematics3z.py` (edit: keep `HEADER`/`FOOTER` names as defaults built from the templates), `tools/atomize.py` (edit `Parameters`: `bed_temp=55`, `nozzle_temp=210` optional JSON keys), `tools/toolpath_to_gcode.py` (edit: optional `--bed-temp/--nozzle-temp` args, defaults unchanged).
- **Steps:** `make_header(profile, bed_temp, nozzle_temp)` / `make_footer(profile)` pick the template by `profile.firmware_dialect`. The `rrf` template must produce text **byte-identical** to today's with the defaults. The `klipper` template is a stub that raises `NotImplementedError("GATE E1")` until P1.5/P6.3.
- **Tests:** `unit`: rrf header with defaults == upstream string (stored as a fixture); temperatures appear where expected. `pipeline`: golden green.

### P1.3 Infill parameters
- **Files:** `tools/sdf_to_isdf.py` (edit: optional CLI args `--infill-period`, `--shell-thickness`, defaults 8 / 2), `tools/atomize.py` (edit: JSON keys → command line).
- **Tests:** `unit`: argument parsing and defaults. `pipeline`: golden green with defaults; a run with `infill_period=12` gives a different total extrusion (sanity check that the parameter reaches the stage).

### P1.4 G-code validator ★ used by P3, P4, P5, P7
- **Files:** `src/atom/gcode_check.py`, `tools/validate_gcode.py`, `tests/test_gcode_check.py`, `tests/fixtures/gcode/*.gcode`.
- **Steps:**
  1. Streaming parser for `G0/G1` with X Y Z U V E F (modal: missing words keep their last value), comments, `M`-codes. **Strip quoted strings before reading words** (hazard 2); reuse the tokeniser from `tools/gcode_stats.py` rather than writing a second one.
  2. Checks, each with a stable ID, returning a JSON report `{ok, violations: [{id, line, detail}], stats}`:
     - `AXIS_RANGE`: X/Y within `[0, MAX_X/Y]`; Z/U/V within the profile's screw range. The header purge lines are real moves and must pass too (hazard 8).
     - `TILT_LIMIT`: bed tilt implied by (Z, U, V), through FK or a closed-form plane fit through the three ball points, ≤ `MAX_TILT_ANGLE_DEG` under `tilt_limit_shape`.
     - `NAN`: no NaN/inf.
     - `FEED`: every F in `(0, profile max]`.
     - `EXTRUSION`: with `M83` relative, no single E > a threshold (param); retract/prime pairs balanced.
     - `STRUCTURE`: header and footer present; the 3Z enable macro appears before the first U/V move.
  3. The CLI exits non-zero if not ok; `--json` prints the report.
- **Tests (`unit`):** one small fixture file per violation that triggers exactly that ID; a clean fixture passes; line numbers are reported correctly. `pipeline`: the golden calibration-cube G-code passes with zero violations.
- **Done when:** golden G-code validates clean.

### P1.7 Stamp provenance on every report (small, unblocks cross-machine work)
- **Why:** §0 rule 10. The 48 baseline runs all used the stock backend mix, but nothing in the files says so, and the team may buy time on university CPU machines.
- **Files:** `tools/overhang_report.py` (and any tool that writes a report): add a `provenance` block — machine name, OS, `ATOM_TI_ARCH` (or "stock mix"), the arch each Taichi stage actually reported, Taichi version, `ATOM_MACHINE` profile, git commit, `metrics_version`.
- **Steps:** backfill the 48 existing reports with the known values (laptop, stock mix, CUDA) rather than re-running them; `--summarize` prints a warning when a table mixes provenances.
- **Tests (`unit`):** the block is present and complete; `--summarize` warns on a mixed table; backfilled reports validate.
- **Done when:** every report in `reports/` carries provenance and the summary warns on mixing.

### P1.5 Firmware dialect templates (partly gated by E1)
- **Steps:** ask the operator for the E1 answers. If still open, commit only the `rrf` path plus a `docs/firmware.md` listing the E1 questions (firmware choice, axis names for the three screws, what enable/disable-3Z must do, homing/bed-calibration command). No macro files are written until E1 is answered.
- **Done when:** either the E1 answers are recorded and the templates + `firmware/enable3Z.*` / `disable3Z.*` exist, or `docs/firmware.md` lists the open questions and the task stays open (it does not block P2–P5).

### P1.6 `order_atoms` profiler switch (cheap speed-up, golden-guarded)
- **Why:** `order_atoms` is 86 % of runtime and passes `kernel_profiler=True` to `ti.init`, which has a cost. Removing it cannot change the output, and the golden SHA proves that.
- **Files:** `tools/order_atoms.py` (vendored edit: route `kernel_profiler` through an env var `ATOM_TI_PROFILER`, default **off**; keep any profiler printing behind the same switch).
- **Tests:** `unit`: the env var parsing. `pipeline`: golden SHA unchanged; record the new `order_atoms` time in the `tests/golden/baseline.md` regression table.
- **Done when:** golden passes and the measured speed-up (or none) is recorded. If it is faster, re-estimate the matrix times in §0.2 and `docs/handoff.md`.

**P1 exit criteria:** kinematics suite green; validator merged and passing on the golden output; temps/infill params reach their stages with golden unchanged at defaults.

---

## P2 — Overhang-aware continuous orientation field (core contribution)

**Outcome:** Atomizer's own tool-orientation field, and therefore the real bed tilt, leans into overhangs early and far enough to make them printable without supports, uses the full reachable tilt, and keeps top-surface quality. Everything sits behind one flag (`overhang_aware`, default off), so stock behaviour stays byte-identical.
**Depends on:** P0.6 (tilt contracts), P0.7 (meshes), P0.8 (baseline + metrics). **Lane:** B. Needs laptop runs for every evaluation. **Gates:** D0, D1, D2.

### P2.0 Field-only evaluation mode (skip `order_atoms`) ★ makes P2 iteration affordable
- **Why:** a full run spends 86 % of its time ordering atoms, but the effective overhang angle and the tilt used depend only on the orientation field and the extracted atoms. Iterating P2.2–P2.4 on full runs would cost most of a day per change at size `s`. (Changes to the **metric** don't need this: use `--reanalyse`. P2.0 is for changes to the **field**.)
- **Steps:**
  1. Confirm what `extract_explicit_atoms.py` writes to `data/frame/<part>.npz`, and whether each atom's position and tool orientation (spherical or basis) can be read from it. Record the answer in `docs/plan_corrections.md` if it differs from this assumption.
  2. `tools/overhang_report.py --field-only`: run the stages up to `extract_explicit_atoms` (via `run_stages`, P5.1a, or a local copy of the command list), then compute `effective_overhang_angles` and `max_tool_tilt` on the atoms. `unsupported_deposition` needs a print order, so it is reported as "n/a (field-only)".
  3. Add a field-only column set to the `--summarize` table so field-only and full results are never confused.
- **Tests:** `unit`: the frame reader on a fixture; the report marks unsupported deposition as n/a. `pipeline`: on `ramp60_xs`, field-only `θ_eff` matches the full-run `θ_eff` within 1° (proves the shortcut measures the same thing).
- **Done when:** a field-only run of `ramp60_xs` finishes in a small fraction of the full-run time (record both).

### P2.1 Document the orientation-field pipeline + analytic bound ★ feeds D0
- **Files:** `docs/orientation_field.md` (new), `tools/tilt_bound.py` (new).
- **Steps:**
  1. Read and document, with file/line references: `tools/compute_tool_orientations.py` (init → `SphericalMultigridAligner.align()` → 32 smoothing passes with the first layer re-constrained), `fff3.init_spherical_direction_field_from_sdf` (constraints: first layer `z < layer_height` → up; boundary cells with curvature < `CURVATURE_THRESHOLD` that are ceilings within `CEIL_MAX_ANGLE` → their normal; the floor branch is commented out at L92/L101, while `FLOOR_MAX_ANGLE = 1°` stays defined at L23 and is re-assigned in `compute_tool_orientations.py` but **read by nothing**; say this explicitly so nobody mistakes it for a live constraint), how `MAX_SLOPE_ANGLE` is enforced (`fff3.py` ~L138–154), and how the direction field is consumed by `sdf_df_to_layers.py`, `compute_tangents.py` and `align_atoms.py`.
  2. Record how "constrained" cells work (`direction.constrain(state)`) and how the multigrid aligner treats them. P2.2 adds new constrained cells, so this is the main thing to understand.
  3. `tools/tilt_bound.py --max-overhang 45 --tilt 30` prints the steepest printable overhang (`max_overhang + tilt`), and for each ramp/T part says whether it is inside the bound.
- **Tests (`unit`):** the bound function (45 + 30 = 75; 90° needs ≥ 45°).
- **Done when:** the doc is merged, and the operator is asked to take the bound + P0.8 results to **D0**.

### P2.2 Overhang constraint in the field initialisation
- **Files:** `src/atom/fff3.py` (edit: new branch behind a module flag `OVERHANG_AWARE`, default False; `fff3.py` defines kernels, so **no `from __future__ import annotations`** (hazard 1)), `tools/compute_tool_orientations.py` (new CLI flags `--overhang_aware`, `--max_overhang`), `src/atom/overhang_field.py` (new; pure helper math, unit-testable without Taichi).
- **Rule for a boundary cell with outward normal `n` pointing down** (`n.z < 0`), not in the first layer, curvature below threshold:
  - geometric overhang `θ_geo` = angle between the surface and vertical (0° wall … 90° flat ceiling); only act if `θ_geo > max_overhang_deg`;
  - needed tilt `t = min(θ_geo − max_overhang_deg + margin_deg, max_tilt_here)`, `margin_deg` default 2 (GATE D1);
  - target tool direction `d*` = +Z tilted by `t` **toward the overhang's outward horizontal direction** (the horizontal part of `n`). The field is stored in spherical form, so this is simply **`[theta, phi] = [t, atan2(n_y, n_x)]`**, with no rotation maths needed (`atom.tilt.rotate_toward` gives the same result in Cartesian form for tests). This lowers `θ_eff = 90° − arccos(n·(−d))` by `t`;
  - write it as a constrained direction, exactly like the ceiling branch writes `n_i_pointing_up_sph`.
- **Conflicts:** if a cell also qualifies as a ceiling (thin features), follow `overhang_priority` (default: overhang wins, GATE D1) and log the count of such cells.
- **Tests:** **sign check first** (the stock field tilts away, see P0 findings): after a flag-on field-only run of `ramp60_xs`, the azimuth comparison from P0.9a (tool azimuth vs the overhang's outward normal) must show the tool leaning **toward** the overhang (difference ≈ 0°), and the fitted relation must read `θ_eff ≈ θ_geo − tilt_used`, the mirror of the stock baseline's. `unit` on `overhang_field.py`: a 60° face with max_overhang 45 → `[theta, phi] = [17°, azimuth of n]` (15 + 2), cross-checked against `atom.tilt.rotate_toward` + `overhang_metrics.spherical_to_cartesian`; a 90° ceiling with max tilt 30 → `t = 30`, and it flags "cannot reach 45"; walls and upward faces → no constraint. `unit` (Taichi CPU, tiny 16³ SDF of a ramp): with the flag on, the overhang cells hold `d*`; with the flag off, the field arrays are identical to stock. `pipeline`: golden green with the flag off.

### P2.3 Reachable-tilt map (position-dependent tilt limit)
- **Why:** both Multipole's Archer and the triple-Z machine can tilt less near the nozzle/gantry than further out. The field must never ask for a tilt the machine can't reach at that point, otherwise `toolpath_to_gcode` aborts the whole file.
- **Files:** `src/atom/reachability.py` (new).
- **Steps:** for a grid over the build volume (cell ≈ 10 mm) and a set of tool directions (tilt 0…`MAX_TILT_ANGLE_DEG` in 2.5° steps × 16 azimuths), call `kinematics3z.inverse()` (its built-in axis-range, endstop, nozzle-bed and bed-gantry checks surface only as **`offset` = NaN**; a finite `offset` means reachable, with extra clearance) and the P4.1 clearance model → boolean table → per cell and azimuth, the max reachable tilt. Cache to `data/reachability/<profile>.npz`. The P2.2 rule uses `max_tilt_here` from this map (trilinear lookup, conservative rounding down).
- **Note:** the map depends on where the part sits on the bed. Re-centring changes screw heights **non-uniformly** (hazard 7), so compute the map in the re-centred frame using `contracts.bed_centering_offset`, and store the offset used with the cache. Azimuths in the map follow the spherical `phi` convention; `(a, b)` columns are for reports only.
- **Tests (`unit`, reference profile):** the centre of the bed at low Z reaches ≥ the tilt the golden cube actually uses (5.53°); a point outside travel reaches nothing; the map is monotonic (if tilt t is reachable, every smaller tilt in that azimuth is too); cache round-trip.

### P2.4 Tilt ramp-in (reach the tilt before the overhang starts)
- **Why:** the field is smoothed, and the tilt has to build up gradually from the flat first layer. An overhang close to the base may not get its tilt in time.
- **Steps:** add ramp-in constraints: below each constrained overhang cell, along the build direction, place softly constrained cells whose tilt increases linearly from 0 to `t` over `tilt_ramp_mm = t / max_tilt_rate_deg_per_mm` (rate default 3°/mm, GATE D3). Then re-run the aligner. If the base is too close for the full ramp, cap `t` and log the shortfall per overhang.
- **Tests:** `unit`: ramp length maths, capped case. `pipeline`: on `ramp60_xs`, the tilt measured from the toolpath at the first overhang layer is ≥ `t − 2°`.

### P2.5 Evaluation against the P0.8 baseline ★ feeds D2
- **Steps:** iterate in the field-only tier (P2.0) on `xs` parts. Run the known-answer check (§0 rule 9) before each full run. For the D2 decision, re-run the **full** P0.8 matrix at the same size as the baseline (default `s`, ~17 h) with `overhang_aware` on, `max_slope` = machine limit, via `scripts/run_baseline_matrix.ps1` (add an `-OverhangAware` switch). Add columns to `reports/baseline_overhang.md` so stock and overhang-aware sit side by side. Also report top-surface quality: fraction of ceiling cells whose final direction is within 2° of their target (stock vs new), and the max tilt rate along the toolpath.
- **Same machine, same backend:** the comparison run must be measured on the machine and backend that produced the baseline (§0 rule 10), and its reports must carry provenance (P1.7).
- **Target set by the baseline:** stock passes `ramp45` only, and fails `ramp50` at every budget. The first milestone is `ramp50` and `ramp60` passing at a 30° budget; at 30° stock turns them into 67° and 90°.
- **Success criteria (placeholders, D0/D1):**
  - every ramp with `θ_geo ≤ max_overhang + usable tilt − 2°`: max `θ_eff` ≤ `max_overhang_deg` and < 1% unsupported deposition near the overhang;
  - ramps beyond the bound are reported as out of range, never silently "passed";
  - top-surface constraint satisfaction ≥ stock − 5 percentage points;
  - no new IK failures (every point passes `kinematics3z.inverse()`);
  - `twin_domes` is not worse than stock on any metric (it reports `n/a` for overhangs, P0.9b);
  - `ramp45` still passes at **every** budget, including 30° where stock now fails it (51°, 3.1 %): that regression is the clearest single sign of the sign-wrong mechanism, so removing it is the minimum bar;
  - the measured relation flips: `θ_eff ≈ θ_geo − tilt_used` on the same 18-run subset, with a fit at least as tight as the stock r = 0.992.
- **Done when:** the table is committed and the operator is asked to take it to **D2** (continue / Plan B / reduce scope).

### P2.6 Parameters end to end
- **Files:** `tools/atomize.py` (edit `Parameters` + command strings), `config/schema/params.schema.json`.
- New optional JSON keys: `overhang_aware` (false), `max_overhang_deg` (45, D1), `overhang_margin_deg` (2, D1), `overhang_priority` ("overhang", D1), `max_tilt_rate_deg_per_mm` (3, D3), `use_reachability_map` (true). Absent keys → stock behaviour.
- **Tests:** `unit`: parsing and defaults; `pipeline`: golden green with all keys absent.

**P2 exit criteria:** P2.5 success criteria met on the ramps inside the tilt bound, **or** a recorded D2 decision; the flag-off path is still golden-identical.

---

## P3 — Pipeline at large tilts

**Outcome:** the stages after the orientation field (layers, tangents, alignment, extraction, ordering, smoothing, tessellation, G-code) behave with steep, fast-changing fields, and the machine motion they produce stays within the screw and tilt-rate limits.
**Depends on:** P1. Starts on stock Atomizer at `max_slope = 30` (P0.8 outputs, already the steepest stock fields) and switches to P2 fields as they land. **Lane:** A.

### P3.1 Stage-by-stage stress run
- **Files:** `tools/stage_health.py` (new).
- **Steps:** for every P0.8/P2.5 run, record per stage: success/failure, runtime, and warnings; plus layer-height distribution (`toolpath.height`), gaps/overlaps where `sdf_df_to_layers` produced thin or thick layers, `order_atoms` runtime. Write `reports/stage_health.md`.
- **Tests:** `unit`: log parsing on fixture logs. `benchmark`: the report exists for all runs.

### P3.2 Layer-thickness guard
- **Steps:** flag any deposition whose height is outside `[min_layer_frac, max_layer_frac] × nominal` (defaults 0.5 / 1.5, placeholders). If P2 fields cause violations, feed back to P2.4 (a gentler ramp) before touching layer code.
- **Tests:** `unit` on synthetic height arrays.

### P3.3 IK failure diagnostics
- **Why:** `toolpath_to_gcode` aborts the whole file on the first IK failure ("Fatal Error: collision found!") with no location.
- **Files:** `tools/ik_diagnose.py` (new; don't change the vendored abort).
- **Steps:** use `contracts.MachineToolpath` (it already runs the IK on every point and marks NaN-`offset` points invalid) to list failing indices, positions and tool directions. The kernel reports **only** NaN, not which check failed. To name the check, write a numpy mirror of the individual checks in `inverse()` (axis range, endstop, nozzle-bed, bed-gantry) **without editing the vendored kernel**, and assert on a grid that the mirror's failure pattern equals the kernel's NaN pattern exactly. Report finite, positive `offset` values separately as "needs clearance", never as failures. Cross-reference with the reachability map (P2.3).
- **Tests:** `unit`: a synthetic toolpath with one out-of-range point is reported at the right index with the right check name; mirror vs kernel agreement on a 1 000-point grid; a clearance-only point is not reported as a failure.

### P3.4 Motion-rate limits
- **Facts to confirm first:** `toolpath_to_gcode.calculate_feedrate` already scales feed by (total machine distance incl. Z/U/V/E) / (tool-tip distance). That is the same compensation Open5x publishes as `F′ = F·d/l`. `tesselate_toolpath_orientations.py` already subdivides so the orientation changes ≤ `degree_angle_max_diff` (1°) per segment, which matches the triple-Z paper's ≤ 1° advice.
- **Steps:** add `max_screw_speed_mm_s` and `max_tilt_rate_deg_s` to the machine profile (placeholders, D3), a post-check `SCREW_SPEED` / `TILT_RATE` in the G-code validator, and an optional feed cap (new helper in `src/atom/motion_limits.py`, applied after `calculate_feedrate`) that lowers F on segments that would exceed them.
- **Tests:** `unit`: the helper reproduces `F·d/l` on hand-computed segments (so we know our understanding of the stock code is right); the cap lowers F only where needed; validator fixtures for the two new checks. `pipeline`: golden G-code is unchanged when caps are not hit.

**P3 exit criteria:** `reports/stage_health.md` shows no stage failures on the P2.5 runs that are inside the tilt bound; 0 IK failures; 0 screw-speed/tilt-rate violations.

---

## P4 — Motion safety for continuous tilt

**Outcome:** every move the printer will make (printing and travel) is checked in simulation: the tilted bed plus the part printed so far against the gantry/hotend, and the nozzle against material already printed.
**Depends on:** P1.4 (validator), P0.6 (tilt contracts). Works on stock toolpaths and synthetic ones, so it runs in parallel with P2. **Gate:** M3 for the real envelope; use the reference proxy until then. **Lane:** C.

### P4.1 Clearance model
- **Files:** `src/atom/clearance.py`.
- `ClearanceModel` protocol (`min_clearance(points_machine)`, `violations(points_machine)`); `ReferenceClearance(profile)`: nozzle as a cone (half-angle `toolpath3.NOZZLE_CONE_ANGLE/2`) up to `NOZZLE_TO_GAUNTRY`, plus the gantry as the half-space above it; `BoxesClearance` from `config/machines/<name>.clearance.json` (filled in P6.2).
- **Tests (`unit`):** inside/outside classification and distances against hand calculations.

### P4.2 Swept check along the whole toolpath
- **Files:** `src/atom/tilt_motion_check.py` (**new**; there is no placeholder in this fork).
- **Steps:** start from `contracts.MachineToolpath` built with `center_on_bed=True`, so the machine coordinates match the written G-code (hazard 7). Walk the machine-coordinate toolpath (x, y, z0, z1, z2). For every segment whose tilt changes by more than `check_tilt_step_deg` (default 0.5°), and for every travel move, interpolate the screw states linearly (that is what the firmware does), compute the bed pose via `forward()`, transform a proxy of the printed-so-far part (convex hull of deposited points, updated every N points for speed), and test it against the clearance model, axis ranges and tilt limits.
- **Tests (`unit`, synthetic):** a 5 mm part under a 20° sweep → clear; a part as tall as `NOZZLE_TO_GAUNTRY − 2 mm` under a 30° sweep → violation at the right segment; a zero-height travel over a part → violation; the report names the index, point and clearance. `pipeline`: 0 violations on the golden cube.

### P4.3 Nozzle vs printed material at large tilts
- **Why:** Atomizer's `order_atoms` avoids nozzle collisions using cones in each atom's frame, but it was tuned for small tilts. We verify it independently at large tilts.
- **Steps:** convert each point's spherical tool orientation to Cartesian (hazard 5). For each deposition point, test earlier deposited points (KD-tree, `scipy.spatial.cKDTree`, subsampled) against the nozzle cone placed at that point along its tool orientation, up to `NOZZLE_TO_GAUNTRY`.
- **Tests (`unit`):** a stem-and-arm synthetic case flagged when the arm is printed beside a taller stem; clear when the stem is shorter. `benchmark`: run on the P2.5 outputs and report.

### P4.4 Safe-travel post-process (only if P4.2/P4.3 find problems)
- **Steps:** if travel moves with large tilt changes fail the checks, insert retract → lift (all three screws together) → re-tilt at height → travel → lower → prime, like v2.1's transition planner but applied only to failing travels. Keep it behind `safe_travel` (default on only when violations exist) and re-run P4.2 afterwards.
- **Tests (`unit`):** the inserted sequence starts and ends at the original states and removes the synthetic violation.

**P4 exit criteria:** all synthetic safety cases behave as expected; 0 violations on the golden cube and on every P2.5 run that is inside the tilt bound (after P4.4 if needed).

---

## P5 — Integration: one command, regression, E2E simulation, visualization

**Outcome:** `python tools/atomize_5ax.py data/param/<part>.json` runs the (flagged) pipeline, P3/P4 checks, the validator and the metrics, writes G-code only if everything passes, and produces a report + tilt animation.
**Depends on:** P2, P3, P4 (P5.4 viz can start after P0.6 on stock toolpaths). **Lane:** merge point.

### P5.1 Driver + report
- **Files:** `tools/atomize_5ax.py` (new; calls the stock `atomize.py` stages via the `run_stages` helper from P5.1a below), `reports/run_<part>.md`.
- **P5.1a (minimal vendored edit):** pull the command construction in `tools/atomize.py` into `build_stage_commands(params)` + `run_stages(params, only, skip)`; `__main__` unchanged; golden green.
- **Flags:** `--resume-from <stage>` (the ordering stage is slow), `--machine <profile>`, `--no-write` (checks only).
- **Flow:** schema-validate params → stages → `ik_diagnose` → P4.2/P4.3 (+P4.4 if needed) → G-code → validator (incl. screw speed/tilt rate) → overhang metrics → report (max tilt used, tilt-rate histogram, % time tilted > 10°, estimated print time, violations: all 0).
- **Tests (`unit`):** flow order and refuse-to-write on a failing check (mocks). 

### P5.2 Regression (`pipeline`)
- `calibration_cube` with no new keys through `atomize_5ax.py` → G-code identical to golden (or within golden tolerances).

### P5.3 End-to-end simulation (`benchmark`)
- Ramps inside the bound, the T-shape (geometry per D0), and twin domes on the `reference` profile: exit 0; 0 IK, validator, and safety violations; the overhang metrics meet the P2.5 criteria; reports committed. These are the thesis numbers before hardware.

### P5.4 Visualization ✅ BUILT EARLY (2026-09-22, P5.4a/b + Qt window)
- **As built:** `src/atom/toolpath_view.py` (numpy/scipy view data), `src/atom/bed_motion.py` (bed pose from three `forward` probes), `tools/visualize_5ax.py` (the engine and the classic pyvista window) and `tools/viewer_qt.py` (a Qt desktop window: side panel, colour-mode dropdown, toggles, Z clip, camera presets, timeline with transport and speed). PySide6 + pyvistaqt are an **optional** `gui` extra; without them the classic window opens.
- **It reads** a toolpath `.npz` **or** the G-code, pairing the two through `forward` to 0.0001 mm (plan_corrections 2.12). Colour modes: print order, part/platform, tilt, tilt direction, bead width/height, feed, unsupported (the exact P0.8 metric), shell/infill guess. The machine view animates the bed tilting under a fixed nozzle, with the gantry level drawn and violations in red.
- **Still open (do these when their dependency lands):**
  - side-by-side stock vs overhang-aware — nothing to compare until P2;
  - violations are limited to bed-gantry and IK rejections; the full clearance model arrives with P4.1, then wire it in;
  - curved-layer numbers (the Z clip is flat), and saving the animation to a video file.
- **Deviations and hazards:** plan_corrections 2.11, 2.12, 4.13 (a VTK timer created before the window opens never fires on Windows — the Qt window avoids it entirely).

**P5 exit criteria:** P5.2 and P5.3 green on the laptop; the operator has watched the bed animation for each benchmark and signed off.

---

## P6 — Our machine profile & firmware (gated side track)

**Outcome:** the pipeline runs against **our** machine (real geometry, tilt budget, clearances, firmware) and every test passes against it.
**Depends on:** P1 + gates M1, M2, M3, E1. Can happen **any time after P1**, in pieces as each input arrives. Target: before P5.3 is re-run for the thesis.

- **P6.1 (M1, M2):** fill in `config/machines/ours.json` from the operator's numbers (never estimate; keep integer-typed constants as ints, hazard 3); set `status: "verified"`, `tilt_limit_shape`, and any position-dependent limits the mechanical team gives. Regenerate the reachability map (P2.3). Parametrize the kinematics and tilt suites over `["reference", "ours"]`. Re-run `tools/tilt_bound.py` and tell the operator if D0 needs revisiting.
- **P6.2 (M3):** `config/machines/ours.clearance.json` as boxes in the machine frame; unit tests that spot-check 3 points the operator gives you.
- **P6.3 (E1):** real firmware dialect templates + `firmware/` macros; update the validator's `STRUCTURE` check. Unit test: header/footer snapshot for `ours`.
- **P6.4:** switch the default `ATOM_MACHINE` to `ours`; operator runs `unit` + `pipeline` + `benchmark`; commit re-generated reports.
- **Exit:** full suite green on `ours`; the reference profile stays in the test matrix permanently.

---

## P7 — Hardware bring-up & calibration (gate HW)

**Outcome:** the real machine runs continuous-tilt G-code safely and its measured geometry is fed back into the profile.
**Depends on:** P5, P6, HW. Claude Code prepares test files and analysis scripts **before** the hardware arrives.

- **P7.0 (software, do early):** `tools/make_motion_tests.py` → G-code with no extrusion or heating: (a) home + 3Z enable/disable; (b) single-axis tilt sweeps to ±(limit − 5°); (c) cone-boundary sweep at 8 azimuths; (d) **continuous tilt trajectories** replayed from the P5.3 benchmark toolpaths on an empty bed, at 50% and 100% speed. Each file must pass the validator and P4.2. Also `tools/fit_tilt_calibration.py` (commanded vs measured tilt → least-squares corrections to the ball/rail constants, with a residual report).
- **P7.1 (hw) Dry run:** (a)→(d) with the nozzle raised by an extra safe offset. Pass: no endstop hits, no stalls/missed steps, no contact, and the bed motion looks like the P5.4 animation.
- **P7.2 (hw) Tilt calibration:** measure the real tilt at ≥ 9 commanded tilts (digital inclinometer), including the largest reachable ones. Fit and update `ours.json`. Target (placeholder): residual ≤ 0.5° RMS.
- **P7.3 (hw) Extrusion & thermal:** E-steps, retract, temperatures → profile and param defaults.
- **P7.4 (hw) First prints, in this order:** calibration cube (stock) → `ramp45` (stock) → `ramp60` overhang-aware → the steepest ramp inside the tilt bound. Tune `max_tilt_rate_deg_per_mm`, `max_tilt_rate_deg_s` and tilt acceleration against adhesion/part swing (**D3**); record them in `docs/decisions.md`.
- **P7.5 (hw) Metric check:** print `ramp60` from the stock pipeline too, and compare sagging/drooping (photo) with the P0.8 metric. If the metric disagrees with reality, tell the team before using it in the thesis.
- **Exit:** the calibration cube and an overhang-aware `ramp60` print complete with no collision and no missed steps; D3 recorded.

---

## P8 — Validation benchmarks

**Outcome:** measured, thesis-ready results comparing **overhang-aware continuous tilt** vs **stock Atomizer** vs **planar 3-axis** (same part, same material).
**Depends on:** P7.

- **Software (Claude Code):** `tools/collect_metrics.py` gathers the P0.8 overhang metrics for all three variants, max tilt used, tilt-rate stats, estimated vs actual print time (operator types in the actual), filament used, and the `order_atoms` feature-transition count → `reports/validation.md`.
- **Ramp family (hw):** the steepest ramp printed support-free by each variant; photos.
- **T-shape (hw, geometry per D0):** support material (target 0 g for overhang-aware), overhang quality (photo + 1–5 grade by two teammates), key dimensions vs CAD (placeholder ±0.3 mm), print time.
- **Twin domes (hw):** staircasing vs planar; measured transition count from `order_atoms` (replacing the illustrative 27 vs 2 from the concept visualization).
- **Infill (hw):** gyroid at 2–3 `infill_period` values: mass, visual, and a simple compression/flex test if available.
- **Exit:** `reports/validation.md` filled in, photos under `reports/img/`, reviewed by the team.

---

## Appendix A — Test inventory by phase

| Phase | Unit (CI, CPU) | Pipeline (laptop) | Benchmark / HW |
|---|---|---|---|
| P0 ✅ | 344 unit tests: imports, gcode_stats, ti_env, profile schema + types, tilt/toolpath contracts, 36 meshes + git-tracking guards, overhang metrics | golden cube (SHA, ~457 s) ✅ | P0.8 matrix (24 runs) **pending** |
| P1 | kinematics round trips/limits (NaN vs clearance), header snapshot, infill args, validator fixtures (incl. quoted-string guard), profiler env var | golden unchanged; validator on golden; P1.6 timing | — |
| P2 | frame reader (P2.0); tilt bound; overhang rule maths; tiny-SDF field test (flag on/off); reachability map; ramp-in maths; param parsing | golden with flag off; field-only vs full `θ_eff` agreement on `ramp60_xs`; `ramp60` ramp-in check | P2.5 matrix vs P0.8 |
| P3 | log parsing, layer-height guard, IK diagnostics, `F·d/l` reproduction, feed cap, new validator checks | golden unchanged when caps not hit | stage-health report |
| P4 | clearance model, swept check, nozzle-vs-material, safe-travel insertion | 0 violations on golden cube | checks on P2.5 outputs |
| P5 | driver flow + refuse-to-write; viewer suite (built: view data, bed pose, Qt window, playback timing) | flag-off regression = golden | E2E on ramps, T-shape, twin domes |
| P6 | suites parametrized on `ours`, header snapshot | full suite on `ours` | re-run E2E |
| P7 | motion-test files validate, calibration fit on synthetic data | — | dry run, calibration, first prints, stock-vs-aware `ramp60` |
| P8 | metrics collector | — | benchmark prints |

## Appendix B — What changed from v1 (still valid in v3)

1. Development decoupled from mechanical inputs (machine profiles, P0.5); our numbers slot in at P6.
2. Golden baseline first (P0.2), so every vendored edit is proven not to change behaviour.
3. Pipeline facts corrected: four late stages (`smooth`, `tesselate`, `add_platform`, `ratrig_to_craftware`); `toolpath_to_gcode` re-centres and aborts on the first IK failure; possible RRF-vs-Klipper mismatch (E1).
4. G-code validator (P1.4) as a shared safety net; infill physical validation moved to hardware phases.

## Appendix C — v3 changes (2026-09-19): pivot to continuous reorientation

**Decision:** the team wants continuous, coordinated bed reorientation that follows the geometry (like Open5x and Multipole), not discrete poses.

**What the research showed** (checked against primary sources, 2026-09-19):
- **Multipole Dynamics (Archer / MaxiSlicer):** CoreXY; bed on three ball joints (two on one side, one opposite), each on an independent rail, a near-identical concept to ours. Their site says the mount "allows for continuous changes to the orientation of the print bed", and the slicer changes layer orientation mid-print; they claim "even fully horizontal overhangs can be printed without any supports". Tilt is limited when the extruder is close to the bed and larger further out (Hackaday). Controller: Duet 3 + RepRapFirmware. No tilt numbers, algorithms or code are published; MaxiSlicer is "still in development".
- **Open5x (Hong et al., CHI EA 2022):** a 2-axis **rotary** gantry (U tilt + continuous V rotation, belt-driven) added to a Prusa i3, Duet 2 + RRF 3.1.1. It is not a lead-screw tilting bed, and its range is far larger than ours. The slicer (Grasshopper/Rhino) draws toolpaths conformally onto a user-selected surface (0.2 mm segments), with orientation from the surface normal at each point, usually printed on top of a planar-printed substrate. Useful ideas: feed compensation `F′ = F·d/l` (Atomizer already does this; verified in P3.4) and shortest-path rotation for a 0–360° axis (not applicable to lead screws).
- **Implication:** continuous reorientation on our hardware is what Atomizer already does. The gaps are (1) the field ignores overhangs and (2) the tilt budget. v3 fixes (1) in P2; (2) is mechanical (M2). Multipole's horizontal-overhang claim implies more than 45° of usable tilt somewhere, which is why M2 now asks whether our design can exceed 30°.

**Plan changes:** removed pose sampling, regions/cuts, pose assignment, per-pose atomization, pose transitions and the inter-group check (old P2–P5). Added P2 (overhang-aware continuous field), P3 (pipeline at large tilts), P4 (motion safety for every move), a reachability map for position-dependent tilt, tilt-rate limits, and gates D0–D3 redefined. P0.8 (stock baseline) is now the "before" measurement for P2.

## Appendix D — Plan B: discrete poses (v2.1)

If gate D2 finds the overhang-aware field can't meet its targets, the v2.1 pose design is the fallback: split the part with planar cuts, give each region a fixed base tilt whose "up" is the cut normal, run the stock pipeline per region, merge, and insert safe pose-change moves. The full task-level text is kept in `SLICER_BUILD_PLAN_v2.1_poses.md` (Claude outputs folder). The P0 foundation, P1 toolchain, P4 safety code and P0.8 metrics are shared, so switching loses little work. The v2.1 text predates the P0 corrections (it assumes Cartesian `tool_orientation`, branch-per-task, etc.); apply §0.4 and Appendix E before reusing any of it.

## Appendix E — v3.1 changes (2026-09-21): P0 complete, corrections folded in

Source: `docs/plan_corrections.md` on branch `claude/new-session-l8g46d`. Item numbers follow the file as of commit `84bae2e6` (2026-09-22), which renumbered section 4.

| # | Correction | Where it changed the plan |
|---|---|---|
| 1.1 | `tool_orientation` is `(N,2)` spherical `[theta, phi]`, not `(N,3)`; `theta` is the tilt | §1 facts, §2, hazard 5, P1.1, P2.2 (the target direction is now simply `[t, azimuth of n]`), P4.3, P5.4 |
| 1.2 | `FLOOR_MAX_ANGLE` is defined and re-assigned but read by nothing | Intro, §1 facts, P2.1 |
| 1.3 | The "placeholder" files don't exist | P1.1, P4.2 say "new" |
| 1.4 | P0.8 matrix is 24 runs (not 18) | P0 as-built, §0.2 |
| 1.5 | `order_atoms` is CPU and 86 % of runtime | §0.2 compute budget, §1 facts, §4 critical-path note, new P1.6 (profiler off), new P2.0 (field-only mode) |
| 1.6 | `forward()` has no X-then-Y decomposition; `(a, b)` is a convention we define | §1 facts, §2, P1.1, P2.3 |
| 1.7 | IK failure = NaN `offset`; finite `offset` = clearance, not an error | Hazard 6, §1 facts, P1.1, P2.3, P3.3 (numpy mirror of the checks to name which one failed) |
| 2.1 | Golden baseline is the fork's `main` | §1 facts, P0 as-built |
| 2.2–2.4 | conda from conda-forge; `scipy` added; `trimesh <6` | P0 as-built |
| 2.5 | 9 parts × 4 sizes, `<part>_<size>` names, non-cubic proportions | P0 as-built, §2 naming rule, §0.2 tiers |
| 2.6–2.8 | `overhang_metrics` takes arrays; `from_toolpath` validates the profile, adds `center_on_bed`; metrics read `_smoothed.npz` | P0 as-built, hazard 12 |
| 2.9 | Single working branch, one commit per task | §0.1, §0.3 |
| 2.10 | Golden archive is the `_platform` toolpath | P0 as-built |
| 3.1 | Taichi falls back to CPU silently | Hazard 11, §1 facts |
| 3.2 / 4.9 | No `from __future__ import annotations` with Taichi kernels | Hazard 1, P2.2 |
| 3.3 | Blender 5.2.1 LTS works | §1 facts |
| 3.4 / 3.6 | Pipeline deterministic; P0.5 refactor byte-identical | §1 facts, P0 as-built |
| 3.5 | `order_atoms` on GPU is not faster | §0.2, P0 as-built |
| 4.1 | `M98 P"…"` parses as an `E3` word | Hazard 2, P1.4 |
| 4.2 / 4.3 | Constant types and misspelled names are load-bearing | Hazards 3–4, P6.1 |
| 4.4 / 4.10 | `.gitignore` whitelisting; "committed" means `git ls-tree HEAD` | Hazards 9–10 |
| 4.7 / 4.8 | Re-centring is non-uniform; purge lines extend ranges | Hazards 7–8, P1.4, P2.3, P4.2 |
| 4.11 | New machines need a git identity | Hazard 13 |
| §5 | Open items: P0.8 matrix not run, gates deferred, thresholds are placeholders | P0 "Next action", §3 D0, §4 |

**New in v3.1, not from the corrections file:**
- **P2.0 field-only evaluation.** Measure `θ_eff` and tilt on extracted atoms before `order_atoms`, so each P2 iteration doesn't cost a night.
- **P1.6 profiler switch.** Turn off `kernel_profiler` in `order_atoms`, guarded by the golden SHA.
- **§0 rule 7.** `docs/plan_corrections.md` is now the standing channel for plan contradictions.

## Appendix F — v3.2 changes (2026-09-22): first baseline results, metric fixes

Source: `docs/plan_corrections.md` and `docs/handoff.md` §7b at commit `84bae2e6`.

| Item | Change | Where in the plan |
|---|---|---|
| plan_corrections 1.8 | Support is decided by the 65° cone; search radius 2.5 × height (was 1.5 ×, which cut off at 48.2°) | Hazard 14, P0 as-built |
| plan_corrections 4.5 | Bed contact anchored to the lowest deposition point, not z = 0 | Hazard 15, P0 as-built |
| plan_corrections 4.6 | Faces subdivided to ≤ one deposition width before sampling | Hazard 16, P0 as-built |
| plan_corrections 3.7 | Known-answer check: `ramp45_xs` at 7° = 0.24 % unsupported, printable | §0 rule 9, §0.2, §0.3, P2.5 |
| handoff 7b | Stock Atomizer tilts **away** from overhangs; worse with larger budget | P0 findings, new P0.9a verification, P2.2 sign check, P2.5 criterion |
| handoff 7b | `order_atoms` scales as points^1.5; `s` matrix 17 h, `m` ~4 days, `l` out of reach | §0.2 compute table, §3 D2, §4 critical path, P2.0, P2.5 |
| plan_corrections §5 | `twin_domes` reported "not printable" with nothing to measure | New P0.9b (`n/a` verdict) |
| new tooling | Toolpaths archived; `--reanalyse` re-scores in seconds | §0.2 tier "Re-score", §0.3, P2.0 |
| housekeeping | Repo renamed `5AX3D_printer_slicer`; plan_corrections section 4 renumbered | P0 "Where it lives", Appendix E |

## Appendix G — v3.3 changes (2026-09-23): P0 closed, viewer built, next phase set

Source: `docs/plan_corrections.md`, `docs/handoff.md` and `reports/baseline_overhang.md` at commit `a003735d`.

| Item | Change | Where in the plan |
|---|---|---|
| P0.8 complete (48/48, metrics v2) | Final baseline table; support-free limit **45°**; `θ_eff = θ_geo + tilt_used` (r = 0.992, MAD 0.57°); `ramp45` passes at 7°/15° and fails at 30° | P0 results section, D0 inputs, P2.5 targets and criteria |
| P0.9a, P0.9b | Both closed: the exact relation plus the viewer confirm the direction; reports carry `verdict.assessable` and "– no overhang" | P0 results section |
| plan_corrections 1.9 | The kinematics realise a requested direction only to ~0.14° at 30°; the IK↔FK round trip cannot see it | P1.1 step 5 (new physical-pose check via `atom.bed_motion`) |
| plan_corrections 3.8, 3.9 | The golden hash holds only on the captured backend mix; forcing the CPU changes the toolpath structurally (+2.4 % G1 moves, 3 fewer deposition runs) | §0 rule 10, hazard 18, P1.1 note, P2.5 same-machine rule |
| plan_corrections 3.9 consequence 4 | Every number needs its backend recorded; university CPU machines can't be pooled | §0 rule 10, §0.2, **new P1.7** (provenance block + mixed-table warning) |
| plan_corrections 3.1a | `conda run … python -c` rejects multi-line scripts; use a file or an exit code | Hazard 17 |
| plan_corrections 3.10 | Decode subprocess output as UTF-8 explicitly | Hazard 19 |
| plan_corrections 4.7 | Version metrics separately from schema; `-Resume`, `--status`, progress CSV | §0 rule 11, §0.2 |
| plan_corrections 2.11, 2.12, 4.13 | P5.4 built early (viewer + bed animation + Qt window), with three parts still open and the VTK-timer hazard | P5.4 rewritten as built |
| `reports/baseline_overhang.md` | Measured per-run times: `xs` 6–10 min, `s` 30–88 min; matrices ~2.8 h and ~17 h | §0.2 compute table |
| Sequencing | P0 closed; P1 + P4 next in parallel; P2 after D0; D0 ready to take | P0 "Next action", §4 critical path |

