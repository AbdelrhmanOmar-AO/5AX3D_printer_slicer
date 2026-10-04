# Corrections and deviations from SLICER_BUILD_PLAN v3

The plan itself is `docs/SLICER_BUILD_PLAN.md`, currently **v3.3** (2026-09-23).

Everything found so far that contradicts the build plan, or where the
implementation deliberately departs from it. Build plan section 0 rule 6 says
to report contradictions rather than work around them silently; this is that
report.

**Keep this file updated as further items are found.** Items marked
**PLAN EDIT** should be corrected in the plan document itself, because a later
task will be written against the wrong fact.

Last updated: 2026-09-23. Phase P0 complete, P0.8 matrix included, and merged
into `main` (pull request #2). P1 and P4 are now built in two parallel
sessions: **each session adds its items only under its own heading in
section 7**, and they are renumbered into sections 1–4 once, when both
branches are merged.

---

## 1. Factual errors in the plan

### 1.1 `tool_orientation` is `(N, 2)` spherical, not `(N, 3)` Cartesian ★ PLAN EDIT

The facts table in section 1 states that `toolpath3.Toolpath` stores
`tool_orientation (N,3)`.

Evidence:

| Where | What it shows |
|---|---|
| `src/atom/toolpath3.py`, `Toolpath.allocate` | allocates `shape=(size, 2)` |
| `src/atom/kinematics3z.py`, `get_vertical_offset_kernel` | reads two components, then calls `direction.spherical_to_cartesian` |
| `src/atom/direction.py:309` | the convention: `[theta, phi]`, `theta` polar from +Z, `phi` azimuth from +X, giving `(cos phi sin theta, sin phi sin theta, cos theta)` |

Consequence: `theta` **is** the tilt from vertical, so the "tilt of a point"
needs no conversion. Anything needing the Cartesian direction must convert.

**Affects:** P0.6 (`contracts.MachineToolpath` assumes the Cartesian form), P2.2,
P4.2, P5.4. Handled in `atom.overhang_metrics.spherical_to_cartesian`, which
mirrors the Taichi function and is tested against the convention.

### 1.2 `FLOOR_MAX_ANGLE` is live, only its use is commented out ★ PLAN EDIT

The plan says "the floor/overhang constraint is commented out". More precisely:

- `src/atom/fff3.py:22` — `# FLOOR_MAX_ANGLE = MAX_SLOPE_ANGLE` (commented)
- `src/atom/fff3.py:23` — `FLOOR_MAX_ANGLE = 1.0 * ti.math.pi / 180.0` (**live**)
- `tools/compute_tool_orientations.py` — re-assigns the same live 1.0-degree value at runtime
- `src/atom/fff3.py:92` — `# is_floor = ...` (commented)
- `src/atom/fff3.py:101` — `# if is_ceiling or is_floor:` (commented)

So the constant exists and holds 1.0 degree, but nothing reads it. It looks
active on a grep and is not. P2.1 and P2.2 should say so explicitly.

**Partly wrong (P2-7):** the orientation field does not read it, but the
layer and tangent stages do.

### 1.3 The "placeholder" files do not exist ★ PLAN EDIT

P1.1 says `tests/test_kinematics3z.py` ("replace the placeholder") and P4.2 says
`src/atom/tilt_motion_check.py` ("replace the placeholder"). Neither file exists
in this fork. Those tasks **create** rather than replace.

### 1.4 The P0.8 matrix is 24 runs in one place and 18 in another ★ PLAN EDIT

P0.8 step 5: "parts `ramp45 … ramp90, tshape, twin_domes` x `max_slope in {7, 15, 30}`
= 24 runs". The tests bullet of the same task: "`benchmark`: the full 18-run
matrix". 8 parts x 3 slopes = 24. 18 would be the 6 ramps alone. **24 is correct.**

### 1.5 `order_atoms` runs on the CPU, and it dominates runtime ★ PLAN EDIT

The plan treats the pipeline as GPU work throughout and notes only that
`order_atoms` is "slow". Measured on the calibration cube (see
`tests/golden/baseline.md`), it is 331.6 s of 383.7 s of logged stage time,
**86 %**, and `tools/order_atoms.py` initialises Taichi with
`arch=ti.cpu`. Backends per stage:

| Backend | Stages |
|---|---|
| GPU | `bpn_to_sdf`, `compute_tool_orientations`, `sdf_df_to_layers`, `compute_tangents`, `align_atoms`, `extract_explicit_atoms`, `add_platform`, `toolpath_to_gcode` |
| CPU | `obj_to_bpn`, `sdf_to_isdf`, `order_atoms`, `smooth_toolpath_point`, `tesselate_toolpath_orientations` |

Consequences: long runs are CPU-bound, not GPU-bound; single-thread and
memory-bandwidth performance matter more than the GPU; and `order_atoms` also
passes `kernel_profiler=True`, which is not free and is worth measuring before
the big matrix.

---

### 1.6 `forward()` imposes no X-then-Y tilt decomposition ★ PLAN EDIT

Section 2 of the plan says the tilt parameters `(tilt_a_deg, tilt_b_deg)` are
"rotation about machine X then machine Y", and asks P0.6 to "confirm this order
against `kinematics3z.forward()`, and change it if the code implies a different
one".

Neither is possible: **`forward` uses no such decomposition at all.** It
composes three rotations about axes derived from the ball geometry
(`src/atom/kinematics3z.py`, the `forward` closure):

1. about an axis perpendicular to ball0->ball1 in the xy-plane, by the angle
   that matches `z0 - z1`;
2. about that vector once rotated, by the angle matching `z0 - z2`;
3. about +Z by a third angle solved from the slot constraints.

None of those is machine X or Y, and the sequence depends on the machine's ball
positions. The `(a, b)` pair is therefore a **reporting convention we choose**,
not one the code constrains. P0.6 fixes it explicitly as X-then-Y applied to
+Z, documented in `docs/conventions.md`, so the reachability map (P2.3) and the
reports agree with each other. The plan should say "define" rather than
"confirm".

### 1.7 The inverse kinematics signals failure as NaN, in a value named `offset`

Not contradicted by the plan, which simply never says how failure is reported.
P1.1 step 4 asks to "read the code first to learn how invalidity is signalled
(NaN? flag?)", so recording the answer here:

`kinematics3z.inverse` returns `(x, y, z0, z1, z2, offset)`. **`offset` is NaN
when the point cannot be reached.** There is no flag and no exception.

When it is not NaN, `offset` is a non-negative **required vertical clearance**
in millimetres — how far the part must be raised for the move to be safe.
`kinematics3z.get_plaftorm_size` takes the maximum over a toolpath to size the
sacrificial platform. So the same value carries two meanings, and code that
treats a non-zero `offset` as an error is wrong.

Full details, including which conditions produce NaN and which only raise
`offset`, are in `docs/conventions.md` section 5.

### 1.8 The support search radius of 1.5 x height overrides the 65-degree cone ★ PLAN EDIT

P0.8 defines a deposition point as supported when earlier material lies "inside
the cone with apex at `p_i`, axis `-d_i`, half-angle
`toolpath3.SUPPORTING_REGION_CONE_ANGLE / 2` (65 degrees), **within
`1.5 x height_i`**".

Those two conditions conflict. On a surface at angle `t` from vertical,
successive layers step `h*tan(t)` sideways, so the nearest earlier bead sits at
`h/cos(t)`. A radius of `1.5h` therefore stops reaching at
`arccos(1/1.5) = 48.2` degrees — well inside the cone, which consequently never
decides anything. The metric measures the radius, not the geometry.

Measured consequence: `ramp45_xs`, an overhang any 3-axis printer manages,
reported **16.35 %** of its deposition near the overhang as printing into air,
and everything past about 48 degrees reported 30–45 %, so no part could pass
the `< 1 %` threshold and the pass/fail column carried no information.

Two independent checks say the **cone** is the right criterion and the radius
is the accident: it is Atomizer's own `SUPPORTING_REGION_CONE_ANGLE`, and the
classic bead-overlap limit for fused filament (supported while `h*tan(t) < w`)
gives **63.4 degrees** at this part's 0.45 / 0.9 geometry, close to the cone's
65.

`SUPPORT_SEARCH_HEIGHTS` is therefore **2.5**, which stops binding at 66.4
degrees, just past the cone. The plan should specify a radius that does not
bind before the cone, or drop the radius and state that it exists only to bound
the neighbour search.

Decided with the operator on 2026-09-22.

### 1.9 The kinematics realise the requested tool direction only approximately ★ PLAN EDIT

Found while building P5.4b. `kinematics3z.inverse` turns the requested build
direction into the bed's normal by flipping its x and y
(`docs/conventions.md` section 2), and `forward` flips them back. That is
exact only if the bed does not also turn about its own normal, and the three
slot constraints do make it turn a little.

The bed's real pose can be recovered from `forward`'s positions:

* at fixed screws, a change in X or Y moves only the nozzle, so three
  `forward` calls give the bed's rotation (`atom.bed_motion`);
* that pose reproduces the screw height differences at the three ball joints
  to within 0.001 mm, up to 30 degrees of tilt.

The nozzle axis under that pose is the build direction the machine actually
produces. It differs from the requested one by:

| Tilt | Largest difference (at diagonal azimuths) |
|---|---|
| 5.5 degrees (golden cube) | 0.0003 degrees |
| 25 degrees | 0.08 degrees |
| 30 degrees | 0.14 degrees |

This is far below the 1-degree tessellation step, so it does not matter for
printing. But **P1.1's IK→FK round trip on the normal (1e-4 rad) cannot catch
it**, because both directions use the same flip and so agree with each other
exactly. P1.1 should also check the physical pose: build the pose with
`atom.bed_motion` and require `R @ d` to be within a stated tolerance of +Z.
`tests/test_bed_motion.py` pins the gap below 0.2 degrees. The kinematics
maths is left unchanged.

### 1.10 Plan v3.3 still carries stale P0 text ★ PLAN EDIT

Found when reading v3.3 against the repository after P0 was merged:

| Where in the plan | What it says | What is true |
|---|---|---|
| §0.1 and the P0 "Where it lives" line | working branch `claude/new-session-l8g46d` | P0 is in `main`; each session has its own branch (2.13) |
| §4 lane diagram | `P0 ✅ (P0.8 matrix run pending)` | the matrix is done (48 of 48, metrics v2) |
| §4 phase table, P0 row | "**Pending:** running the P0.8 matrix" | done |
| Appendix A, P0 row | "344 unit tests" and "P0.8 matrix (24 runs) **pending**" | **630** unit tests (2026-09-24); matrix done, and 48 runs rather than 24 (1.4) |
| Appendix E source line | corrections "on branch `claude/new-session-l8g46d`" | now in `main` |
| §0.4 hazard list | numbered 1–15, then 17, 18, 19, then 16 | hazard 16 is out of order |

None of these changes a task. They mislead a cold reader about what is done.

A test count in a document goes stale the week it is written. The live figure is
whatever `pytest -q` prints; `docs/handoff.md`'s header carries the last one
recorded and the date it was recorded on.

## 2. Deliberate deviations

### 2.1 Golden baseline taken at this fork's `main`, not upstream

P0.2 says to capture "at the unmodified upstream commit". The baseline is taken
at this fork's `main` (`e7b71ea89046281ccff89ec8b14429ef92418a6d`) instead,
because the fork already carries the triple-Z G-code generation and the infill
work, and those are part of the behaviour being preserved. Recorded in
`tests/golden/baseline.md`.

### 2.2 conda, not `python -m venv`

P0.1 says `scripts/setup_laptop.ps1` should create a `.venv`. It creates a conda
environment instead: `README.md` already prescribed conda, Taichi requires
Python < 3.11, and conda can produce a 3.10 interpreter on a machine with no
system Python. Confirmed necessary — `pip install -e .` on Python 3.11 is
refused by the `requires-python` pin.

### 2.3 Environment created from conda-forge

Recent conda releases refuse to install from Anaconda's own channels until their
Terms of Service are accepted (`CondaToSNonInteractiveError`), and those
channels need a paid licence for larger organisations. `setup_laptop.ps1`
defaults to `--channel conda-forge --override-channels`; `-Channel defaults`
restores the old behaviour.

### 2.4 `scipy` added to the dev dependencies

Not listed in P0.1's dependency list, but P0.8 specifies "numpy + scipy only"
for `overhang_metrics.py` and P4.3 requires `scipy.spatial.cKDTree`. trimesh
also needs it. `trimesh` is pinned `>=4.0,<6.0` rather than `<5.0`, since 5.x is
current.

### 2.5 Benchmark meshes: four sizes, non-cubic, size-suffixed names

P0.7 says "use small parts (<= 40 mm) until the operator supplies dimensions"
and names meshes `ramp45.stl` and so on. Instead, each part is generated at four
sizes (`xs`/`s`/`m`/`l` = 30/50/75/100 mm principal dimension) named
`<part>_<size>.stl`, on the operator's instruction to gather data on how
pipeline cost scales with part size.

Proportions are deliberately non-cubic (depth 0.45x, height 0.60x the length):
`order_atoms` scales with volume, so a 100 mm cube costs roughly 11x a 100 mm
long but slimmer part, for no extra information about overhangs.

### 2.6 `overhang_metrics` takes mesh arrays, not a mesh object

P0.8 signature is `effective_overhang_angles(mesh, toolpath)`. Implemented as
`effective_overhang_angles(face_normals, face_centres, face_areas, toolpath, ...)`
to honour the same task's "numpy + scipy only" constraint. A caller with a
trimesh object passes `mesh.face_normals, mesh.triangles_center, mesh.area_faces`.

### 2.7 `contracts.from_toolpath` validates the profile rather than selecting it

P0.6 specifies `from_toolpath(tp, profile)`. The `profile` argument cannot
select a machine: `atom.kinematics3z` bakes its constants into Taichi closures
at import time, so a profile passed afterwards could not take effect. Passing
one that differs from the imported profile raises, rather than silently solving
against the wrong machine. To change machines, set `ATOM_MACHINE` before
importing.

A `center_on_bed` argument was added for the same reason `bed_centering_offset`
exists (see 4.7).

### 2.8 Overhang metrics are taken from `<part>_smoothed.npz`

P0.8 says only "the toolpath". `tools/atomize.py` produces four, and
`tools/overhang_report.py` uses `_smoothed`: after ordering and smoothing, but
before `tesselate_toolpath_orientations`, which only subdivides segments so the
orientation steps stay small, and before `add_platform`, which adds sacrificial
material below the part that is not part of its geometry and would distort both
the support test and the surface sampling.

### 2.9 Single branch instead of one branch per task (superseded by 2.13)

Plan section 0.1 wants `p<phase>/<task-id>-<short-slug>` branches with one pull
request each. All work is on `claude/new-session-l8g46d`, one commit per task,
at the operator's choice: the session is restricted to that branch, and the
operator is new to git and preferred a single review surface. `main` is
untouched.

### 2.10 Golden archive is the `_platform` toolpath

P0.2 says "copy the final toolpath file". The final toolpath stage before G-code
is `data/toolpath/<part>_platform.npz` (`tools/atomize.py`), not the `_smoothed`
one.

---

### 2.11 P5.4 started early, in two parts, as `visualize_5ax.py` P5.4a/P5.4b

The operator asked for a way to look at Atomizer's output while the P0.8
matrix re-runs. P5.4 says it "can start after P0.6 on stock toolpaths", so it
has been started now and split in two:

* **P5.4a:** a static viewer. It scrubs through the print order, clips by Z,
  and colours the lines by tilt, tilt direction, bead width or height, feed
  rate, unsupported points or a shell/infill guess. It can overlay the STL and
  draw the nozzle cone. It reads the toolpath `.npz` or the G-code. Code is in
  `atom.toolpath_view` (numpy only, unit-tested) and `tools/visualize_5ax.py`
  (the window).
* **P5.4b:** the bed-motion animation. Built as a "machine view" toggle in
  the same window, together with a Play/Pause button and a speed slider (the
  operator's requests).

The operator's choices differ from the plan text in three places:

* **Z clip:** Atomizer's layers are curved and no layer number is stored, so
  the "layer slider" is a print-order slider plus a flat Z clip. Recovering
  true curved-layer numbers is left for later.
* **Shell/infill colouring:** added at the operator's request. It is a
  heuristic: Atomizer records no bead type, so it uses the infill stage's own
  rule (solid within `shell_thickness = 2` deposition widths of the surface).
* **Side by side deferred:** the stock vs overhang-aware comparison was not
  asked for in this round. It has nothing to compare until P2 exists.

**UI (operator's request after P5.4b).** The main window is now a Qt desktop
app, `tools/viewer_qt.py`. It has a side panel with a colour-mode dropdown,
toggle switches, Z clip, camera presets and the current point; a timeline bar
with transport buttons and speed; and tooltips. It embeds the same engine
(`visualize_5ax.Viewer`) through `pyvistaqt`. PySide6 and pyvistaqt are new
**optional** dependencies (`pyproject.toml` `gui` extra); without them the
classic pyvista window opens instead. Playback there runs on a Qt timer, not
a VTK one, which avoids 4.13 entirely. CI installs neither, so
`tests/test_viewer_qt.py` skips there and runs where Qt and a display exist.

P5.4b differs from the plan text in three places:

* **Only two kinds of violation are shown in red:** a bed corner above the
  gantry level, and a point the IK rejects. Both come straight from the
  kinematics' own checks. The plan's "clearance model" violations need P4.1,
  which does not exist yet. The part-so-far is drawn as the actual printed
  lines, not a convex-hull proxy.
* **The plan's "frame count = points / stride" test is replaced.** Playback
  runs live and advances by wall-clock time at the chosen speed (points per
  second). The tests check that timing instead. Saving the animation to a
  video file is not built.
* **Toolpath input solves its own screw values.** An `.npz` carries no screw
  values, so they are solved after re-centring on the bed, as
  `toolpath_to_gcode` does. For a `_smoothed` toolpath (no platform yet)
  they can differ from the final G-code's by the platform lift, and the
  window says so. Opening the G-code shows the exact values.

### 2.12 The G-code is paired with its toolpath through the forward kinematics

The viewer reads G-code by taking the moves between the header's `M83` and
the footer's `M82`. It recovers each nozzle position and tilt from X, Y, Z, U
and V with `kinematics3z.forward` (the vendored
`toolpath_to_cartesian_toolpath` kernel). It then subtracts
`contracts.bed_centering_offset` to get back to the part frame. On the golden
cube this reproduces the toolpath the G-code was written from within
**0.0001 mm**, so the reader checks that the G-code and the `.npz` belong to
the same run before it pairs them.

### 2.13 One branch per session, merged into `main` by pull request ★ PLAN EDIT

2.9 described P0: one branch for everything, `main` untouched. On 2026-09-23 the
operator merged that work (P0, P5.4 and the plan) into `main` through pull
request #2, merge commit `d2d39c3`. From then on:

* each Claude Code session works on **its own branch**, started from `main`,
  with one commit per task as before;
* the operator merges a session's branch into `main` through a pull request;
  Claude Code never merges into `main` and never pushes to another session's
  branch;
* a session that needs another session's finished work gets it by merging
  `main` into its own branch, never by merging the other branch directly.

Plan §0 rule 2 ("if a dependency is not merged into `main`, stop") therefore
now applies literally: P4 can use P1.4 once the P1 branch carrying it has been
merged into `main`.

The golden baseline commit is unchanged: `e7b71ea`, the fork's original `main`
(2.1). Merging P0 did not move it.

Plan §0.1 should describe this protocol and stop naming a single branch.

## 3. Environment facts the plan asked to establish

### 3.1 Taichi falls back to CPU rather than failing (answers P0.4 step 1)

P0.4 asks to verify what `ti.init(arch=ti.gpu)` does with no GPU. Observed on
Taichi 1.7.4: it does **not** raise. It logs `libcuda.so lib not found` and
`Arch=[<Arch.cuda: 3>] is not supported, falling back to CPU`, then reports
`Arch.x64`. A stage asking for the GPU on a GPU-less machine therefore degrades
silently and runs many times slower. Any check must compare the resulting arch,
not catch an exception.

### 3.1a `conda run` rejects a multi-line `python -c`

`conda run --name X python -c "<two or more lines>"` aborts with
`NotImplementedError: Support for scripts where arguments contain newlines not
implemented` and dumps a full conda crash report. It has now caused two bugs:
the GPU probe in `setup_laptop.ps1`, and the `-Resume` check in
`run_baseline_matrix.ps1`, where every check would have errored, no
combination would have matched, and an interrupted matrix would have re-run all
48 runs instead of the 9 outstanding — twenty hours instead of six.

Pass a script file, or give the tool a flag whose **exit code** carries the
answer (`overhang_report.py --check-done PART SLOPE`). An exit code cannot be
confused by stray output, which matters when the command runs inside a
`Tee-Object` pipeline.

### 3.2 `from __future__ import annotations` breaks Taichi kernels

That flag turns annotations into strings; Taichi reads kernel argument
annotations as live objects and fails with
`TaichiSyntaxError: Invalid type annotation`. Verified on Taichi 1.7.4. **Any
module defining `@ti.kernel` must not use it** — relevant to P2.2, which adds
kernels.

### 3.3 Blender 5.2.1 LTS works

`README.md` records 4.4 and 4.5 as tested. `tools/process_for_atomizer.py` ran
end to end under 5.2.1 LTS on the calibration cube: `bpy.ops.wm.stl_import`, the
voxel remesh, the SMOOTH modifier and `bpy.ops.wm.obj_export` with its named
arguments all still work. Blender is invoked exactly once in the pipeline, so
that single check covers every use of it.

### 3.4 The pipeline is deterministic on one backend, not across backends

Two consecutive runs of the calibration cube produced identical statistics, so
golden comparisons use the G-code SHA-256 directly rather than float tolerances.

**That holds per backend only.** Change which backend a stage runs on and the
toolpath changes — see 3.9 for the measurement. The hash is a valid check of
"nothing regressed"; it is not a valid check of "these two machines agree".

---

### 3.5 `order_atoms` on the GPU is not faster

Tested via `ATOM_TI_ARCH=cuda` (P0.4). The run was abandoned after it had far
exceeded the 457 s the same test takes on the CPU, so the result is
**"not a win", not a measured factor**. Some of the excess is one-off
compilation of every kernel on a new backend.

The structure explains it: the ordering loop is sequential (one atom per
iteration, 31 630 of them for the calibration cube, each choice depending on
all previous), and `compute_cost_and_find_best_next` returns to Python every
iteration, forcing a host synchronisation each time. Many tiny kernels
punctuated by host round-trips is the pattern GPUs handle worst. The CPU
default stands.

Untested and cheaper: `tools/order_atoms.py` passes `kernel_profiler=True`,
which is not free and could be removed with no effect on determinism.

### 3.6 The P0.5 refactor was confirmed byte-identical

The golden test passed after the machine constants moved into profiles: 3
passed in 457 s, G-code SHA-256 unchanged. This validates the working method
as well as the change — the golden baseline does catch what it was built to
catch, so later vendored edits can proceed behind it. Recorded in
`tests/golden/baseline.md` as a regression history table.

### 3.7 The overhang metric is now validated against a known case

Three separate flaws (4.5, 4.6 and 1.8) each produced confident but wrong
figures for unsupported deposition, and each was hidden by the one before it.
The sequence on `ramp45_xs` at `max_slope 7`, an overhang any 3-axis printer
manages:

| State | Unsupported near overhangs | Verdict |
|---|---:|---|
| Original | 28.70 % | not printable |
| After the bed-contact fix (4.5) | 28.70 % | unchanged — the bug was in the *overall* figure |
| After dense surface sampling (4.6) | 16.35 % | not printable |
| After the cone governs the radius (1.8) | **0.24 %** | **printable** |

0.24 % is the answer physics predicts, and it is the first passing result in
the study: before this the metric returned "not printable" for everything,
whatever Atomizer did.

**The standard worth keeping:** a metric should be checked against a case whose
answer is known before its output is put in a table. Each of these was found by
noticing a figure that did not match physical intuition and chasing it rather
than explaining it away, and twice the check that exposed it cost seven
minutes against a twenty-hour run.

### 3.8 G-code written on the CPU differs from the CUDA golden in the last digits

Running `tools/toolpath_to_gcode.py` on the golden toolpath with
`ATOM_TI_ARCH=cpu` gives a G-code with the same lines and structure as the
golden file. Axis values differ by up to about 4e-5 mm and total extrusion by
2e-5 mm. That is float32 rounding in the IK kernel on a different backend. The
SHA-256 therefore differs, so **the golden SHA check only holds on the
backend the baseline was captured on** (CUDA, on the operator's laptop).
Comparisons across backends need tolerances, not hashes.

### 3.9 Forcing every stage onto the CPU changes the toolpath, not just its digits ★ PLAN EDIT

P0.4's exit criterion asks for the golden test under `ATOM_TI_ARCH=cpu`, and
says a difference is fine to "note". It is worth more than a note.

The run (2026-09-23, operator's laptop, 19 min 54 s) **failed the hash check**,
and the differences are structural:

| Statistic | Golden (stock mix) | `ATOM_TI_ARCH=cpu` | Change |
|---|---:|---:|---:|
| Lines | 47 910 | 49 057 | +1 147 |
| `G1` | 47 837 | 49 004 | +1 167 (+2.44 %) |
| `M106` (fan toggles) | 50 | 30 | -20 |
| Total extrusion | 3231.197 mm | 3223.138 mm | -8.06 mm (-0.25 %) |
| Total retraction | 1060.0 mm | 1054.0 mm | -6.0 mm |
| Retracts / primes | 530 / 529 | 527 / 526 | -3 / -3 |
| X, Y moves | 46 777 | 47 950 | +1 173 |
| Z, U, V moves | 46 778 | 47 951 | +1 173 |
| U max | 87.042015 | 87.247643 | +0.21 mm |
| V max | 110.051544 | 109.963348 | -0.09 mm |

Three fewer retractions means the planner built **three fewer deposition
runs**: a different chaining of the atoms, not a rounding difference. Twenty
fewer fan toggles means the path crosses `z_fan_on = 2.0` mm twenty times less
often. This is a different toolpath.

**Why.** The stages do not all default to the same backend:

| Backend by default | Stages |
|---|---|
| `gpu` (CUDA on the laptop) | `bpn_to_sdf`, `compute_tool_orientations`, `sdf_df_to_layers`, `compute_tangents`, `align_atoms`, `extract_explicit_atoms`, `add_platform`, `toolpath_to_gcode` |
| `cpu` | `obj_to_bpn`, `sdf_to_isdf`, **`order_atoms`**, `smooth_toolpath_point`, `tesselate_toolpath_orientations` |

So `ATOM_TI_ARCH=cpu` moves exactly the **field solvers** off CUDA, and leaves
the planner where it already was. Those solvers are iterative
(`direction.py`, `phasor3.py`, `triphasor3.py` each run a multigrid loop) and
their kernels use `sqrt`, `atan2` and fused multiply-add, which CUDA's
libdevice and LLVM's x64 lowering do not compute to the same last bit. The
fields therefore converge to very slightly different solutions;
`extract_explicit_atoms` then applies a **threshold**
(`frame_field_filter_point_too_close_to_boundary`), so a handful of atoms fall
on the other side of it; and `order_atoms` is a **sequential greedy chain**, so
a few flipped atoms near the start redirect the rest of the path. Small input
difference, amplified by a threshold, amplified again by a greedy planner.

The planner itself is not the culprit and is not at fault: its counters
(`toolpath3.Field.insert`, `set.insert_u32`) increment in Python, sequentially,
so given the same atoms it produces the same path. Nothing here is a bug.

**Consequences.**

1. **The golden SHA-256 is valid only on the captured mix.** `test_golden.py`
   now skips the hash assertion when `ATOM_TI_ARCH` is set, and checks
   invariants plus a 6 % / 2 % drift tolerance instead
   (`test_forced_backend_stays_within_tolerance`). A tolerance loose enough to
   pass the table above would not guard anything, so the two cases are separate
   tests rather than one loosened one.
2. **The 48-run baseline matrix is unaffected.** `scripts/run_baseline_matrix.ps1`
   never sets `ATOM_TI_ARCH`, so all 48 runs used the same stock mix as the
   golden capture. The baseline is internally consistent and comparable to it.
3. **Results from a CPU-only machine cannot be pooled with the laptop's.** This
   matters for the plan to buy time on university CPU machines: a matrix run
   there is a different computation, so its numbers belong in their own group,
   or the comparison run has to be redone on the same machine. Comparing a
   stock run from one machine against an overhang-aware run from another would
   attribute a backend difference to the contribution.
4. **The plan should say which backend a reported number came from.** Every
   table in `reports/` and in the paper needs the backend recorded beside it.

**Cheap diagnostic if this needs pinning down further.** `data/frame/<part>.npz`
holds the extracted atoms; the golden run has 31 630 of them
(`tests/golden/baseline.md`). If a CPU run's count differs, the divergence is
upstream of `order_atoms`, as argued above; if it matches exactly and the
G-code still differs, the argument is wrong and the planner is where to look.
The file from the 2026-09-23 CPU run is still on the laptop.

### 3.10 The golden test lost its own failure message on Windows

The same CPU run raised, in the middle of the test, a
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d in position 49`
from `subprocess`'s reader thread. `subprocess.run(..., text=True)` decodes
with the locale codec, which is cp1252 on the operator's laptop, and one
non-ASCII byte in a stage's output is enough to raise.

It was harmless here only because the run's exit code was 0. Had a stage
failed, the exception would have destroyed `stdout` and `stderr` — the whole
diagnostic the assertion was written to print. `pipeline_stats` now passes
`encoding="utf-8", errors="replace"` and sets `PYTHONIOENCODING=utf-8` for the
child, which `run_baseline_matrix.ps1` already did for its own runs.

### 3.11 A fresh session container is missing runtime dependencies

The first unit-test run in a new session container (Python 3.11, 2026-09-23)
reported **48 failures**, all `ModuleNotFoundError: No module named 'tqdm'`.
`tqdm` is a runtime dependency in `pyproject.toml`, but `atom` cannot be
pip-installed there (the `< 3.11` pin), so its dependencies are not pulled in.
After installing them the suite gave 451 passed, 13 skipped, matching the
laptop.

It looks like broken code, not a missing package. Before trusting a failing
first run, install:

```
pip install pytest pytest-timeout numpy scipy trimesh jsonschema taichi tqdm
```

## 4. Implementation hazards found while building

### 4.1 `M98 P"/macros/enable3Z.g"` parses as an `E3` word

A naive G-code word regex reads `E3` out of the filename in the real header,
corrupting extrusion totals. `tools/gcode_stats.py` strips quoted strings before
parsing words. Any later G-code parser, notably P1.4's validator, needs the same
guard.

### 4.2 Machine constants must keep their Python types

`MAX_X_AXIS`, `MAX_Y_AXIS`, `MAX_Z_AXIS`, `BALL_TO_CORNER`, `NOZZLE_TO_GAUNTRY`,
`DEPOSITON_FEEDRATE`, `TRAVEL_FEEDRATE` and `RETRACT_SPEED` are **integers**
upstream. Taichi compiles a kernel against the Python type of each global it
captures, so `config/machines/reference.json` writes them without decimal points
and a test asserts the type of all 21 constants, not only their values.

### 4.3 Upstream name spellings are load-bearing

`NOZZLE_TO_GAUNTRY` and `DEPOSITON_FEEDRATE` are misspelled upstream and are read
by `tools/toolpath_to_gcode.py` as module attributes. `kinematics3z.Toolpath`
also defines `_init_` with single underscores. None of these may be "fixed"
without touching every caller.

### 4.4 `data/mesh/.gitignore` ignores `*.stl` and whitelists by name

A newly generated mesh is skipped by `git add` with no warning, leaving a
committed parameter file pointing at a mesh nobody else has. This happened once.
Two tests now guard it by checking `git ls-files` and `git check-ignore` rather
than the filesystem.

---

### 4.5 The bed-contact rule must follow the part, not a fixed height

The unsupported-deposition metric credits the bed with supporting the first
layer. Taking "first layer" to mean one layer height above the origin reported
**5.86 % of a plain calibration cube as printing into air**.

The cause is that Atomizer's layers are conformal, so the first layer is a
shell of finite thickness rather than a plane: on the cube its deposition
centres span z = 0.4148 upward past 0.45. A threshold of 0.45 cuts through the
middle of it and leaves the upper half with nothing beneath it but bed it is
not credited for.

Anchoring to the lowest deposition point plus one layer height gives 2.78 %,
and the residual was checked rather than assumed: all 852 remaining points have
no material in the support cone at all, spread through the print rather than
clustered at the bottom, which is the gyroid infill bridging its own voids.

The first baseline matrix (20 hours) was measured before this was found, so its
unsupported column is inflated by roughly three percentage points.

**Update (2026-09-22, found while building P5.4a).** The 2.78 % and the 852
points above were measured with the old 1.5 x height search radius. With the
2.5 x radius from 1.8, the same golden toolpath gives **0.27 % (82 points)**.
So most of those 852 points were the radius cutting off the cone, not the
gyroid bridging its own voids. The 82 that remain are clustered near the top
of the cube and around its internal features.

### 4.6 Sampling a surface by face centroid under-measures large flat faces

Both overhang metrics search for deposition points near each mesh face's
**centroid**. A ramp's overhang is two triangles covering 172 mm^2, so
searching two deposition widths around their two centroids covers about
20 mm^2 — 12 % of the surface, at two arbitrary spots.

The first baseline matrix therefore computed "unsupported deposition near
overhangs" from **108 deposition points** out of roughly 35 000. Reported as
28.70 %, it was 31 points out of 108.

`tools/overhang_report.py` now subdivides every face to at most one deposition
width before sampling (`trimesh.remesh.subdivide_to_size`). Splitting a
triangle in its own plane changes neither its normal nor the total area, both
asserted in the tests, and on `ramp45_xs` it takes the sampled region from 524
points to 7 130, a 13.6x improvement.

The metrics module stays numpy + scipy only; the subdivision lives in the tool,
which may use trimesh.

This was found because a 7-minute check before a 20-hour re-run returned
exactly the same 28.70 % as the run it was meant to correct. A figure that does
not move when the thing it depends on changes is worth chasing.

### 4.7 An overwriting run makes an interrupted re-run look complete

Each matrix run writes its report to the same path for a given part and slope,
so a re-run overwrites the previous one. When the operator's laptop restarted
12 hours into a re-run, the combinations already redone held new results and
the rest still held the **old** ones — and every file existed, so nothing
looked missing.

Reports now carry a `metrics_version` separate from `schema_version`: the file
layout did not change, the meaning of the numbers did. `--status` counts a
report from an older definition as missing, and `-Resume` skips only genuinely
current results.

Two smaller safeguards from the same incident: `load_reports` reports a
truncated file rather than raising, since a power cut can catch a write in
progress, and each run appends a row to `reports/matrix_progress.csv`, flushed
and fsynced, as a trail that survives whatever happens to the process.

### 4.8 Re-centring on the bed is not a constant offset

`tools/toolpath_to_gcode.py` re-centres the whole toolpath on the bed before
solving. It is tempting to treat that as a translation whose effect cancels in
any comparison of ranges. It does not.

Bed tilt pivots about the three ball joints, so how far a screw must travel for
a given tilt depends on where the point sits relative to them. Solving the
golden toolpath in the part frame gave screw travel 7 to 13 mm larger than the
G-code's on all three axes; solving it re-centred reproduced the G-code's
maxima exactly. `contracts.bed_centering_offset` mirrors the arithmetic and
`from_toolpath(..., center_on_bed=True)` applies it.

Anything comparing computed machine axes against written G-code must solve in
the same frame.

### 4.9 The header's purge lines extend the G-code's axis ranges

`kinematics3z.HEADER` draws two priming lines at fixed coordinates
(`X0.1 Y20`, `Y200.0`, all three screws at `0.3 + Z_OFFSET`). Those points are
not in the toolpath, so any statistic taken over a whole G-code file includes
them. On the calibration cube they set the file's X minimum, both Y extremes
and the V minimum.

They can only extend a range, never narrow it, so a comparison against toolpath
output should assert equality on the maxima and an inequality on the minima.
`tests/test_contracts.py` does exactly that.

### 4.10 `from __future__ import annotations` was hit in practice

Recorded as a hazard in 3.2, then walked into two commits later while writing
`src/atom/contracts.py`: the module had the import, and its Taichi kernel
failed with `Invalid type annotation (argument 0) of Taichi kernel:
ti.types.ndarray()`.

Modules that define Taichi kernels now carry a comment saying why the import is
absent. Affected so far: `contracts.py`, `tests/test_contracts.py`,
`tests/test_ti_env.py`, `tools/overhang_report.py`,
`tests/test_overhang_report.py`.

### 4.11 `git ls-files` reports the index, not what is committed

Two guard tests were written against the wrong notion of "tracked" and passed
while the thing they guarded was broken:

- the first checked the working tree, and passed while 36 benchmark meshes sat
  uncommitted because `data/mesh/.gitignore` had silently skipped them;
- the second used `git ls-files`, which lists staged files, and passed while
  the golden baseline had been `git add`-ed but never committed.

Both now use `git ls-tree -r --name-only HEAD`. Only a commit survives a push,
a clone, or a move to another machine.

### 4.12 A fresh Windows machine has no git identity

`git commit` fails with "Author identity unknown" until
`git config --global user.name` and `user.email` are set. The error is easy to
miss in a scrollback, and the files simply stay staged, so a later `git push`
reports "Everything up-to-date" while nothing has been committed. Worth setting
during setup on any new machine.

### 4.13 A VTK timer created before the window opens never fires on Windows

The viewer's Play button did nothing on the operator's laptop, although it
worked in every test here. The playback timer was created while the scene was
built, before `show()`. `vtkWin32RenderWindowInteractor::InternalCreateTimer`
calls `SetTimer(this->WindowId, ...)`, and `WindowId` is only set in
`Initialize()`, which pyvista first calls inside `show()`. Created earlier,
the timer belongs to no window and its messages never reach VTK. On Linux
(X11) VTK keeps its own timer list, so the same code works there, and **no
Linux test can tell the two orders apart**.

The fix is `Viewer.start_timer`: initialise the interactor, then create the
timer. `tests/test_visualize_5ax.py` now drives the real event loop, but that
only guards the playback path. The Windows behaviour has to be checked on the
laptop.

### 4.14 Two sessions built provenance independently, and one wasted a day

Not a bug in the code. A bug in how the work was split, and the most expensive
one so far.

On 2026-09-24 the P0.8 session was told that reports not naming their machine
was urgent — a run on the lab machine had just overwritten a committed,
laptop-measured cell with nothing in the file to show it. It designed and built
`src/atom/provenance.py`, wired it through `tools/overhang_report.py`, backfilled
the 48 reports and wrote 22 tests.

**Plan task P1.7 had already done all of it**, in another session, and merged it
to `main` the day before — more thoroughly:

| | P1.7, merged | The duplicate |
|---|---|---|
| Backend per stage | **All 14, recorded at run time** via `ATOM_TI_ARCH_LOG` | One `ti_arch` field for the whole run |
| Uncommitted code | `code_modified` | Not recorded |
| How the record was obtained | `source`: pipeline / skip-pipeline / backfilled | A `recorded` flag, backfill only |
| Verified on hardware | **Yes** — a laptop run caught a wrong host name in the backfill | No |
| CPU model | Not recorded | `cpu`, `cpu_logical` |

Everything but the last row is strictly better, so the duplicate was discarded
whole rather than merged. The CPU model is worth adding to P1.7's block: a host
name distinguishes machines, but `Intel(R) Xeon(R) Gold 6254` is what a reader
comparing two timings actually needs, and `platform.processor()` will not give
it (bare `x86_64` on Linux).

**What caused it, stated accurately.** Not invisibility. Every session pushes to
`origin`, so `git fetch --all` shows `main` and every other branch at any moment,
and P1.7 sat in `main` in public for a day. **The P0.8 session did not look.**

An earlier draft of this entry said "nothing in its own working tree could have
told it that P1.7 existed". That was wrong, and the wrong version is worse than
useless: it frames an avoidable mistake as an unavoidable one.

What is true is narrower. Nothing *prompts* a fetch; it has to be a habit. And a
stale document reads exactly like a current one — the P0.8 session's own
open-items table said provenance was open, which was true when it was written
and false by the time it was read.

**What to do instead**, for whoever is in this position next:

> **Before building anything that sounds like infrastructure — provenance,
> validation, a runner, a report format — fetch and look at every branch, not
> just `main`.**
>
> ```
> git fetch --all --prune
> git log --oneline HEAD..origin/main
> git for-each-ref --sort=-committerdate \
>     --format='%(committerdate:short) %(refname:short) %(subject)' refs/remotes/
> ```
>
> `main` alone is not enough: P4.1 to P4.3 exist only on
> `claude/phase-p4-build-fzqw55`, so a session checking only `main` could
> duplicate those next. The third command lists every branch with its latest
> commit, and would have caught this one in seconds.

The same applies to the plan: `docs/SLICER_BUILD_PLAN.md` is now committed, and
P1.7 is *in it*. Reading the plan's task list would have prevented this. The
P0.8 session was working from the handoff's summary of the plan rather than the
plan, and the summary did not enumerate P1's subtasks.

**Not wasted entirely.** The same merge kept four things the duplicate session
built that nothing else covers: the parallel matrix runner (4.17 and handoff
section 3), the failing-stage check (4.16), the test-file import fix (4.15), and
the display-test opt-out (4.13's neighbour, `ATOM_SKIP_DISPLAY_TESTS`).

### 4.15 No single test file could be run on its own

Found while adding the tests for 4.14. `pytest tests/test_tilt.py` failed with
`ModuleNotFoundError: No module named 'atom'` while the full suite passed, on
every test file in the repository.

`tests/conftest.py` put `tools/` on `sys.path` but not `src/`. During a full
collection, one of the `tools/` modules inserts `src/` as an import side effect,
and every module imported after it rides on that. So whether an import worked
depended on **collection order** — invisible while running the whole suite, and
broken the moment anyone runs one file, which is the normal way to work on one.

`conftest.py` now adds `src/` explicitly, next to `tools/`.

### 4.16 `atomize.py` ignores every stage's exit code

Found when the first parallel run failed. `tools/atomize.py` runs all 13 stages
with `os.system` and **checks no return code.** One early failure therefore
produces twelve more, and the only thing that finally errors is
`overhang_report.py` finding no toolpath — behind a log that is almost entirely
the later stages complaining about inputs that were never made. Diagnosing the
first parallel failure took a directory listing per stage to find the one real
cause.

The dangerous version is not that one. **In a working tree that has already run
that part, the previous run's outputs are still there.** A failed stage then
leaves a file that *looks* like its output, the pipeline carries on, and the
report is quietly wrong. A fresh worker tree turns that into a loud missing-file
error; the operator's own repository would not.

`overhang_report.run_pipeline` now checks, after `atomize.py` returns, that every
stage's artifact exists **and was written by this run**, and names the first one
that was not:

    Stage 6 tangents produced nothing: data/basis/twin_domes_xs.npz

The freshness half is the part that matters: it catches the stale-output case,
where nothing looks wrong.

The vendored `atomize.py` is left alone. Making it check its own exit codes would
be the better fix and would need the golden test re-run; this covers the same
ground from outside without touching a vendored file.

### 4.17 `compute_tangents.py` needs `data/image/0.png` for every run

`tools/compute_tangents.py` takes optional `--top_lines` and `--bottom_lines`
PNGs. With neither given — which is every benchmark part — it loads
**`data/image/0.png`** as the default and then disables the constraint. So that
committed file is a hard dependency of the whole pipeline, not of an optional
feature.

This broke the first parallel run. The runner built worker trees with a
hand-written list of inputs (`*.stl` and `*.json`), stage 6 died in every worker,
and 4.16 turned that into twelve failures.

The general lesson, having got this wrong twice within an hour — first by copying
whole folders and dragging in Blender's `.obj` intermediates, then by narrowing
to two patterns and dropping this PNG:

> **Do not enumerate the pipeline's inputs by hand.** Everything a run generates
> is gitignored, so `git ls-files -- data` *is* the list of inputs, and it stays
> correct as stages change.

`run_matrix_parallel.data_input_paths` does that. It uses `git ls-files` rather
than `git ls-tree HEAD` deliberately, and despite 4.11: that correction is about
"has this been committed", where the index lies; this asks "is this an input",
and a newly added input that is not yet committed is still one a worker needs.

### 4.18 A cache warm-up run must be the shortest job, not the longest

The parallel runner fills one Taichi cache with a single run before copying it to
every worker (4.17's neighbour: a cold cache inflates the GPU stages five to
sevenfold). That run is **alone**, so its length is pure serial time added to the
job.

It was taking the first job of a longest-first queue — which is the **longest**.
On the first real run that cost **13:21 of a 50:31 total**, a quarter of the
wall-clock. On the full matrix it would have taken `ramp90_s` at max_slope 30,
**1:27:31 spent alone with fifteen workers idle.**

Any job fills the cache equally well. It now takes the shortest.

Worth recording because the symptom is *only slowness*, and slowness on a new
machine is easy to explain away — which is exactly what happened when the same
run's cold-cache figures were first read as the machine being slow (3.9's
neighbour, handoff section 3). A test pins the choice.

### 4.19 Taichi reserves 1 GiB of host memory per process, and it is not tunable

The full 48-run matrix at 16 workers **failed 24 of 48 runs**, all within 90
seconds of the workers starting:

```
[host_memory_pool.cpp] Virtual memory allocation (1073741824 B) failed
RuntimeError: ... Virtual memory allocation (1073741824 B) failed
```

`1073741824 B` is exactly 1 GiB, and `host_memory_pool` is **RAM, not VRAM**.
Taichi commits that pool at `materialize_runtime()`, in every process. A worker
runs two Taichi processes at once — `atomize.py` and whichever stage it has
launched — and on Windows a committed reservation counts against the commit
limit whether or not it is touched.

The same machine ran **8 workers with 24 of 24 succeeding**. The 17
`FileNotFoundError: data/sdf/*.npz` entries in the logs are downstream noise
from 4.16: the stage never ran, so its output was missing.

**Taichi 1.7.4 exposes no setting for this pool.** `device_memory_GB` and
`device_memory_fraction` are the GPU side; there is no host equivalent in
`ti.init`. So **worker count is the only lever**, and `run_matrix_parallel.py`
now warns when it is set beyond what RAM has been shown to support.

**That warning is a calibration, not a calculation, and the difference matters.**
Counting Taichi's reservations alone — 1 GiB per process, two processes per
worker — gives 2 GiB per worker and would have permitted 26 on this machine. The
observation says 16 is too many. The gap is everything that is not a Taichi
reservation: Blender during stage 1, the arrays a large part actually needs, two
Python interpreters per worker, and a Windows commit limit set by RAM plus
whatever the pagefile allows. None of that is knowable from inside the tool, so
the constant is 8 GiB per worker because 64 / 8 is the count that worked. It is a
floor from one machine, and `--workers` still overrides it.

The general lesson, and the second time this session has learned it: **a model
of a machine's behaviour is worth less than one measurement of it.** The first
time, a cold cache made a better GPU look five times slower and the figures were
nearly read as the machine being slow (handoff section 3).

### 4.20 `--resume` was blind to which machine wrote a report

Found immediately after 4.19, while recovering from it. `--resume` skipped any
combination whose report was current by `metrics_version` — and the 24 that had
just failed still held the **laptop's** committed reports, which are current. So
`--resume` would have reported "Nothing to run" and done nothing, with 24 runs
outstanding.

Correct while one machine wrote every report. Wrong from the moment two do, which
is exactly what P1.7's provenance exists to record.

`build_jobs` now compares each report's machine against this one, using the
subset of `provenance.COMPARABILITY_FIELDS` knowable *before* a run — machine,
`ti_arch_setting`, `taichi`, `machine_profile`. `stage_arches` cannot be in it:
which backend each of the 14 stages started on is only known once they have run.

Three cases deliberately count as **not from here**, so they are re-run rather
than skipped: a report with no provenance (the 48 predating P1.7), an incomplete
block, and one written by `--skip-pipeline`, where the toolpath's origin is
unknown. Unknown is never "mine".

Two existing tests broke on this, and they were right to: they asserted that the
committed reports counted as done, which quietly depended on the test running on
the operator's laptop. They now say which machine they are pretending to be.

## 5. Open plan items not yet resolved

| Item | Status |
|---|---|
| P0.8 matrix | **Done** (2026-09-23): 48 of 48 at metrics v2. The first run, 48 runs in 19.9 h, produced sound tilt and runtime data but unusable unsupported figures; `--reanalyse` recovered 39 of those from their archived toolpaths. |
| Gates M1, M2, M3 | Deferred by the team until the mechanical design is settled |
| Gate E1 | **Answered** (2026-09-28, `docs/firmware.md`): RepRapFirmware, axes X Y Z U V, `G32` kept while levelling is undecided. The macro files wait for the authors' files or our board (P6.3) |
| Gate D0 (tilt budget, benchmark geometry) | Matrix done. Still needs P2.1's analytic bound, the team's decision, and ideally M2 |
| T-shape `underside_angle_deg` | Defaults to 90; gate D0 picks the real value |
| Thresholds (45 deg effective, 1 % unsupported) | Placeholders, gate D0. Now meaningful: with the metric fixed, parts can actually pass. |
| `twin_domes` verdict | **Closed** (P0.9b): reports carry `verdict.assessable`, and the summary prints "– no overhang". |
| Which stage first diverges across backends | 3.9 argues the field solvers, from which stages change backend and how the planner works. Not measured stage by stage. The atom count in `data/frame/<part>.npz` settles it in one command if it ever matters. |
| Backend recorded beside each number | **Closed** by plan task **P1.7**: `provenance.stage_arches` records the backend every stage actually started on, and `ti_arch_setting` records whether anything was forced. |
| `order_atoms` with `kernel_profiler=False` | **Closed** by P1.6 (2026-09-27): output unchanged, and only **1.1 %** faster (335.2 s mean against 338.8 s). The profiler was not what made the stage slow |
| P1.5 firmware templates | **Done** (2026-09-28, P1-17): E1 recorded, `rrf` is the dialect, `docs/firmware.md` specifies the macros. The macro files themselves are deferred by the operator to P6.3 |
| P4 vs P1.4 ordering | **Closed.** P4 is complete and in `main` (pull request #17); P1.4 reached `main` first. |
| P4 branch not merged | **Closed** (2026-09-28): merged through pull request #17, viewer included. |
| P0.8 follow-on branch not merged | **Closed** (2026-09-28): `claude/new-session-l8g46d` is fully in `main` through pull requests #9 and #12, and that session is closed. Handoff 0c records what it delivered and its two declared convention breaches. |
| P2 | **The next work, and nothing blocks starting it.** No session yet. P2.0 and P2.1 do not need gate D0, and D0 now has everything it needs but the team's decision. |
| Parallel matrix runner | **Built** (2026-09-24) as `tools/run_matrix_parallel.py`: a copy of the working tree per worker, so no two runs share a `data/` path. Handoff section 3 has the design and the numbers. Tested with the slicing stubbed out; **contention between workers is still unmeasured**, so its projection is arithmetic rather than an observation. |
| Contention between parallel workers | **Measured twice at 8 workers, and it depends on job length**: 65 % on short `xs` jobs (2026-09-24), **79 %** on the mostly-`s` resume (2026-09-28). The runner reports the range rather than interpolating. **Above 8 workers it cannot be measured on the 64 GiB machine at all** — 16 workers exhausts committed memory first (4.19). |
| Workers per GiB of RAM | Calibrated from two points on one machine (8 works on 64 GiB, 16 does not). A machine with a large pagefile, or an `xs`-only run, may take more. 4.19. |
| Parallel efficiency on the laptop | Unmeasured. The lab machine's 65 % came from 36 cores and two sockets; a 6-core 45 W laptop will throttle instead, which is a different limit. `--sizes xs --workers 3` settles it in about an hour. |
| CPU model in the provenance block | P1.7 records the machine's **host name**, not its processor. A host name distinguishes machines; `Intel(R) Xeon(R) Gold 6254` is what a reader comparing two timings needs. `platform.processor()` will not give it — the Windows registry or `/proc/cpuinfo` will. Worth adding to `atom.provenance`; see 4.14. |
| Correction numbers 4.14 to 4.20 | **Taken** by the P0.8 follow-on session, straight into section 4 rather than a per-session heading in section 7. Seven numbers, all merged. A later session must not reuse them; the next free number is **4.21**. |
| A third concurrent session | The plan allows two. Three ran, and 4.14 is what it cost: a day of duplicated work. Two have since closed, so P2 starts alongside P1 alone. |

## 6. Quick index

The items a later task is most likely to get wrong if it trusts the plan:

| # | In one line |
|---|---|
| 1.1 | `tool_orientation` is `(N,2)` spherical; `theta` is the tilt directly |
| 1.5 | `order_atoms` is CPU and 86 % of runtime; long runs are not GPU-bound |
| 1.6 | `forward()` has no X-then-Y tilt decomposition to confirm against |
| 1.7 | IK failure is NaN in `offset`; a non-zero `offset` is a clearance, not an error |
| 1.8 | The plan's 1.5 x height support radius overrides its own 65-degree cone |
| 1.9 | IK→FK agrees with itself by construction; check the physical pose too |
| 3.2 | `from __future__ import annotations` breaks any module with a Taichi kernel |
| 3.9 | The golden SHA-256 holds only on the backend mix it was captured on |
| 4.5 | The bed-contact rule must anchor to the part's lowest point |
| 4.6 | Sampling by face centroid under-measures large flat faces; subdivide first |
| 4.7 | An overwriting run makes an interrupted re-run look complete; version the metrics |
| 4.8 | Bed re-centring changes screw heights non-uniformly; compare in one frame |
| 4.11 | `git ls-files` reports the index, not what is committed |
| 4.13 | Create VTK timers after the interactor is initialised, or they never fire on Windows |
| 4.14 | Fetch **every branch**, not just `main`, before building infrastructure; two sessions built provenance twice |
| 4.15 | Add `src/` in `conftest.py`, or no single test file can be run alone |
| 4.16 | `atomize.py` ignores stage exit codes, so one failure becomes twelve — and a stale output can pass for a fresh one |
| 4.17 | Never hand-write the pipeline's input list; `git ls-files -- data` is the list |
| 4.18 | A warm-up run is serial time: make it the shortest job, never the longest |
| 4.19 | Taichi commits 1 GiB of RAM per process; worker count is the only lever, and calibrate it, do not model it |
| 4.20 | `--resume` must compare the machine, not just the metrics version |

And the habits that caught most of them:

* Before a long run, do one short run and compare against what you expect. A
  figure that does not move when the thing it depends on changes is worth
  chasing (4.6 was found exactly this way).
* Check a metric against a case whose answer is known before putting its
  output in a table (3.7).
* **When a measurement contradicts the hardware, the measurement is wrong.** A
  better GPU running five times slower was a cold kernel cache, not a slow GPU
  (handoff section 3). Never benchmark a fresh machine on its first run.
* **Fetch every branch and read the plan's task list before building
  infrastructure.** Nothing is hidden — every session pushes to `origin`, so one
  `git fetch --all` shows all of it. What is missing is the habit, and a stale
  document reads exactly like a current one (4.14).

## 7. Parallel sessions: items found during P1 and P4

Two sessions run at once from 2026-09-23 (`docs/handoff.md` section 0). To
avoid both editing the same numbered list (the 4.12/4.13 collision,
`handoff.md` 7c), each session adds items **only under its own heading
below**, numbered `P1-1`, `P1-2`, … or `P4-1`, `P4-2`, …, using the same
categories as sections 1–4 (factual error, deliberate deviation, environment
fact, hazard) and the same **PLAN EDIT** marker. When both branches are in
`main`, the items move into sections 1–4 with ordinary numbers, and every
citation of them is updated in the same commit.

### 7a. P1 session (`claude/vibrant-rubin-waln7y`)

#### P1-1 A plane through the balls' fixed XY under-reads the tilt ★ PLAN EDIT

*Factual error.* P1.4's `TILT_LIMIT` says to take the bed tilt "through FK or
a closed-form plane fit through the three ball points". Taken literally (the
balls at their profile XY, `ball_2dpos_*`, heights Z, U, V), that plane is
wrong, because the contact points slide in their slots as the bed tilts. Their
horizontal spacing shrinks by `cos(t)`, so the fixed-XY fit reads
`arctan(sin t)`: **26.57 degrees at a true 30**, and 9.85 at a true 10.

The exact closed form keeps the bed-frame ball spacing `l1`, `l2` (rigid) and
solves `g . l1 = z0 - z1`, `g . l2 = z0 - z2` for the in-plane gradient `g`.
Then `tilt = arcsin(|g|)`. It agrees with `kinematics3z.forward` to 1e-7
degrees (`atom.screw_tilt.total_tilt_deg`, `tests/test_screw_tilt.py`). On
the golden cube it gives 5.525987 degrees; the toolpath's largest `theta` is
5.525986.

A `box` limit needs the direction of the tilt as well. `screw_tilt` ports
`forward`'s orientation part to numpy for that. It returns the direction the
toolpath requested, which differs from the bed's physical pose by up to 0.14
degrees at 30 degrees of tilt (1.9). That gap is a turn about the bed's own
normal, so the `cone` check is exact and the `box` check is exact to 0.14
degrees. Taichi's own `forward` is 0.012 degrees off the port on random states,
because it runs in float32.

The plan should say "`arcsin` of the in-plane gradient, with the bed-frame
ball spacing", not "plane fit through the three ball points".

#### P1-2 The profile has no maximum feed rate, and the golden exceeds the travel feed ★ PLAN EDIT

*Factual error.* P1.4's `FEED` check says "every F in `(0, profile max]`". The
machine profile has no maximum feed. The highest feed it names is
`travel_feedrate` (3000 mm/min), and the golden cube's G-code goes well
beyond it: **F10725** at most, with 23 lines above 3000.

That is by design. `toolpath_to_gcode.calculate_feedrate` scales the
deposition feed by (machine distance over X, Y, Z, U, V, E) / (tool-tip
distance), so a move where the screws travel much further than the nozzle tip
gets a proportionally higher F (§1 facts; P3.4 verifies it).

Decided with the operator on 2026-09-23: `FEED` always checks that F is finite
and above zero, and applies an upper limit only when one is passed
(`--max-feed`, `max_feed=`). A real limit needs a machine number: P3.4's
`max_screw_speed_mm_s` (gate D3) or a `max_feedrate` from gate M1/P6. It must
not be invented.

#### P1-3 `docs/conventions.md` gave (25, 25) degrees as 34.6 in total; it is 34.78

*Factual error, corrected.* `cos(total) = cos(a) cos(b)` gives
`arccos(cos^2 25) = 34.78` degrees. `atom.tilt.total_tilt_from_tilts_deg` already
computed 34.78, so no code was wrong. Only the document and a comment in
`tests/test_tilt.py` were. Found when a validator test built from that figure
disagreed by 0.2 degrees. Both are corrected.

#### P1-4 The shared tokeniser moved from `tools/gcode_stats.py` into `atom.gcode_check`

*Deliberate deviation.* P1.4 step 1 says to "reuse the tokeniser from
`tools/gcode_stats.py`". The package cannot sensibly import from a script
directory, so the direction is reversed. `strip_comment`, `parse_words` and
the quoted-string guard now live in `atom.gcode_check`, and
`tools/gcode_stats.py` imports them, so they are still importable from there
(`tools/visualize_5ax.py` uses them). The code moved unchanged. The
`gcode_stats` tests pass, and the golden statistics of a regenerated
calibration-cube G-code are identical.

#### P1-5 `Xnan` is invisible to the word parser

*Hazard.* Python writes a NaN as `nan`, so a failed solve would appear in
G-code as `Xnan`. The word pattern needs digits after the letter, so it does
not match `Xnan` at all. The word is **silently dropped**, and the axis keeps
its previous value, which looks perfectly valid. The validator's `NAN` check
therefore scans the raw text for non-finite words (`nan`, `inf`, `infinity`,
any case, with a sign) rather than looking at parsed values. Any other G-code
reader has the same blind spot.

#### P1-6 What "retracts and primes balanced" means in the validator

*Definition the plan left open.* The pipeline writes retract/prime pairs of
`retract_length` (2 mm), plus one unpaired retract in the header (E30 to E28,
absolute) and one in the footer. The golden cube has 530 retracts and 529
primes. So balance cannot mean equal counts. The validator tracks how much
filament is currently retracted and flags three things: a prime with nothing
retracted, a prime larger than what is retracted, and a printing move while
filament is still retracted. Ending the file retracted is correct.

#### P1-7 How provenance is recorded, and what the plan did not specify

*Deliberate deviations and definitions for P1.7.* The block's fields and their
meaning are in `atom.provenance`. Decisions beyond the plan text, all agreed
with the operator on 2026-09-23 unless marked:

* **Each stage's actual backend is recorded by the stage itself.**
  `atom.ti_env.init_taichi` appends `{stage, requested, actual}` to the file
  named by `ATOM_TI_ARCH_LOG` after `ti.init`. `tools/overhang_report.py`
  sets it for the pipeline it launches. Unset, nothing happens. This is an
  edit to a shared file, approved by the operator. The alternative, parsing
  Taichi's console lines, cannot tell which stage printed which line.
* **Machine name** is the host name (`platform.node()`), compared
  case-insensitively.
* **Comparability**: machine, backend (`ATOM_TI_ARCH` plus every stage's
  actual backend), Taichi version, and machine profile. The git commit is
  recorded but does not split groups, since code changes between runs are what
  a comparison is for.
* **`scored` is separate from the run.** `--reanalyse` re-scores an archived
  toolpath, possibly elsewhere and at another commit, and keeps the block that
  describes the run which produced it. *(Not specified by the plan.)*
* **`source`** says whether the tool ran the pipeline itself, measured a
  toolpath already on disk (`--skip-pipeline`: origin unknown, flagged in the
  summary), or was backfilled. *(Not specified by the plan.)*
* **Backfill**: all 48 baseline reports got the laptop's host name
  (`AbdoYasser`), the stock mix and the reference profile. `stage_arches` is
  **inferred** from each stage's default backend, with the GPU stages on CUDA
  because the laptop's CUDA works. The commit and run times were not recorded
  and are `null`.
* **Checked on the laptop, 2026-09-23.** The known-answer run
  (`ramp45_xs` at 7 degrees: 0.24 %, printable) recorded all 14 stage
  backends exactly as inferred. It also recorded the host name as
  `AbdoYasser`, not `Abdelrahman-personal-laptop`, the name first given for
  the backfill. `--summarize` then warned that the table mixed two machines,
  which is the warning working as designed. The operator confirmed it is the
  same laptop, and the backfill was corrected. **Lesson:** take a machine's
  name from what the code records (`platform.node()`), not from a name a
  person reads off the computer.
* **`schema_version` stays 1.** `provenance` is an added key, and bumping the
  version would make `load_reports` skip every existing report.
* **Not given a provenance block** *(not specified by the plan)*:
  `tests/golden/*.stats.json`, which is the golden record itself, with its
  environment in `tests/golden/baseline.md`; and
  `reports/matrix_progress.csv`, which keeps its columns so the file stays one
  consistent table. Both remain readable next to the per-run JSON, which does
  carry it.

#### P1-8 The direction round trip is 4.5e-4 rad in 32-bit floats, not 1e-4 ★ PLAN EDIT

*Factual error in a tolerance.* P1.1 step 2 asks the IK->FK round trip to
return the build direction within **1e-4 rad**. On the CPU backend, in the
32-bit floats the pipeline uses, it comes back within **4.5e-4 rad** (0.026
degrees; 99th percentile 3.8e-4). The error is about the same at every tilt,
from 0 to 29 degrees.

It is rounding, not the maths. The same kernels run with Taichi's
`default_fp=f64` return the direction to 3e-16 rad, positions to 1e-13 mm and
screws to 9e-14 mm. The other two round trips meet the plan in 32-bit floats:
screws 1.2e-4 mm, positions 7.8e-5 mm, against 1e-3 mm.

For scale: 0.026 degrees is 40 times below the 1-degree tessellation step, a
fifth of the 0.14-degree pose gap (1.9), and about 0.05 mm at the nozzle for a
point 100 mm from the pivot. It does not matter for printing.

Decided with the operator on 2026-09-23: test both precisions.
`tests/test_kinematics3z.py` asserts 1e-3 rad in 32-bit floats, and
`tests/kinematics_f64_roundtrip.py` (run in its own process, since the
precision is fixed when Taichi starts) asserts the plan's 1e-4 rad and, beyond
it, 1e-9. The kinematics maths is unchanged.

The plan should state the direction tolerance per precision: 1e-3 rad in
32-bit floats on the CPU, 1e-4 rad (in fact exact) in 64-bit. CUDA's 32-bit
rounding has not been measured; it runs on the laptop only.

#### P1-9 A positive `offset` is a first estimate of the lift, not the exact lift

*Factual error in `docs/conventions.md`, corrected.* The document said a
positive `offset` is "how much the part must be raised". It is how much to
raise it before solving again. Raising the part moves the tilted bed's corners
by less than the lift, so one step falls short. At 20 degrees of tilt near the
bed: 6.10 mm, then 0.37, 0.022, 0.0013, 0.00008, then 0, which is 6.49 mm in
total over five steps. `kinematics3z.get_plaftorm_size` already loops until the
offset is zero, so the pipeline is right. Only the description was wrong.
Anything else that uses `offset` as a lift (P3.3's diagnostics, P4.4's
safe-travel insertion) must iterate the same way. `tests/test_kinematics3z.py`
pins that the loop converges.

#### P1-10 Editors strip the trailing space the golden header depends on

*Hazard.* Upstream's header and footer each contain `M400 ; wait ` **with a
trailing space**, and that space is part of the golden G-code byte for byte.
When the header moved into a template (`atom.gcode_templates`, P1.2), saving
the new source file silently stripped it, and the rebuilt header stopped
matching upstream. The byte-identity check caught it before anything was
committed. The line is now inserted from code (`WAIT_LINE = "M400 ; wait" + " "`)
with a comment saying why, and a test pins it. Any other text that must match
upstream exactly should be built the same way, or stored in a fixture file
rather than in source.

#### P1-11 How P1.2 wires the temperatures

*Deliberate choices within P1.2.*

* **The option is added only when set.** `atomize.py` appends
  `--bed-temp`/`--nozzle-temp` to the `toolpath_to_gcode.py` command only
  when the parameter file has the key, so a file without them runs exactly the
  upstream command. The logged command is unchanged too.
* **Values are checked when the parameter file is read**, not an hour later at
  the G-code stage. The check allows only a finite number of zero or more. No
  upper limit is imposed: that would be a machine number, and none is given
  (§0 rule 4).
* **Formatting:** whole numbers are written as integers (`S60` for 60 or 60.0),
  others as given (`S60.5`).
* **One source for the macro strings.** `atom.gcode_check` now takes its 3Z
  enable/disable markers from `gcode_templates.MACRO_CALLS`, the strings the
  header writes, rather than keeping a copy.

#### P1-12 How P1.3 wires the infill settings, and one knock-on for the viewer

*Deliberate choices within P1.3, and a follow-up for another owner.*

* **Units are deposition widths**, as `fff3.sdf_generate_infill` uses them.
  The plan does not say. With 0.9 mm beads the defaults are a 7.2 mm period
  and a 1.8 mm shell.
* **Fractional values are accepted**, since the kernel takes floats. The
  period must be above zero, because the kernel takes the position modulo it.
  The shell must be zero or more. No upper limit is set.
* **Defaults stay upstream's ints (8, 2)** when no option is given, passed to
  the kernel exactly as before. The options are added to the stage's command
  only when the parameter file has the key.
* **`tools/sdf_to_isdf.py` gained `parse_args()`** (vendored edit), so the
  command line can be tested without Taichi. Taichi now starts after the
  arguments are parsed, so a bad option fails before initialisation. The GUI
  path is unchanged.
* **Knock-on for the viewer (not changed here):** `atom.toolpath_view`'s
  shell/infill colouring hard-codes a 2-width shell
  (`tests/test_toolpath_view.py`, `plan_corrections.md` 2.11). For a part
  printed with another `shell_thickness`, that colouring will be off. The
  viewer is the P4 session's area (`docs/handoff.md` section 0), so this is
  recorded for it, not edited.

#### P1-13 `atomize.py`'s log lists only 10 of its 14 stage commands

*Hazard, found on the laptop 2026-09-27.* `tools/atomize.py` writes a
"Pipeline commands" section to `data/log/<part>.log`, and it is natural to read
that as the list of what ran. It is not. It omits the infill stage
(`sdf_to_isdf.py`) and the last four stages (`tesselate`, `add_platform`,
`toolpath_to_gcode`, `ratrig_to_craftware`), although all of them run.

P1.3's first pipeline test asserted that `--infill-period 12` appeared in that
log. It failed on the laptop after a complete, successful run, because the
option is on the one command the log never lists. The test was wrong, not the
feature. It now relies on the plan's own check, a different total extrusion,
and says why. Anything else that wants to confirm what a stage was given
should not use this log either.

#### P1-14 A parallel run's report lost its commit, and its runtime looked like the machine's speed

*Found reviewing `tools/run_matrix_parallel.py` (the P0.8 follow-on session's
runner) against P1.7, 2026-09-24; fixed 2026-09-27.* Three gaps, none in the
metrics themselves:

1. **No commit.** Each worker runs in a copy of the working tree with no
   `.git`, so `provenance.git_state` found no repository and recorded
   `git_commit: null`. Had the worker root sat inside *another* repository,
   it would have recorded **that** one's commit. The runner now states the
   parent's commit and dirty flag in `ATOM_GIT_COMMIT` / `ATOM_CODE_MODIFIED`,
   which `git_state` uses instead of asking git. Set but empty means "known to
   be unknown", so git is not asked then either.
2. **Runtimes under load were indistinguishable.** The lab machine's first
   parallel run measured each run **1.47x slower** with eight at once than
   alone. Those runtimes flow into the summary's runtime table and into the
   points^1.5 scaling result (handoff 7b), and nothing in a report said the
   machine was shared. Reports now carry `parallel_workers` (the pool size;
   1 for a run on its own, and for the runner's warm-up) and
   `taichi_cpu_threads`. The runtime table has a Run column and a note when
   the conditions differ. These are **not** comparability fields: the metrics
   are deterministic on one machine, so pooled and solo runs of one machine
   stay in one group, and only their times are kept apart.
3. **The mixed-table warning was not printed.** The runner wrote the summary
   with the warning inside it but never showed it on screen, unlike
   `--summarize`. It now prints it.

The 48 baseline reports are backfilled with `parallel_workers: 1`, since
`scripts/run_baseline_matrix.ps1` runs one combination at a time. The two new
fields are optional, so reports that predate them stay complete; the table
shows "not recorded" for them.

#### P1-15 How P1.6 switches the profiler

*Deliberate choices within P1.6.*

* **The switch lives in a new module, `atom.ti_profiler`,** not in
  `atom.ti_env`, which is on the shared "ask first" list (`docs/handoff.md`
  section 0). It needs no Taichi import, so it is testable anywhere.
* **Unknown values are refused**, as `ATOM_TI_ARCH`'s are. `1 true yes on`
  mean on; `0 false no off` or unset mean off. A typo that silently left the
  profiler on would distort the very timing this task measures.
* **Its printed table goes behind the same switch.** Upstream printed the
  kernel table after every run; with the profiler off there is nothing to
  print.
* **The time is measured on the stage alone**, `order_atoms.py` on the golden
  run's own `data/sdf` and `data/frame` files, run off / on / off so that the
  laptop warming up cannot pass for a speed-up. A full golden run carries four
  minutes of other stages and the laptop's thermal state, which swamp a
  difference of this size.
* **Result (laptop, 2026-09-27):** 335.5 / 338.8 / 334.9 s. The profiler cost
  **3.6 s, 1.1 %**. The two runs without it agree within 0.6 s, so the
  difference is real, but it is smaller than the laptop's day-to-day drift:
  the baseline, made *with* the profiler, recorded 331.6 s. The plan's "if
  faster, re-estimate the matrix times" does not apply at 1 %.

#### P1-16 The lab machine is the reference for overhang results

*A decision by the operator (2026-09-28), recorded here with what follows from
it. Changes plan §0 rule 10, §0.2, the compute budget, P0.8's note and P2.5
(plan v3.4, Appendix H).*

**What was decided.** Pull request #12 re-measured all 48 baseline runs on the
lab machine (`cad-p07-2065-9`, 2 x Xeon Gold 6254, RTX A6000) with
`tools/run_matrix_parallel.py`, as one single-machine set. The operator adopted
it as the committed baseline in `reports/baseline_overhang/`. Every comparison
against the baseline, P2.5's first of all, therefore runs on the lab machine.
That is correction 3.9 applied, not relaxed: a comparison is still only valid
on the machine that measured the baseline; the machine changed.

**Why it is safe.** The laptop and lab sets agree on every conclusion: **0
verdict changes in 48**, the worst effective overhang moves by at most
**0.008 degrees**, the tilt used by at most **0.21 degrees**. Only the
unsupported fraction near overhangs moved: by +0.5 to +1.3 percentage points
on 7 cells (all upward), and by less than half a point, either way (-0.3 to
+0.5), on 25 more. The overall fraction moved by under 0.05 points everywhere.
The closest cell to the 1 % threshold (`ramp45_xs` at 7 degrees, 0.24 %)
keeps 0.76 points of margin. PR #12 reported only the 7 larger moves; the
count of 32 was found when comparing the two sets for this entry. So both P0.8 results (the 45-degree limit, and theta_eff = theta_geo +
tilt_used) hold on two machines. The fraction's movement is also a warning
for P2.5: it is the one metric sensitive to the machine, so a P2.5 change in
it smaller than about 1.3 points would mean nothing if the two sides came
from different machines.

**What stays on the laptop.**

* **The golden test.** Its SHA-256 was captured on the laptop's backend mix
  and is valid only there (3.9). Moving it would mean a new golden record,
  which nobody has asked for.
* **Timing.** The lab set's runtimes mix 16, 8 and 1 workers (23 / 23 / 2
  runs), and a run is about 1.4x slower under 8-way load. The laptop's serial
  runs are the project's only clean timings, and the points^1.5 result for
  `order_atoms` comes from them.

**The laptop set is kept, not deleted:** `reports/baseline_overhang_laptop/`
holds its 48 reports, its summary and a README. The folder is visible so the
robustness finding and the scaling result stay traceable. No tool reads it,
and two tests pin that: the live baseline in `reports/baseline_overhang/` is
never the archive, and `tools/backfill_provenance.py`, whose values describe
the laptop, now writes only to the archive. Left pointed at
`reports/baseline_overhang/`, it would have been one command away from
stamping the laptop's host name on the lab machine's reports.

**The tests that pinned the laptop.** Three tests in `tests/test_provenance.py`
required the committed baseline to be the laptop's host name and
`parallel_workers == 1`, so #12 failed them. They now require what must stay
true whatever the reference machine: one machine across all 48, a recorded
worker count on every report, and a summary that says so when the worker
counts differ. The per-stage backends stay pinned.

**Compute budget, lab machine.** Measured (#12 and handoff section 3):
`xs` only, 8 workers: **0:50:31**; 24 mostly-`s` runs at 8 workers: **2:58:44**
(6.31x, 79 % efficiency); 16 workers exhaust Taichi's host memory pool (4.19).
A full 48 at 8 workers is therefore **roughly 3.5 to 4 h**, against about 20 h
serially on the laptop. That is an estimate from those two runs, not a
measurement of a full 48 at 8.

#### P1-17 Gate E1 answered; P1.5 built without the macro files

*The operator's answers (2026-09-28), and what the paper behind Atomizer's
printer adds. Changes plan §3 (E1), P1.5 and P6.3 (plan v3.5, Appendix I).*

**The answers.** RepRapFirmware; axis letters X, Y, Z, U, V ("as RRF allows",
which is also what Atomizer writes); for the macros, find what they do from RRF,
then from Atomizer's own documentation; homing and levelling not decided, so
`G32` stays. `docs/firmware.md` records
them.

**Why no macro files.** The plan's P1.5 writes `firmware/enable3Z.*` and
`disable3Z.*` once E1 is answered. Their contents are not published: the
operator supplied the paper (Cocco et al., SCF '25, doi:10.1145/3745778.3766652),
and it says only that "RepRap firmware can be configured to interpret 5-axis
G-code without any modifications". It gives no `M584` lines, macros or axis
letters. Writing them from general RRF practice would be a guess, and every
`M584` line needs our board's driver numbers, which do not exist yet. The
operator chose to build P1.5 now and add the files later (P6.3). So
`docs/firmware.md` specifies what the macros must **do**, taken from the G-code
and the kinematics, instead:

* `enable3Z` runs after `G32` and leaves Z, U, V as independent axes that
  **agree on height**. The G-code's first move after it is Z = U = V, which
  means a level bed only if the split kept one zero and direction;
* `disable3Z` runs last and joins the screws into one Z again, so the next
  `G32` works.

**Klipper** stays accepted in a profile but refused when G-code is written or
checked, now with a message saying E1 chose RepRapFirmware rather than "GATE
E1". Removing it from the valid dialects would change what a profile may say,
which nobody asked for.

**Tests** (`tests/test_gcode_templates.py`, "Gate E1"): the validator's axis
letters are E1's; every move the pipeline writes uses only X Y Z U V E F; the
header levels (`G32`) before the enable macro, and no U or V word comes before
it; the macro is followed by `M400`; the footer ends with the disable macro;
and `docs/firmware.md` names the macro paths and commands the G-code uses.

**What the paper adds, for later phases** (recorded in `docs/firmware.md`
section 5):

* its printer ran RRF **3.6.0** on a RatRig V-Core 3.1;
* the Z axes are limited to **1900 mm/min**. RRF slows a move so no axis
  exceeds its limit, so fast tilting moves run slower than the G-code asks.
  Relevant to D3 (max tilt rate) and to any print-time estimate;
* at most **1 mm and 1 degree** between G-code points keeps the firmware's
  straight-line interpolation of machine coordinates within 0.05 mm and
  0.01 degrees. Atomizer tessellates orientation to 1 degree; point spacing
  was not checked;
* its ball positions, ball height, build area and 30-degree tilt match the
  `reference` profile exactly. Two rail angles are stated in the opposite
  direction (150.11 and -90 against -29.89 and 90). `kinematics3z.inverse`
  built with the paper's angles gives **identical** screw heights on 200 random
  poses up to 25 degrees, so the difference is a sign convention only;
* backlash in the bed makes layer alignment depend on tilt direction, and the
  balls are held on their rails by gravity alone, so an ungreased rail can
  lift the bed off. Both matter at P7/P8.

### 7b. P4 session (branch recorded in `docs/handoff.md` section 0b)

#### P4-1 The clearance model takes world-frame points and the head position (deliberate deviation)

P4.1 specifies `min_clearance(points_machine)` and `violations(points_machine)`
without saying which frame "machine" means. `atom.clearance` fixes it: points
are in the **world frame** of `atom.bed_motion` (fixed to the machine, +Z up,
nozzle tip at `(X, Y, 0)`), and every method also takes the head position
`head_xy`. The reference proxy moves entirely with the head, so it only needs
points relative to the tip, but the real envelope (gate M3) will not: a frame
member stays put while the head moves. With `head_xy` defaulting to `(0, 0)`,
head-frame points work unchanged.

Three additions to the plan's interface, all needed by P4.2's report:
`body_distances` (signed distance to each named body: `nozzle`, `gantry`),
`nearest_body` (which one is closest) and `describe` (the parameters, for a
report's provenance). Two more came with P4.2, which scores thousands of
machine states in one call: `head_xy` may be one row per point, and
`body_distances(..., bodies=[...])` computes only the bodies asked for (the
swept check never scores the hull against the nozzle, see P4-3). Distances are exact, not bounds: the cone's is computed
in its meridian plane against a triangle, and a test checks it against a
brute-force sampling of the surface.

The box format for `config/machines/<name>.clearance.json` is defined but no
such file exists (gate M3, filled in P6.2). Each box says which head axes it
follows (`"moves_with": ["x", "y"]` for the hotend, `[]` for the frame,
`["y"]` for a beam riding on Y). `load_clearance` falls back to the reference
proxy when a machine has no file, as the P4 gate line prescribes, and a test
asserts no file exists until M3 is answered.

**Verified against the kinematics.** Putting the bed corners in the world
frame with `atom.bed_motion` and measuring them against the reference
model's gantry half-space reproduces the lift `kinematics3z.inverse` asks for
to within 0.002 mm, over 24 states up to 29.9 degrees of tilt
(`tests/test_clearance.py`). The two therefore share one world frame, which
P4.2 relies on.

#### P4-2 P4.3 checks every nozzle position, exactly, not only deposition points subsampled (deliberate deviation)

P4.3 says "for each deposition point, test earlier deposited points
(KD-tree, subsampled)". As built (`src/atom/nozzle_material_check.py`,
`tools/check_motion_safety.py`; the plan names no files for P4.3):

* **Every toolpath point is checked**, travel included. A travel that ends
  inside material is as much a collision as a print move. Reports split the
  two. The travel *between* points stays with P4.2.
* **"Earlier" is defined by the moves.** `travel_type[i]` describes the move
  *to* point `i`, so a printing move `j` makes both `p[j-1]` and `p[j]`
  material. With the nozzle at `p[i]`, material from moves up to `i - 1`
  counts; move `i`'s own bead ends under the nozzle.
* **No subsampling by default.** Exact is fast enough: the whole golden cube
  (46 773 points) takes about 2 s on the session container. The cone is covered
  by a chain of spheres along its axis, so a KD-tree query returns only
  material near the cone. `--subsample MM` remains as an option; it keeps the
  earliest point per voxel, so everything it reports is a real collision.
* The cone is `atom.clearance`'s nozzle (40 degrees, capped at
  `nozzle_to_gantry`), so P4.1, P4.2 and P4.3 share one nozzle.
* **A nozzle buried in material stops early.** The first version examined
  every blocker of every position. On a stand-in where four copies of a part
  stood 5.7 mm apart, nearly every position of copies 2–4 had thousands of
  genuine blockers, and the check did not finish in ten minutes. Each nozzle is
  now searched slab by slab from the tip, and one that has already hit
  material *and* examined more than `exhaustive_limit` (4 096) candidates
  stops there. Whether a position collides stays exact (a nozzle with no hit
  searches every slab; a test forces this path everywhere and compares with
  the exhaustive answer). Only its depth and blocker count then describe the
  first material up the nozzle, and the report says so
  (`first_slab_only`, `collisions_summarised_from_the_first_slab`). Cost:
  about 9 ms per buried position. A file that collides almost everywhere
  takes minutes, not hours, and one that is clear takes seconds.
* **Measured speed** (session container, both P4.2 and P4.3): golden cube
  about 6 s; a 38 000-point 30-degree `ramp60_xs` about 9 s; 150 000 points
  (the size of an `s` part) about 42 s.

**Checked against known answers.** The golden cube is clear. The zero means
something: the top 3 000 points printed in reverse order (top down) collide at
more than a third of positions, and 12 000 of them at 83 %. A brute-force
comparison over 400 scattered points (330 colliding) matches the fast check
exactly (positions, depths, blocker counts) at three block sizes.

**The 50-degree cap is now a test.** The plan's introduction says the tilt is
"capped at 50 degrees by the nozzle cone". With a 40-degree half-angle, a
flat layer printed at more than 90 - 40 = 50 degrees of tilt puts its own
earlier beads inside the nozzle; at 45 degrees it is clear
(`tests/test_nozzle_material_check.py`).

**Hazard for a wider nozzle (gate M3).** The covering spheres stay above the
nozzle tip's plane only for half-angles below 45 degrees
(`q <= 1 / tan^2(half_angle)` for a slab ratio `q`). If the real hotend
turns out to need a cone of 45 degrees or wider, the check stays correct but
slows to comparing most point pairs. A wide hotend should be modelled as a
narrow cone plus boxes (P6.2) rather than as one wide cone.

**Not yet run on large tilts.** P4.3's benchmark is "run on the P2.5
outputs", which do not exist. The P0.8 archive on the laptop
(`reports/toolpaths/`, stock toolpaths up to 30 degrees of tilt) is the
nearest real input and is a laptop request (`docs/handoff.md` 0b).

#### P4-3 P4.2 as built: hull for the gantry only, axis and tilt checks deferred to P1.4 (deliberate deviation)

P4.2 says to "transform a proxy of the printed-so-far part (convex hull of
deposited points, updated every N points for speed) and test it against the
clearance model, axis ranges and tilt limits". As built
(`src/atom/tilt_motion_check.py`), with the operator's decisions of
2026-09-23:

* **The convex hull is tested against the machine bodies only** (the gantry
  half-space, for the reference proxy). Against a flat body the hull is
  exact, not a proxy: a convex solid's highest point is a vertex. Between
  rebuilds (every 500 material points) the raw points laid since are added,
  so nothing printed is ever missing. A test checks this against brute force
  under random orientations.
* **The nozzle is tested against the real printed points**, through the same
  code as P4.3 (`nozzle_material_check.cone_hits`). The hull fills hollows,
  so it would put the nozzle "inside" material whenever it works between two
  features or inside a cup.
* **Axis ranges and the tilt limit between points were deferred to P1.4**,
  so those rules exist once. **Done 2026-09-28, see P4-6**: every state is
  now held to the validator's own rules, and `not_checked` is empty. Moves
  touching a point the IK rejects (NaN `offset`) are still skipped and
  counted.
* **Two checks added** that the plan's list does not name: the bed corners
  against every body at every state (the IK does this only at the points),
  and the nozzle tip against the bed plane between points.
* **Which states:** every reachable point, plus interior states of each move
  that turns the tool by more than `check_tilt_step_deg` (0.5 degrees, the
  plan's default) and of each travel longer than `travel_step_mm` (0.5 mm).
  Interpolation is linear in the five axes, as the firmware moves. The
  nozzle check at the points themselves is P4.3's; P4.2 does it between
  them. `tools/check_motion_safety.py` runs both.
* **Tolerance 0.01 mm.** The kinematics run in float32 (about 1e-3 mm of
  rounding); a violation must be deeper than this.

**Verified against hand calculations and the IK.** A 68 mm tower 80 mm from a
nozzle working 30 mm up, tilted 30 degrees toward it, reaches
`38 cos 30 + 80 sin 30 = 72.91` mm, 2.91 mm into the gantry; the check
reports 2.909. A travel with no lift through a 5 mm block is caught at that
move, at `(4 - 0.001) sin 40` deep, while P4.3 at the two clear endpoints
sees nothing. Where the check finds a bed corner in the gantry at a point,
the depth equals the lift `kinematics3z.inverse` asks for there. The golden
cube has no violations over 49 063 states (2 290 between points); the part
comes no closer than 68.9 mm to the gantry.

**A fact about the reference machine worth knowing (from the IK, not new
maths).** Its bed corners reach the gantry at large tilts when the nozzle is
near the bed: 20 degrees toward a diagonal with the nozzle 5 mm up puts a
corner 4 mm into it, and 30 degrees does so in every direction once the work
sits 40 mm off the bed's centre. Along X or Y at 20 degrees it is clear. The
IK asks for lift at such points and `add_platform` raises the part to give
it, so a `_platform` toolpath is clear; a `_smoothed` one checked directly
will show these as `bed` violations. Gate M2 (usable tilt by position) is
where this matters.

**Test tier.** The golden-cube swept test takes about 3.6 s on the session
container (the P4.3 one about 2 s), above the unit tier's "~2 s". They are
kept in the unit tier so CI guards them; the operator may prefer them in
`pipeline`.

#### P4-4 The platform is sized in one frame and the G-code written in another, so large tilts can abort the G-code (hazard, vendored code, fixed 2026-09-28)

Found by running the P4 checks on a 30-degree stock toolpath. `toolpath_to_gcode`
printed `Fatal Error: collision found!` and wrote no G-code for `ramp60_xs`
at `max_slope 30`, although every point is reachable.

The cause, measured:

1. `add_platform` sizes the platform with `kinematics3z.get_plaftorm_size`,
   which re-centres the **part's** bounding box on the bed and raises the
   part until no point asks for lift.
2. The platform it then adds starts at x = 0, y = 0, so the **part plus
   platform** has a different bounding box: here x from 0 instead of 0.401.
3. `toolpath_to_gcode` re-centres **that** box, 0.187 mm away in x from the
   frame the platform was sized in. Re-centring changes screw heights
   non-uniformly (hazard 7), so near the limit the bed corners come back:
   solved in the sizing frame the largest lift is 0.000 mm; in the G-code
   frame it is 0.036 mm, at points 7892–7895 (29.4 degrees of tilt).
4. `kinematics3z.toolpath_from_cartesian_toolpath` treats **any** non-zero
   lift as invalid (`collision != 0`), so 0.036 mm aborts the whole file.

P4.2 reports the same three moves (7893–7895) as `bed` vs `gantry`, 0.015 to
0.036 mm deep, and nothing else; P4.3 is clear. So this toolpath is safe to
within a few hundredths of a millimetre, and the abort is a frame mismatch,
not a collision.

**Caveat:** this run was made in the session container with a stand-in for
the Blender remesh and every stage on the CPU (development only; no number
from it is reported anywhere). The mechanism is independent of both: it
needs only a part whose bed corners are near the gantry at its largest tilt,
which becomes common at a 30-degree budget and is exactly where P2 works.
**Update (2026-09-28): the baseline did not hit it.** Since correction 4.16,
`overhang_report.run_pipeline` fails a run whose stage left no fresh artifact,
the G-code (stage 11) included, and `toolpath_to_gcode` aborts before opening
its file. All 48 lab-machine baseline runs, the 16 at `max_slope 30` among
them, were made with that check (commit `fadcab3`) and none failed. So P4-4 is
latent: real, and a hazard for P2's larger tilts, but no committed number is
affected.

**Fixed (2026-09-28, the operator asked the P4 session to).** A vendored
edit to `kinematics3z.get_plaftorm_size`, the first of the three options:
size the platform in the frame the G-code will use. The platform's footprint
(`ceil(max / width) * width` from the origin) is known before its height, so
after solving the lift in the part's frame as before, the function checks
whether `add_platform` will draw platform layers (it does only when the
platform is at least two layers high, by the same expression). If it will,
the lift is solved again with the part-plus-platform box re-centred, the way
`toolpath_to_gcode` will re-centre it. Without drawn layers the frames are
already the same and nothing changes, which includes the golden cube (no
platform), so its G-code is unaffected; the golden test is still owed on the
laptop, as for any vendored edit.

Checked three ways: the stand-in `ramp60_xs` that aborted now gets a
10.80 mm platform instead of 10.35, `toolpath_to_gcode` writes its G-code, and
both P4 checks are clear on the result; two synthetic parts (25 degrees toward
135, 28 degrees toward 225, 40 mm from the origin) left 5.1 and 7.2 mm of lift
in the G-code's frame with the old sizing and none with the new
(`tests/test_platform_sizing.py`); and the golden cube's platform size is
unchanged. Neither `add_platform.py` nor `toolpath_to_gcode.py` was edited,
and `toolpath_to_gcode` still aborts on any lift, so a real collision is
still refused.

A small, deliberate cost: parts that needed the second solve get a slightly
taller platform (0.45 mm on the stand-in).

#### P4-5 The viewer shows the P4 checks (P5.4's open item, as built)

P5.4 left "the full clearance model arrives with P4.1, then wire it in". The
operator asked for it after P4.2, so the viewer shows exactly what
`tools/check_motion_safety.py` reports rather than a separate reading of the
model:

* a colour mode **Collisions (P4)**: a point is red when the nozzle there has
  earlier material inside it (P4.3), or when the move to it clashes between
  the points (P4.2). Flagged points are also drawn as dots, travel included,
  since a single red segment is easy to miss;
* the current-point card and the status line name each collision (kind,
  depth, how far along the move);
* in the machine view the bed turns red on a flagged move, as well as on the
  two conditions it already showed (IK rejection, bed corner above the
  gantry); the notes panel gives the totals and says what is not checked yet.

The checks run once per file, when the collision mode or the machine view is
first opened (a few seconds for the golden cube). G-code is checked with the
screw values as written; a toolpath is solved re-centred on the bed, as for
the machine view. Code: `toolpath_view.CollisionMarks` / `collision_marks`
(numpy, tested without a display) and `visualize_5ax.compute_collisions`.
The Qt window needed no change: its dropdown and card are built from the
same lists. Checked here under a virtual display (Qt window included); the
laptop check is still to do.

#### P4-6 P4.2 applies the validator's own axis-range and tilt-limit rules (operator's decision)

P1.4 put both rules inside `gcode_check.check_lines`, which reads G-code line
by line, so they could not be called on the machine states *between* toolpath
points. With the operator's permission (2026-09-28), the P4 session moved
them, unchanged, into shared functions in `src/atom/gcode_check.py` (a P1
file): `axis_limits`, `axis_range_detail`, `axis_range_problems(states,
profile)` and `tilt_limit_problems(screws, profile)`. `check_lines` now calls
them. Its output is **byte-identical**: the validator's reports on all 11
G-code fixtures, under both profiles and with the tilt limit as a cone and as
a box (44 reports), were compared before and after.

`tilt_motion_check` applies the same functions to every state it visits, as
kinds `axis_range` (body = the axis, depth in mm) and `tilt_limit` (depth in
degrees as `excess_deg`, no `clearance_mm`), each with the validator's own
message as `detail`. Reports no longer list anything under `not_checked`.

**What these can and cannot find between points.** A move is linear in the
five axes. The axis ranges are a box in those axes; the total tilt is a
monotone function of a norm that is linear in the screw differences, so it
peaks at an end; and a box tilt limit can only be broken once the total
passes the cone. So a move between two points that pass cannot break either
rule. What the check does find is **points** the IK accepts and the
validator would reject: a screw below zero, which `inverse` answers with a
lift rather than NaN. On the 30-degree `ramp60_xs` stand-in (P4-4), 676 moves
of the `_smoothed` toolpath have one, and none of the `_platform` toolpath:
the platform supplies exactly that lift. The tool's before-platform hint now
covers these too.

A hazard noted, not seen: the IK's tilt tolerance is 1e-4 degrees and the
screws are float32, so a point requested at exactly the limit could read a
little over it from its screws. Atomizer's `max_slope` stays at or below the
limit and its toolpaths have peaked at 29.4 degrees at a 30-degree budget.

#### P4-7 The validator's 5 mm single-move limit rejects the platform's own edges (finding, P1's decision)

Found while checking the P4-4 fix. The first G-code this repository has
written *with* a platform (the golden cube has none) fails `validate_gcode`
with 46 `EXTRUSION` violations, all on platform lines: `add_platform` draws
each platform layer's outline as four straight moves, and the long side of
a 29.7 mm platform extrudes 5.0009 mm of filament at 0.9 x 0.45 mm, just over
the 5 mm limit agreed on 2026-09-23 (`gcode_check.DEFAULT_MAX_E_MM`). Nothing
else in the file breaks any rule.

So every part at least about 30 mm long that needs a platform will fail the
validator as it stands: all `s` parts would (about 8.4 mm per 50 mm line).
The toolpath is fine; the limit was chosen from the golden cube, whose
largest move was a 2 mm prime. Options, P1's file and the operator's call:
raise the default, scale it with the move's length (a limit per mm of travel
rather than per move), or have `add_platform` split long lines. Not changed
here.

**Resolved 2026-09-28, the operator's call: a limit per mm of travel.** In
relative mode an extruding move (one with an axis word and E > 0) may push
out at most `max_e_per_mm` mm of filament per mm of machine travel,
sqrt(dX^2 + dY^2 + dZ^2 + dU^2 + dV^2) (`gcode_check.DEFAULT_MAX_E_PER_MM`
= 0.5, `--max-e-per-mm`). Measured before choosing:

| E per mm of travel | Golden cube | 30-degree G-code with a platform (`ramp60_xs`, max tilt 29.4) |
|---|---|---|
| Tip distance (what `calculate_extrusion` uses; needs forward kinematics) | 0.1684 | median 0.1684, max 0.1684 |
| 5-axis machine distance (read from the words alone; chosen) | 0.1684 | median 0.1376, p99.9 0.1770, max 0.1779 |
| XY plus mean screw distance | | median 0.1548, max 1.1304 (not usable) |

The machine distance keeps the validator numpy only and needs no kinematics.
It differs from the tip distance only when the bed tilts during a move, and
by at most 6 percent at the 30-degree limit. 0.5 is three times the nominal
0.9 x 0.45 mm bead on 1.75 mm filament (0.168): it leaves room for wider
beads (0.5 mm per mm is a 1.2 mm^2 bead cross-section) and still catches a
blob, which the old limit let through on any move short enough. A move whose
axis words go nowhere is zero travel, so any E on it is rejected. An E with
no axis word (a retract or a prime) has no travel and keeps the absolute
limit, `max_e_mm` = 5 mm (`--max-e`). The same `ramp60_xs` G-code now
validates clean; the golden cube and every fixture still pass.

#### P4-8 The lab baseline under both checks: 43 of 48 clear, the rest lift or hairline grazes (result)

Run 2026-09-28 on the lab machine (`cad-p07-2065-9`), `tools/check_motion_safety.py`
on the 48 baseline toolpaths (`reports/toolpaths/`, the `_smoothed` stage, so
before `add_platform`), reference clearance model. About 15 minutes.

* **At the points (P4.3): all 48 clear.**
* **Between the points (P4.2): 43 of 48 clear.** All 5 flagged files are at
  `max_slope 30`:
  * `ramp60_s` (4 975 `bed`, 3 291 `axis_range` moves), `ramp60_xs` (1 063,
    556) and `twin_domes_s` (8 `bed`): the lift `add_platform` adds, a bed
    corner in the gantry or a screw below zero. All 48 runs wrote G-code
    under the stage check (4.16), which `toolpath_to_gcode` refuses if any
    point still needs lift after the platform.
  * `tshape_xs` (1 move) and `twin_domes_xs` (4): `nozzle_vs_material`,
    depths **0.0120, 0.0122, 0.0155, 0.0221, 0.0464 mm**, at 5.3 to 14.6
    degrees of tilt, 20 to 67 % along the move; four printing moves and one
    travel. None buried (`first_slab_only` false).

**The five nozzle hits are hairline grazes, not collisions.** They pass the
0.01 mm tolerance by 0.002 to 0.036 mm. The nozzle is an idealised 40-degree
cone (Atomizer's planning proxy, gate M3 for the real shape) and material is
represented by bead-centre points of beads 0.9 x 0.45 mm, so an overlap of a
few hundredths of a millimetre is below what either model resolves. The
32-bit direction error (P1-8, 0.026 degrees) would need a blocker about
100 mm away to reach 0.046 mm, so rounding does not explain them. They happen
at low tilt, between two points that are each clear, and the `s` versions of
both parts are clear.

Kept as they are, deliberately: the tolerance stays at 0.01 mm (a float32
margin, not a physical one), and the reports carry the depth, so that P2's
steeper toolpaths are measured on the same strict check and a real problem
(depths of tenths of a millimetre or more) stands out against these.


### 7c. P2 session (`claude/brave-ramanujan-7xolkm`)

Started 2026-09-29, the only session running. Items are numbered `P2-1`,
`P2-2`, … as in 7a and 7b.

#### P2-1 `max_slope` is not enforced on the field; P2.1 points at the wrong lines ★ PLAN EDIT

*Factual error.* P2.1 step 1 asks to document "how `MAX_SLOPE_ANGLE` is
enforced (`fff3.py` ~L138–154)". Those lines are
`orthogonolize_direction_wall_region`, which runs only with `--ortho_to_wall`,
an option no benchmark part sets.

On the path every part takes, `max_slope` reaches the orientation field in one
place only: `CEIL_MAX_ANGLE = MAX_SLOPE_ANGLE` (`tools/compute_tool_orientations.py:58-62`),
which decides which upward-facing boundary cells count as "ceilings" and are
fixed to their normal (`src/atom/fff3.py:91`, `:102-104`).
`MAX_SLOPE_ANGLE` itself is `min(max_slope, (180 - NOZZLE_CONE_ANGLE) / 2)`,
the 50-degree nozzle cap.

There is no clamp. The field stays inside the budget because every constraint
does (the first layer at 0 degrees, ceilings below `CEIL_MAX_ANGLE`), and the
aligner only forms normalised positive-weight averages of unit vectors, which
cannot leave a cone of half-angle below 90 degrees around +Z. All 48 baseline
runs agree: none used more tilt than its `max_slope`.

**Consequence for P2.2:** a constraint written beyond the budget would widen
the field's tilt beyond it, and nothing between the field and the G-code
stops that except the inverse kinematics, whose NaN makes
`toolpath_to_gcode` abort the whole file. P2.2's
`t = min(theta_geo - max_overhang + margin, max_tilt_here)` is therefore the
only limit, and must be applied where the constraint is written.

#### P2-2 The final 32 smoothing passes also smooth the constrained cells (hazard for P2.2)

*Hazard, found reading the code; not measured.* After the multigrid solve,
`tools/compute_tool_orientations.py:107-112` runs 32 passes of
`align_one_level_one_time(0, True)`. The `True` is `smooth_constraints`:
`src/atom/direction.py:274` keeps a constrained cell's direction only when it
is false, so in these passes every constrained cell is replaced by the
average of its 3 x 3 x 3 neighbourhood like any other. Only the first layer is
put back after each pass (`spherical_field_constrain_fisrt_layer_up`).
Upstream's comment says this is deliberate ("Smooth everything - including
constraints - except the first layer", Chermain et al. 2025, Section 6).

The constrained band is thin: `is_boundary_region(sdf, layer_height)` selects
cells **inside** the part within one layer height of the surface, about 2.7
cells at 0.9 mm beads (0.169 mm cells). So 32 passes average each constrained
cell with unconstrained interior cells many times over.

**Why it matters:** P2.2 says to write the overhang direction "as a
constrained direction, exactly like the ceiling branch". Written that way it
will be smoothed exactly like the ceiling branch, and how much of the intended
tilt survives at the overhang is unknown. P2.2 needs to either re-apply the
overhang constraint after each pass, as the first layer is, or measure the
loss and compensate. P2.2's planned tiny-SDF unit test and P2.0's field-only
run can both measure it. Not decided here.

**Update (P2.1):** one measurement. For the infill's inner surface above
`ramp70_xs` at 30 degrees (P2-3), the 32 passes changed the mean tilt next to
the underside from 19.85 to 19.77 degrees: that constrained surface is broad
and uniform, so averaging it with itself changes little. A thin band of
overhang constraints facing an opposite constraint 1.8 mm away (P2-8) is the
case that matters, and it is still unmeasured.

**Update (P2.2): measured.** Overhang constraints on the `xs` ramps lose 0.1
to 1.8 degrees of tilt in the 32 passes, more the more tilt they ask for; an
option `hold_overhang` puts them back after each pass and removes the loss
(P2-12, `docs/orientation_field.md` section 7.2).

#### P2-3 Where the stock tilt near an overhang comes from (found in P2.1: the infill, see the update)

*Open question.* The per-surface data in the lab baseline
(`reports/baseline_overhang/ramp*_s_ms*.json`, `surfaces[0]`) shows the tilt
near each ramp's underside, which P0.8 summarised as
`theta_eff = theta_geo + tilt_used`:

| Part | max_slope | Flipped-normal tilt (90 - theta_geo) | Tilt near the underside | Mean theta_eff there |
|---|---:|---:|---:|---:|
| `ramp45_s` | 7 / 15 / 30 | 45 | 0.00 / 0.00 / 6.29 | 45.0 / 45.0 / 49.0 |
| `ramp50_s` | 7 / 15 / 30 | 40 | 0.00 / 0.00 / 17.14 | 50.0 / 50.0 / 63.2 |
| `ramp60_s` | 7 / 15 / 30 | 30 | 0.00 / 4.82 / 29.65 | 60.0 / 63.3 / 88.5 |
| `ramp70_s` | 7 / 15 / 30 | 20 | 0.00 / 9.68 / 20.64 | 70.0 / 78.3 / 89.5 |
| `ramp80_s` | 7 / 15 / 30 | 10 | 3.00 / 10.18 / 13.04 | 82.0 / 89.7 / 89.6 |
| `ramp90_s` | 7 / 15 / 30 | 0 | 0.69 / 3.49 / 10.14 | 89.9 / 89.7 / 89.2 |

The tilt always leans **away** from the overhang, is zero there at 7 degrees on
the four shallower ramps, and grows with the budget. In three cells it
matches the underside's normal flipped upward (`ramp60` at 30, `ramp70` at 30,
`ramp80` at 15), but it is not tied to that direction: it exceeds it on
`ramp80` at 30 (13.0 against 10) and `ramp90` at 30 (10.1 against 0).

**The code as read does not explain it.** The only branch that writes a
non-vertical constraint is the ceiling branch, which excludes cells whose
normal points down (`fff3.py:84-91`), so no underside cell is constrained. The
SDF is negative inside and its gradient points outward
(`solid3.sdf_create_from_bpn`, `sdf_compute_gradient_central`), so the sign is
as expected. The ramps' only upward faces are the flat top and the edges
rounded by the Blender remesh. The tool orientation passes unchanged from the
direction field to the toolpath (basis, triphasor and extraction copy the
normal; `order_atoms` writes it at `toolpath3.py:2228`), so the tilt is in the
field itself.

**Why it matters:** if an existing mechanism drives the field away from
overhangs, P2.2 must override or remove it, not only add a constraint beside
it. P2.1 should find it, most cheaply by running the field stage on a small
analytic ramp SDF on the CPU and looking at which cells are constrained.

**Update (P2.1, 2026-09-30): found. It is the infill.** `sdf_to_isdf.py`
overwrites the part's SDF with a shell (`fff3.sdf_generate_infill`,
`hollow_sdf = -sdf - wall_width`), which makes every cell deeper than 1.8 mm
"outside" again, apart from the gyroid walls. The SDF therefore has an inner
surface 1.8 mm under every outer one, facing the other way. Above an overhang
it faces up and away, at `90 - theta_geo` from vertical; it is not "pointing
down", so the ceiling rule constrains the field to it whenever that angle is
within `max_slope`. Measured by running stage 4 itself on exact ramp SDFs on
the CPU (`experiment/experiment_orientation_field_ramp.py`, bit-identical to
`tools/compute_tool_orientations.py`): `ramp70` at 30 degrees gets 3 910
constrained cells at 20.3 degrees, azimuth 179, 1.5 mm deep, and a tilt of
19.8 degrees next to the underside (baseline 20.6); **with infill off, 0 tilted
constraints and exactly zero tilt** on `ramp60`, `ramp70` and `ramp90`. Full
table and reasoning: `docs/orientation_field.md` section 4. What it means for
P2.2: P2-8.

#### P2-4 A stale figure in `overhang_metrics`'s docstring

*Documentation only, corrected.* The module docstring still said a point is
supported by material "within `1.5 x height`". The constant has been 2.5 since
correction 1.8, and the constant's own comment says why. Fixed with this
entry; no behaviour changed.

#### P2-5 P5.1a built early, for P2.0, with a `--stop-after` option (deliberate deviation)

*Operator's decision, 2026-09-29.* P2.0 has to run the pipeline up to atom
extraction and stop. The plan offers `run_stages` (P5.1a) or a local copy of
the command list; the operator chose P5.1a now, so the commands exist in one
place and P2.2's new orientation options will be added once.

As built (`tools/atomize.py`, a vendored edit):

* `STAGE_NAMES`: the 15 stages in order, named after their tools
  (`process_for_atomizer` … `ratrig_to_craftware`). `sdf_to_isdf` exists only
  for a part with infill.
* `build_stage_commands(params)` returns `Stage` records (name, command, the
  step heading it prints, its log heading, whether `--warmup` repeats it).
  The command strings are upstream's, unchanged.
* `run_stages(params, only=None, skip=None, warmup=False, stages=None)` runs
  them in pipeline order, prints each step heading once and writes the log
  headings as upstream did. Beyond the plan's `(params, only, skip)`: `warmup`
  (upstream's `--warmup`) and `stages` (to run a list already built). An
  unknown name raises `ValueError`.
* **New option:** `atomize.py PARAMS --stop-after STAGE`. It writes the same
  log header as a full run, runs the stages up to and including `STAGE`, and
  then prints and logs "Stopped after STAGE (--stop-after); the later stages
  did not run." Without it the script behaves as before.
* **Still not checked: stage exit codes** (4.16). `run_stages` runs each
  command with `os.system` like upstream and carries on after a failure.
  `overhang_report.first_failed_stage` remains the check; P5.1's driver needs
  one too.

**How "unchanged" was shown without a laptop.** `tests/_atomize_record.py`
runs the script with every stage stubbed out and records each command, each
printed line and the log. `tests/fixtures/atomize/recorded.json` is that record
taken from the script **before** the change (commit `f5ffcd2`), for four
parameter sets: the calibration cube, a part without infill, a part using
every optional key, and `--warmup`. `tests/test_atomize_stages.py` requires the
refactored script to reproduce it byte for byte, and it does. Breaking one
heading's capital letter and adding one space to one command made 9 of its
tests fail. The golden test on the laptop remains the check that counts,
through the G-code.

**Also found:** a fresh session container needs **Pillow** and **pyvista**
as well as the packages listed in 3.11: `atom.line` imports Pillow and
`atom.solid3` imports pyvista, so any test that runs a pipeline stage's
imports fails without them. Both are runtime dependencies in
`pyproject.toml`, so CI has them.

#### P2-6 P2.0 as built: what a field-only report is, and where it lives

*Definitions and choices within P2.0.*

**Step 1 answered: the frame file is what the plan assumed.**
`extract_explicit_atoms` writes `data/frame/<part>.npz` through
`frame3.Field.save_active_frame_set`: `point (M, 3)` float32 mm, `normal
(M, 2)` float32 spherical `[theta, phi]`, `phi_t (M,)`, one row per atom
that survived extraction, NaN rows already removed. `normal` is the
orientation field at the atom's cell, copied unchanged through the basis,
triphasor and extraction stages; `order_atoms` copies it unchanged into
`tool_orientation` (`toolpath3.py:2228`), and `smooth_toolpath_point` moves
positions only. So every deposition point of `_smoothed.npz` carries exactly
one atom's orientation. `atom.frame_atoms` reads the file and presents it with
the attributes `atom.overhang_metrics` reads from a toolpath, so both modes are
measured by the same functions.

Two differences from a full run's measurement, both expected to be small and
both covered by the plan's 1-degree check: every atom counts, while a full
run's deposition mask leaves out the first atom of each deposition run
(reached by a travel move; 30 603 deposition points against 31 630 atoms on
the golden cube, though that toolpath is also tessellated); and positions are
the atoms' own, before smoothing moves them along the path.

**What a field-only report says.** `mode: "field_only"` (a report without
`mode` is a full run's, as all existing ones are); `atoms.count`; both
unsupported fractions `null` with `unsupported_note` "n/a (field-only) …";
`atoms_near_overhangs` in place of `deposition_points_near_overhangs`. The
verdict cannot be "printable": `printable` is **False** when the worst
effective angle is over the threshold, which rules a part out on its own, and
**null** otherwise, with `effective_within_threshold` saying which. In the
summary that is ❌ or ❔, never ✅.

**Where it lives (the plan's "field-only column set", step 3).** Reports in
`reports/field_only/`, atoms archived to `reports/frames/` (gitignored) for
`--reanalyse`. `--summarize` adds a separate section and table, "Field-only
runs (build plan P2.0)", after the conclusion, saying the two are not
comparable; the main table is byte-identical to before (tested against the
committed summary). A separate table rather than extra columns in the main
one, so no cell can be read across. `load_reports` refuses a report of the
wrong kind in either folder. `--status`, `--check-done` and
`run_matrix_parallel.py --resume` read only the full-run folder, so a
field-only report can never count as a completed full run.

**Not written to `reports/matrix_progress.csv`.** That log has no column for
how a run was made, so a field-only row would read as a full run's.

**Provenance.** A field-only run records the backends of the stages it ran
only, so it never shares a comparability group with a full run. Consequence
for P2 iterations: an overhang-aware field-only result must be compared with a
**stock field-only** result from the same machine, not with the committed
baseline. No stock field-only set exists yet (open; see handoff 0d).

**The pipeline check** (`tests/test_field_only.py`, laptop): `ramp60_xs` at a
30-degree budget, where stock Atomizer tilts by about 29 degrees, so the check
is not met trivially by an untilted field. It runs field-only, then the full
pipeline, and requires: the atoms of the two runs identical; every orientation
the full run deposits with to be one of the atoms'; the field-only maximum tilt
to be at least the full run's (the atoms are a superset); the worst, and each
surface's maximum and mean, effective angle within 1 degree; and the
field-only run under half the full run's time. It prints both times, which
P2.0's "Done when" asks to record.

**Laptop result (2026-09-30).** Every correctness check passed on
`ramp60_xs` at 30 degrees: the atoms of the two runs identical, every deposited
orientation an atom's own, worst effective overhang **89.097 field-only against
89.097 full**, maximum tilt 29.097 both, surfaces within the 1-degree criterion.
The timing check alone failed: field-only 303.3 s against 585.0 s, ratio
0.518 against the test's 0.5. **The test was wrong, not the feature**: the
field-only run went first and paid Taichi's kernel compilation, its direction
and layer stages taking 46.1 and 45.4 s against 10.0 and 12.4 s in the full run
straight after, on identical input. The test now runs field-only, full, then
field-only again, times only the third, and also checks that a field-only run
leaves the toolpath file untouched, which catches a broken `--stop-after`
without relying on time.

**Warm timing, recorded (laptop, 2026-09-30, P2.0's "Done when").**
`python tools/overhang_report.py data/param/ramp60_xs.json --max-slope 30
--field-only`, straight after the runs above: **126 s**, against **585 s** for
the full run the same day on the same laptop, a ratio of **0.22**. Its stages
took 10.9 (direction), 13.3 (layers), 20.0 (tangents), 40.7 (alignment) and
1.3 s (extraction); the full run's planner alone took 425.1 s. Against the
laptop's archived full run of the same part (411 s, 2026-09-22, when its
planner took 298.6 s) the ratio is 0.31. The same run measured 25 862 atoms,
worst effective overhang 89.1 degrees, max tilt 29.10 degrees, "not
printable", as the full run did. It was the first use of the command line on
the laptop: stop, archive, report and verdict all worked.

**Not built:** a `--field-only` option on `tools/run_matrix_parallel.py`.
Single field-only runs are a few minutes each at size `xs`; whether the P2
iterations need a parallel field-only matrix on the lab machine is the
operator's call.

#### P2-7 `FLOOR_MAX_ANGLE` is read, by stages 5 and 6 ★ PLAN EDIT

*Factual error, in correction 1.2 and in the plan.* Correction 1.2 says the
constant "exists and holds 1.0 degree, but nothing reads it", and P2.1 asks to
"say this explicitly so nobody mistakes it for a live constraint". Only the
first half is right: the **orientation field** does not read it (its floor
branch is commented out, `fff3.py:92`, `:101`). But:

* `fff3.init_phasor3_field_from_sdf` (stage 5, layers) pins layer positions
  on floors within `FLOOR_MAX_ANGLE` of straight down (`fff3.py:293`, `:313`);
* `fff3.init_deposition_tangent_field` (stage 6, tangents) uses it for bottom
  surfaces when a `bottom_lines` image is given (`fff3.py:235`).

`sdf_df_to_layers.py` and `compute_tangents.py` both set it to 1 degree
explicitly. So a flat underside (within 1 degree) fixes where a layer lies,
and a sloped one does not. P2.2 should not reuse the name for an overhang
threshold without deciding what that does to the layers.
`docs/orientation_field.md` section 3.

#### P2-8 With infill on, P2.2's overhang constraint would face a stock constraint pointing the other way (for P2.2's design)

*Hazard for P2.2; found in P2.1.* P2.2 constrains the outer underside cells to
`[t, azimuth of n]`, toward the overhang. With infill (every benchmark part,
and the calibration cube), stock's ceiling rule constrains the infill's inner
surface 1.8 mm above them to `-n`, away from it (P2-3). On `ramp60` at 30
degrees with `t = 17`, the two directions are 47 degrees apart across ten
cells, and the 32 final passes average both (P2-2). The plan's P2.2 text does
not anticipate this.

Options, for the operator with P2.2's design (`docs/orientation_field.md`
section 5):

* **(a)** when `overhang_aware`, compute the orientation field on the SDF
  **before** infill; the later stages keep the infilled one, so the infill
  still prints. Needs the pre-infill SDF kept (stage 3 bis overwrites it in
  place today);
* **(b)** keep the infilled SDF but exclude inner surfaces from the ceiling
  rule, which needs the pre-infill SDF to tell them apart;
* **(c)** leave it and let the constraints fight: not recommended.

(a) and (b) also change stock's behaviour wherever the gyroid's surfaces were
acting as ceilings, so P2.5's top-surface check should cover it. Both are
behind the `overhang_aware` flag, so the golden output is untouched.

**Decided by the operator, 2026-09-30: option (a).** When `overhang_aware` is
on, the orientation field is computed on the SDF before infill; the later
stages keep the infilled SDF. To be built as part of P2.2.

A related question for the team: stock **without** infill is exactly vertical
on the ramps (`theta_eff = theta_geo`), a fairer 3-axis-like reference than
stock with infill, which tilts away. Whether to add it to the comparison is
open; it is cheap in the field-only tier.

#### P2-9 P2.1 as built

*Definitions within P2.1.*

* **`tools/tilt_bound.py`**: steepest printable overhang =
  `max_overhang + usable tilt`, at most 90; usable tilt = the machine's limit
  capped at **50 degrees by the nozzle** (`fff3.MAX_SLOPE_ANGLE`, from the
  80-degree `NOZZLE_CONE_ANGLE`; the plan's bound omits the cap). Each part is
  **inside** (P2.5's criterion, `theta_geo <= max_overhang + usable tilt -
  margin`), **edge** (inside only without the margin; not counted) or **out of
  range**. The 90-degree cap is applied after the margin, so at 50 degrees of
  tilt a flat ledge (needing 47 with the margin) is inside. Parts are read
  from `atom.benchmark_meshes` (the T-shape at its default underside unless
  `--tshape-underside` is given); the default tilt is the machine profile's;
  several `--tilt` values give a column each, for gate D0.
* **`docs/orientation_field.md`**: the field from SDF to toolpath with line
  references, the constraint rules, the multigrid aligner, what the later
  stages read, the mechanism of P2-3 and what it means for P2.2.
* **`experiment/experiment_orientation_field_ramp.py`**: the P2-3 experiment,
  kept so the finding can be re-checked. Development numbers only.

#### P2-10 Gate D0: the mechanical target is 60 degrees per axis, not yet a value

*Input from the operator, 2026-09-30; not a decision.* The mechanical team is
still drawing the machine; their **goal is 60 degrees of tilt on each axis**.
So D0 has no value yet, and the operator asked whether the slicer can take the
tilt limit as a variable that everything else follows.

What was checked before answering (reference geometry, the tilt check in the
inverse kinematics lifted to 89 degrees, a throwaway profile, CPU): the
reference machine's screws can reach 60 degrees, but only with the part raised
off the bed, because the bed's corners swing into the gantry. The lift the
kinematics ask for:

| Where the nozzle is | First tilt needing lift | Lift at 30 deg | Lift at 60 deg |
|---|---|---|---|
| bed centre, 2 mm up | 20 deg | 46 mm | 139 mm |
| bed centre, 30 mm up | 25 deg | 22 mm | 125 mm |
| bed centre, 80 mm up | 40 deg | none | 100 mm |
| 50 mm off centre, 2 mm up | 20 deg | 64 mm | 168 mm |

(Worst of 16 tilt directions; the IK's lift is a first estimate, P1-9, and
`add_platform` would print a platform at least that tall. At 80 mm up, 5 of
the 16 directions are unreachable outright from 55 degrees, a screw running
out of travel.) So how much of a 60-degree bed is usable depends on the bed's size and
the nozzle's length below the gantry (gates M1, M3), not only on the tilt
mechanism. Separately, Atomizer caps the field at 50 degrees for its 80-degree
nozzle cone whatever the bed allows (P2-9); using 60 needs a nozzle cone of 60
degrees or narrower.

#### P2-11 The nozzle cone is a machine constant; a `dev60` development profile (operator's decisions)

*Decisions by the operator, 2026-09-30, answering P2-10.* The tilt limit
is a **cone** of 60 degrees (60 degrees total in any direction), the target,
not yet a value (D0 stays open). The operator asked for:

1. **`nozzle_cone_angle_deg` in every machine profile.** Upstream's
   `toolpath3.NOZZLE_CONE_ANGLE = 80.0 * pi / 180.0` is now
   `load_profile().nozzle_cone_angle_deg * pi / 180.0` (a vendored edit;
   `reference` and `ours` say 80.0, which reproduces the literal bit for bit,
   tested). It sets the planner's collision cone and, through
   `fff3.MAX_SLOPE_ANGLE`, the field's tilt cap `90 - cone / 2`. Like the
   kinematics constants, it is fixed when `toolpath3` is imported (plan hazard
   12). The P4 clearance model and nozzle check (`clearance.nozzle_half_angle_deg`,
   `NozzleCheckSettings.for_profile`) and `tools/tilt_bound.py` now take it
   from the profile too. The golden test is owed on the laptop.
2. **`config/machines/dev60.json`**: `reference`'s geometry with
   `max_tilt_angle_deg` 60 (cone) and `nozzle_cone_angle_deg` **60**, the
   widest cone that lets the field use 60 degrees (the operator chose it over
   upstream's 80, which would cap the field at 50). A requirement on the
   hotend, not a measured nozzle. `status: PLACEHOLDER`, so it warns. A test
   pins that it differs from `reference` in those two numbers only. Select it
   with `ATOM_MACHINE=dev60`. On this geometry large tilts near the bed need
   the part lifted (P2-10), so its G-code is for inspection only; field-only
   work (P2.0) does not use the kinematics at all.
3. **Overhang-aware runs take their tilt budget from the profile**: built with
   P2.2 (P2-12).

#### P2-12 P2.2 as built: a separate kernel, the field on the solid part, hold, and the budget from the profile

*Definitions and deviations within P2.2; the results are in
`docs/orientation_field.md` section 7.*

* **No `fff3.py` flag.** The plan puts the overhang branch in `fff3.py` behind
  a module flag `OVERHANG_AWARE`. Built instead: a new kernel,
  `init_overhang_aware` in `src/atom/orientation_field.py`, which is
  upstream's initialisation line for line plus the overhang branch, and
  `compute_direction_field`, upstream's stage-4 steps plus P2.2's options.
  `tools/compute_tool_orientations.py` (vendored) takes the new path only when
  given `--overhang_aware` or `--solid_sdf`, before any of its own code runs;
  otherwise it runs upstream's code, untouched. Reasons: `fff3.py` stays
  byte-identical, and a module flag is baked in at compile time (hazard 12),
  which makes flag-off identity harder to prove than a separate path. The
  plan's flag-off test is met twice: the kernel with the rule off equals
  upstream's initialisation, and `compute_direction_field` with its options
  off equals the stage's output bit for bit (`tests/test_orientation_field.py`).
* **Option (a) routing** (P2-8). With `overhang_aware` and `infill`,
  `atomize.py` has `bpn_to_sdf` write `data/sdf/<part>_solid.npz`,
  `sdf_to_isdf` read it and write `data/sdf/<part>.npz`, and stage 4 get
  `--solid_sdf data/sdf/<part>_solid.npz`. The field is computed on the solid
  SDF, then NaN where the infilled SDF is outside, the cells upstream masks.
  Measured: without it the rule fires all through the hollow and the field
  next to `ramp60`'s underside is 57 degrees effective, with it 44.
* **`hold_overhang`** (new, not in the plan; P2-2's first option). Re-applies
  the overhang constraints after each final pass. Measured: the passes cost
  0.1-1.8 degrees of tilt without it, none with it. **On by default: the
  operator's decision, 2026-09-30.** `"hold_overhang": false` switches it off;
  the stage takes `--hold_overhang` / `--no_hold_overhang` (two explicit
  options: argparse's `BooleanOptionalAction` would have spelled the second
  `--no-hold_overhang`, which a test caught). Upstream smooths its
  constraints on purpose, for smoother tool motion, so whether holding makes
  the nozzle turn less smoothly next to overhangs is for P2.5's check.
* **The budget.** `max_tilt_here` waits for P2.3's map; until then the cap is
  `fff3.MAX_SLOPE_ANGLE`, `min(max_slope, 90 - cone / 2)`, which is also the
  ceiling threshold. By the operator's decision (b) of 2026-09-30,
  `max_slope` comes from the machine profile when the parameter file has none
  (`atomize.py`; a vendored edit), and `overhang_report.py --overhang-aware`
  uses the profile's limit unless `--max-slope` is given, because the part
  files' `max_slope` (7) is stock Atomizer's.
* **`overhang_priority` is not built.** The plan's conflict ("a cell also
  qualifies as a ceiling") cannot happen: a cell has one closest normal,
  pointing up (ceiling) or down (overhang). The real conflict is between
  neighbouring cells through the smoothing, which the solid SDF and hold
  address. No count is logged; there is nothing to count. P2.6's key list
  still names it: whether it should mean something else is for D1.
* **Parameter keys built now** (P2.6 in part): `overhang_aware`,
  `max_overhang_deg`, `overhang_margin_deg`, `hold_overhang`. The rule's keys
  without `overhang_aware`, and `overhang_aware` with `ortho_to_wall` or
  `all_up` (not supported on the new path), are refused when the file is
  read, before any stage. `config/schema/params.schema.json` does not exist
  in this repository, so there is no schema to extend. The stage refuses the
  same combinations on its own command line.
* **The report tool.** `overhang_report.py --overhang-aware` writes
  `<part>_ms<deg>_aware.json` (and archives) beside the stock ones, records
  `"overhang_aware": true`, is kept out of `matrix_progress.csv` (it has no
  column for it; P2.5 adds one), and gets its own section of the summary; the
  field-only section splits stock from overhang-aware. Without the option
  nothing changes, and the committed summary is reproduced exactly.
* **The tests' ramp is not 16³.** The plan's tiny SDF has no room for the
  infill's 1.8 mm shell to leave a hollow, which is the case that matters. The
  tests use a 12 x 5.4 x 7.2 mm ramp (72 x 33 x 43 cells); on the CPU a field
  costs about 20 s whatever the size, because Taichi compiles the aligner's
  kernels for every new field, so the file shares three fields and takes
  one to three minutes. **It stays in the default (unit) tier, the
  operator's choice of 2026-09-30**, although `pytest.ini` describes that tier
  as tests of about 2 s: the whole default run is about 3 minutes.
* **Found while measuring, for D1:** the rule is a threshold (at 45.1 degrees
  it asks for 2.1, at 45 for nothing); a flat underside's azimuth is decided
  by rounding, harmless on the analytic ramps, untested on a remeshed part;
  and even with hold the worst cell next to the underside is 0.6-3.4 degrees
  above 45 on `ramp70` to `ramp90`, so the 2-degree margin covers the mean,
  not the worst cell. **The operator's answer, 2026-09-30: keep it** (the
  2-degree margin, and the rule switching on at 45 with no fade-in).
* **Laptop, 2026-09-30, on `0c7bad9`:** golden test 5 passed, 1 skipped
  (460 s), so the output with the option off is unchanged after the vendored
  edits. The sign check on real output: field-only `ramp60_xs` at 30,
  overhang-aware, **worst effective overhang 56.1 against stock's 89.1**, max
  tilt 21.6. The field leans the right way; the remaining excess sits where
  the column's wall turns into the overhang, which the tilt has not finished
  turning to reach (reproduced on the CPU: worst 52.1 there, mean 43.2 over
  the middle of the underside; `docs/orientation_field.md` 7.4). That is
  P2.4's case (tilt ramp-in). The plan's fitted relation
  (`theta_eff = theta_geo - tilt_used` over the ramps) needs the P2.5 runs.

#### P2-13 P2.4 as built: the ramp-in steepens instead of capping; the metric counts supported atoms at a corner

*Definitions within P2.4, one operator decision, and a finding about the P0.8
metric. Results in `docs/orientation_field.md` sections 7.4 and 7.5.*

* **The ramp-in** (`src/atom/ramp_in.py`, numpy, applied in
  `atom.orientation_field` between the initialisation and the solve). "Below
  each constrained overhang cell, along the build direction" is read as:
  step back from the cell along its own direction `-d`, half a cell at a
  time, through the material printed before it. Ramp cells get the cell's
  azimuth and `t - rate * s`; they never replace an existing constraint, and
  where walks cross the largest tilt wins. A walk ends at air (under most of
  an overhang, immediately) or at the first layer.
* **Too little room: steepen, not cap (operator's decision, 2026-09-30).**
  The plan says to cap `t` when the base is too close. On `ramp60_xs` (the
  corner 4.5 mm above the bed, 5.7 mm needed at 3 degrees per mm) that
  would give the corner 13 degrees instead of 17. Asked, the operator chose
  to keep the full tilt and build it up faster in the room there is; the
  stage logs how many walks were steepened and the steepest rate (3.7
  degrees per mm on `ramp60_xs`). Not the plan's per-overhang shortfall log:
  one line per run.
* **"Softly constrained"** is read as: constrained for the multigrid solve,
  smoothed in the 32 final passes like upstream's own constraints; hold
  restores only the overhang cells.
* **Keys:** `ramp_in` (default true; not in the plan's P2.6 list, added so
  the ramp can be switched off for comparisons) and
  `max_tilt_rate_deg_per_mm` (P2.6's name, default 3, gate D3). Stage
  options `--ramp_in` / `--no_ramp_in`, `--max_tilt_rate`.
* **Measured** (CPU, exact SDF, `ramp60_xs` at 30): the reported worst
  effective overhang goes from 52.1 to 48.8; the atoms really out over the air
  are unchanged (worst 46.4, mean 43.5).
* **Finding: the effective-overhang metric counts supported atoms at a
  corner.** `overhang_metrics.effective_overhang_angles` takes every atom
  within one bead width of an overhang face. Where a wall turns into an
  overhang, that includes atoms printed on top of the wall, which are
  supported: on `ramp60_xs` they, not the overhang, give the worst value
  (52.1 against 46.4 for the atoms over the air). The same applies to the
  stock baseline (its column atoms: 82.5 against 89.3 over the air, so its
  worst is unchanged there). Counting only atoms with air below them would
  change the P0.8 metric, so the baseline would be re-scored with it
  (`overhang_report.py --reanalyse`, on the machine holding the archived
  toolpaths). **The operator chose to change it: P2-14.**
* **The plan's pipeline test** ("the tilt at the first overhang layer >= t -
  2", 15 on `ramp60_xs`): over the first 0.5 mm of overhang the tilt is 16.1
  on average and 13.6 at least (CPU). Not written yet; it waits for the
  laptop's numbers on the remeshed part.

#### P2-14 The effective overhang angle counts only points out over the air (metrics version 3) ★ METRIC CHANGE

*The operator's decision, 2026-09-30, answering P2-13.* The P0.8 effective
overhang angle took every deposition point within one bead width of an
overhang face. From now on a point counts only if it is **out over the air**:
the spot one layer height (half the deposition width) back along its build
direction `-d`, where the layer below it would be, is outside the part and
above the bed. Points printed onto material (the wall below a corner, the
deeper layers above the outermost one) or onto the bed are skipped, and
counted per surface in the report as `supported_samples_skipped`.

* **Code:** `overhang_metrics.points_over_air` (inside test: the mesh's
  generalised winding number, `winding_number`, numpy only; `rtree` is not
  installed, so trimesh's own test is unavailable) and new optional arguments
  of `effective_overhang_angles`; `overhang_report.py` passes the part's mesh
  and layer height for full and field-only reports. `METRICS_VERSION` 2 -> 3.
  The unsupported-deposition metric is unchanged.
* **Effect, CPU runs of `ramp60_xs` at 30 (exact SDF):** stock 89.3 before and
  after; overhang-aware 52.1 -> **43.2**, with the ramp-in 48.8 -> **43.3**.
  Only the outer layer's edge now counts: about 180 atoms along the underside
  instead of about 650.
* **The committed baseline must be re-scored.** All 48 reports in
  `reports/baseline_overhang/` are the lab machine's (`cad-p07-2065-9`),
  version 2. Until they are re-scored they count as stale:
  `run_matrix_parallel.py --resume` would re-run them. On the lab machine,
  which holds their archived toolpaths: `git pull`, then
  `python tools/overhang_report.py --reanalyse`, then commit and push
  `reports/baseline_overhang/` and `reports/baseline_overhang.md`.
* **A guard found on the way:** archives are looked up by part and slope
  only, and the laptop holds its own older toolpaths under the same names.
  `--reanalyse` there would have scored the lab's reports on the laptop's
  toolpaths. It now skips any report whose provenance names another
  machine ("measured on ...; re-score it there").
* Two runner tests that impersonate the committed reports' machine now also
  take the committed reports' metrics version, so they keep testing the
  machine logic whether or not the re-score has happened.
* **`tools/overhang_where.py`** (new, diagnostic): for each overhang surface,
  the points over air and on material, and the worst points with position and
  tilt. For finding where a part's worst value comes from.

#### P2-15 One atom at the ramp's tip corner: the overhang rule does not reach sharp corners

*Found on the laptop, 2026-09-30.* `ramp60_xs` at 30, overhang-aware with
ramp-in, re-measured with metrics version 3: worst 56.1, from **one** atom
of 148 over the air, at (29.76, 0.32, 9.72), the corner where the underside
meets the end face (`x` = 30) and the side face (`y` = 0), tilted 4
degrees. The other 147 are at most 44.7 (mean 42.9). The overhang rule reads
each cell's closest surface normal; in that corner the closest surface is
the side or end face, so no overhang constraint is set and the field there
stays near vertical. The same will happen at every corner where an overhang
meets two walls (the T-shape has four). The CPU run on the exact SDF did not
show it (43.2), so it depends on the remeshed surface. **The operator's
decision, 2026-09-30: leave it for now** and see in P2.5's runs (every part,
the T-shape's four corners included) how often and how badly it happens,
then decide on a fix with those numbers.

#### P2-16 P2.5 prepared: the matrix runner's `--overhang-aware`, and stock beside overhang-aware in the summary

*Built 2026-09-30 at the operator's request; the run itself is the lab
machine's.*

* **`tools/run_matrix_parallel.py --overhang-aware`**: the same matrix with
  the overhang-aware field. Every job's report, archived toolpath and worker
  log is named `<part>_ms<deg>_aware`, so it lands beside the stock files and
  never over them (the runner named everything by part and slope only, so
  without this an overhang-aware matrix would have overwritten the stock
  archives the comparison needs). `--resume` looks at overhang-aware reports
  only; a job with no overhang-aware runtime yet is estimated from the stock
  run of the same part and slope. Dry run for `--sizes xs s --workers 8`: 48
  runs, **4:39 to 5:39** at the measured 79-65 % efficiency (the plan said
  3.5-4 h, from the lower bound).
* **Which runs:** the plan says "re-run the full P0.8 matrix (xs + s, as the
  baseline) with overhang_aware on, max_slope = machine limit". Its success
  criteria compare at every budget ("`ramp45` still passes at every budget",
  "the same 18-run subset"), so the runner's default is the baseline's own 48
  combinations, slopes 7, 15 and 30, each against the stock run with the same
  slope. `--slopes 30` runs the machine limit alone (16 runs).
* **The summary:** the overhang-aware section is now **Stock vs
  overhang-aware**: one cell per part and slope, `stock -> overhang-aware`,
  each as mark, worst effective overhang and unsupported deposition, with a
  count of printable pairs. Above it, a warning when the pairs are not all in
  one provenance group (the plan's "one provenance group" requirement) or
  were scored with different metrics versions (the case until the lab
  re-scores the stock baseline to version 3, P2-14).
* **The progress log** gains an `overhang_aware` column (older rows padded
  empty; all of them are stock), so overhang-aware full runs are now logged.
* **Not yet built** (the plan's other P2.5 numbers): top-surface quality and
  the maximum tilt rate along the toolpath. Both can be measured from the
  archived toolpaths and the mesh, so they can be added after the lab run and
  applied with `--reanalyse` without re-running anything.

#### P2-17 Metrics version 3 was flawed; version 4 counts the points whose nearest surface is an overhang ★ METRIC CHANGE

*Found in the lab machine's P2.5 data, 2026-10-01; the operator's decision
the same day.* Version 3 (P2-14) counted a point only if the spot one layer
height straight back along `-d` was outside the part. On an overhang of 45
degrees or less, each bead hangs only half over the one below, so the spot
under its centre is inside the bead below: version 3 called such beads
supported and dropped them. On `ramp45_xs` it kept **none** of about 4 000
samples at four of six budgets, so the table read "no overhang", hiding
stock's known 51-degree failure at 30. Gentle overhangs became unmeasurable.

**Version 4:** a point near an overhang face counts when its **nearest
surface is an overhang face**, nearer by 0.05 mm than any other face
(`overhang_metrics.nearest_surface_is_overhang`, exact point-to-triangle
distances on the part's mesh, `triangle_distances`). A point against the wall
below a corner is nearer the wall and is skipped; a point at the corner's
shared edge is equally near both and is skipped; every bead along an
underside counts, whatever its angle. Same as version 2 everywhere except
next to other surfaces. Report key `skipped_samples` (was
`supported_samples_skipped`).

On the CPU `ramp60_xs` runs (exact SDF): stock 89.3 (unchanged), overhang-aware
with ramp-in **45.3** (mean 43.2), keeping 5 923 of 6 834 samples. The
version 3 code (`points_over_air`, `winding_number`) is removed.

**The lab machine re-scores again** (archived toolpaths, seconds): the 48
stock and 44 overhang-aware reports are version 3.

#### P2-18 P2.5's first overhang-aware matrix (lab machine, 2026-10-01; metrics version 3)

`run_matrix_parallel.py --sizes xs s --workers 8 --overhang-aware`, commit
`3ce2cfe`, 3:42 for 48 runs; **44 completed, 4 failed**. Stock and
overhang-aware in one provenance group (cad-p07-2065-9, stock mix, Taichi
1.7.4, reference). Selected cells, worst effective overhang stock ->
overhang-aware (`reports/baseline_overhang.md` has them all):

| Part | 7 | 15 | 30 |
|---|---|---|---|
| `ramp45_s` | 45 -> 43 | 45 -> 43 | **51 -> 43** |
| `ramp60_s` | 60 -> 53 | 65 -> 45 | **90 -> 45** |
| `ramp70_s` | 70 -> 63 | 80 -> 55 | **90 -> 44** |
| `ramp80_s` | 83 -> 73 | 90 -> 65 | 90 -> 50 |
| `ramp90_s` | 90 -> 83 | 90 -> 75 | failed |

* **The relation flips**, as P2.5 asks: wherever the budget is short of what
  the rule wants, the effective angle is the geometric angle minus the budget
  (60 - 7 = 53, 70 - 15 = 55, 80 - 30 = 50); where it is enough, 43-45.
  Stock was the geometric angle **plus** its tilt.
* **Printable: stock 3, overhang-aware 5** of the 44 pairs. Most
  overhang-aware runs within 45 degrees still fail on **unsupported deposition
  near the overhang, 0.8-2.4 % against the 1 % limit** (`ramp50` at every
  budget, `ramp60_xs` and `ramp70_xs` at 30). Where those points are is not
  known yet.
* **The 4 failures** are all at 30 degrees on the flat undersides
  (`ramp90_xs`, `ramp90_s`, `tshape_xs`, `tshape_s`): stage 9, `order_atoms`,
  wrote no toolpath. Its only exit is the planner finding no valid next atom
  ("maybe two adjacent atoms are constraining each other"). Being reproduced.
* These numbers are version 3 (P2-17): `ramp45_xs`'s "no overhang" cells are
  that flaw, not results.

**Update, re-scored with metrics version 4 on the lab machine (`a29df7f`,
all 92 reports):** printable **stock 4, overhang-aware 6** of 44 (`ramp45_xs`
is measured again: stock fails at 30 with 51 degrees, overhang-aware 44).
Per surface, the **mean** effective angle of the overhang-aware runs is
42.5-43.2 wherever the budget allows the rule's tilt (`ramp45` to `ramp70`
at 30, `ramp50` at every budget), and exactly the geometric angle minus the
budget where it does not (`ramp60` at 7: 53.0; `ramp70` at 15: 55.1; `ramp80`
at 30: 50.1; `ramp90` at 15: 75.4). The **worst** point sits 4-11 degrees
above the mean (`ramp60_s` at 30: 49 against 42.7; `ramp70_xs` 50 against
43.1; `ramp80_xs` 61 against 50.2; the flat undersides 89-90 against 75-83):
a few spots per part, the edge and corner effect of P2-15, which now decides
whether a part passes. Unsupported deposition near the overhangs, where the
budget suffices: 0.1-2.0 %.

#### P2-19 The overhang constraint carried to the overhang's edges (operator's decision, 2026-10-01)

*Answering P2-15 with P2-18's numbers.* The overhang rule, like upstream's
ceiling rule, constrains only low-curvature boundary cells whose own normal
is steeper than `max_overhang`. Along an overhang's edges neither holds: the
curvature is high and the central-difference normal blends the two faces.
Measured on the 60-degree test ramp: within 0.5 mm of the side walls **26 %**
of the boundary cells next to the underside were constrained (43 % low
curvature, 49 % steeper than 45 degrees), near the tip 44 %, in the middle
100 %. So the worst point of most parts in P2.5 sat at an edge, 4-11 degrees
above the surface's mean.

**`atom.overhang_edges.fill_overhang_edges`**, between the rule and the
ramp-in: every unconstrained boundary cell (inside, within one layer height
of the surface, not in the first layer) within **two layer heights** of an
overhang cell takes the nearest overhang cell's direction (`scipy.ndimage`
distance transform), **unless it lies behind that cell along its build
direction**: that is the material printed before it, such as the wall below a
corner, which is the ramp-in's job (copying the full tilt there put 27
degrees a millimetre above the bed on the test ramp and asked the ramp for 40
degrees per mm). The filled cells are marked `MARK_OVERHANG_EDGE`, held with
the overhang cells, and the ramp-in starts below them too. Key
`overhang_edges` (default true), stage options `--overhang_edges` /
`--no_overhang_edges`; the stage logs the count.

On the test ramp (exact SDF): the boundary cells next to the underside within
0.5 mm of the side walls are now all constrained, and the field there leans
0.9 degrees further (25.9 -> 26.8 of the rule's 27). The exact SDF has
cleaner edges than the remeshed one, so the laptop's remeshed `ramp60_xs`
is the real test.

**Update (2026-10-01, after the laptop's runs):** on the laptop's remeshed
`ramp60_xs` the fill above changed nothing at the worst point (56.1, one atom
at the tip's corner, tilt 4.0 before and after). The reason: at the tip the
rule *does* fire, but on a normal blended with the end face, which it reads
as a gentler overhang (about 50 degrees), so it asks for 9-13 degrees of tilt
where the underside asks 17; the same at the column's corner (12-14). The fill
only touches cells the rule skipped. **Revised:** `fill_overhang_edges` now
also corrects the overhang cells near another surface. The overhang's
**core** is its cells further than two layer heights from any other surface;
both kinds of edge cell (skipped, and blended) take the direction of the
nearest core cell within twice that distance. Spreading the strongest nearby
lean instead was tried and rejected: on the remeshed surface it carried the
noise peaks everywhere and raised the whole underside from 17.0 to 21.7
degrees. With the core: the underside's middle is unchanged (17.0 on average),
the edge strip 17.1 (was 16.8, with a tail down to 4), the tip corner 17.6-18.4.

#### P2-20 One outward lean under a flat overhang; the four `order_atoms` failures

*Found 2026-10-01, reproduced without the laptop.* **The reproduction:** a
pipeline-like SDF built here without Blender: the STL subdivided to 0.12 mm
triangles with vertex normals averaged across edges (as Blender's remesh and
smooth shading round them), then the real stages 2, 3 and 3 bis on the CPU.
Same grid as the laptop's (178 x 81 x 107 for the `xs` ramps). On it,
`ramp90_xs` at 30 overhang-aware **deadlocks in `order_atoms` exactly as on
the laptop** ("No more valid next atom ...", at atom 25 233 here, 25 672 on
the laptop); the exact SDF did not.

**The cause:** under a flat underside the normal's horizontal part is noise,
so the rule's azimuth `atan2(n_y, n_x)` points every way (on `ramp90_xs`,
evenly over all eight 45-degree sectors), and the field a millimetre above
the underside averages to 2 degrees of tilt where the constraints ask for 30.
Neighbours leaning apart can each wait for the other in the planner.

**The fix, `atom.overhang_azimuth.aim_flat_overhangs_outward`:** overhang
cells at least 80 degrees steep (horizontal normal at most 0.17) take the
azimuth of the walls' and the overhang's own edge normals around them,
smoothed with a 3 mm Gaussian: away from the material that holds the overhang
up, toward its free edges. Their tilt stays the rule's. On a flat underside
every lean lowers the effective angle equally, so only consistency matters,
and outward is the direction in which each layer reaches past the one below
it. On `ramp90_xs`: all azimuths within 90 degrees of +x, the field above the
underside at 26.3 degrees of tilt (was 2.1). Key `flat_overhangs_outward`
(default true), stage options `--flat_overhangs_outward` /
`--no_flat_overhangs_outward`; the stage logs the count.

**Confirmed on the pipeline-like SDFs (CPU, stages 4-9, `ramp90_xs` and
`ramp60_xs` at 30, metrics version 4), before (`4563f68`) and after
(`7cb5cbf`):**

| | Before | After |
|---|---|---|
| `ramp90_xs`: `order_atoms` | deadlock at atom 25 233 | **completes, 27 036 atoms** |
| `ramp90_xs`: atoms, worst / mean | 89.9 / 67.2 | **64.7 / 60.5** (the bound is 90 - 30 = 60) |
| `ramp90_xs`: full toolpath, worst / mean / unsupported | - | 63.0 / 60.3 / 4.2 % |
| `ramp60_xs`: atoms, worst / mean / over 45 | 45.1 / 43.3 / 16 of 495 | 46.5 / 43.0 / 5 of 504 |

On `ramp60_xs` the tip (45.1, tilt 14.9) is fixed; the five atoms left over 45
(45.3-46.5, tilt 13.5) sit where the underside starts at the column, by the
side walls. There the rule marks the blended cells on the column side as
overhang too, so the core test (distance from *non-overhang* surfaces)
counts them as core and corrects nothing. **Left open**: 5 atoms of 504, 1.5
degrees over. This pipeline-like SDF does not reproduce the laptop's tip
atom (56.1), so the laptop's run of the same part is the test of that.

**Laptop, 2026-10-01, on `a042fbc`:** golden test passed (5 passed, 1 skipped,
508 s). Field-only `ramp60_xs` at 30, overhang-aware: **worst 45.4** (was
56.1), mean 42.7 over 519 counted atoms; the 20 over 45 are 45.3-45.4, along
the side walls (tilt 14.6-14.7 where the underside asks 17). Full `ramp90_xs`
at 30: **`order_atoms` completes** (27 068 atoms, 301 s; it deadlocked
before), worst 65.0, unsupported 3.75 %, max tilt 30. A flat underside needs
47 degrees of tilt for 45; at 30 the bound is 60, so the part is out of range
there as P2.1's bound says, and is reported so, not passed.

#### P2-21 P2.5's two missing numbers: top-surface quality and the tilt rate, from the toolpath (operator's decisions, 2026-10-01)

*Built 2026-10-01.* Both are computed from the toolpath the report already
measures and the part's mesh, so `--reanalyse` adds them to every archived
run (stock and overhang-aware) in seconds, with no re-slicing. They change
no existing number, so the metrics version stays 4; a report scored before
them shows `not scored` in the summary.

* **Top-surface quality** (`overhang_metrics.top_surface_quality`, report key
  `metrics.top_surface`). The plan asks for "the fraction of ceiling cells
  whose final direction is within 2 degrees of their target". The field is
  not archived by the matrix runner, so the operator chose the toolpath
  form: the share of the **top layer's deposition points** printed within 2
  degrees of the top surface's normal. A top surface is a face whose normal
  is within `max_slope` of vertical (capped by the nozzle cone: upstream's
  `CEIL_MAX_ANGLE`); the top layer is the points within one layer height of
  it, and a point counts only if a top face is nearer it than any other face
  (as metrics version 4 does for overhangs). The target is the **smooth**
  normal (each face's corner normals averaged with its neighbours within 30
  degrees): `twin_domes`' facets are about 10 degrees apart, five times the
  tolerance, so the facet normal is no target. Unlike upstream's ceiling rule
  there is no curvature condition, so a small dome's top counts here. The
  summary puts stock and overhang-aware side by side and flags a pair more
  than 5 points below stock (P2.5's criterion; `TOP_SURFACE_ALLOWANCE_POINTS`,
  a placeholder).
* **Tilt rate** (`overhang_metrics.tilt_rate`, `metrics.tilt_rate`): the
  largest turn of the tool within any **1 mm of continuous printing**, in
  degrees per mm, and the share of printing moves whose 1 mm turns faster
  than 3 degrees per mm (the D3 placeholder, the ramp-in's default). Travel
  moves end a run. Per move (the plan's literal "max tilt rate along the
  toolpath") the number depends on how finely the path is cut: on the
  T-shape below, smoothing the points alone took it from 70 to 111 degrees
  per mm with no more turning; over 1 mm, from 42 to 51. The operator asked
  why per move was first recommended; on that evidence the 1 mm form was
  chosen. Reported, not judged: P2.5 sets no limit on it.

**`tshape_xs` at 30, overhang-aware, on the pipeline-like SDF (CPU, the
answer to the handover's question (a)):** `order_atoms` **completes**
(19 990 atoms, 235 s; it deadlocked in the lab's first matrix, P2-18), so the
flat-underside fix (P2-20) cleared the T-shape as it did `ramp90_xs`. Worst
effective angle 63.7, mean 60.3 (the bound at 30 degrees is 60: out of
range, as P2.1 says), unsupported near the overhang 4.9 %, top surface 100 %
within 2 degrees (mean 0.8), tilt rate 51 degrees per mm (26 % of printing
moves over 3). The fast turns sit along the side walls just above the
column, where the two arms' outward leans meet: about 100 single moves turn
13-29 degrees in under half a millimetre. Development numbers, not
comparable with the lab's.

#### P2-22 Where the unsupported deposition near overhangs comes from: the outermost bead steps a whole width

*Found 2026-10-01 with `tools/unsupported_where.py` (new).* In the first
overhang-aware matrix most runs within 45 degrees still failed on
unsupported deposition near the overhang, 0.8-2.4 % against 1 % (P2-18). The
tool sorts each such point by cause: **printed too early** (material lies in
its support cone but is printed later: an ordering problem) or **nothing
beneath** (no material in the cone at all), and by depth, nearest surface,
tilt and whether it starts a bead run.

**`ramp50_xs` at 30, overhang-aware, pipeline-like SDF (CPU):** worst
effective angle 46.0, mean 43.0, unsupported near the overhang **2.18 %**
(the lab: 0.8-2.4 % across budgets). All 32 such points are **nothing
beneath**, none printed too early; all are nearest the underside, within two
layer heights of it, spread along the whole ramp, at the rule's tilt (median
6.8). The material that holds them up *is* there, 44-50 degrees off the tool
axis (inside the 65-degree cone), but 1.25-1.44 mm away, past the metric's
reach of 2.5 layer heights (1.125 mm): it is **two layers down**. The layer
directly below has no bead there.

A cross-section shows why. The outermost bead of each layer, the one by the
underside, sits 0.3-0.8 mm inside the surface: beads come on a spacing of
about a width, so where the last one falls depends on the phase of that
spacing against the surface. Most layers step out a little past the one
below; every few layers one stalls (its last bead a width short of where it
could be) and the next jumps a whole width. For all 32 points the nearest
bead of the layer directly below is **0.97-1.08 mm sideways** (median 1.00),
against 0.04 mm for supported points: the bead sits entirely past the one
below it, over air. The cone allows at most about 0.96 mm sideways at 0.45
mm layers (65 degrees). Across-bead spacing near the underside has median
0.92 mm but runs 0.43-1.10 (10-90 %), so a full-width jump can land either
side of that limit. **So the cause is the discretisation of the outermost
beads at the overhang surface, not the tilt and not the print order.**

**`ramp60_xs` at 30, overhang-aware (CPU):** worst 46.5, mean 43.2,
unsupported near the overhang **1.06 %** (the lab, on older code: 1.73 %);
the same picture: all 14 nothing beneath, all within two layer heights of
the underside.

**Against stock** (`ramp45_xs` at 7, the known answer, CPU: 0.00 % here,
0.24 % in the lab): the outermost bead jumps a whole width there too, and as
often, but the jump lands at 0.8-0.9 mm sideways, just inside the cone; in
the overhang-aware runs it lands at 0.9-1.05:

| Sideways step to the layer below, share of near-overhang points | 0.8-0.9 | 0.9-0.96 | 0.96-1.05 | 1.05-1.5 |
|---|---|---|---|---|
| `ramp45_xs` stock at 7 | 10.2 % | 2.1 % | 0.1 % | 0.0 % |
| `ramp50_xs` overhang-aware at 30 | 2.0 % | 4.4 % | 3.4 % | 0.4 % |
| `ramp60_xs` overhang-aware at 30 | 5.2 % | 3.9 % | 1.1 % | 0.2 % |

Bead spacing across the beads is the same in all three (median 0.88-0.92
mm). The overhang-aware layers are not quite parallel by the underside (the
tilt changes 0.3-0.6 degrees from one layer to the next, against 0.0 in
stock), but the size of a jump does not follow that change (correlation
0.11 and -0.03), so that is not the reason.

**Part of it is the metric.** The support test looks for earlier
*points* in the cone, but a bead is a line with points about 0.45 mm apart
along it. In stock the points of successive layers sit in step; in the
overhang-aware runs they do not, so the nearest point below can be a little
along the bead from the nearest part of the bead, and a 0.9 mm jump measures
up to about 1.0. Measured to the bead *line* below instead (the segments
between consecutive deposition points), the share stepping past 0.96 mm
falls from 1.37 % to **0.61 %** on `ramp60_xs` (its unsupported points have a
bead line a median 0.82 mm sideways, inside the cone) and from 3.81 % to
2.79 % on `ramp50_xs` (whose jumps stay about 0.98 mm: real), and from 0.12
to 0.06 % on stock.

**So, two causes, neither of them the tilt or the print order:** (1) the
outermost bead at an overhang surface steps a whole bead width every few
layers, which in stock falls just inside the 65-degree cone and in the
overhang-aware runs often just outside (a margin of about 0.06 mm); (2) the
support test checks points rather than bead lines, which counts some
supported beads as unsupported when the points of successive layers are out
of step, as they are with a tilted field. The 1 % criterion therefore turns
on a difference of a few hundredths of a millimetre at the jumps. The options were: change the metric to test bead
lines (a version 5, a ★ metric change, which moves stock too, a little),
work on the jumps in the slicer, or leave both. **The operator's decision,
2026-10-01: leave both for now**, run the lab matrix on Sunday as planned,
read its unsupported column with this in mind, and decide on a change with
the full new table in hand.

#### P2-23 P2.3 as built: the reachable-tilt map on `reference`, viewed only (operator's decisions, 2026-10-01)

*Built 2026-10-01.* `atom.reachability`, `tools/reachability_map.py`,
`tests/test_reachability.py`. Two decisions by the operator: **map only**,
the field unchanged (`use_reachability_map` waits for the real machine's
geometry: on `reference` it would change no test part at the bed centre, and
wiring it in is a vendored edit needing a laptop golden run); and **both
meanings kept**, with the lift.

* **What it holds.** Over the bed every 10 mm (31 x 30 x 29 points up to
  `max_z_axis`) and 13 tilts (0-30 in 2.5-degree steps) x 16 azimuths: the
  lift each pose needs (NaN unreachable, 0 as the part stands), and whether
  the bed's corners at the lifted pose clear the P4.1 clearance model. From
  it: the largest tilt reachable with **any platform**, and with **none**.
  About 20 s on a CPU; cached in `data/reachability/<profile>.npz`
  (gitignored, 1.2 MB). The map is in the bed frame, so it is the same for
  every part; a lookup adds the part's placement
  (`contracts.bed_centering_offset`, hazard 7) and any platform height.
  Trilinear between grid points, the smaller of the two azimuths either side,
  rounded down to the tilt step; NaN (nothing known reachable) off the grid.
* **The lift is iterated** (P1-9): the kinematics' offset is a first
  estimate, and the map raises the part by it and solves again until no more
  is asked. At the bed centre 2 mm up and 30 degrees the converged lift is
  **10-57 mm** by direction; P2-10's table gave 46 mm there from the first
  estimate, about 10 mm short at the worst direction. Without iterating, the
  clearance model then rejected 6.8 % of the poses the kinematics accepted
  (the corners still 1-6 mm into the gantry); with it, none.
* **The raw kinematics are not monotonic in tilt** (the plan's test assumed
  they are): 1.8 % of reachable (point, direction) pairs have a smaller tilt
  in the same azimuth that is unreachable, 95 % of them within 10 mm of the
  X or Y travel limits (tilting shifts the axes, bringing some edge points
  into range), the rest high up where a screw nears its limit. The field's
  tilt grows from the vertical first layer, so the map's max tilt is the
  largest **whose every smaller step is reachable**: monotonic by
  definition, and the test checks that instead.
* **On `reference`** (`python tools/reachability_map.py`): near the bed,
  interior points reach 28-30 degrees with a platform and **10-15 without**;
  at 100 mm up, 20-30 in the middle of the bed and 10-12 near its edges, the
  same with or without a platform; the row at x = 0 reaches only the
  vertical and x = 300 nothing. `ramp60_s` placed at the bed centre reaches
  30 degrees in every direction with a platform, 15 without: the lab's
  30-degree runs would need a platform on this geometry. The clearance proxy
  rejects nothing the kinematics accept (its gantry is the same plane); a
  clearance file (gate M3) can.
* **Tests** (the plan's, adjusted): the bed centre near the bed reaches the
  golden cube's 5.53 degrees, with and without a platform; points outside
  travel reach nothing; every tilt up to a cell's max is reachable; the lift
  is converged; the reference proxy agrees with the kinematics; cache round
  trip; the lookup's interpolation, azimuth and rounding on a hand-made map.

#### P2-24 P2.4's pipeline test: every atom in the first 0.5 mm of overhang at t - 2 or more (operator's decisions, 2026-10-01)

*Written 2026-10-01; not yet run on the laptop.* `tests/test_ramp_in_pipeline.py`
(`pipeline` marker): a field-only, overhang-aware run of `ramp60_xs` at 30,
then the tilt of the atoms by the underside in the first 0.5 mm past the
column's edge (`overhang_metrics.overhang_start_tilts`: the effective
angle's own points, measured horizontally from where the underside starts,
toward the overhang), and the same for one bead width (0.9 mm), with the five
lowest atoms' positions. The plan's bar is `t - 2` = 15 (the rule's `t` is 17).
The atoms stand for the toolpath: every orientation the toolpath deposits with
is an atom's own (`test_field_only.py`), and a field-only run is a fraction of
a full one.

**Why it does not judge yet.** On the CPU (pipeline-like SDF, current code)
the first 0.5 mm holds 13 atoms: mean 15.0, lowest 13.5, 62 % at 15 or over;
the first 0.9 mm, 50 atoms: mean 16.2, lowest 13.5, 90 %. The lowest sit by
the side walls (y within 1.3 mm of a wall), where P2-20 left residuals. Every
point at 15 or more would fail; the mean would pass by a hair. The operator
chose to see the laptop's numbers on the remeshed part first and set the pass
rule then. Until then the test asserts only that there is something to
measure. Laptop: `pytest --run-pipeline tests/test_ramp_in_pipeline.py`
(about 3-5 minutes).

**Laptop, 2026-10-01, on `3d5a495`** (181 s): the first 0.5 mm holds 12
atoms, mean **16.4**, lowest **15.7**, all at 15 or over; the first 0.9 mm,
57 atoms, mean 16.7, lowest 14.6, 98 % at 15 or over. The remeshed part does
better than the CPU's pipeline-like one (lowest 15.7 against 13.5); the
lowest sit at y = 1.3-3.5 and 12.3, not right against the side walls.
**The operator's pass rule:** every atom in the first 0.5 mm at `t - 2` or
more (the plan read strictly; 0.7 degrees to spare). The average was offered
and not chosen (a few low points could hide behind it), and so was the 0.9
mm strip (it fails today, 14.6). The 0.9 mm numbers are still printed. The
test fails on the CPU atoms (13.5), which is expected, since pipeline tests
run only on the laptop. **Laptop, 2026-10-01, with the rule, on
`1163851`:** **1 passed** in 176 s, the same numbers (lowest 15.7).

#### P2-25 P3.2 built early: the layer thickness is measured from the geometry, because a toolpath's `height` is the nominal everywhere ★ PLAN EDIT

*Built 2026-10-01 at the operator's request, before the lab's Sunday run, so
`--reanalyse` adds it to all 96 runs.* The plan's P3.2 flags "any deposition
whose height is outside [0.5, 1.5] x nominal". **A toolpath's `height` is the
nominal layer (0.45 mm for 0.9 mm beads) on every point**, in every toolpath
checked (stock `ramp45_xs`, overhang-aware `ramp50_xs`, `ramp60_xs`,
`tshape_xs`): Atomizer writes the nominal, not the real spacing. That check
could never fire.

`atom.layer_thickness` measures it instead: for each deposition point, the
nearest deposition point below it along `-d` and within half a bead width
sideways (the bead it sits on); the distance along `d` is the thickness
there. Searched to 2.5 layers, so a bead over a one-layer gap is found;
points within half a layer of the lowest one rest on the bed and are not
judged; closer than a quarter layer is the same layer. Thin and thick at the
plan's 0.5 and 1.5 (placeholders). Report key `metrics.layers`, overall and
near the overhangs; a summary table beside stock.

On the CPU toolpaths (development numbers): the real layers have a median of
0.45-0.46 mm everywhere. **Thick** (mostly about 0.85-0.9 mm: the bead under
it is missing, a one-layer gap, mostly in the solid shell about 1.4 mm in,
where it meets the infill): stock `ramp45_xs` at 7 1.29 %, overhang-aware
`ramp50_xs` and `ramp60_xs` at 30 1.24 and 1.31 %, `tshape_xs` at 30 2.44 %.
**Thin**: 0.00-0.03 % on the ramps, 0.14 % on the T-shape (1.07 % near its
underside, about 0.14 mm, where the tilted layers converge on the flat
underside). So the tilted field leaves the ramps' layers as stock's; P3.2's
feedback to P2.4 is not needed there.

#### P2-26 The machine's reach and the platform in every report; archive before scoring

*Built 2026-10-01 at the operator's request, before Sunday.*

* **Reach and platform** (`atom.machine_reach`, report key
  `metrics.machine`). P2.5 asks for "no new IK failures"; nothing counted
  them. A run with points out of reach stops in `add_platform` (the
  vendored `get_plaftorm_size` asserts), so every completed run has none, and
  the report now says so explicitly. It also records the **platform**
  `add_platform` prints under the part so the tilted bed's corners clear the
  gantry, which no report gave. It follows the pipeline step for step: the
  toolpath tesselated to 1 degree as stage 10 does, then the vendored lift
  loop, both re-centrings (P1-9, P4-4), counting instead of asserting; the
  layer height as the pipeline passes it (float32). Checked against the
  vendored function: identical. On the CPU toolpaths: stock `ramp45_xs` at
  7 and overhang-aware `ramp60_xs` at 30 need **no platform**; `tshape_xs` at
  30 needs **42.75 mm**. A run made on another profile than the one scoring
  it says so rather than being scored against the wrong machine.
* **A hazard found doing it:** the vendored tesselation
  (`toolpath3.tesselate_orientation`) sizes its output at three times its
  input and writes past it, unchecked, when more points are needed; on the
  CPU that is a protection fault that kills the process (seen with synthetic
  random-tilt toolpaths). The pipeline's own stage has survived every real
  toolpath so far, but `machine_reach` counts the points first and checks the
  toolpath untesselated when they would not fit (`tesselated: false`), so
  `--reanalyse` can never crash on it.
* **Archive before scoring.** `overhang_report.py` scored a run before
  archiving its toolpath, so a scoring error lost 30+ minutes of slicing for
  good; it now archives first, and `run_matrix_parallel.py` brings a failed
  job's archive back too.
* **Checked before Sunday:** every matrix part (16) at 7 and 30 degrees,
  scored from synthetic toolpaths with all the new numbers: no errors, at
  most 2.5 s a run. `--reanalyse` on the lab's 96 reports should take a few
  minutes.

#### P2-27 P2.5's second overhang-aware matrix (lab machine, 2026-10-04, `e3f3e4f`)

`run_matrix_parallel.py --sizes xs s --workers 8 --overhang-aware`, 3:40:35,
**48 of 48 completed** (the first matrix: 44). Stock and overhang-aware in one
provenance group (cad-p07-2065-9, stock mix, Taichi 1.7.4, `reference`,
metrics version 4). `--reanalyse` was not run after it, so the 48 stock
reports do not have top-surface quality, the tilt rate, reach and platform or
layer thickness yet (owed on the lab machine).

* **The flat undersides complete** at 30 (`ramp90`, `tshape`, both sizes; all
  four failed in `order_atoms` in the first matrix): worst 62.1-65.1, mean
  60.2-60.3 against P2.1's bound of 60, so out of range and reported so.
* **Printable: stock 4, overhang-aware 6 of 48** (unchanged from the first
  matrix): `ramp45` at every budget, both sizes. At 30 stock fails it (51°,
  2.7-2.9 %) and overhang-aware passes (43.5°, 0.2-0.5 %): P2.5's minimum bar.
* **The edge fixes (P2-19, P2-20) lowered the worst points**, worst
  effective angle first matrix -> second: `ramp60_s` at 30 48.7 -> 45.7,
  `ramp60_s` at 15 50.5 -> 46.3, `ramp60_xs` at 30 46.4 -> 45.4, `ramp70_xs`
  at 30 50.0 -> 48.1, the flat undersides at 7 and 15 89.5 -> 83.7 (now the
  geometric angle minus the budget). Not everywhere: `ramp80` at 7 and 15
  rose by 2.6-4.8 (73.5 -> 76.1, 66.3 -> 69.6 on `xs`), and `ramp70_s` at 30
  47.2 -> 48.3.
* **Inside the tilt bound** (`theta_geo <= 45 + budget - 2`: `ramp45` and
  `ramp50` at every budget, `ramp60` and `ramp70` at 30; 16 runs), 6 pass
  and 10 fail, all narrowly: `ramp60_s` at 30 on the angle alone (45.7,
  unsupported 0.9 %); `ramp60_xs` at 30 on both (45.4, 1.1 %); `ramp50_s`
  on both (45.3, 1.1-1.3 %); `ramp50_xs` on unsupported deposition alone
  (44.5, 2.3-4.5 %, worse than the first matrix's 1.6-2.1); `ramp70` at 30
  on both (48.1-48.3, 1.1-2.0 %). The worst points are the edge residuals
  (handoff, known residuals); the unsupported share is P2-22's full-width
  bead jumps.
* **The relation flips**: where the budget falls short, the mean effective
  angle is the geometric angle minus the budget (`ramp60` at 7: 53.0;
  `ramp70` at 15: 55.0; `ramp80` at 30: 50.8-51.3; flat undersides at 30:
  60.2-60.3). Fit over the 36 overhang-aware ramp runs using more than 2
  degrees of tilt and not horizontal, predicted `theta_geo - tilt_used`
  (each surface's largest tilt): r = 0.986 against the mean angle (mean
  absolute deviation 1.5), 0.969 against the worst (3.8). Stock's 0.992 was
  P0.8's 18 runs under metrics version 2, so not strictly the same
  measurement; the reports carry each surface's largest tilt, not its
  typical one, which loosens this fit.
* **Reach and platform** (P2-26): no point out of reach in any run (P2.5's
  "no new IK failures": met). Platforms only at 30 on the steep parts:
  `ramp70_xs` 6.3 mm, `ramp80` 38.7-46.8 mm, the flat undersides 33.7-47.2
  mm; `ramp45` to `ramp60` need none at any budget.
* **Top-surface quality** (overhang-aware only, until stock is re-scored):
  the ramps and T-shape 97-100 % (T-shape `xs` at 30: 90 %); `twin_domes`
  falls with the budget, `xs` 98 / 57 / 28 % and `s` 96 / 84 / 61 % at 7 /
  15 / 30, because a larger `max_slope` makes more of the domes count as top
  surface. Whether that is worse than stock is the criterion, and needs the
  stock numbers.
* **Tilt rate**: 1 degree per mm on `ramp45`, rising with the slope and the
  budget to 79-85 on `ramp80` at 30. **Layer thickness**: 0.9-1.7 % thick
  (beads over a one-layer gap), `tshape_xs` at 30 2.4 %.

**Update, 2026-10-04: the lab re-scored all 96 reports (`--reanalyse`,
`2150258`), so stock has the new numbers too:**

* **Top-surface quality: 47 of 48 pairs within 5 points of stock.** The one
  miss is `tshape_xs` at 30 (100 -> 90.4 %), a part out of range there.
  `twin_domes` is the same as stock at every budget (`xs` 98.2 -> 98.2, 55.2 ->
  56.7, 26.7 -> 28.1; `s` 95.6 -> 95.8, 82.7 -> 84.0, 61.5 -> 60.9): its fall
  with the budget is the measure's (more of the domes count as top surface at
  a larger `max_slope`), not the overhang-aware field's.
* **Tilt rate: much higher than stock.** Stock turns at most 17 degrees per
  mm (mostly 0-8); overhang-aware up to 85 (`ramp80_s` at 30), and
  `tshape_xs` at 30 turns faster than 3 degrees per mm on 25 % of its
  printing moves. P2.5 sets no limit; it is gate D3's question.
* **Platform:** stock needs one at `ramp60` at 30 (9.4 and 10.8 mm, because
  it tilts away by 30 degrees there), overhang-aware none; overhang-aware
  needs one on the steep parts at 30 (above). No point out of reach on either
  side.
* **Layer thickness:** much the same; thick 0.8-1.5 % stock, 0.9-1.7 %
  overhang-aware (`tshape_xs` at 30 2.4 %); thin under 0.15 % everywhere.

**P2.5's success criteria on this matrix:**

| Criterion | Result |
|---|---|
| Every ramp inside the bound: worst <= 45 and unsupported < 1 % | **Not met**: 6 of 16 (10 narrow failures, above) |
| Ramps beyond the bound reported out of range, never passed | Met |
| Top-surface quality >= stock - 5 points | Met on 47 of 48; the miss is out of range |
| No new IK failures | Met (0 points out of reach) |
| `twin_domes` not worse than stock on any metric | Met within noise (top surface equal; thick layers +0.1-0.3 points) |
| `ramp45` passes at every budget, 30 included | Met |
| The relation flips, fit at least as tight as stock's r = 0.992 | Flips; r = 0.986 (mean angle), not strictly comparable (above) |

By the plan's own words P2.5 is now done ("the table is committed and the
operator is asked to take it to D2"); P2 ends with the first criterion met,
or a recorded D2 decision.
