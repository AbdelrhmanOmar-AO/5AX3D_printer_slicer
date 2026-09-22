# Corrections and deviations from SLICER_BUILD_PLAN v3

Everything found so far that contradicts the build plan, or where the
implementation deliberately departs from it. Build plan section 0 rule 6 says
to report contradictions rather than work around them silently; this is that
report.

**Keep this file updated as further items are found.** Items marked
**PLAN EDIT** should be corrected in the plan document itself, because a later
task will be written against the wrong fact.

Last updated: 2026-09-22. Phase P0 complete; the P0.8 matrix is being re-run
against corrected metrics.

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

### 2.9 Single branch instead of one branch per task

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
a VTK one, which avoids 4.12 entirely. CI installs neither, so
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

## 3. Environment facts the plan asked to establish

### 3.1 Taichi falls back to CPU rather than failing (answers P0.4 step 1)

P0.4 asks to verify what `ti.init(arch=ti.gpu)` does with no GPU. Observed on
Taichi 1.7.4: it does **not** raise. It logs `libcuda.so lib not found` and
`Arch=[<Arch.cuda: 3>] is not supported, falling back to CPU`, then reports
`Arch.x64`. A stage asking for the GPU on a GPU-less machine therefore degrades
silently and runs many times slower. Any check must compare the resulting arch,
not catch an exception.

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

### 3.4 The pipeline is deterministic

Two consecutive runs of the calibration cube produced identical statistics, so
golden comparisons use the G-code SHA-256 directly rather than float tolerances.

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

### 4.7 Re-centring on the bed is not a constant offset

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

### 4.8 The header's purge lines extend the G-code's axis ranges

`kinematics3z.HEADER` draws two priming lines at fixed coordinates
(`X0.1 Y20`, `Y200.0`, all three screws at `0.3 + Z_OFFSET`). Those points are
not in the toolpath, so any statistic taken over a whole G-code file includes
them. On the calibration cube they set the file's X minimum, both Y extremes
and the V minimum.

They can only extend a range, never narrow it, so a comparison against toolpath
output should assert equality on the maxima and an inequality on the minima.
`tests/test_contracts.py` does exactly that.

### 4.9 `from __future__ import annotations` was hit in practice

Recorded as a hazard in 3.2, then walked into two commits later while writing
`src/atom/contracts.py`: the module had the import, and its Taichi kernel
failed with `Invalid type annotation (argument 0) of Taichi kernel:
ti.types.ndarray()`.

Modules that define Taichi kernels now carry a comment saying why the import is
absent. Affected so far: `contracts.py`, `tests/test_contracts.py`,
`tests/test_ti_env.py`, `tools/overhang_report.py`,
`tests/test_overhang_report.py`.

### 4.10 `git ls-files` reports the index, not what is committed

Two guard tests were written against the wrong notion of "tracked" and passed
while the thing they guarded was broken:

- the first checked the working tree, and passed while 36 benchmark meshes sat
  uncommitted because `data/mesh/.gitignore` had silently skipped them;
- the second used `git ls-files`, which lists staged files, and passed while
  the golden baseline had been `git add`-ed but never committed.

Both now use `git ls-tree -r --name-only HEAD`. Only a commit survives a push,
a clone, or a move to another machine.

### 4.11 A fresh Windows machine has no git identity

`git commit` fails with "Author identity unknown" until
`git config --global user.name` and `user.email` are set. The error is easy to
miss in a scrollback, and the files simply stay staged, so a later `git push`
reports "Everything up-to-date" while nothing has been committed. Worth setting
during setup on any new machine.

### 4.12 A VTK timer created before the window opens never fires on Windows

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

## 5. Open plan items not yet resolved

| Item | Status |
|---|---|
| P0.8 matrix | **Re-running** (2026-09-22) against the corrected metrics. The first run, 48 runs in 19.9 h, produced sound tilt and runtime data but unusable unsupported figures. |
| Gates M1, M2, M3, E1 | Deferred by the team until the mechanical design is settled |
| Gate D0 (tilt budget, benchmark geometry) | Needs the re-run matrix and P2.1's analytic bound |
| T-shape `underside_angle_deg` | Defaults to 90; gate D0 picks the real value |
| Thresholds (45 deg effective, 1 % unsupported) | Placeholders, gate D0. Now meaningful: with the metric fixed, parts can actually pass. |
| `twin_domes` verdict | Reports "not printable" when it has no overhang to measure. Conservative by design but misleading in the table; distinguishing "nothing to measure" from "failed" is worth doing. |
| `order_atoms` with `kernel_profiler=False` | Untested; a possible CPU speedup with no determinism risk |
| P1.5 firmware templates | Blocked on E1; only the `rrf` path exists |

## 6. Quick index

The items a later task is most likely to get wrong if it trusts the plan:

| # | In one line |
|---|---|
| 1.1 | `tool_orientation` is `(N,2)` spherical; `theta` is the tilt directly |
| 1.5 | `order_atoms` is CPU and 86 % of runtime; long runs are not GPU-bound |
| 1.6 | `forward()` has no X-then-Y tilt decomposition to confirm against |
| 1.7 | IK failure is NaN in `offset`; a non-zero `offset` is a clearance, not an error |
| 1.8 | The plan's 1.5 x height support radius overrides its own 65-degree cone |
| 4.5 | The bed-contact rule must anchor to the part's lowest point |
| 4.6 | Sampling by face centroid under-measures large flat faces; subdivide first |
| 4.7 | Bed re-centring changes screw heights non-uniformly; compare in one frame |
| 4.12 | Create VTK timers after the interactor is initialised, or they never fire on Windows |

And the two habits that caught most of them:

* Before a long run, do one short run and compare against what you expect. A
  figure that does not move when the thing it depends on changes is worth
  chasing (4.6 was found exactly this way).
* Check a metric against a case whose answer is known before putting its
  output in a table (3.7).
