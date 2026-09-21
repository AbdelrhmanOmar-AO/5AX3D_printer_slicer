# Corrections and deviations from SLICER_BUILD_PLAN v3

Everything found so far that contradicts the build plan, or where the
implementation deliberately departs from it. Build plan section 0 rule 6 says
to report contradictions rather than work around them silently; this is that
report.

**Keep this file updated as further items are found.** Items marked
**PLAN EDIT** should be corrected in the plan document itself, because a later
task will be written against the wrong fact.

Last updated: 2026-09-21.

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

### 2.7 Single branch instead of one branch per task

Plan section 0.1 wants `p<phase>/<task-id>-<short-slug>` branches with one pull
request each. All work is on `claude/new-session-l8g46d`, one commit per task,
at the operator's choice: the session is restricted to that branch, and the
operator is new to git and preferred a single review surface. `main` is
untouched.

### 2.8 Golden archive is the `_platform` toolpath

P0.2 says "copy the final toolpath file". The final toolpath stage before G-code
is `data/toolpath/<part>_platform.npz` (`tools/atomize.py`), not the `_smoothed`
one.

---

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

## 5. Open plan items not yet resolved

| Item | Status |
|---|---|
| CI sets `ATOM_TI_ARCH=cpu` | Nothing reads it yet; P0.4's `ti_env` helper is not written |
| Gates M1, M2, M3, E1 | Deferred by the team until the mechanical design is settled |
| Gate D0 (tilt budget, benchmark geometry) | Needs P0.8 results and P2.1's analytic bound |
| T-shape `underside_angle_deg` | Defaults to 90; gate D0 picks the real value |
