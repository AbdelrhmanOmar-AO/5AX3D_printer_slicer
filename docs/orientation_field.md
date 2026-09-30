# The tool-orientation field

How Atomizer decides which way the tool points at every point of a print, and
therefore how the bed tilts. Build plan task **P2.1**, written for P2.2 to
P2.4, which change this field. Read with the code; line numbers are as of the
commit that adds this file.

**In short:**

1. The field is computed once, on the grid of the part's signed distance
   field, by `tools/compute_tool_orientations.py` (pipeline stage 4). Every
   later stage carries it unchanged to the atoms and then to the toolpath.
   The tool orientation of a toolpath point *is* the field at its atom's cell.
2. It is fixed ("constrained") in only two kinds of cell: the first layer
   (straight up) and the part's upward-facing, gently curved boundary cells,
   the "ceilings" (their surface normal). Everything else is interpolated.
3. `max_slope` is not a clamp. It only decides which surfaces count as
   ceilings: those within `max_slope` of horizontal.
4. **With infill on, the "boundary" includes the inside of the part's shell.**
   The infill stage hollows the SDF 1.8 mm under every surface. Above an
   overhang, that inner surface faces up and away from the overhang, so the
   ceiling rule pins the field to it. **This is where the P0.8 baseline's
   "tilts away from the overhang, degree for degree" comes from.** Without
   infill the stock field on the ramps is exactly vertical (section 4).
5. So P2.2's overhang constraint, 1.8 mm below, would contradict a stock
   constraint pointing the other way. P2.2 has to deal with the infill's
   inner surfaces, not only add a constraint (section 5).
6. **P2.2 is built** (section 7): behind `overhang_aware`, the field is
   computed on the part before infill (the operator's option (a)) with the
   overhang constraint added, and it leans toward the overhang. On the `xs`
   ramps, CPU, the mean effective overhang next to the underside comes down
   to 43-46 degrees wherever the budget allows the tilt the rule asks for.

---

## 1. Where the field sits in the pipeline

| Stage | Tool | Reads | Does to the field |
|---|---|---|---|
| 3 | `bpn_to_sdf.py` | point cloud | builds the SDF the field lives on (grid origin at 0, cell 0.169 mm for 0.9 mm beads) |
| 3 bis | `sdf_to_isdf.py` (infill parts) | SDF | **rewrites the same SDF file** with the shell and gyroid (`atomize.py:194`: input and output are both `params.sdf_path`) |
| **4** | **`compute_tool_orientations.py`** | **the SDF, infilled if infill is on** | **computes it** (section 2) |
| 5 | `sdf_df_to_layers.py` | SDF + field | reads it: layers are orthogonal to it; not changed |
| 6 | `compute_tangents.py` | SDF + field | copies it as the basis normal (`solid3.py:189`); aligns only the tangent angle |
| 7 | `align_atoms.py` | + basis | copies the basis normal (`solid3.py:206`); aligns only phases |
| 8 | `extract_explicit_atoms.py` | + triphasor | each atom takes the normal of its cell (`triphasor3.py:112`) |
| 9 | `order_atoms.py` | atoms | writes it as `tool_orientation` (`toolpath3.py:2228`); travel points between atoms get an nlerp of the two |
| 10 | `smooth_toolpath_point.py` | toolpath | moves positions only |
| 10 | `tesselate_toolpath_orientations.py` | toolpath | subdivides segments so the orientation changes at most 1 degree per segment |
| 11 | `toolpath_to_gcode.py` | toolpath | inverse kinematics per point; a NaN aborts the file |

Stages 6 and 7 run multigrid aligners of their own, but they write normals only
into their coarser levels (`fff3.basis3_field_restrict`, `fff3.py:461-465`;
`triphasor3.py:336-340`). Prolongation only *reads* the fine normal, so the
field reaches the atoms unchanged. That is what lets P2.0 measure the
orientation on the atoms before `order_atoms` (`atom.frame_atoms`).

## 2. Stage 4, step by step

`tools/compute_tool_orientations.py`:

### 2.1 Parameters (lines 58-64)

| Global in `fff3` | Value | Used for |
|---|---|---|
| `MACHINE_MAX_SLOPE_ANGLE` | `max_slope` | only to compute the next one |
| `MAX_SLOPE_ANGLE` | `min(max_slope, (180 - NOZZLE_CONE_ANGLE) / 2)` = `min(max_slope, 50)` for the 80-degree reference nozzle (the cone is the profile's `nozzle_cone_angle_deg` since plan_corrections P2-11: 60 for `dev60`) | the ceiling threshold; the opt-in ortho-to-wall branch |
| `CEIL_MAX_ANGLE` | `MAX_SLOPE_ANGLE` | **which surfaces are ceilings** |
| `FLOOR_MAX_ANGLE` | 1 degree | **not read by this stage** (the floor branch is commented out); read by stages 5 and 6 (section 3) |

These are module globals that Taichi bakes into the kernels when it first
compiles them, which is why every stage sets them before running anything
(plan hazard 12 is the same mechanism for the machine profile).

### 2.2 Initialisation: which cells are constrained (`fff3.py:60-106`)

`init_spherical_direction_field_from_sdf` looks at every cell once:

* **Masked:** a cell outside the part (SDF >= 0) gets NaN (`:105-106`) and is
  ignored by everything after.
* **Boundary region:** a cell *inside* the part within one layer height
  (0.45 mm) of the surface, i.e. SDF in (-0.45, 0) (`:79`, `is_boundary_region`
  at `:886`). Only these cells can be constrained. For each:
  * the surface normal `n` is the normalised central-difference gradient of
    the SDF (`solid3.py:292-352`), pointing out of the part;
  * if `n.z < 0` the surface faces down ("pointing down") and `n` is flipped
    upward for the angle test (`:84-89`);
  * the curvature is the mean over x, y, z of `(1 - cos a) / 2` between the
    normals two cells apart (`solid3.py:356-406`). The threshold 0.1
    (`CURVATURE_THRESHOLD`) therefore excludes only sharp features, where
    the normal turns more than about 37 degrees across two cells.
* **First layer** (cell centre z < one layer height, `:97-99`): constrained to
  straight up, `[theta, phi] = [0, 0]`.
* **Ceiling** (`:91`, `:100-104`): not in the first layer, curvature below 0.1,
  **not pointing down**, and the upward normal within `CEIL_MAX_ANGLE` of
  vertical: constrained to that normal.
* **Everything else** — the interior, walls, overhangs, sharp edges — is left
  unconstrained, holding the Taichi field's initial zero, `[0, 0]` = up.

The floor branch (`# is_floor = ...` at `:92`, `# if is_ceiling or is_floor`
at `:101`) is commented out: **no downward-facing surface is ever
constrained.** That is the gap P2.2 fills.

### 2.3 Constrained cells and the multigrid aligner (`direction.py`)

A cell's `state` holds a constraint bit (`constrain`, `is_constrained`,
`direction.py:376-392`). `SphericalMultigridAligner.align()` (`:86-101`):

1. **Restrict** level by level to the coarsest (`spherical_field_restrict`,
   `:140-182`): a coarse cell becomes constrained if any of its 8 fine cells
   is, and takes the normalised mean of **the constrained fine cells only**.
   A coarse cell whose 8 fine cells are all masked is masked.
2. From the coarsest level to the finest: **64 alignment passes** on every
   level except the coarsest (`if level_i < level_count_m1`, `:95`), then
   **prolong** (`spherical_field_prolong`, `:185-215`), which copies each
   coarse value into its **unconstrained**, unmasked fine cells only.
3. One alignment pass (`spherical_field_align_one_level_one_time`,
   `:218-276`) sets each cell to the normalised weighted mean of its 26
   neighbours' directions. The weights are a triangle filter of radius two
   cell diagonals (`:259-264`); masked neighbours are skipped; during the
   first 32 passes each neighbour is ignored at random, with probability 0.5
   falling to 0 (`:247-256`, fixed seed, so deterministic). **A constrained
   cell keeps its value** unless `smooth_constraints` is set (`:274-275`).

The result is the smoothest field that honours the constraints: every
unconstrained cell ends up an average of the constrained directions around
it.

### 2.4 The 32 final passes (`compute_tool_orientations.py:107-112`)

After the multigrid solve, 32 more passes on the finest level with
`smooth_constraints=True`: **constrained cells are averaged too**, ceilings
included. After each pass only the first layer is put back to vertical
(`spherical_field_constrain_fisrt_layer_up`, `fff3.py:160-175`; note it uses
`z <= layer height` where the initialisation used `<`). Upstream's comment
cites Chermain et al. 2025, Section 6, "Implementation details".

How much this changes a constraint depends on what surrounds it. For the
infill's inner shell surface above `ramp70_xs` (section 4) it moved the tilt
next to the underside by 0.08 degrees, because that constrained surface is
broad and uniform. For a thin band of overhang constraints facing an opposite
constraint 1.8 mm away (section 5) it has not been measured
(plan_corrections P2-2).

### 2.5 `max_slope` is not a clamp

Nothing limits the field's tilt directly. It stays within `max_slope` because
every constraint does (0 for the first layer, below `CEIL_MAX_ANGLE` for
ceilings), and averaging unit vectors with positive weights cannot leave a
cone of half-angle below 90 degrees around +Z. **A P2.2 constraint beyond the
budget would therefore take the field beyond it**, and nothing but the
inverse kinematics would object, by aborting the G-code (plan_corrections
P2-1).

### 2.6 `--ortho_to_wall` (off for every benchmark part)

`orthogonolize_direction_wall_region` (`fff3.py:109-156`) constrains wall
cells to the current direction made orthogonal to the wall, where that stays
within `MAX_SLOPE_ANGLE`, and the multigrid solve runs a second time. The
build plan's P2.1 text pointed here for "how `MAX_SLOPE_ANGLE` is enforced";
this is the only place it is compared against a direction, and only with
this option on.

## 3. How the later stages use the field, and their own constraints

These do not change the field, but their constraints use the same "ceiling"
and "floor" tests, so the same surfaces matter to them.

* **Stage 5, layers** (`sdf_df_to_layers.py`, `fff3.init_phasor3_field_from_sdf`
  at `fff3.py:277-325`): a phase field whose level sets are the layers,
  orthogonal to the orientation field. Layer positions are pinned on ceilings
  (normal within `CEIL_MAX_ANGLE` of up) and **floors (within
  `FLOOR_MAX_ANGLE` = 1 degree of straight down)**, within one cell diagonal of
  the boundary. So `FLOOR_MAX_ANGLE` is read, here: a flat underside fixes
  where a layer lies, a sloped one does not.
* **Stage 6, tangents** (`compute_tangents.py`, `fff3.init_deposition_tangent_field`
  at `fff3.py:178-274`): the deposition direction in each layer. Constrained
  on walls (to run along them) and, when `top_lines` / `bottom_lines` images
  are given, on top surfaces (`CEIL_MAX_ANGLE`) and bottom surfaces
  (**`FLOOR_MAX_ANGLE`**, `:235`).
* **Stage 7, atoms** (`align_atoms.py`, `fff3.init_triphasor3_field` at
  `fff3.py:328-386`): bead positions within each layer, pinned near walls.
* **Stage 8, extraction** (`extract_explicit_atoms.py`): one atom per local
  maximum of the triphasor field, each taking its cell's orientation; atoms
  too close to a wall or top/bottom surface are dropped
  (`frame_field_filter_point_too_close_to_boundary`, `fff3.py:672-702`).

## 4. Where stock Atomizer's tilt near an overhang comes from

The P0.8 baseline found that stock Atomizer tilts the tool **away** from every
ramp's overhang, making it worse by exactly the tilt it uses
(`theta_eff = theta_geo + tilt_used`, r = 0.992), and that the tilt grows with
`max_slope`. The initialisation above constrains no downward-facing cell, so
where does a systematic, sloped direction come from on a part whose only
upward face is a flat top?

### 4.1 The infill makes a second, inner surface

Every benchmark part, and the calibration cube, has `"infill": true`. Stage
3 bis replaces the SDF with (`fff3.sdf_generate_infill`, `fff3.py:705-762`):

```
wall_width = shell_thickness x deposition_width           # 2 x 0.9 = 1.8 mm
infill     = max(sdf, gyroid_walls)                       # gyroid, inside the part
hollow_sdf = -sdf - wall_width   if sdf < -wall_width / 2
             sdf                 otherwise
output     = min(infill, hollow_sdf)                      # shell  or  gyroid
```

`hollow_sdf` is a shell: solid from the surface down to 1.8 mm, then
**"outside" again**, apart from the gyroid walls. The SDF therefore has a
second boundary, 1.8 mm inside every outer surface, whose outward normal is
`-n`, facing into the part.

Above an overhang whose outer normal `n` points down and outward, that inner
surface faces **up and away from the overhang**, at `90 - theta_geo` from
vertical. It is not "pointing down", it is flat, and its cells are within one
layer height of a boundary. So wherever `90 - theta_geo < max_slope`, the
ceiling rule of section 2.2 **pins the field to it**, and the layers print
parallel to the underside: the effective overhang goes to 90 degrees.

### 4.2 Measured: the stage itself, with and without infill

`experiment/experiment_orientation_field_ramp.py` builds the exact SDF of an
`xs` ramp (no Blender), applies the infill kernel with `sdf_to_isdf`'s
arguments or not, and runs stage 4's steps on the CPU. Its field is
**bit-identical** to `tools/compute_tool_orientations.py` run on the same SDF
(checked for `ramp70` at 30 degrees). Development numbers only: CPU, an
analytic SDF instead of the remesh, and field cells within 0.9 mm of the
underside rather than atoms.

| Ramp, `max_slope` | Infill | Tilted constraints above the underside | Tilt next to the underside, mean / max | Mean theta_eff there | Lab baseline, `s` (tilt near / mean theta_eff) |
|---|---|---|---|---|---|
| `ramp60`, 30 | on | 1 887, at 29.8 deg, azimuth 180 (away) | 28.2 / 29.3 | 88.2 | 29.65 / 88.5 |
| `ramp60`, 30 | **off** | 0 | **0 / 0** | **60.0** | – |
| `ramp70`, 30 | on | 3 910, at 20.3 deg, azimuth 179, 1.5 mm deep | 19.8 / 20.4 | 89.6 | 20.64 / 89.5 |
| `ramp70`, 30 | **off** | 0 | **0 / 0** | **70.0** | – |
| `ramp70`, 15 | on | 16, at 11.7 deg (the inner surface is at 20: not a ceiling) | 7.4 / 8.4 | 77.4 | 9.68 / 78.3 |
| `ramp45`, 30 | on | 6 (the inner surface is at 45: not a ceiling) | 5.1 / 6.6 | 50.1 | 6.29 / 49.0 |
| `ramp90`, 30 | on | 13; max tilt anywhere 11.6 | 0.6 / 3.6 | 89.4 | max tilt 11.9 |
| `ramp90`, 30 | **off** | 0 | **0 / 0** | **90.0** | – |

What it shows:

* **Without infill, the stock field on the ramps is exactly vertical.** Stock
  Atomizer then prints these overhangs like a 3-axis printer, at their
  geometric angle.
* **With infill, the inner shell surface above the underside is constrained
  to the direction pointing away from the overhang**, whenever that direction
  is within the budget, and the field next to the underside follows it. The
  numbers match the lab baseline's to within a degree or two, from a
  different SDF on a different backend.
* When the inner surface is steeper than the budget (`ramp45` and `ramp70` at
  lower budgets), the smaller tilts come from the gyroid's own surfaces
  where they happen to face up within the budget: a few cells, all
  directions, which is also where `ramp90`'s 11-12 degrees come from.

The P0.8 conclusion stands as a measurement of stock Atomizer as it is run,
with infill. Its cause is not the smoothing and not a sign error in the
kinematics: it is the infill's inner surface meeting a ceiling rule written
for outer top surfaces.

## 5. What this means for P2.2

P2.2 constrains the **outer** underside cells to lean toward the overhang,
`[theta, phi] = [t, azimuth of n]`. With infill on, the stock rule constrains
the **inner** surface 1.8 mm above them to lean away, at `90 - theta_geo`. On
`ramp60` at a 30-degree budget, with P2.2's `t = 60 - 45 + 2 = 17`, the two
directions are 47 degrees apart and 1.8 mm apart. The field between them
would have to turn by 47 degrees across ten cells, and the 32 final passes
average both. Neither P2.2's text nor the P0.8 analysis anticipated this.

Options, **not decided here** (plan_corrections P2-8):

| Option | What it does | Cost |
|---|---|---|
| (a) Compute the field on the SDF **before** infill, when `overhang_aware` | The field sees only the part's real surfaces; the later stages still use the infilled SDF, so the infill still prints | The pre-infill SDF must be kept (today stage 3 bis overwrites it); stages 4 and 5 read different files. Behind the flag, so the golden output is untouched |
| (b) Keep the infilled SDF, but **exclude inner surfaces** from the ceiling rule when `overhang_aware` | Same effect on the constraints | Needs to tell inner from outer surfaces, which again needs the pre-infill SDF |
| (c) Leave it | P2.2's constraint fights the stock one | Not recommended: the conflict is geometric, not a tuning matter |

Either (a) or (b) changes what stock Atomizer does on top surfaces too,
wherever the infill's inner surfaces were acting as ceilings. That is part of
the overhang-aware field's behaviour and belongs in P2.5's top-surface check.

Two more consequences:

* **The baseline comparison.** P2.5 compares against stock *with* infill, as
  measured. Stock *without* infill would be a fairer "3-axis-like" reference
  for the ramps (0 tilt, `theta_eff = theta_geo`). Whether to add it is a
  question for the team; it is cheap in the field-only tier.
* **A faster loop for P2.2.** The experiment script runs stage 4 on an `xs`
  ramp in about a minute on four CPU cores, with no Blender and no GPU, and
  reports the field at the underside directly. P2.2's field changes can be
  tried there first, then confirmed with P2.0's field-only runs.

## 6. The analytic bound (`tools/tilt_bound.py`)

Tilting by `t` toward the overhang lowers the effective angle by `t`, so the
steepest printable overhang is `max_overhang + usable tilt`, where the usable
tilt is the machine's limit capped by the nozzle at `90 - cone / 2`: 50
degrees for the reference nozzle, 60 for `dev60`'s (section 2.1).
P2.5 counts a part only up to that minus the 2-degree margin.

```
python tools/tilt_bound.py --tilt 30 40 45 50
```

| Machine tilt | Steepest printable | Counted by P2.5 up to | `ramp80` | `ramp90`, flat T-shape |
|---|---|---|---|---|
| 30 (reference) | 75 | 73 | out of range | out of range |
| 40 | 85 | 83 | inside | out of range |
| 45 | 90 | 88 | inside | edge (needs 47 with the margin) |
| 50 | 90 | 90 | inside | inside |

`ramp45` to `ramp70` are inside at 30 degrees. A flat ledge needs 45 degrees of
usable tilt, 47 with the margin. The bound is an upper limit: it assumes the
full tilt is available where the overhang is (gate M2) and reached before the
overhang starts (P2.4). These are the numbers for gate D0.

## 7. The overhang-aware field (build plan P2.2)

### 7.1 What was built

Switched on by `"overhang_aware": true` in the parameter file (or
`overhang_report.py --overhang-aware`). Without it the pipeline runs
upstream's code and commands exactly (`tests/test_atomize_stages.py`).

* **The rule** (`init_overhang_aware` in `src/atom/orientation_field.py`;
  the same rule in numpy in `src/atom/overhang_field.py`): upstream's
  initialisation of section 2.2, plus one branch. A boundary cell whose
  closest normal points down, with curvature below the threshold and not in
  the first layer, and whose surface is steeper than `max_overhang_deg`, is
  constrained to `[t, atan2(n_y, n_x)]`, with
  `t = min(theta_geo - max_overhang_deg + overhang_margin_deg, budget)`.
  The budget is section 2.1's `MAX_SLOPE_ANGLE`, and `max_slope` is the
  machine profile's limit when the parameter file does not give one (the
  operator's decision b, plan_corrections P2-11). `max_overhang_deg` 45 and
  `overhang_margin_deg` 2 are gate D1 placeholders. P2.3's reachability map
  will give a budget per position; until then it is one number.
* **The field on the part before infill** (option (a), plan_corrections
  P2-8). `atomize.py` has `bpn_to_sdf` write `data/sdf/<part>_solid.npz`,
  `sdf_to_isdf` read it and write the usual `data/sdf/<part>.npz`, and stage
  4 get `--solid_sdf`. The field is computed on the solid SDF and then set to
  NaN wherever the infilled SDF is outside: exactly the cells upstream
  masks, so no later stage sees a difference in which cells have a
  direction. Without infill there is one SDF and nothing to route.
* **Holding the overhang constraints** (`"hold_overhang"`, **on by
  default**, the operator's decision of 2026-09-30; `false` switches it off):
  after each of the 32 final passes the overhang cells are put back to their
  constraint, as upstream does for the first layer (P2-2).
* Stage 4 runs all this through `atom.orientation_field.compute_direction_field`
  when it is given `--overhang_aware` or `--solid_sdf`, and upstream's own
  code otherwise. With every option off, `compute_direction_field` gives
  upstream's field **bit for bit** (tested on the ramp, with and without
  infill), so the new path is upstream's steps plus the additions, nothing
  else. `fff3.py` is not edited (the plan placed a module flag there,
  plan_corrections P2-12).

### 7.2 Measured on the `xs` ramps

`experiment/experiment_orientation_field_ramp.py` with `--infill
--overhang-aware --solid [--no-hold]`: the exact SDF, Atomizer's infill kernel,
CPU. Cells inside the part within 0.9 mm of the underside, away from its ends
(as section 4.2). Development numbers, not comparable with the lab baseline.
Budget 30 is the reference machine's; 60 is `dev60`'s (60-degree nozzle).

| Ramp | Budget | Field on | Hold | Overhang constraints (capped) | Constraint above the underside | Tilt there, after the multigrid -> after the 32 passes | theta_eff mean / max | Lean toward |
|---|---|---|---|---|---|---|---|---|
| `ramp45` | 30 | solid | no | 11 678 (0) | 2.0 @ azimuth 0 | 2.0 -> 1.9 | 43.1 / 43.4 | +0.03 |
| `ramp50` | 30 | solid | no | 13 686 (0) | 7.0 @ 0 | 6.9 -> 6.6 | 43.4 / 44.1 | +0.11 |
| `ramp50` | 30 | solid | yes | 13 686 (0) | 7.0 @ 0 | 6.9 -> 6.9 | 43.1 / 43.8 | +0.12 |
| `ramp60` | 30 | solid | no | 12 163 (0) | 17.0 @ 0 | 16.7 -> 16.0 | 44.0 / 45.3 | +0.27 |
| `ramp60` | 30 | solid | yes | 12 163 (0) | 17.0 @ 0 | 16.7 -> 16.7 | 43.3 / 44.8 | +0.29 |
| `ramp60` | 30 | **infilled** | no | 29 341 (**17 178**) | 19.3 @ 32 | 11.8 -> 3.4 | **57.3 / 64.6** | +0.05 |
| `ramp70` | 30 | solid | no | 11 086 (0) | 27.0 @ 0 | 26.5 -> 25.4 | 44.6 / 46.5 | +0.43 |
| `ramp70` | 30 | solid | yes | 11 086 (0) | 27.0 @ 0 | 26.5 -> 26.5 | 43.5 / 45.6 | +0.45 |
| `ramp80` | 60 | stock (infilled, no rule) | no | 0 | 10.2 @ 178 (the inner surface) | 10.6 -> 10.6 | 89.3 / 90.0 | -0.18 |
| `ramp80` | 60 | solid | no | 10 653 (0) | 37.0 @ 0 | 36.3 -> 34.9 | 45.1 / 48.2 | +0.57 |
| `ramp80` | 60 | solid | yes | 10 653 (0) | 37.0 @ 0 | 36.3 -> 36.3 | 43.7 / 46.5 | +0.59 |
| `ramp90` | 60 | solid | no | 7 724 (0) | 47.0 @ 0 | 45.9 -> 44.1 | 45.9 / 50.9 | 0 (flat) |
| `ramp90` | 60 | solid | yes | 7 724 (0) | 47.0 @ 0 | 45.9 -> 45.9 | 44.1 / 48.4 | 0 (flat) |

`ramp70` at 60 is identical to `ramp70` at 30: the rule asks for 27 degrees,
inside both budgets. Stock with infill at 30 (section 4.2) was 88.2 on
`ramp60` and 89.6 on `ramp70`.

What it shows:

* **The sign is right.** Every overhang-aware row leans toward the overhang
  (stock: away), and the mean effective overhang next to the underside is
  43-46 degrees on every ramp whose rule fits the budget, from `ramp45` to
  the flat `ramp90` at 60. `tests/test_orientation_field.py` pins this on
  `ramp70` at 30.
* **Option (a) is necessary, not a refinement.** On the infilled SDF the rule
  also fires on the hollow's and the gyroid's downward-facing surfaces:
  17 178 more constraints than on the solid part, and 17 178 capped, so
  surfaces facing nearly straight down, all through the interior. The field
  next to the underside ends at 3.4 degrees of tilt and 57 degrees effective.
* **The 32 final passes cost 0.1 to 1.8 degrees of tilt** (P2-2, now
  measured), growing with the tilt the rule asks for. **Holding removes the
  loss entirely**: the tilt after the passes equals the tilt after the
  multigrid, and the worst effective angle drops by 0.3 to 2.5 degrees.
* **The worst cell is still above 45 on `ramp70` to `ramp90`**, even with hold
  (45.6 on `ramp70`, 46.5 on `ramp80`, 48.4 on `ramp90`). With hold, a
  constrained cell sits at `theta_geo - t`, about 43, so these are cells of
  the 0.9 mm band that carry no constraint (the constraints are 0.22 mm deep
  on average), where the aligner blends toward the interior's direction. The 2-degree margin (gate D1) is about what these ramps
  need on the mean, and not enough for the worst cell. Whether the metric on
  atoms and toolpaths (P2.0, P2.5) sees the same is the next measurement.

### 7.3 Behaviours to know

* **A threshold, not a ramp.** A surface at `max_overhang_deg` gets nothing,
  one just above it gets the whole margin: at 45.1 degrees the rule asks for
  2.1. `ramp45`'s underside measures a little above 45 on the grid, so the
  rule fires there with 2 degrees (harmless: its effective angle goes from 45
  to 43). Whether the margin should fade in is for D1.
* **A flat underside has no azimuth.** For `n = (0, 0, -1)` any lean lowers
  the effective angle by the same amount, and `atan2(n_y, n_x)` is decided by
  rounding. On the analytic ramp `n_y` is exactly 0 and the azimuth comes out
  0 everywhere, so the constraints agree and the smoothing keeps them
  (`ramp90` above). On a remeshed part the noise could point neighbouring
  cells in different directions and the smoothing would average them toward
  vertical. Not seen yet, because no remeshed flat underside has been run;
  the T-shape part is the test, and a rule for it (a common azimuth per
  underside, for example) would be P2.2 follow-up work.
* **`overhang_priority` has nothing to decide cell by cell.** A cell has one
  closest normal, so it is a ceiling (pointing up) or an overhang (pointing
  down), never both; the "conflict" the plan describes happens between
  neighbouring cells, through the smoothing, which is what the solid SDF and
  hold address. No count of conflicting cells is logged because there are
  none to count (plan_corrections P2-12).
* **The cost** should be the stock stage's: the same multigrid solve and
  passes, plus one copy per pass with hold. Not timed against stock yet; the
  stage logs `Direction computation took` as upstream does, so the first
  field-only run on the laptop will show it.

### 7.4 The first run on real output (laptop, 2026-09-30)

`python tools/overhang_report.py data/param/ramp60_xs.json --max-slope 30
--field-only --overhang-aware`, the laptop, stock backend mix, `reference`
profile, hold on (commit `0c7bad9`):

| | Stock (P2.0 run) | Overhang-aware |
|---|---|---|
| Worst effective overhang | 89.1 | **56.1** |
| Max tilt used | 29.1 | 21.6 |
| Overhang constraints | – | 12 099 cells, none capped |
| Direction computation | – | 36.3 s |

**Where the 56 comes from.** The same stages 4-8 run here on the CPU from the
exact SDF (no Blender) give a worst of 52.1 and a mean of 43.6 over the 6 671
atom samples next to the underside. By position along the underside:

| Part of the underside | Mean | Worst | Samples over 45 | Mean tilt |
|---|---|---|---|---|
| First 5 %, where the column's vertical wall turns into the overhang | 46.1 | **52.1** | 58 % | 13.9 |
| 5-95 % | 43.2-43.6 | 46.4 | 0-24 % | 16.4-16.8 |
| Last 5 %, at the tip | 44.0 | 46.0 | 47 % | 16.0 |

The worst atoms sit just below the corner, at `x` = 20.5 mm against the
corner's 21, with 8-10 degrees of tilt instead of 17: the field turns from
the column's vertical direction to the overhang's 17 degrees over about a
millimetre, and the first strip of overhang is printed before it has turned.
This is the case build plan **P2.4** (tilt ramp-in) exists for: start the
tilt below the overhang so it is complete where the overhang begins.

The CPU reproduction is trustworthy for this: run the same way with the stock
field, it gives a worst of **89.3 and a max tilt of 29.3, against the laptop's
89.1 and 29.1**. The laptop's 56 against 52 here is the remeshed SDF against
the exact one; the laptop's per-position numbers have not been measured.

## 8. Open

| Question | Where it is decided |
|---|---|
| ~~How P2.2 handles the infill's inner surfaces~~ | decided: option (a), built (section 7.1) |
| Whether a stock-without-infill reference joins the comparison | the team, with D0 |
| ~~How much of an overhang constraint survives the 32 final passes~~ | measured: 0.1-1.8 degrees lost, none with hold (section 7.2) |
| ~~Whether `hold_overhang` should be on by default~~ | decided: on (operator, 2026-09-30). Not yet checked: whether it makes the nozzle turn less smoothly near overhangs (P2.5) |
| `max_overhang_deg` and the margin: is 2 degrees enough, should it fade in | kept as they are for now (operator, 2026-09-30); gate D1 can revisit with P2.5's numbers |
| The azimuth of a flat or nearly flat underside on a remeshed part | P2.2 follow-up, after a T-shape run (section 7.3) |
| The corner where a wall turns into an overhang prints before the tilt is complete (worst 52-56 on `ramp60_xs`) | build plan P2.4, tilt ramp-in (section 7.4) |
| Whether the ceiling threshold should stay tied to `max_slope` once overhangs are constrained too | gate D1 |
