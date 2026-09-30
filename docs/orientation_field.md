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
| `MAX_SLOPE_ANGLE` | `min(max_slope, (180 - NOZZLE_CONE_ANGLE) / 2)` = `min(max_slope, 50)` | the ceiling threshold; the opt-in ortho-to-wall branch |
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
tilt is the machine's limit capped at 50 degrees by the nozzle (section 2.1).
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

## 7. Open

| Question | Where it is decided |
|---|---|
| How P2.2 handles the infill's inner surfaces: (a), (b) or other | P2.2 design, with the operator (plan_corrections P2-8) |
| Whether a stock-without-infill reference joins the comparison | the team, with D0 |
| How much of an overhang constraint survives the 32 final passes | measured in P2.2's tiny-SDF test (P2-2) |
| Whether the ceiling threshold should stay tied to `max_slope` once overhangs are constrained too | gate D1 (`overhang_priority`) |
