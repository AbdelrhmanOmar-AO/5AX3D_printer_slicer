# Coordinate and tilt conventions

Derived from `src/atom/kinematics3z.py` and verified by running it, not
assumed. Build plan task P0.6. Every later lane codes against this document.

Units: **millimetres** for length, **degrees** at every public interface,
radians only inside functions.

---

## 1. Frames

### Machine frame

X and Y as the CoreXY head moves; +Z away from the bed at zero tilt. This is
the frame `kinematics3z` works in. Travel limits come from the active machine
profile (`max_x_axis`, `max_y_axis`, `max_z_axis`).

### Part frame

Where the toolpath is stored. The part sits near the origin: the calibration
cube's points span roughly x 0..21, y 0..25.

**These are not the same.** `tools/toolpath_to_gcode.py` re-centres the whole
toolpath on the bed before solving:

```python
offset = 0.5 * ((MAX_X_AXIS, MAX_Y_AXIS, 0) + min - max) - min
```

This matters more than a translation normally would. Bed tilt pivots about the
three ball joints, so how far a screw must travel for a given tilt depends on
where the point is relative to them. Solving the same toolpath in the part
frame and in the re-centred frame gives **different screw heights**, not values
differing by a constant. `contracts.from_toolpath(..., center_on_bed=True)`
reproduces the re-centred frame; `contracts.bed_centering_offset` is the same
arithmetic.

---

## 2. Tool orientation

### What `inverse(position, normal)` takes

`normal` is the **build direction**: a unit vector pointing the way the nozzle
faces away from the layer being printed. `+Z` means untilted.

Evidence:

* `kinematics3z.inverse` rejects a point when
  `acos(normal.z) / pi * 180 > MAX_TILT_ANGLE_DEG`, so `normal.z = cos(tilt)`
  with the tilt measured from +Z.
* `kinematics3z.get_vertical_offset_kernel` feeds it
  `direction.spherical_to_cartesian(tool_orientation)`, the per-point tool
  orientation from the toolpath.

Internally `inverse` immediately forms
`normalize(vec3(-normal.x, -normal.y, normal.z))`, mirroring x and y. That
converts the tool direction into the bed's own normal — tilting the bed one way
points the tool the other. `forward` mirrors back before returning, so the two
are exact inverses of each other in the build-direction convention. **Callers
never see the mirrored form.**

### How it is stored

`toolpath3.Toolpath.tool_orientation` is `(N, 2)`, **spherical**, not `(N, 3)`
Cartesian. (The build plan's facts table says otherwise; see
`docs/plan_corrections.md` 1.1.)

From `direction.spherical_to_cartesian`:

```
theta = orientation[0]      polar angle from +Z
phi   = orientation[1]      azimuth from +X in the xy-plane

d = (cos(phi) sin(theta), sin(phi) sin(theta), cos(theta))
```

So **`theta` is the tilt from vertical directly**; no conversion is needed for
that one quantity. `atom.overhang_metrics.spherical_to_cartesian` and
`tool_directions` do the full conversion in numpy.

### Total tilt

`degrees(arccos(d.z))`. This is exactly what `inverse` compares against
`MAX_TILT_ANGLE_DEG`, so it is the quantity that decides reachability.
`atom.tilt.total_tilt_deg`.

---

## 3. Tilt parameters (a, b)

For reporting and for the reachability map (P2.3), a build direction is also
described by a pair of angles: rotate about machine X by `a`, then about
machine Y by `b`, applied to +Z.

```
d = [sin(b)cos(a), -sin(a), cos(a)cos(b)]

a = -arcsin(d.y)
b =  arctan2(d.x, d.z)
```

`atom.tilt.rotation_from_tilts`, `direction_from_tilts`, `tilts_from_direction`.

**This ordering is a convention chosen here, not one the code imposes.** The
build plan asks to "confirm this order against `kinematics3z.forward()`". The
answer is that `forward` uses no such decomposition: it composes rotations
about axes derived from the ball geometry — first about an axis perpendicular
to ball0→ball1 in the xy-plane, then about that vector once rotated, then about
+Z by a third angle. None of those is machine X or Y. Any consistent
parameterisation would therefore do; this one is fixed so the reachability map
and the reports agree.

### The two angles do not add

`cos(total) = cos(a) cos(b)`. Tilting 20° on each axis gives **27.99°** from
vertical, not 40°. This is why the limit's shape matters:

| Shape | Rule | Reference machine |
|---|---|---|
| `cone` | total angle from vertical ≤ limit | ✅ declared |
| `box` | each axis ≤ limit independently | — |

A box permits more on the diagonal: (25°, 25°) is 34.6° total, inside a 30° box
but outside a 30° cone. Which one our machine has is **gate M2**.
`atom.tilt.within_limit`.

---

## 4. The three screws

`inverse` returns `(x, y, z0, z1, z2, offset)`. In G-code, `z0 → Z`,
`z1 → U`, `z2 → V`, matching the order in `kinematics3z.HEADER`.

At zero tilt all three equal `position.z + z_offset`, where `z_offset` is the
machine profile's value (75.0 mm on the reference machine).

Screw travel per degree of tilt is large, because the balls are far apart
(~309 mm between ball 0 and ball 1 on the reference machine). See the worked
example below: 10° of tilt spreads the screws over more than 50 mm.

---

## 5. How the inverse kinematics signals failure

**The sixth return value, `offset`, is NaN when the point cannot be reached.**
There is no flag and no exception.

When it is not NaN, `offset` is a non-negative **required vertical clearance**
in mm: how much the part must be raised for the move to be safe.
`kinematics3z.get_plaftorm_size` takes the maximum over a toolpath to size the
sacrificial platform.

| `offset` | Meaning |
|---|---|
| `NaN` | Unreachable. See the causes below. |
| `0.0` | Reachable as-is. |
| `> 0.0` | Reachable only if the part is raised by this much. |

Causes of NaN, in the order `inverse` checks them:

1. `position.z < 0` — the nozzle would be below the bed.
2. Tilt exceeds `MAX_TILT_ANGLE_DEG` (with a 1e-4 tolerance).
3. X or Y outside travel, or any screw above `MAX_Z_AXIS`.

Two further conditions raise `offset` rather than invalidating the point: a
screw going below zero (endstop), and a bed corner rising above
`NOZZLE_TO_GAUNTRY` (bed-gantry clearance, using `BALL_TO_CORNER` to locate the
corners).

`tools/toolpath_to_gcode.py` aborts the whole file on the first NaN, printing
"Fatal Error: collision found!" with no location.
`atom.contracts.from_toolpath` runs the same solve and marks failures in
`valid` instead, which is what the diagnostics (P3.3), the safety checks (P4)
and the visualisation (P5.4) are built on.

---

## 6. Worked example

Reference profile, produced by calling `inverse` and `forward` directly.

### Zero tilt

Position (150, 145, 10) in the machine frame, build direction (0, 0, 1):

```
x = 150.0000   y = 145.0000
z0 = 85.0000   z1 = 85.0000   z2 = 85.0000
offset = 0.0
```

All three screws equal, at `10 + 75 (z_offset) = 85`.

### Ten degrees toward +X

Same position, build direction `(sin 10°, 0, cos 10°)`:

```
x = 140.3344   y = 146.3492
z0 = 110.9078  z1 = 57.2505   z2 = 84.0791
offset = 0.0
```

X and Y move even though the commanded position did not: tilting the bed
carries the part with it, and the head must follow. The screws span 53.7 mm for
10° of tilt.

### Round trip

Build direction `(0, sin 12°, cos 12°)` at (150, 145, 10), through `inverse`
then `forward`:

```
position in  = (150.0,      145.0,      10.0)
position out = (150.0000,   145.0000,   10.0000)
normal in    = (0.000000,   0.207912,   0.978148)
normal out   = (0.000000,   0.207912,   0.978148)
```

Exact to the printed precision, confirming that the internal mirroring cancels
and that `forward` is the true inverse of `inverse` in this convention.

### Beyond the limits

Each of these returns `offset = NaN`:

| Input | Cause |
|---|---|
| tilt 35° (limit is 30°) | exceeds `MAX_TILT_ANGLE_DEG` |
| position z = −1 | nozzle below the bed |
| x = 400 (limit is 300) | outside travel |

---

## 7. Where each convention lives in code

| Concept | Module |
|---|---|
| Machine constants | `atom.machine_profile`, `config/machines/*.json` |
| Tilt angles, rotations, limits | `atom.tilt` |
| Spherical → Cartesian orientation | `atom.overhang_metrics` |
| Machine state per toolpath point | `atom.contracts` |
| Effective overhang angles | `atom.overhang_metrics` |
| The kinematics themselves | `atom.kinematics3z` (vendored; edit only where a task says so) |
