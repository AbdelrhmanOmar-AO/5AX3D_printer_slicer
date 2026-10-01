"""The reachable-tilt map: how far the machine can tilt, where (build plan P2.3).

For a grid over the bed (``GRID_STEP_MM`` apart) and a fan of tool directions
(tilt 0 to the profile's ``max_tilt_angle_deg`` in ``TILT_STEP_DEG`` steps, at
``AZIMUTH_COUNT`` azimuths), this asks the inverse kinematics
(`kinematics3z.inverse`) whether the machine reaches each point at each
direction, and how far the part must then be lifted:

* its ``offset`` NaN: **unreachable** (an axis out of range, the tilt over
  the limit, the nozzle below the bed);
* 0: reachable as the part stands;
* above 0: reachable only with the part raised that far on a printed
  platform (`add_platform`), because a bed corner would otherwise hit the
  gantry or a screw its endstop. On the `reference` profile that is most
  tilts over 20 degrees near the bed (plan_corrections P2-10).

Every reachable entry is then posed at its lifted height and the bed's
corners are tested against the clearance model (P4.1, `atom.clearance`), as
the swept check does (`atom.tilt_motion_check`). On the reference proxy this
rejects nothing the kinematics accepted (its gantry is the plane the
kinematics test the corners against); a machine with a clearance file
(gate M3) can reject more.

The operator's decisions, 2026-10-01 (plan_corrections P2-23): the map is
kept with both meanings, the largest tilt reachable at all and the largest
reachable with no platform, plus the lift each tilt needs; it is built and
viewed only, not yet used by the field (`use_reachability_map` waits for the
real machine's geometry).

Monotonic by definition
-----------------------
The raw kinematics are not monotonic in tilt: on `reference`, 1.8 % of the
reachable (point, direction) pairs have a smaller tilt in the same azimuth
that is unreachable, 95 % of them within 10 mm of the X or Y travel limits
(tilting shifts the axes, which brings some edge points into range) and the
rest high up, where a screw nears its limit. The field's tilt grows
continuously from the vertical first layer, so a tilt is only usable if every
smaller one is too. `max_tilt_deg` is therefore the largest tilt whose every
smaller step is reachable, which is monotonic by construction.

Frame
-----
The **bed frame**, the one `kinematics3z.inverse` takes positions in: the
part as `toolpath_to_gcode` places it, re-centred on the bed
(`contracts.bed_centering_offset`), z up from the bed. The map is therefore
the same for every part; `max_tilt_here` takes the part's own points and adds
its offset (and any platform height). Re-centring changes the screw heights
non-uniformly (build plan hazard 7), which is why the offset must be applied
rather than the part frame looked up directly. Azimuths follow the spherical
``phi`` convention (from +X, counter-clockwise in xy), as the field does.

Requires Taichi to be initialised (`atom.ti_env.init_taichi`), and the active
profile must be the one `atom.kinematics3z` was imported with
(`contracts.from_toolpath`'s rule): its constants are baked into Taichi
closures at import time.

Units: millimetres and degrees.
"""

# No `from __future__ import annotations`: this module drives Taichi kernels
# through atom.contracts. See docs/plan_corrections.md 3.2.

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bed_motion, clearance, contracts, machine_profile

#: Grid spacing over the bed, mm (build plan P2.3: "cell about 10 mm").
GRID_STEP_MM = 10.0
#: Tilt step, degrees.
TILT_STEP_DEG = 2.5
#: Azimuths, evenly spaced from +X.
AZIMUTH_COUNT = 16
#: A bed corner may sit this far inside a clearance body before it counts:
#: the forward kinematics run in float32 (`tilt_motion_check`'s tolerance).
CLEARANCE_TOLERANCE_MM = 0.01
#: The cache's layout; bump on any change to what is stored.
FORMAT_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "reachability"

#: Solves per batch, to bound memory.
_CHUNK = 200_000


@dataclass
class ReachabilityMap:
    """The map. Arrays are indexed ``[x, y, z, tilt, azimuth]``."""

    profile_name: str
    #: Grid coordinates along each axis, bed frame, mm.
    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    tilts_deg: np.ndarray
    azimuths_deg: np.ndarray
    #: The lift each (point, direction) needs, mm: NaN unreachable, 0 as it
    #: stands. float32.
    lift_mm: np.ndarray
    #: False where the clearance model rejects the lifted pose (and wherever
    #: the kinematics do).
    clear: np.ndarray
    #: The clearance model's `describe()` name.
    clearance_model: str

    @property
    def reachable(self) -> np.ndarray:
        """Reachable at all, any platform."""
        return np.isfinite(self.lift_mm) & self.clear

    @property
    def reachable_without_lift(self) -> np.ndarray:
        return self.reachable & (np.nan_to_num(self.lift_mm, nan=np.inf) <= 0.0)

    def max_tilt_deg(self, without_lift: bool = False) -> np.ndarray:
        """``[x, y, z, azimuth]``: the largest tilt whose every smaller step is
        reachable; NaN where not even the vertical is."""
        ok = self.reachable_without_lift if without_lift else self.reachable
        prefix = np.logical_and.accumulate(ok, axis=3)
        steps = prefix.sum(axis=3)
        return np.where(steps > 0, self.tilts_deg[np.maximum(steps - 1, 0)], np.nan)

    def max_tilt_here(self, points_mm, azimuth_deg, without_lift: bool = False, offset_mm=(0.0, 0.0, 0.0)):
        """The largest usable tilt at each point toward each azimuth, degrees.

        ``points_mm`` ``(N, 3)`` in the part's frame; ``offset_mm`` its
        placement on the bed (`contracts.bed_centering_offset`, plus any
        platform height in z). Trilinear between the grid points, the
        smaller of the two azimuths either side, rounded down to the tilt
        step (the plan's conservative lookup). NaN outside the grid, or where
        a grid point it needs reaches nothing: the safe answer, "no tilt is
        known to be reachable here".
        """
        points = np.atleast_2d(np.asarray(points_mm, dtype=np.float64)) + np.asarray(offset_mm, dtype=np.float64)
        azimuth = np.broadcast_to(np.asarray(azimuth_deg, dtype=np.float64), (len(points),))
        table = self.max_tilt_deg(without_lift)
        step = 360.0 / len(self.azimuths_deg)
        position = np.mod(azimuth - self.azimuths_deg[0], 360.0) / step
        below = np.floor(position).astype(np.int64) % len(self.azimuths_deg)
        above = (below + 1) % len(self.azimuths_deg)
        axes = (self.x_mm, self.y_mm, self.z_mm)
        best = np.minimum(_trilinear(table, axes, points, below), _trilinear(table, axes, points, above))
        tilt_step = float(self.tilts_deg[1] - self.tilts_deg[0]) if len(self.tilts_deg) > 1 else 1.0
        return np.floor(best / tilt_step + 1e-9) * tilt_step

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(FORMAT_VERSION),
            profile_name=np.array(self.profile_name),
            x_mm=self.x_mm, y_mm=self.y_mm, z_mm=self.z_mm,
            tilts_deg=self.tilts_deg, azimuths_deg=self.azimuths_deg,
            lift_mm=self.lift_mm, clear=self.clear,
            clearance_model=np.array(self.clearance_model),
        )

    @classmethod
    def load(cls, path) -> "ReachabilityMap":
        with np.load(path) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: reachability map format {version}, expected {FORMAT_VERSION}; rebuild it")
            return cls(
                profile_name=str(data["profile_name"]),
                x_mm=data["x_mm"], y_mm=data["y_mm"], z_mm=data["z_mm"],
                tilts_deg=data["tilts_deg"], azimuths_deg=data["azimuths_deg"],
                lift_mm=data["lift_mm"], clear=data["clear"],
                clearance_model=str(data["clearance_model"]),
            )


def _trilinear(table, axes, points, azimuth_index):
    """``table[x, y, z, azimuth]`` interpolated at ``points``, each at its own azimuth.

    NaN outside the grid, or where a grid point that carries weight is NaN.
    """
    inside = np.ones(len(points), dtype=bool)
    cells, fractions = [], []
    for axis, coordinates in enumerate(axes):
        value = points[:, axis]
        inside &= (value >= coordinates[0] - 1e-9) & (value <= coordinates[-1] + 1e-9)
        if len(coordinates) == 1:
            cells.append(np.zeros(len(points), dtype=np.int64))
            fractions.append(np.zeros(len(points)))
            continue
        cell = np.clip(np.searchsorted(coordinates, value, side="right") - 1, 0, len(coordinates) - 2)
        cells.append(cell)
        fractions.append(np.clip((value - coordinates[cell]) / (coordinates[cell + 1] - coordinates[cell]), 0.0, 1.0))
    total = np.zeros(len(points))
    poisoned = ~inside
    for corner in range(8):
        weight = np.ones(len(points))
        index = []
        for axis in range(3):
            bit = (corner >> axis) & 1
            weight = weight * (fractions[axis] if bit else 1.0 - fractions[axis])
            index.append(np.minimum(cells[axis] + bit, len(axes[axis]) - 1))
        value = table[index[0], index[1], index[2], azimuth_index]
        counts = weight > 0
        poisoned |= counts & np.isnan(value)
        total += np.where(counts, weight * np.nan_to_num(value, nan=0.0), 0.0)
    return np.where(poisoned, np.nan, total)


def directions(tilts_deg, azimuths_deg) -> np.ndarray:
    """``[tilt, azimuth, 3]`` unit vectors, spherical ``phi`` convention."""
    tilt = np.radians(np.asarray(tilts_deg, dtype=np.float64))[:, None]
    azimuth = np.radians(np.asarray(azimuths_deg, dtype=np.float64))[None, :]
    return np.stack(
        np.broadcast_arrays(np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)),
        axis=-1,
    )


def _solve(points, dirs):
    """`kinematics3z.inverse` for each pair: ``(machine (N, 5), lift (N,))``."""
    machine = np.empty((len(points), 5), dtype=np.float64)
    lift = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), _CHUNK):
        p = np.ascontiguousarray(points[start:start + _CHUNK], dtype=np.float32)
        d = np.ascontiguousarray(dirs[start:start + _CHUNK], dtype=np.float32)
        out = np.zeros((len(p), 6), dtype=np.float32)
        if len(p):
            contracts._solve_inverse_kernel(p, d, out)
        machine[start:start + len(p)] = out[:, :5]
        lift[start:start + len(p)] = out[:, 5]
    return machine, lift


#: The lift loop stops when a step asks for less than this, mm.
LIFT_CONVERGED_MM = 1e-3
#: ... or after this many steps (P1-9 measured five at 20 degrees).
LIFT_MAX_STEPS = 30


def _converged_lift(points, dirs, first):
    """Raise each point by the kinematics' offset until they ask for no more.

    Returns ``(lift, machine)`` per point: the converged total lift (NaN for
    any that becomes unreachable on the way up, an axis running out of
    travel, or does not converge), and the machine state at that height.
    """
    total = np.maximum(np.asarray(first, dtype=np.float64), 0.0)
    machine = np.full((len(points), 5), np.nan)
    active = np.arange(len(points))
    for _ in range(LIFT_MAX_STEPS):
        if not len(active):
            break
        raised = points[active].copy()
        raised[:, 2] += total[active]
        state, more = _solve(raised, dirs[active])
        machine[active] = state
        lost = ~np.isfinite(more) | ~np.isfinite(state).all(axis=1)
        total[active[lost]] = np.nan
        step = np.where(lost, 0.0, np.maximum(more, 0.0))
        total[active] += step
        active = active[~lost & (step > LIFT_CONVERGED_MM)]
    total[active] = np.nan  # did not converge: not known to be reachable
    return total, machine


def _corners_clear(machine, profile, model, tolerance_mm=CLEARANCE_TOLERANCE_MM):
    """For each machine state, whether the bed's corners clear the model."""
    from .tilt_motion_check import poses

    corners = bed_motion.bed_corners(profile)
    result = np.zeros(len(machine), dtype=bool)
    for start in range(0, len(machine), _CHUNK // 4):
        chunk = machine[start:start + _CHUNK // 4]
        if not len(chunk):
            continue
        rotation, translation, _ = poses(chunk)
        world = np.matmul(corners[None], np.transpose(rotation, (0, 2, 1))) + translation[:, None, :]
        heads = np.repeat(chunk[:, :2], len(corners), axis=0)
        distance = model.signed_distance(world.reshape(-1, 3), heads).reshape(len(chunk), len(corners))
        result[start:start + len(chunk)] = distance.min(axis=1) >= -tolerance_mm
    return result


def _axis(limit, step):
    count = int(math.floor(limit / step + 1e-9))
    return np.arange(count + 1, dtype=np.float64) * step


def build_map(profile=None, step_mm=GRID_STEP_MM, tilt_step_deg=TILT_STEP_DEG,
              azimuth_count=AZIMUTH_COUNT, x_mm=None, y_mm=None, z_mm=None,
              model=None) -> ReachabilityMap:
    """Build the map for the active profile.

    The grid defaults to the whole bed (0 to ``max_x_axis`` and
    ``max_y_axis``) from the bed up to ``max_z_axis``, every ``step_mm``;
    ``x_mm``, ``y_mm`` and ``z_mm`` replace an axis (tests use small grids).
    The tilts run from 0 to ``max_tilt_angle_deg``.
    """
    active = machine_profile.load_profile() if profile is None else profile
    from . import kinematics3z

    if active.name != kinematics3z._PROFILE.name:
        raise ValueError(
            f"atom.kinematics3z was imported with the {kinematics3z._PROFILE.name!r} profile, "
            f"but {active.name!r} was requested; set ATOM_MACHINE before importing."
        )
    model = clearance.load_clearance(active) if model is None else model
    xs = _axis(active.max_x_axis, step_mm) if x_mm is None else np.asarray(x_mm, dtype=np.float64)
    ys = _axis(active.max_y_axis, step_mm) if y_mm is None else np.asarray(y_mm, dtype=np.float64)
    zs = _axis(active.max_z_axis, step_mm) if z_mm is None else np.asarray(z_mm, dtype=np.float64)
    tilts = _axis(active.max_tilt_angle_deg, tilt_step_deg)
    azimuths = np.arange(azimuth_count, dtype=np.float64) * (360.0 / azimuth_count)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    fan = directions(tilts, azimuths).reshape(-1, 3)
    points = np.repeat(grid, len(fan), axis=0)
    dirs = np.tile(fan, (len(grid), 1))
    _, lift = _solve(points, dirs)

    # The kinematics' offset is a first estimate (plan_corrections P1-9):
    # raise by it and solve again until no more is asked, as
    # `kinematics3z.get_plaftorm_size` does. Then pose each entry at its
    # lifted height and test the bed's corners.
    found = np.flatnonzero(np.isfinite(lift))
    converged, machine = _converged_lift(points[found], dirs[found], lift[found])
    lift = np.full(len(points), np.nan)
    lift[found] = converged
    clear = np.isfinite(lift)
    kept = np.isfinite(converged)
    clear[found[kept]] = _corners_clear(machine[kept], active, model)

    shape = (len(xs), len(ys), len(zs), len(tilts), len(azimuths))
    return ReachabilityMap(
        profile_name=active.name,
        x_mm=xs, y_mm=ys, z_mm=zs, tilts_deg=tilts, azimuths_deg=azimuths,
        lift_mm=lift.reshape(shape).astype(np.float32),
        clear=clear.reshape(shape),
        clearance_model=str(model.describe().get("model", type(model).__name__)),
    )


def cache_path(profile_name) -> Path:
    """``data/reachability/<profile>.npz``."""
    return CACHE_DIR / f"{profile_name}.npz"


def load_or_build(profile=None, rebuild: bool = False) -> ReachabilityMap:
    """The cached map for the active profile, built and saved if missing."""
    active = machine_profile.load_profile() if profile is None else profile
    path = cache_path(active.name)
    if path.is_file() and not rebuild:
        try:
            cached = ReachabilityMap.load(path)
        except (ValueError, KeyError, OSError):
            cached = None
        if cached is not None and cached.profile_name == active.name:
            return cached
    built = build_map(active)
    built.save(path)
    return built
