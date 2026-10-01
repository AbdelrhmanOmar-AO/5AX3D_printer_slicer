"""Can the machine print this toolpath, and on how tall a platform? (build plan P2.5)

P2.5 asks for "no new IK failures (every point passes `kinematics3z.inverse()`)".
Nothing measured that: a run whose points the machine cannot reach stops in
`add_platform`, whose `kinematics3z.get_plaftorm_size` ends with an
assertion, so a run that completes has none by construction, and the report
said nothing either way. Nor did any report say how tall a platform
`add_platform` prints under the part so that the tilted bed's corners clear
the gantry, which on the `reference` profile grows quickly past 20 degrees of
tilt near the bed (plan_corrections P2-10, P2-23).

`platform_and_reach` answers both from the archived toolpath, so
`overhang_report.py --reanalyse` adds them to every run. It follows the
pipeline step for step: the toolpath is tesselated to 1-degree orientation
steps as `tesselate_toolpath_orientations.py` does (the IK sees those
points), then `get_plaftorm_size`'s loop runs: re-centre the part's box on
the bed as `toolpath_to_gcode` will, raise it until no point asks for more
lift (plan_corrections P1-9), and solve again in the frame of the platform's
larger box when the platform has layers (P4-4). Where the vendored loop stops
with an assertion, this counts the points no lift can reach and reports no
platform.

Requires Taichi to be initialised, and the active profile must be the one
`atom.kinematics3z` was imported with (its constants are baked in at import).

Units: millimetres and degrees.
"""

# No `from __future__ import annotations`: this module drives Taichi kernels.
# See docs/plan_corrections.md 3.2.

import math
from dataclasses import dataclass

import numpy as np

#: `tools/atomize.py`'s ``degree_angle_max_diff``: the tesselation stage's step.
TESSELATE_DEG = 1.0
#: The vendored loop has no bound; this one stops here and says so.
MAX_LIFT_STEPS = 200


@dataclass(frozen=True)
class MachineReach:
    """What `add_platform` and `toolpath_to_gcode` will make of a toolpath."""

    profile: str
    #: Points checked: the toolpath after tesselation, travel moves included
    #: (the G-code stage needs every one).
    points_checked: int
    #: Points no lift can bring within reach. Non-zero means the pipeline
    #: stops at `add_platform`.
    unreachable_points: int
    #: The platform `add_platform` prints under the part, mm: the lift
    #: rounded up to whole layers. 0 when none; NaN when unreachable.
    platform_mm: float
    #: The lift before rounding, mm.
    lift_mm: float
    #: True when the lift loop stopped at `MAX_LIFT_STEPS` without settling.
    unsettled: bool = False
    #: False when the points were checked as given, untesselated: the
    #: vendored tesselation sizes its output at three times its input and
    #: writes past it when more is needed (`tesselated_count`), which the
    #: pipeline's own stage would not have survived either.
    tesselated: bool = True


def tesselated_count(orientations, max_angle_deg):
    """How many points `toolpath3.toolpath_tesselate_orientation` makes from these.

    A move turning more than ``max_angle_deg`` becomes ``ceil(turn / max)``
    moves; the kernel writes them into an array three times the input's
    size, unchecked.
    """
    from .overhang_metrics import spherical_to_cartesian

    directions = spherical_to_cartesian(np.asarray(orientations, dtype=np.float64))
    if len(directions) < 2:
        return len(directions)
    turn = np.arccos(np.clip(np.einsum("nk,nk->n", directions[1:], directions[:-1]), -1.0, 1.0))
    limit = math.radians(max_angle_deg)
    return int(np.where(turn > limit, np.ceil(turn / limit), 1).sum()) + 1


def _shifted_offsets(points, orientations, box_min, box_max, z_shift):
    """`get_vertical_offset_kernel`'s value, and the points it found unreachable."""
    from . import contracts, kinematics3z
    from .overhang_metrics import spherical_to_cartesian

    shift = 0.5 * (np.array((kinematics3z.MAX_X_AXIS, kinematics3z.MAX_Y_AXIS, 0.0)) + box_min - box_max) - box_min
    offset = kinematics3z.get_vertical_offset_kernel(points, orientations, shift[0], shift[1], z_shift)
    if not math.isnan(offset):
        return offset, 0
    moved = np.ascontiguousarray((points + np.array([shift[0], shift[1], z_shift])).astype(np.float32))
    directions = np.ascontiguousarray(spherical_to_cartesian(orientations).astype(np.float32))
    results = np.zeros((len(points), 6), dtype=np.float32)
    contracts._solve_inverse_kernel(moved, directions, results)
    return offset, int(np.count_nonzero(np.isnan(results[:, 5])))


def _lift(points, orientations, box_min, box_max, z_shift):
    """`kinematics3z._lift_needed`, counting instead of asserting: ``(z, unreachable, settled)``."""
    for _ in range(MAX_LIFT_STEPS):
        offset, unreachable = _shifted_offsets(points, orientations, box_min, box_max, z_shift)
        if math.isnan(offset):
            return z_shift, unreachable, True
        if offset > 0:
            z_shift += offset
        else:
            return z_shift, 0, True
    return z_shift, 0, False


def platform_and_reach(toolpath, nozzle_width, layer_height, tesselate_deg=TESSELATE_DEG):
    """The platform `add_platform` would print, and the points out of reach.

    ``toolpath`` is a `toolpath3.Toolpath` (the smoothed one the report
    measures); it is copied, not changed. ``nozzle_width`` and
    ``layer_height`` are the pipeline's (`add_platform`'s arguments).
    """
    from . import kinematics3z, toolpath3

    work = toolpath3.Toolpath()
    count = int(np.asarray(toolpath.point_count).item())
    work.from_numpy({
        "point": np.ascontiguousarray(np.asarray(toolpath.point[:count], dtype=np.float32)),
        "travel_type": np.ascontiguousarray(np.asarray(toolpath.travel_type[:count], dtype=np.int32)),
        "tool_orientation": np.ascontiguousarray(np.asarray(toolpath.tool_orientation[:count], dtype=np.float32)),
        "width": np.ascontiguousarray(np.asarray(toolpath.width[:count], dtype=np.float32)),
        "height": np.ascontiguousarray(np.asarray(toolpath.height[:count], dtype=np.float32)),
        "point_count": np.array(count),
    })
    tesselated = bool(tesselate_deg) and tesselated_count(work.tool_orientation[:count], tesselate_deg) < 3 * count - 16
    if tesselated:
        work.tesselate_orientation(tesselate_deg)
    count = int(work.point_count)
    points = np.ascontiguousarray(work.point[:count], dtype=np.float32)
    orientations = np.ascontiguousarray(work.tool_orientation[:count], dtype=np.float32)
    profile = kinematics3z._PROFILE.name
    if count == 0:
        return MachineReach(profile, 0, 0, 0.0, 0.0, tesselated=tesselated)

    # kinematics3z.get_plaftorm_size, step for step.
    low, high = points.min(axis=0).astype(np.float64), points.max(axis=0).astype(np.float64)
    platform_x = np.ceil(high[0] / nozzle_width) * nozzle_width
    platform_y = np.ceil(high[1] / nozzle_width) * nozzle_width
    z_shift, unreachable, settled = _lift(points, orientations, low, high, 0.0)
    if not unreachable and int(np.ceil(z_shift / layer_height) * layer_height / layer_height) - 1 > 0:
        box_min = np.minimum(low, np.array((0.0, 0.0, low[2])))
        box_max = np.maximum(high, np.array((platform_x, platform_y, high[2])))
        z_shift, unreachable, settled_again = _lift(points, orientations, box_min, box_max, z_shift)
        settled = settled and settled_again
    if unreachable:
        return MachineReach(profile, count, unreachable, float("nan"), float("nan"), not settled, tesselated)
    platform = float(np.ceil(z_shift / layer_height) * layer_height)
    return MachineReach(profile, count, 0, platform, float(z_shift), not settled, tesselated)
