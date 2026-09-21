"""The toolpath type every later lane consumes (build plan task P0.6).

`MachineToolpath` is a `toolpath3.Toolpath` with the machine state attached:
the five axis values the printer will actually be commanded to, the tilt used
at each point, and whether the inverse kinematics could solve the point at all.

Why it exists
-------------
`tools/toolpath_to_gcode.py` aborts the entire file on the first point whose IK
fails ("Fatal Error: collision found!"), with no indication of where. Every
later task needs to ask questions of the machine state without that: which
points fail (P3.3), what the bed does over time (P4.2), how fast it tilts
(P3.4), what to draw (P5.4). `from_toolpath` runs the same IK over every point
and **never aborts** — failures are marked in `valid` and the run continues.

Machine axes
------------
``machine`` is ``(N, 5)``: ``x, y, z0, z1, z2``. X and Y are the CoreXY head;
Z, U and V in the G-code are the three independent bed lead screws, in that
order. At zero tilt all three are equal to ``point.z + profile.z_offset``.

Note: these are in the **part frame**, as `kinematics3z.inverse` returns them.
`tools/toolpath_to_gcode.py` re-centres the whole toolpath on the bed before
writing G-code, so absolute positions there differ by a constant offset. See
`docs/conventions.md`.
"""

# NOTE: no `from __future__ import annotations` in this module. That flag
# turns annotations into strings, and Taichi reads kernel argument annotations
# as live objects, so `_solve_inverse_kernel` would fail with
# "Invalid type annotation (argument 0) of Taichi kernel: ti.types.ndarray()".
# See docs/plan_corrections.md 3.2.

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import machine_profile

#: Column indices into `MachineToolpath.machine`.
AXIS_X, AXIS_Y, AXIS_Z0, AXIS_Z1, AXIS_Z2 = range(5)

#: Version of the on-disk format, so a stale file is recognised as stale.
SCHEMA_VERSION = 1


@dataclass
class MachineToolpath:
    """A toolpath together with the machine state it implies.

    Attributes mirror `toolpath3.Toolpath` (``point``, ``travel_type``,
    ``tool_orientation``, ``width``, ``height``, ``point_count``,
    ``platform_height``) and add:

    machine
        ``(N, 5)`` float — x, y, z0, z1, z2 per point.
    tilt_deg
        ``(N,)`` float — angle of the build direction from +Z.
    valid
        ``(N,)`` bool — False where the inverse kinematics could not solve the
        point. `kinematics3z.inverse` signals this by returning NaN in its
        sixth value.
    lift_mm
        ``(N,)`` float — the vertical clearance `inverse` asks for at that
        point, 0 when none is needed and NaN where invalid. `add_platform`
        uses the maximum of these to size the sacrificial platform.
    """

    point: np.ndarray
    travel_type: np.ndarray
    tool_orientation: np.ndarray
    width: np.ndarray
    height: np.ndarray
    point_count: int
    platform_height: float

    machine: np.ndarray
    tilt_deg: np.ndarray
    valid: np.ndarray
    lift_mm: np.ndarray

    profile_name: str = "reference"

    # -- convenience ------------------------------------------------------

    @property
    def invalid_count(self) -> int:
        return int(np.count_nonzero(~self.valid))

    @property
    def invalid_indices(self) -> np.ndarray:
        """Indices of points the machine cannot reach, in print order."""
        return np.flatnonzero(~self.valid)

    @property
    def max_tilt_deg(self) -> float:
        reachable = self.tilt_deg[self.valid]
        return float(np.max(reachable)) if reachable.size else 0.0

    @property
    def max_lift_mm(self) -> float:
        finite = self.lift_mm[np.isfinite(self.lift_mm)]
        return float(np.max(finite)) if finite.size else 0.0

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write to a `.npz`, losslessly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            schema_version=np.array(SCHEMA_VERSION),
            point=self.point,
            travel_type=self.travel_type,
            tool_orientation=self.tool_orientation,
            width=self.width,
            height=self.height,
            point_count=np.array(self.point_count),
            platform_height=np.array(self.platform_height),
            machine=self.machine,
            tilt_deg=self.tilt_deg,
            valid=self.valid,
            lift_mm=self.lift_mm,
            profile_name=np.array(self.profile_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MachineToolpath":
        data = np.load(Path(path), allow_pickle=False)

        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema version {version}, expected {SCHEMA_VERSION}"
            )

        return cls(
            point=data["point"],
            travel_type=data["travel_type"],
            tool_orientation=data["tool_orientation"],
            width=data["width"],
            height=data["height"],
            point_count=int(data["point_count"]),
            platform_height=float(data["platform_height"]),
            machine=data["machine"],
            tilt_deg=data["tilt_deg"],
            valid=data["valid"],
            lift_mm=data["lift_mm"],
            profile_name=str(data["profile_name"]),
        )


def bed_centering_offset(points, profile) -> np.ndarray:
    """The XY shift `tools/toolpath_to_gcode.py` applies before writing G-code.

    That stage re-centres the whole toolpath on the bed, so the machine axis
    values in the G-code are not the ones a part-frame solve produces. This
    reproduces its arithmetic exactly (`toolpath_to_gcode.toolpath_to_gcode`,
    which is also mirrored in `kinematics3z.get_plaftorm_size`)::

        offset = 0.5 * ((MAX_X, MAX_Y, 0) + min - max) - min

    The difference is not a constant added to the screw heights: bed tilt
    pivots about the ball joints, so how far a screw must travel for a given
    tilt depends on where the point sits relative to them. Comparing against
    the G-code therefore requires solving in the same frame.
    """
    points = np.asarray(points, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    limits = np.array([profile.max_x_axis, profile.max_y_axis, 0.0])
    offset = 0.5 * (limits + low - high) - low
    offset[2] = 0.0
    return offset


def from_toolpath(toolpath, profile=None, center_on_bed: bool = False) -> MachineToolpath:
    """Run the inverse kinematics over a toolpath, marking rather than aborting.

    Parameters
    ----------
    toolpath
        A `toolpath3.Toolpath`, or anything with the same attributes.
    profile
        The machine to solve against. Defaults to the active profile.
    center_on_bed
        When True, apply the same XY re-centring `tools/toolpath_to_gcode.py`
        does before solving, so the machine axis values match the G-code. The
        default of False keeps the part frame, which is what the geometry-side
        analyses want.

        **This must be the profile `atom.kinematics3z` was imported with.**
        That module bakes its constants into Taichi closures at import time, so
        a different profile cannot be applied afterwards; passing one raises
        rather than silently solving against the wrong machine. To change
        machines, set ``ATOM_MACHINE`` before importing.

    Requires Taichi to be initialised (see `atom.ti_env.init_taichi`).
    """
    from . import kinematics3z, overhang_metrics

    active = machine_profile.load_profile() if profile is None else profile
    if active.name != kinematics3z._PROFILE.name:
        raise ValueError(
            f"atom.kinematics3z was imported with the {kinematics3z._PROFILE.name!r} "
            f"profile, but {active.name!r} was requested. Machine constants are "
            "baked into Taichi closures at import time, so set ATOM_MACHINE "
            "before importing rather than passing a different profile here."
        )

    count = int(np.asarray(toolpath.point_count).item())
    points = np.asarray(toolpath.point[:count], dtype=np.float64)
    if center_on_bed and count:
        points = points + bed_centering_offset(points, active)
    points = np.ascontiguousarray(points.astype(np.float32))
    directions = np.ascontiguousarray(
        overhang_metrics.tool_directions(toolpath)[:count].astype(np.float32)
    )

    machine = np.zeros((count, 5), dtype=np.float64)
    lift = np.zeros(count, dtype=np.float64)
    if count:
        results = np.zeros((count, 6), dtype=np.float32)
        _solve_inverse_kernel(points, directions, results)
        machine[:] = results[:, :5]
        lift[:] = results[:, 5]

    valid = np.isfinite(lift)
    tilt_deg = np.degrees(
        np.arccos(np.clip(directions[:, 2].astype(np.float64), -1.0, 1.0))
    )

    return MachineToolpath(
        point=points if center_on_bed else np.asarray(toolpath.point[:count]),
        travel_type=np.asarray(toolpath.travel_type[:count]),
        tool_orientation=np.asarray(toolpath.tool_orientation[:count]),
        width=np.asarray(toolpath.width[:count]),
        height=np.asarray(toolpath.height[:count]),
        point_count=count,
        platform_height=float(getattr(toolpath, "platform_height", 0.0) or 0.0),
        machine=machine,
        tilt_deg=tilt_deg,
        valid=valid,
        lift_mm=lift,
        profile_name=active.name,
    )


def _solve_inverse_kernel(points, directions, results):
    """Call `kinematics3z.inverse` for every point.

    Defined lazily so importing this module does not require a Taichi runtime;
    the kernel is compiled on first use.
    """
    global _INVERSE_KERNEL
    if _INVERSE_KERNEL is None:
        import taichi as ti

        from . import kinematics3z

        @ti.kernel
        def solve(
            point: ti.types.ndarray(),
            direction: ti.types.ndarray(),
            out: ti.types.ndarray(),
        ):
            for i in range(point.shape[0]):
                position = ti.math.vec3(point[i, 0], point[i, 1], point[i, 2])
                normal = ti.math.vec3(
                    direction[i, 0], direction[i, 1], direction[i, 2]
                )
                x, y, z0, z1, z2, offset = kinematics3z.inverse(position, normal)
                out[i, 0] = x
                out[i, 1] = y
                out[i, 2] = z0
                out[i, 3] = z1
                out[i, 4] = z2
                out[i, 5] = offset

        _INVERSE_KERNEL = solve

    _INVERSE_KERNEL(points, directions, results)


#: Compiled on first use by `_solve_inverse_kernel`.
_INVERSE_KERNEL = None
