"""Tilt geometry: the shared vocabulary for bed orientation (build plan P0.6).

Pure numpy. No Taichi, no machine model, no I/O — just the angle and rotation
maths that the orientation field (lane B), the safety checks (lane C) and the
reporting tools all need to agree on.

Conventions
-----------
See `docs/conventions.md` for the full derivation from `atom.kinematics3z`.
In brief:

Build direction ``d``
    A unit vector, the direction the nozzle points away from the layer being
    printed. ``+Z`` means untilted. This is what `kinematics3z.inverse` takes
    as its ``normal`` argument, and what `toolpath3.Toolpath.tool_orientation`
    stores (in spherical form; see `atom.overhang_metrics`).

Total tilt
    The angle of ``d`` from ``+Z``: ``degrees(arccos(d.z))``. This is the
    quantity `kinematics3z.inverse` compares against ``MAX_TILT_ANGLE_DEG``.

Tilt parameters ``(a, b)``
    Rotation about machine X by ``a``, then about machine Y by ``b``, applied
    to ``+Z``. **This is a reporting convention chosen here, not one imposed by
    the code.** `kinematics3z.forward` does not use an X-then-Y decomposition
    at all: it composes rotations about axes derived from the ball positions
    (perpendicular to ball0->ball1 in xy, then along that vector, then about
    +Z). Any consistent parameterisation would do; this one is fixed here so
    the reachability map and the reports agree.

Units are degrees at every boundary, radians only inside functions.
"""

from __future__ import annotations

import numpy as np

#: Directions closer than this to parallel cannot define a rotation plane.
_PARALLEL_TOLERANCE = 1e-9


def rotation_from_tilts(a_deg: float, b_deg: float) -> np.ndarray:
    """Rotation matrix for a tilt of ``a`` about machine X then ``b`` about Y.

    Returns a 3x3 orthonormal matrix with determinant +1. Applying it to
    ``+Z`` gives the corresponding build direction::

        d = rotation_from_tilts(a, b) @ [0, 0, 1]
          = [sin(b)cos(a), -sin(a), cos(a)cos(b)]
    """
    a, b = np.radians(a_deg), np.radians(b_deg)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
    )
    rot_y = np.array(
        [[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]]
    )
    return rot_y @ rot_x


def direction_from_tilts(a_deg: float, b_deg: float) -> np.ndarray:
    """The build direction produced by tilts ``(a, b)``."""
    return rotation_from_tilts(a_deg, b_deg) @ np.array([0.0, 0.0, 1.0])


def tilts_from_direction(direction: np.ndarray) -> tuple[float, float]:
    """Decompose a build direction into ``(a, b)`` degrees.

    The inverse of `direction_from_tilts`, from
    ``d = [sin(b)cos(a), -sin(a), cos(a)cos(b)]``::

        a = -arcsin(d.y)
        b = arctan2(d.x, d.z)

    ``a`` lands in [-90, 90] and ``b`` in (-180, 180].
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < _PARALLEL_TOLERANCE:
        raise ValueError("cannot take the tilt of a zero-length direction")
    direction = direction / norm

    a = -np.arcsin(np.clip(direction[1], -1.0, 1.0))
    b = np.arctan2(direction[0], direction[2])
    return float(np.degrees(a)), float(np.degrees(b))


def total_tilt_deg(direction: np.ndarray) -> float:
    """Angle of a build direction from +Z, in degrees.

    This is the quantity `kinematics3z.inverse` tests against
    ``MAX_TILT_ANGLE_DEG``, so it is the one that decides reachability.
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < _PARALLEL_TOLERANCE:
        raise ValueError("cannot take the tilt of a zero-length direction")
    return float(np.degrees(np.arccos(np.clip(direction[2] / norm, -1.0, 1.0))))


def total_tilt_from_tilts_deg(a_deg: float, b_deg: float) -> float:
    """Total tilt from +Z for the tilt pair ``(a, b)``.

    ``cos(total) = cos(a) cos(b)``, so the total is **not** the sum: tilting 20
    degrees on each axis gives 27.9 degrees from vertical, not 40.
    """
    a, b = np.radians(a_deg), np.radians(b_deg)
    return float(np.degrees(np.arccos(np.clip(np.cos(a) * np.cos(b), -1.0, 1.0))))


def within_limit(
    a_deg: float, b_deg: float, max_deg: float, shape: str = "cone"
) -> bool:
    """Whether a tilt pair is inside the machine's limit.

    Parameters
    ----------
    shape
        ``"cone"`` limits the total angle from vertical, whatever the
        direction. ``"box"`` limits each axis independently, which permits
        more total tilt on the diagonal. Which one a machine has is
        **GATE M2**, a mechanical question; the reference profile declares
        ``"cone"``.
    """
    if shape == "cone":
        return total_tilt_from_tilts_deg(a_deg, b_deg) <= max_deg + 1e-9
    if shape == "box":
        return abs(a_deg) <= max_deg + 1e-9 and abs(b_deg) <= max_deg + 1e-9
    raise ValueError(f"tilt_limit_shape must be 'cone' or 'box', got {shape!r}")


def rotate_toward(
    direction: np.ndarray, target_horizontal: np.ndarray, angle_deg: float
) -> np.ndarray:
    """Rotate a build direction by ``angle_deg`` toward a horizontal direction.

    This is the operation the overhang-aware field performs (build plan P2.2):
    take ``+Z`` and lean it by ``t`` degrees toward the horizontal part of an
    overhang's outward normal, which lowers the effective overhang angle by
    exactly ``t``.

    Parameters
    ----------
    direction
        The build direction to rotate; need not be unit length.
    target_horizontal
        The direction to lean toward. Two or three components; any vertical
        component is ignored, since tilt is a horizontal choice.
    angle_deg
        How far to rotate. Negative leans away.

    Returns a unit vector. If the target is zero-length, or already parallel to
    ``direction``, there is no rotation plane and ``direction`` is returned
    unchanged (normalised).
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < _PARALLEL_TOLERANCE:
        raise ValueError("cannot rotate a zero-length direction")
    unit = direction / norm

    target = np.asarray(target_horizontal, dtype=np.float64).ravel()
    if target.size == 2:
        target = np.array([target[0], target[1], 0.0])
    elif target.size == 3:
        target = np.array([target[0], target[1], 0.0])
    else:
        raise ValueError(
            f"target_horizontal must have 2 or 3 components, got {target.size}"
        )

    target_norm = np.linalg.norm(target)
    if target_norm < _PARALLEL_TOLERANCE:
        return unit
    target = target / target_norm

    axis = np.cross(unit, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < _PARALLEL_TOLERANCE:
        return unit
    axis = axis / axis_norm

    # Rodrigues' rotation formula. Rotating about `unit x target` moves `unit`
    # toward `target`, because (unit x target) x unit is target's component
    # perpendicular to unit.
    angle = np.radians(angle_deg)
    rotated = (
        unit * np.cos(angle)
        + np.cross(axis, unit) * np.sin(angle)
        + axis * np.dot(axis, unit) * (1.0 - np.cos(angle))
    )
    return rotated / np.linalg.norm(rotated)
