"""The bed tilt implied by three screw heights, in numpy (build plan task P1.4).

The G-code validator has to check every move's bed tilt against the machine's
limit, and a G-code file carries only the three screw heights (Z, U, V), not
the tilt. `atom.kinematics3z.forward` recovers the tilt, but it is a Taichi
function, and the validator is meant to run anywhere, on any file, without
initialising Taichi. This module provides the same answer in numpy.

Two routes, which agree
-----------------------
``total_tilt_deg``
    Closed form, total angle from vertical only. The three ball contact points
    are rigid on the bed, so the bed-frame vectors ball0->ball1 and
    ball0->ball2 (``l1``, ``l2``) keep their lengths. For a plane tilted by
    ``t``, the height gained along an in-plane vector ``l`` is ``g . l``, where
    ``g`` is the in-plane gradient with ``|g| = sin(t)``. ``forward`` makes the
    rotated ``l1`` rise by ``z0 - z1`` and the rotated ``l2`` by ``z0 - z2``,
    so ``g`` solves a 2x2 linear system and ``t = arcsin(|g|)``.

    **Not** a plane fit through the ball positions at their fixed XY. The
    contact points slide in their slots as the bed tilts, so the horizontal
    distance between them shrinks by ``cos(t)``. Fitting a plane through fixed
    XY gives ``arctan(|g|)`` instead: 26.57 degrees at a true 30. See
    ``docs/plan_corrections.md`` section 7a, P1-1.

``build_direction``
    A line-by-line numpy port of the orientation part of ``forward``: the two
    rotations that match the screw differences, then the rotation about +Z
    that the slot constraints fix. Needed when the limit is a ``box``, which
    depends on the direction of the tilt as well as its size. Returns the
    build direction ``forward`` returns, in the machine frame, x and y flipped
    back exactly as ``forward`` does (``docs/conventions.md`` section 2).

Both depend only on the screw differences, never on X, Y or the common
height, and both are checked against the Taichi ``forward`` in
``tests/test_screw_tilt.py``.

Physical versus requested direction
-----------------------------------
``build_direction`` is the direction ``forward`` reports, which is the one the
toolpath requested. The bed's physical pose differs from it by up to 0.14
degrees at 30 degrees of tilt (``docs/plan_corrections.md`` 1.9). That gap is a
rotation about the bed's own normal, so it changes the direction of the tilt
slightly but **not its total angle**: the ``cone`` check is exact, and a
``box`` check is exact to that 0.14 degrees.

Units: millimetres for screw heights, degrees at the public API.
"""

from __future__ import annotations

import numpy as np


def _geometry(profile):
    """The constant vectors of ``KinematicTaichi``, from a machine profile."""
    b0 = np.asarray(profile.ball_2dpos_0, dtype=float)
    b1 = np.asarray(profile.ball_2dpos_1, dtype=float)
    b2 = np.asarray(profile.ball_2dpos_2, dtype=float)
    l1 = b1 - b0
    l2 = b2 - b0

    def slot_normal(angle_deg):
        a = np.radians(angle_deg)
        return np.array([-np.sin(a), np.cos(a)])

    s0 = slot_normal(profile.rail_angle_0)
    s1 = slot_normal(profile.rail_angle_1)
    s2 = slot_normal(profile.rail_angle_2)
    return l1, l2, s0, s1, s2


def bed_gradient(z0, z1, z2, profile) -> np.ndarray:
    """In-plane height gradient of the bed, in the bed frame, shape ``(N, 2)``.

    Its length is ``sin`` of the total tilt. Screw heights in mm.
    """
    l1, l2, *_ = _geometry(profile)
    z0, z1, z2 = (np.atleast_1d(np.asarray(z, dtype=float)) for z in (z0, z1, z2))
    rise = np.stack([z0 - z1, z0 - z2], axis=-1)
    return np.linalg.solve(np.stack([l1, l2]), rise.T).T


def total_tilt_deg(z0, z1, z2, profile) -> np.ndarray:
    """Total bed tilt from vertical, in degrees, for each set of screw heights.

    NaN where the screw differences are larger than the bed can realise (the
    gradient would exceed 1): no rigid bed has that state.
    """
    sine = np.linalg.norm(bed_gradient(z0, z1, z2, profile), axis=-1)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.where(sine <= 1.0, np.arcsin(np.minimum(sine, 1.0)), np.nan))


def _solve_weierstrass(A, B, C):
    """``kinematics3z``'s ``solve_weierstrass``: A cos x + B sin x + C = 0.

    Returns the root of smaller magnitude, with the same special case and the
    same sign convention (``np.sign(0) == 0``, as ``ti.math.sign``).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        root = 2.0 * np.arctan2(
            (C + A) / (-B - np.sign(B) * np.sqrt(B**2 + (A + C) * (A - C))), 1.0
        )
    return np.where(A + C == 0, 0.0, root)


def _rotate(vector, axis, angle):
    """Rodrigues rotation, as ``kinematics3z``'s ``rotate``. Rows are vectors."""
    angle = np.asarray(angle)[..., None]
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.sum(axis * vector, axis=-1, keepdims=True) * (1.0 - np.cos(angle))
    )


def _angle_to_match_z(vector, axis, target_z):
    """``kinematics3z``'s ``get_angle_to_match_z``, row-wise."""
    C = axis[..., 2] * np.sum(vector * axis, axis=-1)
    A = vector[..., 2] - C
    C = C - target_z
    B = np.cross(axis, vector)[..., 2]
    return _solve_weierstrass(A, B, C)


def build_direction(z0, z1, z2, profile) -> np.ndarray:
    """The build direction ``forward`` returns for these screw heights.

    Unit vectors in the machine frame, shape ``(N, 3)``. NaN rows where the
    screw differences are not realisable.
    """
    l1, l2, s0, s1, s2 = _geometry(profile)
    z0, z1, z2 = (np.atleast_1d(np.asarray(z, dtype=float)) for z in (z0, z1, z2))
    n = z0.shape[0]

    # The constants KinematicTaichi precomputes for forward.
    k1 = -np.dot(l1, s1)
    k2 = -np.dot(l2, s2)
    s0_dir = np.array([-s0[1], s0[0]])
    s0d_s1n = np.dot(s0_dir, s1)
    s0d_s2n = np.dot(s0_dir, s2)
    Cf = np.dot(k2 * s1 - k1 * s2, s0_dir)

    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3))

    # First rotation: about the horizontal axis perpendicular to ball0->ball1,
    # by the angle that makes ball1 sit (z0 - z1) below ball0.
    vector1 = np.broadcast_to(np.array([l1[0], l1[1], 0.0]), (n, 3))
    axis1 = np.array([-l1[1], l1[0], 0.0])
    axis1 = np.broadcast_to(axis1 / np.linalg.norm(axis1), (n, 3))
    angle1 = _angle_to_match_z(vector1, axis1, z0 - z1)

    # Second rotation: about ball0->ball1 once rotated, to match z0 - z2.
    vector2 = _rotate(np.broadcast_to(np.array([l2[0], l2[1], 0.0]), (n, 3)), axis1, angle1)
    vector1 = _rotate(vector1, axis1, angle1)
    axis2 = vector1 / np.linalg.norm(vector1, axis=-1, keepdims=True)
    angle2 = _angle_to_match_z(vector2, axis2, z0 - z2)

    # Third rotation: about +Z, by the angle the slot constraints fix.
    l1p = _rotate(vector1, axis2, angle2)[:, :2]
    l2p = _rotate(vector2, axis2, angle2)[:, :2]
    l1p_rot = np.stack([-l1p[:, 1], l1p[:, 0]], axis=-1)
    l2p_rot = np.stack([-l2p[:, 1], l2p[:, 0]], axis=-1)
    l1_n1 = l1p @ s1
    l1_rot_n1 = l1p_rot @ s1
    A = s0d_s1n * (l2p @ s2) - s0d_s2n * l1_n1
    B = s0d_s1n * (l2p_rot @ s2) - s0d_s2n * l1_rot_n1
    theta = _solve_weierstrass(A, B, np.full(n, Cf))

    normal = _rotate(_rotate(_rotate(up, axis1, angle1), axis2, angle2), up, theta)
    # forward mirrors x and y back to the build direction.
    return normal * np.array([-1.0, -1.0, 1.0])
