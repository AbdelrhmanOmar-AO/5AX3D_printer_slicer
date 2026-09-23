"""Where the tilted bed is for a given machine state (build plan P5.4b).

The toolpath viewer's machine view draws the printer the way it moves: the
nozzle stays vertical and the bed, carrying the part, tilts and slides beneath
it. This module turns machine states ``(X, Y, Z, U, V)`` into the bed's rigid
pose. numpy only; the forward kinematics are run by the caller.

Frames
------
Bed frame
    The frame `kinematics3z.inverse` and `forward` work in: the (bed-centred)
    part frame, bed surface at z = 0, ball joints at ``(ball_2dpos_i,
    ball_z)``. Toolpath points re-centred with
    `contracts.bed_centering_offset` are in this frame.
World frame
    Fixed to the machine, nozzle tip at ``(X, Y, 0)``: the frame
    `kinematics3z.inverse` uses for its bed-gantry check, where the gantry sits
    ``nozzle_to_gantry`` above the nozzle tip.

How the pose is found
---------------------
With the screws held, `forward` moves only the nozzle when X or Y change: the
bed's rotation depends on Z, U and V alone. So evaluating `forward` at
``(X, Y)``, ``(X + h, Y)`` and ``(X, Y + h)`` gives the bed-frame images of the
world X and Y axes, and hence the whole rotation, using the vendored kinematics
rather than a re-implementation of them. On the golden cube the resulting pose
maps every build direction to +Z within 0.0003 degrees, and on synthetic
states up to 30 degrees of tilt it reproduces the screw height differences at
the three ball joints within 0.001 mm.

It also shows that the tool direction the machine actually realises differs
slightly from the one requested, by up to about 0.14 degrees at 30 degrees of
tilt (`docs/plan_corrections.md` 1.9). `realised_tool_direction` returns the
former.

Units: millimetres; rotations as 3 x 3 matrices.
"""

from __future__ import annotations  # no Taichi kernels in this module

import numpy as np

#: Distance, mm, between the X and Y probes. The forward kinematics run in
#: float32, so a longer baseline keeps rounding out of the rotation.
PROBE_STEP_MM = 10.0


def ball_positions(profile) -> np.ndarray:
    """The three ball joints in the bed frame, ``(3, 3)`` mm."""
    return np.array(
        [
            [*profile.ball_2dpos_0, profile.ball_z],
            [*profile.ball_2dpos_1, profile.ball_z],
            [*profile.ball_2dpos_2, profile.ball_z],
        ],
        dtype=np.float64,
    )


def bed_corners(profile) -> np.ndarray:
    """The bed's four corners in the bed frame, ``(4, 3)`` mm, at z = 0.

    Exactly the corners `kinematics3z.inverse` checks against the gantry:
    ``ball_to_corner`` outside the ball joints. Order: (-x, -y), (+x, -y),
    (+x, +y), (-x, +y), so they form a closed outline.
    """
    b0, b1, b2 = (np.asarray(b, dtype=np.float64) for b in (
        profile.ball_2dpos_0, profile.ball_2dpos_1, profile.ball_2dpos_2))
    c = float(profile.ball_to_corner)
    return np.array(
        [
            [b0[0] - c, b0[1] - c, 0.0],
            [b1[0] + c, b1[1] - c, 0.0],
            [b1[0] + c, b2[1] + c, 0.0],
            [b0[0] - c, b2[1] + c, 0.0],
        ]
    )


def probe_states(machine: np.ndarray, step: float = PROBE_STEP_MM) -> np.ndarray:
    """``(3N, 5)`` machine states to run through `forward`: as given, X + step, Y + step."""
    machine = np.asarray(machine, dtype=np.float64)
    shift_x = np.zeros(5)
    shift_x[0] = step
    shift_y = np.zeros(5)
    shift_y[1] = step
    return np.vstack([machine, machine + shift_x, machine + shift_y])


def poses_from_probes(machine, probed_points, step: float = PROBE_STEP_MM):
    """Bed poses from the `forward` results of `probe_states`.

    Returns ``(R, t)``, shapes ``(N, 3, 3)`` and ``(N, 3)``, such that a
    bed-frame point ``p`` is at ``R @ p + t`` in the world frame. The rotation
    is made exactly orthonormal (nearest rotation by SVD) to remove float32
    rounding.
    """
    machine = np.asarray(machine, dtype=np.float64)
    count = len(machine)
    probed = np.asarray(probed_points, dtype=np.float64)
    p0, px, py = probed[:count], probed[count : 2 * count], probed[2 * count :]

    # Columns: the world X, Y and Z axes expressed in the bed frame.
    ex = (px - p0) / step
    ey = (py - p0) / step
    ez = np.cross(ex, ey)
    bed_from_world = np.stack([ex, ey, ez], axis=2)
    u, _, vt = np.linalg.svd(bed_from_world)
    bed_from_world = u @ vt
    rotation = np.transpose(bed_from_world, (0, 2, 1))

    nozzle = np.column_stack([machine[:, 0], machine[:, 1], np.zeros(count)])
    translation = nozzle - np.einsum("nij,nj->ni", rotation, p0)
    return rotation, translation


def pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """One pose as a 4 x 4 homogeneous matrix (what VTK's user matrix takes)."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def to_world(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Bed-frame points into the world frame under one pose."""
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def realised_tool_direction(rotation: np.ndarray) -> np.ndarray:
    """The nozzle axis (world +Z) in the bed frame: the build direction the
    machine actually produces. ``(N, 3)`` for ``(N, 3, 3)`` rotations."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return rotation[..., 2, :]


def corner_heights(rotation, translation, corners) -> np.ndarray:
    """World height of each bed corner above the nozzle tip, mm."""
    return to_world(corners, rotation, translation)[:, 2]


def last_valid_index(valid: np.ndarray, index: int) -> int | None:
    """The latest index at or before ``index`` whose machine state is valid."""
    valid = np.asarray(valid, dtype=bool)
    index = int(np.clip(index, 0, len(valid) - 1)) if len(valid) else -1
    candidates = np.flatnonzero(valid[: index + 1])
    return int(candidates[-1]) if len(candidates) else None
