"""Tests for the bed pose behind the viewer's machine view (build plan P5.4b).

The synthetic tests check the arithmetic. The kinematic ones check that a pose
recovered from Atomizer's own `forward` is physically right: it turns every
build direction into the vertical nozzle axis, and it puts the three ball
joints at the height differences the screws command.

No `from __future__ import annotations` here; the kinematic tests drive the
Taichi kernels in `atom.kinematics3z`.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import visualize_5ax as vt
from atom import bed_motion, contracts, machine_profile
from atom import overhang_metrics as om
from atom import toolpath_view as tv


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


def rotation_about(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    angle = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))


# --------------------------------------------------------------------------
# Geometry from the profile
# --------------------------------------------------------------------------


def test_bed_corners_are_the_ones_inverse_checks_against_the_gantry(reference):
    """kinematics3z.inverse: ball_to_corner (10 mm) outside the ball joints."""
    corners = bed_motion.bed_corners(reference)
    np.testing.assert_allclose(corners, [
        [-14.07, -22.16, 0.0],
        [314.93, -22.16, 0.0],
        [314.93, 306.84, 0.0],
        [-14.07, 306.84, 0.0],
    ])


def test_ball_joints_sit_at_ball_z(reference):
    balls = bed_motion.ball_positions(reference)
    assert balls.shape == (3, 3)
    np.testing.assert_allclose(balls[:, 2], reference.ball_z)
    np.testing.assert_allclose(balls[2, :2], reference.ball_2dpos_2)


# --------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------


def _probes_for(rotation, translation, machine, step):
    """What `forward` would return for a bed at a known pose."""
    world = [np.array([m[0], m[1], 0.0]) for m in machine]
    inverse = rotation.T
    p0 = np.array([inverse @ (w - translation) for w in world])
    px = np.array([inverse @ (w + [step, 0, 0] - translation) for w in world])
    py = np.array([inverse @ (w + [0, step, 0] - translation) for w in world])
    return np.vstack([p0, px, py])


def test_a_known_pose_is_recovered_from_its_probes():
    rotation = rotation_about([1, 2, 0], 17.0) @ rotation_about([0, 0, 1], 3.0)
    translation = np.array([12.0, -4.0, -80.0])
    machine = np.array([[150.0, 140.0, 90.0, 80.0, 70.0]])

    probed = _probes_for(rotation, translation, machine, bed_motion.PROBE_STEP_MM)
    found_r, found_t = bed_motion.poses_from_probes(machine, probed)

    np.testing.assert_allclose(found_r[0], rotation, atol=1e-12)
    np.testing.assert_allclose(found_t[0], translation, atol=1e-9)


def test_the_nozzle_point_maps_to_the_nozzle_tip():
    """The bed point under the nozzle is at (X, Y, 0) in the world frame."""
    rotation = rotation_about([0, 1, 0], 25.0)
    translation = np.array([3.0, 5.0, -60.0])
    machine = np.array([[100.0, 120.0, 0, 0, 0]])
    probed = _probes_for(rotation, translation, machine, bed_motion.PROBE_STEP_MM)
    found_r, found_t = bed_motion.poses_from_probes(machine, probed)

    world = bed_motion.to_world(probed[:1], found_r[0], found_t[0])
    np.testing.assert_allclose(world[0], [100.0, 120.0, 0.0], atol=1e-9)


def test_rounding_noise_is_removed_from_the_rotation():
    rotation = rotation_about([1, 0, 0], 10.0)
    machine = np.array([[100.0, 100.0, 0, 0, 0]])
    probed = _probes_for(rotation, np.zeros(3), machine, bed_motion.PROBE_STEP_MM)
    probed += np.random.default_rng(1).normal(scale=1e-5, size=probed.shape)

    found, _ = bed_motion.poses_from_probes(machine, probed)

    np.testing.assert_allclose(found[0] @ found[0].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(found[0]) == pytest.approx(1.0)


def test_pose_matrix_is_homogeneous():
    rotation = rotation_about([0, 0, 1], 90.0)
    matrix = bed_motion.pose_matrix(rotation, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)


def test_corner_heights_follow_the_tilt(reference):
    """Tilting about Y raises one side of the bed and lowers the other."""
    corners = bed_motion.bed_corners(reference)
    flat = bed_motion.corner_heights(np.eye(3), np.array([0, 0, -10.0]), corners)
    np.testing.assert_allclose(flat, -10.0)

    tilted = bed_motion.corner_heights(rotation_about([0, 1, 0], -10.0), np.zeros(3), corners)
    assert tilted[1] > 0 > tilted[0] or tilted[0] > 0 > tilted[1]


def test_last_valid_index_holds_the_last_reachable_pose():
    valid = np.array([False, True, True, False, False, True])
    assert bed_motion.last_valid_index(valid, 4) == 2
    assert bed_motion.last_valid_index(valid, 5) == 5
    assert bed_motion.last_valid_index(valid, 0) is None
    assert bed_motion.last_valid_index(valid, 99) == 5


# --------------------------------------------------------------------------
# Against the real kinematics
# --------------------------------------------------------------------------


def _solve(points, tilts_deg, azimuths_deg):
    count = len(points)
    toolpath = SimpleNamespace(
        point=np.asarray(points, np.float32),
        travel_type=np.zeros(count, np.int32),
        tool_orientation=np.column_stack(
            [np.radians(tilts_deg), np.radians(azimuths_deg)]).astype(np.float32),
        width=np.ones(count), height=np.ones(count), point_count=count,
    )
    result = contracts.from_toolpath(toolpath)
    assert result.valid.all()
    return result.machine


def _poses(machine):
    probed, _ = vt.machine_to_build_frame(bed_motion.probe_states(machine))
    return bed_motion.poses_from_probes(machine, probed)


def test_golden_cube_build_directions_all_point_up_the_nozzle(ti_cpu, repo_root):
    """The bed pose turns each point's build direction into world +Z."""
    with open(repo_root / "tests/fixtures/gcode/calibration_cube_head.gcode") as handle:
        moves = vt.read_gcode_moves(handle)
    golden = tv.load_toolpath_arrays(repo_root / "tests/golden/calibration_cube.toolpath.npz")
    rotation, _ = _poses(moves.machine)
    directions = om.spherical_to_cartesian(golden["tool_orientation"][: len(moves.line)])

    up = np.einsum("nij,nj->ni", rotation, directions)

    assert angle_deg(up, [0, 0, 1]).max() < 0.001


def test_ball_joints_rise_and_fall_by_what_the_screws_command(ti_cpu, reference):
    """Up to 30 degrees of tilt in eight directions, to 0.001 mm."""
    tilts = np.repeat([10.0, 20.0, 29.9], 8)
    azimuths = np.tile(np.arange(0, 360, 45.0), 3)
    points = np.tile([150.0, 145.0, 10.0], (len(tilts), 1))
    machine = _solve(points, tilts, azimuths)
    rotation, translation = _poses(machine)
    balls = bed_motion.ball_positions(reference)

    for index, state in enumerate(machine):
        heights = bed_motion.to_world(balls, rotation[index], translation[index])[:, 2]
        # A higher screw value lowers the bed at that ball joint.
        np.testing.assert_allclose(heights[1] - heights[0], state[2] - state[3], atol=1e-3)
        np.testing.assert_allclose(heights[2] - heights[0], state[2] - state[4], atol=1e-3)


def test_realised_tool_direction_stays_within_a_fifth_of_a_degree(ti_cpu):
    """plan_corrections 1.9: the kinematics realise the requested build
    direction only approximately, by up to about 0.14 degrees at 30 degrees of
    tilt (diagonal azimuths). This pins the size of that gap."""
    tilts = np.repeat([15.0, 29.9], 8)
    azimuths = np.tile(np.arange(0, 360, 45.0), 2)
    machine = _solve(np.tile([150.0, 145.0, 10.0], (len(tilts), 1)), tilts, azimuths)
    rotation, _ = _poses(machine)
    requested = om.spherical_to_cartesian(np.radians(np.column_stack([tilts, azimuths])))

    gap = angle_deg(bed_motion.realised_tool_direction(rotation), requested)

    assert gap.max() < 0.2
