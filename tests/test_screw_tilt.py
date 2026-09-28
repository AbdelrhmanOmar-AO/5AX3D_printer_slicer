"""Tests for the bed tilt implied by three screw heights (build plan P1.4).

`atom.screw_tilt` is the numpy route the G-code validator uses instead of the
Taichi `forward`. The first tests need no Taichi: they pin the arithmetic
against the worked example in `docs/conventions.md`. The last ones check the
port against Atomizer's own kernels on random states.

No `from __future__ import annotations` here; the kinematic tests drive the
Taichi kernels in `atom.kinematics3z`.
"""

import math

import numpy as np
import pytest

from atom import machine_profile, screw_tilt

#: docs/conventions.md section 6: position (150, 145, 10), build direction
#: (sin 10, 0, cos 10), reference machine.
TEN_DEGREES_TOWARD_X = (110.9078, 57.2505, 84.0791)


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))


def test_equal_screws_mean_no_tilt(reference):
    assert screw_tilt.total_tilt_deg(85.0, 85.0, 85.0, reference)[0] == 0.0
    np.testing.assert_allclose(
        screw_tilt.build_direction(85.0, 85.0, 85.0, reference)[0], [0.0, 0.0, 1.0], atol=1e-12
    )


def test_worked_example_reads_ten_degrees(reference):
    """The screws of the conventions document's 10-degree example."""
    tilt = screw_tilt.total_tilt_deg(*TEN_DEGREES_TOWARD_X, reference)[0]
    assert tilt == pytest.approx(10.0, abs=1e-3)

    direction = screw_tilt.build_direction(*TEN_DEGREES_TOWARD_X, reference)[0]
    expected = [math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))]
    assert angle_deg(direction, expected) < 1e-3


def test_only_the_screw_differences_matter(reference):
    z = np.array(TEN_DEGREES_TOWARD_X)
    for lift in (-40.0, 0.0, 100.0):
        assert screw_tilt.total_tilt_deg(*(z + lift), reference)[0] == pytest.approx(
            screw_tilt.total_tilt_deg(*z, reference)[0], abs=1e-9
        )


def test_the_two_routes_agree(reference):
    """Closed form and the port of forward give the same total tilt."""
    rng = np.random.default_rng(7)
    z0 = rng.uniform(60.0, 200.0, 500)
    z1 = z0 + rng.uniform(-120.0, 120.0, 500)
    z2 = z0 + rng.uniform(-120.0, 120.0, 500)
    closed = screw_tilt.total_tilt_deg(z0, z1, z2, reference)
    ported = np.degrees(np.arccos(screw_tilt.build_direction(z0, z1, z2, reference)[:, 2]))
    reachable = np.isfinite(closed)
    assert reachable.sum() > 400
    np.testing.assert_allclose(ported[reachable], closed[reachable], atol=1e-7)


def test_a_plane_through_fixed_ball_positions_under_reads_the_tilt(reference):
    """Why the validator does not fit a plane through the balls' XY positions.

    The plan suggested that. The contact points slide in their slots, so the
    horizontal spacing shrinks by cos(tilt) and a fixed-XY fit reads
    arctan(sin t) instead of t: 26.57 degrees at a true 30
    (docs/plan_corrections.md 7a, P1-1).
    """
    l1 = np.subtract(reference.ball_2dpos_1, reference.ball_2dpos_0)
    rise = math.sin(math.radians(30.0)) * np.linalg.norm(l1)
    z0, z1 = 150.0, 150.0 - rise
    # ball2 sits halfway along x, so it drops by half as much for a tilt toward +X.
    l2 = np.subtract(reference.ball_2dpos_2, reference.ball_2dpos_0)
    z2 = 150.0 - rise * l2[0] / l1[0]

    assert screw_tilt.total_tilt_deg(z0, z1, z2, reference)[0] == pytest.approx(30.0, abs=1e-9)
    fixed_xy_fit = math.degrees(math.atan(rise / np.linalg.norm(l1)))
    assert fixed_xy_fit == pytest.approx(26.565, abs=1e-3)


def test_unrealisable_screw_differences_are_nan(reference):
    """A rise longer than the ball spacing: no rigid bed can do it."""
    assert np.isnan(screw_tilt.total_tilt_deg(400.0, 0.0, 200.0, reference)[0])


# --------------------------------------------------------------------------
# Against Atomizer's own kinematics
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def random_states(ti_cpu):
    """Requested build directions up to 30 degrees, solved by the Taichi IK."""
    import atom.kinematics3z as k

    rng = np.random.default_rng(3)
    n = 200
    theta = np.radians(rng.uniform(0.0, 30.0, n))
    phi = rng.uniform(-np.pi, np.pi, n)
    points = np.stack(
        [rng.uniform(120.0, 180.0, n), rng.uniform(120.0, 170.0, n), rng.uniform(1.0, 40.0, n)], 1
    ).astype(np.float32)
    spherical = np.stack([theta, phi], 1).astype(np.float32)
    machine = np.zeros((n, 5), np.float32)
    k.toolpath_from_cartesian_toolpath(points, spherical, machine)

    requested = np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], 1
    )
    return k, machine, requested


def test_port_recovers_the_direction_the_ik_was_given(reference, random_states):
    _, machine, requested = random_states
    direction = screw_tilt.build_direction(machine[:, 2], machine[:, 3], machine[:, 4], reference)
    assert angle_deg(direction, requested).max() < 1e-3

    total = screw_tilt.total_tilt_deg(machine[:, 2], machine[:, 3], machine[:, 4], reference)
    np.testing.assert_allclose(total, np.degrees(np.arccos(requested[:, 2])), atol=1e-3)


def test_port_matches_the_taichi_forward(reference, random_states):
    """Within float32 rounding: Taichi's forward runs in float32, the port in float64."""
    k, machine, _ = random_states
    points = np.zeros((machine.shape[0], 3), np.float32)
    spherical = np.zeros((machine.shape[0], 2), np.float32)
    k.toolpath_to_cartesian_toolpath(machine, points, spherical)
    theta, phi = spherical[:, 0], spherical[:, 1]
    forward = np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], 1
    )

    direction = screw_tilt.build_direction(machine[:, 2], machine[:, 3], machine[:, 4], reference)
    assert angle_deg(direction, forward).max() < 0.05
