"""Kinematics test suite for the triple-Z bed (build plan task P1.1).

Proves `atom.kinematics3z` correct by testing it, without changing its maths:
round trips in both directions, zero tilt, the failure and clearance signals,
the bed's physical pose, and the direction the screws move as the tilt grows.
Reference machine profile throughout.

Precision, stated per backend
-----------------------------
The pipeline runs these kernels in **32-bit floats** on whichever backend each
stage uses. On the CPU backend (`ti_cpu`, as here and in CI) the measured
round-trip errors are:

=====================================  ===========  ===========  =============
Round trip                             plan         measured     asserted here
=====================================  ===========  ===========  =============
FK -> IK, screws and X/Y               1e-3 mm      1.2e-4 mm    1e-3 mm
IK -> FK, position                     1e-3 mm      7.8e-5 mm    1e-3 mm
IK -> FK, build direction              1e-4 rad     4.5e-4 rad   1e-3 rad
=====================================  ===========  ===========  =============

The direction misses the plan's 1e-4 rad **only because of 32-bit rounding**.
The same kernels in 64-bit floats return it to 3e-16 rad and positions to
1e-13 mm (`tests/kinematics_f64_roundtrip.py`, run in its own process below).
4.5e-4 rad is 0.026 degrees: 40 times below the 1-degree tessellation step,
and about 0.05 mm at the nozzle for a point 100 mm from the pivot. The operator
chose to test both precisions (2026-09-23, ``docs/plan_corrections.md`` 7a,
P1-8).

A forced backend (``ATOM_TI_ARCH``) changes the pipeline's output structurally,
not just in its last digits (plan_corrections 3.9, hazard 18). These
tolerances are stated for the CPU backend. CUDA's float32 maths rounds
differently and has not been measured here.

What the IK->FK round trip cannot see
-------------------------------------
``inverse`` turns the requested build direction into the bed normal by
flipping x and y, and ``forward`` flips them back, so the two agree with each
other **by construction**. The bed also turns slightly about its own normal,
so the direction the machine *physically* produces differs from the requested
one: by up to 0.14 degrees at 30 degrees of tilt (plan_corrections 1.9).
`test_the_physical_bed_pose_realises_the_build_direction` measures that
through `atom.bed_motion`.

No `from __future__ import annotations`: this module defines Taichi kernels.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atom import bed_motion, machine_profile

#: Tolerances for the 32-bit CPU kernels (module docstring).
SCREW_TOLERANCE_MM = 1e-3
POSITION_TOLERANCE_MM = 1e-3
DIRECTION_TOLERANCE_RAD_F32 = 1e-3
#: The plan's tolerance, which holds in 64-bit floats.
DIRECTION_TOLERANCE_RAD_PLAN = 1e-4


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


@pytest.fixture(scope="module")
def kernels(ti_cpu):
    """``(inverse, forward, k)``: numpy wrappers around the Taichi functions."""
    ti = ti_cpu
    import atom.kinematics3z as k

    @ti.kernel
    def solve_inverse(point: ti.types.ndarray(), direction: ti.types.ndarray(), out: ti.types.ndarray()):
        for i in range(point.shape[0]):
            x, y, z0, z1, z2, offset = k.inverse(
                ti.math.vec3(point[i, 0], point[i, 1], point[i, 2]),
                ti.math.vec3(direction[i, 0], direction[i, 1], direction[i, 2]),
            )
            out[i, 0] = x
            out[i, 1] = y
            out[i, 2] = z0
            out[i, 3] = z1
            out[i, 4] = z2
            out[i, 5] = offset

    @ti.kernel
    def solve_forward(machine: ti.types.ndarray(), position: ti.types.ndarray(), normal: ti.types.ndarray()):
        for i in range(machine.shape[0]):
            p, n = k.forward(machine[i, 0], machine[i, 1], machine[i, 2], machine[i, 3], machine[i, 4])
            for j in ti.static(range(3)):
                position[i, j] = p[j]
                normal[i, j] = n[j]

    def inverse(points, directions):
        """``(N, 6)``: x, y, z0, z1, z2, offset."""
        points = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float32)
        directions = np.ascontiguousarray(np.atleast_2d(directions), dtype=np.float32)
        out = np.zeros((len(points), 6), np.float32)
        solve_inverse(points, directions, out)
        return out.astype(np.float64)

    def forward(machine):
        """``(positions, normals)``, each ``(N, 3)``."""
        machine = np.ascontiguousarray(np.atleast_2d(machine), dtype=np.float32)
        position = np.zeros((len(machine), 3), np.float32)
        normal = np.zeros((len(machine), 3), np.float32)
        solve_forward(machine, position, normal)
        return position.astype(np.float64), normal.astype(np.float64)

    return inverse, forward, k


def directions(tilt_deg, azimuth_deg):
    """Build directions from spherical ``[theta, phi]`` in degrees, as the
    toolpath stores them (hazard 5): theta from +Z, phi from +X."""
    theta = np.radians(np.atleast_1d(tilt_deg).astype(float))
    phi = np.radians(np.atleast_1d(azimuth_deg).astype(float))
    return np.column_stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])


def angle_rad(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


# --------------------------------------------------------------------------
# 1. FK -> IK
# --------------------------------------------------------------------------


def test_fk_then_ik_recovers_the_screws(kernels, reference):
    """200 seeded machine states inside the limits whose forward result is
    reachable: inverse gives back X, Y and all three screws."""
    inverse, forward, _ = kernels
    rng = np.random.default_rng(11)
    count = 600
    base = rng.uniform(60.0, 220.0, count)
    machine = np.column_stack([
        rng.uniform(0.0, reference.max_x_axis, count),
        rng.uniform(0.0, reference.max_y_axis, count),
        base,
        base + rng.uniform(-120.0, 120.0, count),
        base + rng.uniform(-120.0, 120.0, count),
    ])
    position, normal = forward(machine)
    solved = inverse(position, normal)

    valid = np.flatnonzero(np.isfinite(solved[:, 5]) & np.all(np.isfinite(position), axis=1))
    assert len(valid) >= 200, "too few reachable states to sample"
    chosen = valid[:200]

    error = np.abs(solved[chosen, :5] - machine[chosen].astype(np.float32))
    assert error[:, 2:].max() < SCREW_TOLERANCE_MM
    assert error[:, :2].max() < SCREW_TOLERANCE_MM


# --------------------------------------------------------------------------
# 2. IK -> FK
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ik_fk_round_trip(kernels, reference):
    inverse, forward, _ = kernels
    rng = np.random.default_rng(12)
    count = 400
    tilt = rng.uniform(0.0, reference.max_tilt_angle_deg - 1.0, count)
    requested = directions(tilt, rng.uniform(-180.0, 180.0, count))
    points = np.column_stack(
        [rng.uniform(100.0, 200.0, count), rng.uniform(100.0, 190.0, count), rng.uniform(0.0, 60.0, count)]
    )
    solved = inverse(points, requested)
    position, normal = forward(solved[:, :5])
    return points, requested, solved, position, normal


def test_ik_then_fk_recovers_the_position(ik_fk_round_trip):
    points, _, solved, position, _ = ik_fk_round_trip
    assert np.isfinite(solved[:, 5]).all(), "every tilt below the limit is reachable here"
    assert np.linalg.norm(position - points, axis=1).max() < POSITION_TOLERANCE_MM


def test_ik_then_fk_recovers_the_direction_to_32_bit_precision(ik_fk_round_trip):
    """Measured 4.5e-4 rad (0.026 degrees) on the CPU backend; see the module docstring."""
    _, requested, _, _, normal = ik_fk_round_trip
    assert angle_rad(normal, requested).max() < DIRECTION_TOLERANCE_RAD_F32


def test_the_kinematics_are_exact_in_64_bit_floats(repo_root):
    """The plan's tolerances, and far tighter, hold once rounding is removed.

    Runs `tests/kinematics_f64_roundtrip.py` in its own process, because the
    float precision is fixed when Taichi starts.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root / "src"), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("kinematics_f64_roundtrip.py"))],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    assert measured["valid"] == measured["count"]
    assert measured["ik_fk_normal_rad"] < DIRECTION_TOLERANCE_RAD_PLAN
    assert measured["ik_fk_position_mm"] < POSITION_TOLERANCE_MM
    assert measured["fk_ik_screw_mm"] < SCREW_TOLERANCE_MM
    # Exact, not merely within tolerance: only float64 rounding remains.
    assert measured["ik_fk_normal_rad"] < 1e-9
    assert measured["ik_fk_position_mm"] < 1e-9
    assert measured["fk_ik_screw_mm"] < 1e-9


# --------------------------------------------------------------------------
# 3. Zero tilt
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", [(150.0, 145.0, 10.0), (60.0, 220.0, 0.0), (250.0, 40.0, 120.0)])
def test_zero_tilt_puts_all_screws_at_one_height(kernels, reference, point):
    inverse, _, _ = kernels
    solved = inverse([point], directions(0.0, 0.0))[0]
    expected = point[2] + reference.z_offset
    np.testing.assert_allclose(solved[2:5], [expected] * 3, atol=1e-4)
    assert solved[5] == 0.0


# --------------------------------------------------------------------------
# 4. Limits: NaN means unreachable, a finite offset means clearance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, point, tilt",
    [
        ("tilt beyond the limit", (150.0, 145.0, 10.0), "limit+2"),
        ("X beyond its travel", (400.0, 145.0, 10.0), 0.0),
        ("nozzle below the bed", (150.0, 145.0, -1.0), 0.0),
    ],
)
def test_unreachable_points_return_nan(kernels, reference, case, point, tilt):
    inverse, _, _ = kernels
    if tilt == "limit+2":
        tilt = reference.max_tilt_angle_deg + 2.0
    offset = inverse([point], directions(tilt, 0.0))[0, 5]
    assert math.isnan(offset), case


def test_a_reachable_point_needing_clearance_is_not_a_failure(kernels):
    """A finite positive offset is how far to raise the part, not an error
    (docs/conventions.md section 5). 20 degrees of tilt close to the bed lifts
    a bed corner above the gantry level."""
    inverse, _, _ = kernels
    solved = inverse([(150.0, 145.0, 0.5)], directions(20.0, 30.0))[0]
    assert np.isfinite(solved[5])
    assert solved[5] > 0.0
    assert solved[2:5].min() > 0.0, "this case is the gantry clearance, not the endstop"


def test_the_clearance_offset_is_a_first_estimate_that_converges(kernels):
    """The offset is the lift to try next, not the exact lift needed.

    Raising the point moves the tilted bed's corners by less than the lift, so
    one step under-shoots: here 6.10 mm first, 6.49 mm in total. That is why
    `kinematics3z.get_plaftorm_size` repeats the solve until the offset is
    zero (docs/plan_corrections.md 7a, P1-9). This pins that it converges.
    """
    inverse, _, _ = kernels
    height, first, steps = 0.5, None, 0
    while steps < 10:
        offset = inverse([(150.0, 145.0, height)], directions(20.0, 30.0))[0, 5]
        assert np.isfinite(offset)
        if offset == 0.0:
            break
        first = offset if first is None else first
        height += offset
        steps += 1
    assert offset == 0.0, "the platform-sizing loop would not terminate"
    assert 2 <= steps <= 8
    assert height - 0.5 > first, "one step of the first offset is not enough"


# --------------------------------------------------------------------------
# 5. The physical bed pose
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tilt_deg, bound_deg",
    [
        # Measured on the CPU backend (max over 24 azimuths): 0.0009, 0.018,
        # 0.083 and 0.143 degrees. plan_corrections 1.9 records 0.0003, 0.08
        # and 0.14 at 5.5, 25 and 30 degrees.
        (5.5, 0.005),
        (15.0, 0.03),
        (25.0, 0.1),
        (30.0, 0.2),
    ],
)
def test_the_physical_bed_pose_realises_the_build_direction(kernels, tilt_deg, bound_deg):
    """R @ d close to +Z: the tilted bed really points the requested build
    direction at the nozzle, to within a gap that grows with the tilt.

    The pose comes from three `forward` evaluations per state
    (`atom.bed_motion`), so it is the vendored kinematics' own answer.
    """
    inverse, forward, _ = kernels
    azimuths = np.arange(0.0, 360.0, 15.0)
    requested = directions(np.full(len(azimuths), tilt_deg), azimuths)
    solved = inverse(np.tile([150.0, 145.0, 10.0], (len(azimuths), 1)), requested)
    assert np.isfinite(solved[:, 5]).all()

    machine = solved[:, :5]
    probed, _ = forward(bed_motion.probe_states(machine))
    rotation, _ = bed_motion.poses_from_probes(machine, probed)
    nozzle_axis = np.einsum("nij,nj->ni", rotation, requested)

    gap = np.degrees(angle_rad(nozzle_axis, [0.0, 0.0, 1.0]))
    assert gap.max() < bound_deg


# --------------------------------------------------------------------------
# 6. Which way the screws move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("azimuth_deg", [0, 45, 90, 135, 180, 225, 270, 315])
def test_screws_move_monotonically_as_the_tilt_grows(kernels, reference, azimuth_deg):
    """Tilting toward an azimuth raises ball 0 relative to ball j when the
    tilt direction points along ball0 -> ball j, and lowers it when it points
    away. The difference grows steadily with the tilt: no reversal, no jump.

    A ball perpendicular to the tilt direction stays level.
    """
    inverse, _, _ = kernels
    tilts = np.arange(0.0, reference.max_tilt_angle_deg, 1.0)
    requested = directions(tilts, np.full(len(tilts), float(azimuth_deg)))
    solved = inverse(np.tile([150.0, 145.0, 10.0], (len(tilts), 1)), requested)
    assert np.isfinite(solved[:, 5]).all()

    ball0 = np.asarray(reference.ball_2dpos_0)
    horizontal = requested[-1, :2] / np.linalg.norm(requested[-1, :2])
    for j, ball in ((1, reference.ball_2dpos_1), (2, reference.ball_2dpos_2)):
        rise = solved[:, 2] - solved[:, 2 + j]  # z0 - zj
        along = float(np.dot(horizontal, np.asarray(ball) - ball0))
        if abs(along) < 1.0:
            np.testing.assert_allclose(rise, 0.0, atol=1e-3)
            continue
        steps = np.diff(rise)
        assert (np.sign(steps) == np.sign(along)).all(), (azimuth_deg, j)
        # And by about the amount the geometry says: sin(tilt) times the
        # distance along the tilt direction. The bed also turns slightly about
        # its own normal (plan_corrections 1.9), which moves this by up to
        # 1.4 % at the diagonal azimuths (measured), so 3 % is allowed.
        np.testing.assert_allclose(rise[-1], math.sin(math.radians(tilts[-1])) * along, rtol=0.03)
