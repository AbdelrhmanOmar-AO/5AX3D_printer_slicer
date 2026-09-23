"""Tests for the swept check (build plan P4.2).

The synthetic cases are the plan's: a short part under a 20-degree sweep is
clear; a part as tall as ``nozzle_to_gantry - 2`` mm under a 30-degree sweep
reaches the gantry at the sweep; a travel with no lift through a part hits it.
The expected numbers are worked by hand on the reference machine (gantry
70 mm above the tip, nozzle cone 40 degrees).

No `from __future__ import annotations` here; these tests drive the Taichi
kernels in `atom.kinematics3z`.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from atom import contracts, machine_profile
from atom import nozzle_material_check as nm
from atom import tilt_motion_check as tmc
from atom import toolpath_view as tv


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


def toolpath(points, tilts_deg, azimuths_deg, deposit):
    points = np.asarray(points, dtype=float)
    count = len(points)
    return SimpleNamespace(
        point=points.astype(np.float32),
        travel_type=np.where(deposit, 0, 1).astype(np.int32),
        tool_orientation=np.radians(
            np.column_stack([np.broadcast_to(tilts_deg, count),
                             np.broadcast_to(azimuths_deg, count)])).astype(np.float32),
        width=np.full(count, 0.9), height=np.full(count, 0.45), point_count=count,
    )


def column(x, y, top, step=0.5):
    z = np.arange(0.5, top + 1e-9, step)
    return np.column_stack([np.full_like(z, x), np.full_like(z, y), z])


def sweep_case(part_top, tilt, azimuth, tower=None):
    """Print a column at the origin bottom-up (after an optional tower), then
    re-orient the tool in place at its top, from vertical to ``tilt`` toward
    ``azimuth``: one travel move that only turns. Returns the toolpath and
    the index of the sweep move."""
    parts = [column(0.0, 0.0, part_top)]
    if tower is not None:
        parts.insert(0, tower)
    points = np.vstack(parts)
    points = np.vstack([points, points[-1]])
    count = len(points)
    tilts = np.zeros(count)
    tilts[-1] = tilt
    deposit = np.ones(count, dtype=bool)
    deposit[0] = False
    deposit[-1] = False
    if tower is not None:
        deposit[len(tower)] = False
    return toolpath(points, tilts, azimuth, deposit), count - 1


def block(side=10.0, top=5.0, step=0.5):
    """A solid block printed in flat, back-and-forth layers."""
    grid = np.arange(0.0, side + 1e-9, step)
    rows = []
    for z in np.arange(0.5, top + 1e-9, 0.5):
        for row, y in enumerate(grid):
            xs = grid if row % 2 == 0 else grid[::-1]
            rows.append(np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)]))
    return np.vstack(rows)


def travel_case(travel_z):
    """Print a 10 x 10 x 5 mm block, lift and step aside, drop to
    ``travel_z`` beside it, then travel straight across it at that height.
    Returns the toolpath and the index of the crossing move."""
    solid = block()
    path = np.vstack([solid, [[-10, 5, 6.0], [-10, 5, travel_z], [20, 5, travel_z]]])
    deposit = np.zeros(len(path), dtype=bool)
    deposit[1:len(solid)] = True
    return toolpath(path, 0.0, 0.0, deposit), len(path) - 1


def kinds(result, move=None):
    return {(row["kind"], row["body"]) for row in result.violations
            if move is None or row["move"] == move}


# --------------------------------------------------------------------------
# States (no kinematics)
# --------------------------------------------------------------------------


def test_moves_are_split_by_turn_and_by_travel_distance():
    settings = tmc.SweptCheckSettings()
    up = np.array([0.0, 0.0, 1.0])
    turned = np.array([math.sin(math.radians(2.2)), 0.0, math.cos(math.radians(2.2))])
    points = np.array([[0, 0, 1], [0.4, 0, 1], [0.8, 0, 1], [3.8, 0, 1], [3.9, 0, 1], [9, 0, 1.0]])
    directions = np.array([up, up, turned, turned, turned, turned])
    deposit = np.array([False, True, True, False, False, True])
    valid = np.array([True, True, True, True, True, False])

    steps = tmc.interior_counts(points, directions, deposit, valid, settings)

    # 0.4 mm printed move, no turn: not split. 2.2-degree turn: 5 steps.
    # 3 mm travel: 6 steps. 0.1 mm travel: not split. Unreachable end: skipped.
    np.testing.assert_array_equal(steps, [0, 0, 5, 6, 0, 0])


def test_interior_states_are_linear_in_the_axes():
    machine = np.array([[0, 0, 10, 10, 10], [10, 20, 16, 4, 10], [10, 20, 16, 4, 10.0]])
    states = tmc.build_states(machine, np.array([True, True, True]), np.array([0, 4, 0]))

    np.testing.assert_array_equal(states.move, [0, 1, 1, 1, 1, 2])
    np.testing.assert_allclose(states.fraction, [1, 0.25, 0.5, 0.75, 1, 1])
    np.testing.assert_allclose(states.machine[2], [5, 10, 13, 7, 10])
    assert states.interior.tolist() == [False, True, True, True, False, False]


def test_the_hull_proxy_keeps_the_highest_point_under_any_tilt():
    """Against the flat gantry, hull vertices plus the recent points find
    exactly the part's highest point, whatever the bed's orientation."""
    rng = np.random.default_rng(5)
    points = rng.uniform(-20, 20, size=(3000, 3))
    time = np.arange(3000)
    proxy = tmc._PartProxy(points, time, hull_every=400)
    for _ in range(20):
        visible = int(rng.integers(1, 3000))
        epoch = visible // proxy.every
        subset = np.concatenate([proxy.hull(epoch), proxy.order[epoch * proxy.every:visible]])
        up = rng.normal(size=3)
        up /= np.linalg.norm(up)
        assert (points[subset] @ up).max() == pytest.approx((points[:visible] @ up).max())
        assert len(subset) < visible or visible < 400


# --------------------------------------------------------------------------
# Against the kinematics
# --------------------------------------------------------------------------


def test_states_at_the_points_are_the_points(ti_cpu, reference):
    """Forward kinematics of each point's own state returns the point."""
    tp, _ = sweep_case(5.0, 20.0, 0.0)
    solved = contracts.from_toolpath(tp, reference, center_on_bed=True)
    _, _, tip = tmc.poses(solved.machine)
    np.testing.assert_allclose(tip, solved.point, atol=2e-3)


def test_a_short_part_under_a_20_degree_sweep_is_clear(ti_cpu, reference):
    tp, sweep = sweep_case(5.0, 20.0, 0.0)
    result = tmc.check_toolpath(tp, reference)

    assert result.ok, result.violations
    assert result.interpolated_moves == 1
    assert result.interior_states == 39  # 20 degrees in 0.5-degree steps
    # The 5 mm column's top edge rises by at most ~0.1 mm at 20 degrees.
    assert result.min_part_clearance_mm > 69.0


def test_a_diagonal_sweep_near_the_bed_lifts_a_corner_into_the_gantry(ti_cpu, reference):
    """The reference bed itself: 20 degrees toward a diagonal with the
    nozzle 5 mm up puts a bed corner into the gantry. `inverse` agrees: it
    asks for exactly that much lift at the sweep's end point."""
    tp, sweep = sweep_case(5.0, 20.0, 45.0)
    result = tmc.check_toolpath(tp, reference)
    bed = [row for row in result.violations if row["kind"] == "bed"]

    assert [(row["move"], row["body"], row["fraction"]) for row in bed] == \
        [(sweep, "gantry", 1.0)]
    lift = contracts.from_toolpath(tp, reference, center_on_bed=True).lift_mm[sweep]
    assert lift > 1.0
    assert -bed[0]["clearance_mm"] == pytest.approx(lift, abs=5e-3)


def test_a_part_as_tall_as_the_gantry_reaches_it_during_a_30_degree_sweep(ti_cpu, reference):
    """A tower 68 mm tall (nozzle_to_gantry - 2), 80 mm from where the nozzle
    works at the top of a 30 mm column. Upright, its top is 38 mm above the
    tip: clear. Tilted 30 degrees toward it, the top rises to
    38 cos 30 + 80 sin 30 = 72.91 mm, 2.91 mm into the gantry."""
    tp, sweep = sweep_case(30.0, 30.0, 0.0, tower=column(80.0, 0.0, 68.0))
    result = tmc.check_toolpath(tp, reference)
    part = [row for row in result.violations if row["kind"] == "part"]

    assert [(row["move"], row["body"]) for row in part] == [(sweep, "gantry")]
    expected = 70.0 - (38.0 * math.cos(math.radians(30)) + 80.0 * math.sin(math.radians(30)))
    assert part[0]["clearance_mm"] == pytest.approx(expected, abs=0.05)
    # The report names the offending point: the tower's top, in the bed frame.
    assert part[0]["point_bed"][2] == pytest.approx(68.0)
    assert part[0]["tilt_deg"] == pytest.approx(30.0, abs=0.2)
    assert result.to_dict()["worst"]


def test_a_shorter_tower_stays_below_the_gantry(ti_cpu, reference):
    """60 mm: 30 cos 30 + 80 sin 30 = 65.98 mm, 4.02 mm below the gantry."""
    tp, _ = sweep_case(30.0, 30.0, 0.0, tower=column(80.0, 0.0, 60.0))
    result = tmc.check_toolpath(tp, reference)
    assert ("part", "gantry") not in kinds(result)
    assert result.min_part_clearance_mm == pytest.approx(
        70.0 - (30.0 * math.cos(math.radians(30)) + 40.0), abs=0.05)


def test_a_travel_without_lift_through_a_part_is_caught(ti_cpu, reference):
    """Both ends of the travel are clear, so no check at the points can see
    it (P4.3 finds nothing). Between them the nozzle passes through the
    block, with its top face 4 mm above the tip: (4 - 0.001) sin 40 deep."""
    tp, crossing = travel_case(1.0)
    result = tmc.check_toolpath(tp, reference)

    assert kinds(result) == {("nozzle_vs_material", "nozzle")}
    (row,) = result.violations
    assert row["move"] == crossing and not row["deposit"]
    assert 0 < row["fraction"] < 1
    assert row["clearance_mm"] == pytest.approx(-(4.0 - 0.001) * math.sin(math.radians(40)),
                                                abs=1e-3)
    assert nm.check_toolpath(tp, reference).ok


def test_the_same_travel_above_the_part_is_clear(ti_cpu, reference):
    tp, _ = travel_case(5.5)
    assert tmc.check_toolpath(tp, reference).ok


def test_moves_to_unreachable_points_are_skipped_and_counted(ti_cpu, reference):
    """35 degrees exceeds the reference machine's 30: `inverse` returns NaN."""
    tp, _ = sweep_case(5.0, 35.0, 0.0)
    result = tmc.check_toolpath(tp, reference)
    assert result.skipped_moves == 1
    assert result.interpolated_moves == 0
    assert result.states == tp.point_count - 1


def test_report_says_what_is_not_checked_yet(ti_cpu, reference):
    tp, _ = sweep_case(5.0, 20.0, 0.0)
    report = tmc.check_toolpath(tp, reference).to_dict()
    assert report["check"] == "swept" and report["ok"] is True
    assert set(report["not_checked"]) == {"axis_range", "tilt_limit"}
    assert report["clearance_model"]["model"] == "reference"
    assert report["settings"]["check_tilt_step_deg"] == 0.5


def test_golden_cube_has_no_swept_violations(ti_cpu, reference, repo_root):
    """The plan's exit check for P4.2, on the committed golden toolpath (the
    `_platform` one the G-code was written from). It runs on the CPU in a
    few seconds, so it needs no laptop."""
    arrays = tv.load_toolpath_arrays(repo_root / "tests/golden/calibration_cube.toolpath.npz")
    result = tmc.check_toolpath(SimpleNamespace(**arrays), reference)

    assert result.ok, result.to_dict(limit=5)
    assert result.points == 46773 and result.skipped_moves == 0
    assert result.interior_states > 1000
    # The part never comes within 68 mm of the gantry, nor the bed within 69.
    assert result.min_part_clearance_mm > 68.0
    assert result.min_bed_clearance_mm > 69.0
