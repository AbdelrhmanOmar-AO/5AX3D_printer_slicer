"""Tests for the nozzle-vs-printed-material check (build plan P4.3).

Synthetic cases with a known answer, a brute-force comparison that exercises
the KD-tree and block logic, and the golden cube: printed as Atomizer planned
it, nothing is ever inside the nozzle; printed in reverse (top down), almost
everything is. The second is the check that the first zero means something.
"""

from __future__ import annotations  # no Taichi kernels in this test module

import math

import numpy as np
import pytest

from atom import clearance, machine_profile
from atom import nozzle_material_check as nm
from atom import overhang_metrics as om
from atom import toolpath_view as tv

SETTINGS = nm.NozzleCheckSettings(height_mm=70.0)


def tilted(tilt_deg, azimuth_deg=180.0):
    """Unit build direction tilted from +Z toward ``azimuth`` (180 = toward -x)."""
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    return np.array([math.cos(a) * math.sin(t), math.sin(a) * math.sin(t), math.cos(t)])


def stem_and_arm(stem_top, arm_z=5.0, arm_tilt=0.0, gap=3.0):
    """A vertical stem at x = 0, printed first, then an arm beside it.

    The stem is one bead straight up from z = 0.5 to ``stem_top`` in 0.5 mm
    steps, tool vertical. Then a travel to ``(gap, 0, arm_z)``, and the arm is
    printed outward to x = gap + 7 with the tool tilted ``arm_tilt`` degrees
    toward the stem. Returns ``(points, directions, deposit, first arm index)``.
    """
    stem_z = np.arange(0.5, stem_top + 1e-9, 0.5)
    stem = np.column_stack([np.zeros_like(stem_z), np.zeros_like(stem_z), stem_z])
    arm_x = np.arange(gap, gap + 7.0 + 1e-9, 0.5)
    arm = np.column_stack([arm_x, np.zeros_like(arm_x), np.full_like(arm_x, arm_z)])

    points = np.vstack([stem, arm])
    directions = np.vstack([np.tile([0.0, 0.0, 1.0], (len(stem), 1)),
                            np.tile(tilted(arm_tilt), (len(arm), 1))])
    deposit = np.ones(len(points), dtype=bool)
    deposit[0] = False           # the first point: no move to it
    deposit[len(stem)] = False   # travel from the stem top to the arm
    return points, directions, deposit, len(stem)


# --------------------------------------------------------------------------
# Timing of material
# --------------------------------------------------------------------------


def test_material_time_follows_the_moves():
    # moves:          -     print  print  travel travel print  travel
    deposit = np.array([True, True, True, False, False, True, False])
    time = nm.material_time(deposit)
    never = np.iinfo(np.int64).max
    # 0 starts the first bead (move 1); 4 starts the bead of move 5.
    np.testing.assert_array_equal(time, [1, 1, 2, never, 5, 5, never])


def test_cone_slabs_cover_the_cone():
    """Random points inside the cone all fall in at least one sphere."""
    rng = np.random.default_rng(0)
    for half_angle in (30.0, 40.0, 50.0):
        centres, radii = nm.cone_slabs(70.0, half_angle)
        axial = rng.uniform(0, 70, 5000)
        radial = axial * math.tan(math.radians(half_angle)) * np.sqrt(rng.uniform(0, 1, 5000))
        inside = np.hypot(axial[:, None] - centres[None], radial[:, None]) <= radii[None]
        assert inside.any(axis=1).all(), half_angle


def test_later_spheres_stay_above_the_tip_for_the_real_nozzle():
    """What makes the check fast: past the first slab, no sphere reaches the
    layer being printed on."""
    centres, radii = nm.cone_slabs(70.0, clearance.NOZZLE_HALF_ANGLE_DEG)
    assert np.all(centres[1:] - radii[1:] > 0)
    assert centres[-1] + radii[-1] >= 70.0


# --------------------------------------------------------------------------
# Known answers
# --------------------------------------------------------------------------


def test_arm_beside_a_taller_stem_is_flagged():
    points, directions, deposit, first_arm = stem_and_arm(stem_top=10.0)
    result = nm.check(points, directions, deposit, SETTINGS)

    assert not result.ok
    # The travel to the arm's start is already inside: stem material 3 mm away
    # and up to 5 mm above the tip, within 40 degrees of vertical.
    assert result.index[0] == first_arm
    assert not result.deposit[0]
    assert result.deposit[1:].all()
    # The deepest blocker is stem material, and the report is self-consistent.
    assert np.all(result.blocker < first_arm)
    assert np.all(points[result.blocker, 2] > 5.0)
    assert np.all(result.depth_mm > 0)


def test_the_flagged_depth_matches_a_hand_calculation():
    """At the arm's start (3, 0, 5), vertical tool: the stem point (0, 0, 10)
    is 5 mm up and 3 mm out. Depth inside a 40-degree cone:
    (5 tan 40 - 3) cos 40 from the side, versus 65 mm to the cap."""
    points, directions, deposit, first_arm = stem_and_arm(stem_top=10.0)
    result = nm.check(points, directions, deposit, SETTINGS)
    expected = (5.0 * math.tan(math.radians(40)) - 3.0) * math.cos(math.radians(40))
    # The apex sits 0.001 mm up the axis.
    assert result.depth_mm[0] == pytest.approx(expected - 0.001 * math.sin(math.radians(40)),
                                               abs=1e-9)
    assert result.axial_mm[0] == pytest.approx(5.0 - 0.001)


def test_arm_beside_a_shorter_stem_is_clear():
    points, directions, deposit, _ = stem_and_arm(stem_top=4.5)
    assert nm.check(points, directions, deposit, SETTINGS).ok


def test_tilting_toward_the_stem_collides_past_fifty_degrees():
    """With the stem level with the arm, a vertical nozzle is clear. Tilting
    toward the stem swings its side down toward the arm's level: past
    90 - 40 = 50 degrees the stem top is inside. This is the 50-degree cap on
    tilt the nozzle cone implies."""
    for tilt, collides in [(0.0, False), (30.0, False), (45.0, False), (55.0, True)]:
        points, directions, deposit, first_arm = stem_and_arm(stem_top=5.0, arm_tilt=tilt)
        result = nm.check(points, directions, deposit, SETTINGS)
        assert result.ok != collides, tilt
        if collides:
            assert result.index[0] == first_arm
            assert result.tilt_deg[0] == pytest.approx(55.0)


def test_tilting_away_from_the_stem_is_clear():
    points, directions, deposit, _ = stem_and_arm(stem_top=5.0, arm_tilt=0.0)
    directions[directions[:, 2] < 1] = tilted(55.0, azimuth_deg=0.0)
    directions[len(directions) - 15:] = tilted(55.0, azimuth_deg=0.0)
    assert nm.check(points, directions, deposit, SETTINGS).ok


def layer_stack(tilt_deg=0.0, layers=6, side=4.0, step=0.5, height=0.4):
    """Flat square layers printed bottom-up, every point with one tool tilt."""
    grid = np.arange(0.0, side + 1e-9, step)
    rows = []
    for layer in range(layers):
        z = height * (layer + 1)
        for row, y in enumerate(grid):
            xs = grid if row % 2 == 0 else grid[::-1]
            rows.append(np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)]))
    points = np.vstack(rows)
    directions = np.tile(tilted(tilt_deg, azimuth_deg=0.0), (len(points), 1))
    deposit = np.ones(len(points), dtype=bool)
    deposit[0] = False
    return points, directions, deposit


def test_a_plain_layer_stack_never_collides():
    """Neither the layer below nor earlier beads in the same layer are inside
    the nozzle, and the layers above do not count: they come later."""
    assert nm.check(*layer_stack(), SETTINGS).ok


def test_flat_layers_collide_once_the_tilt_passes_fifty_degrees():
    """Earlier beads of the same flat layer, on the side the tool leans
    toward, enter the cone once the tilt exceeds 90 - 40 degrees."""
    assert nm.check(*layer_stack(tilt_deg=45.0), SETTINGS).ok
    result = nm.check(*layer_stack(tilt_deg=55.0), SETTINGS)
    assert not result.ok
    assert np.all(result.tilt_deg == pytest.approx(55.0))


def test_the_same_stack_printed_top_down_collides():
    points, directions, deposit = layer_stack()
    reverse_deposit = np.zeros(len(points), bool)
    reverse_deposit[1:] = deposit[::-1][:-1]
    result = nm.check(points[::-1], directions[::-1], reverse_deposit, SETTINGS)
    # Everything after the first (top) layer has material above it.
    per_layer = len(points) // 6
    assert result.count >= len(points) - per_layer - 1


def test_material_beyond_the_gantry_level_is_not_the_nozzles():
    """The cone stops at nozzle_to_gantry; above that is the swept check's."""
    points, directions, deposit, first_arm = stem_and_arm(stem_top=10.0)
    short = nm.NozzleCheckSettings(height_mm=4.0)
    result = nm.check(points, directions, deposit, short)
    # With a 4 mm cone, only stem points up to z = 9 can reach it.
    assert np.all(points[result.blocker, 2] <= 5.0 + 4.0)


def test_tolerance_ignores_shallow_contact():
    points, directions, deposit, _ = stem_and_arm(stem_top=10.0)
    exact = nm.check(points, directions, deposit, SETTINGS)
    loose = nm.check(points, directions, deposit,
                     nm.NozzleCheckSettings(height_mm=70.0, tolerance_mm=0.5))
    assert 0 < loose.count < exact.count
    assert loose.depth_mm.min() >= 0.5
    with pytest.raises(ValueError, match="tolerance_mm"):
        nm.check(points, directions, deposit,
                 nm.NozzleCheckSettings(height_mm=70.0, tolerance_mm=-1.0))


# --------------------------------------------------------------------------
# Against brute force
# --------------------------------------------------------------------------


def brute_force(points, directions, deposit, settings):
    """Every earlier material point against every nozzle position."""
    time = nm.material_time(deposit)
    apex = points + settings.apex_offset_mm * directions
    deepest = {}
    for i in range(len(points)):
        k = np.flatnonzero(time <= i - 1)
        if not len(k):
            continue
        radial, axial = nm.cone_coordinates(apex[[i] * len(k)], directions[[i] * len(k)],
                                            points[k])
        distance = clearance.cone_signed_distance_axial(
            radial, axial, settings.half_angle_deg, settings.height_mm)
        if np.any(distance < -settings.tolerance_mm):
            deepest[i] = (float(-distance.min()), int(np.count_nonzero(distance < 0)))
    return deepest


@pytest.mark.parametrize("block_size", [7, 64, 1000])
def test_fast_check_equals_brute_force(block_size):
    rng = np.random.default_rng(11)
    count = 400
    # Points scattered through a box in random order: plenty of collisions and
    # plenty of clear positions, and the box is deeper than the 6 mm cone.
    points = rng.uniform(0, 5, size=(count, 3)) * [1.0, 1.0, 1.6]
    directions = om.spherical_to_cartesian(np.column_stack([
        np.radians(rng.uniform(0, 60, count)), np.radians(rng.uniform(0, 360, count))]))
    deposit = rng.uniform(size=count) < 0.7
    settings = nm.NozzleCheckSettings(height_mm=6.0, block_size=block_size)

    expected = brute_force(points, directions, deposit, settings)
    result = nm.check(points, directions, deposit, settings)

    assert len(expected) > 20
    assert sorted(expected) == result.index.tolist()
    np.testing.assert_allclose(result.depth_mm, [expected[i][0] for i in result.index])
    np.testing.assert_array_equal(result.blocker_count, [expected[i][1] for i in result.index])


def test_buried_nozzles_are_detected_exactly_but_summarised_from_the_first_slab():
    """Forcing the buried path everywhere (limit 10): the colliding
    positions are exactly the exhaustive ones; only depth and count are
    partial, and never deeper than the truth."""
    rng = np.random.default_rng(11)
    count = 400
    points = rng.uniform(0, 5, size=(count, 3)) * [1.0, 1.0, 1.6]
    directions = om.spherical_to_cartesian(np.column_stack([
        np.radians(rng.uniform(0, 60, count)), np.radians(rng.uniform(0, 360, count))]))
    deposit = rng.uniform(size=count) < 0.7
    exact = nm.check(points, directions, deposit,
                     nm.NozzleCheckSettings(height_mm=6.0, block_size=64, exhaustive_limit=None))
    fast = nm.check(points, directions, deposit,
                    nm.NozzleCheckSettings(height_mm=6.0, block_size=64, exhaustive_limit=10))

    np.testing.assert_array_equal(fast.index, exact.index)
    assert not exact.partial.any()
    assert fast.partial.sum() > 100
    assert np.all(fast.depth_mm <= exact.depth_mm + 1e-12)
    assert np.all(fast.blocker_count <= exact.blocker_count)
    assert fast.to_dict()["collisions_summarised_from_the_first_slab"] == fast.partial.sum()


def test_subsampling_only_reports_real_collisions():
    """Thinned material keeps the earliest point of each voxel, so everything
    it finds is also found without thinning."""
    points, directions, deposit = layer_stack(tilt_deg=55.0)
    exact = nm.check(points, directions, deposit, SETTINGS)
    thinned = nm.check(points, directions, deposit,
                       nm.NozzleCheckSettings(height_mm=70.0, subsample_mm=1.0))
    assert 0 < thinned.count <= exact.count
    assert set(thinned.index.tolist()) <= set(exact.index.tolist())


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_settings_come_from_the_profile():
    reference = machine_profile.load_profile("reference")
    settings = nm.NozzleCheckSettings.for_profile(reference, subsample_mm=0.5)
    assert settings.height_mm == 70.0
    assert settings.half_angle_deg == 40.0
    assert settings.apex_offset_mm == 0.001
    assert settings.subsample_mm == 0.5


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        nm.check(np.zeros((3, 3)), np.zeros((2, 3)), np.ones(3, bool), SETTINGS)


def test_an_empty_toolpath_is_clear():
    result = nm.check(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, bool), SETTINGS)
    assert result.ok and result.checked == 0


def test_report_lists_the_deepest_first():
    points, directions, deposit, _ = stem_and_arm(stem_top=10.0)
    report = nm.check(points, directions, deposit, SETTINGS).to_dict(limit=3)
    assert report["check"] == "nozzle_vs_material"
    assert report["ok"] is False
    assert report["collisions"] == report["collisions_while_printing"] + \
        report["collisions_while_travelling"]
    depths = [entry["depth_mm"] for entry in report["deepest"]]
    assert depths == sorted(depths, reverse=True) and len(depths) == 3
    assert report["max_depth_mm"] == pytest.approx(depths[0], abs=1e-4)
    assert report["settings"]["height_mm"] == 70.0


# --------------------------------------------------------------------------
# The golden cube
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden(repo_root):
    return tv.load_toolpath_arrays(repo_root / "tests/golden/calibration_cube.toolpath.npz")


def test_golden_cube_is_clear(golden):
    """Atomizer's own plan at 5.5 degrees of tilt: nothing inside the nozzle."""
    from types import SimpleNamespace

    reference = machine_profile.load_profile("reference")
    result = nm.check_toolpath(SimpleNamespace(**golden), reference)
    assert result.checked == 46773
    assert result.ok


def test_golden_cube_top_printed_top_down_collides(golden):
    """The zero above means something: the same material in the opposite
    order puts earlier material above the nozzle almost everywhere."""
    count = 3000
    points = golden["point"][-count:].astype(float)[::-1]
    directions = om.spherical_to_cartesian(golden["tool_orientation"][-count:])[::-1]
    forward_deposit = golden["travel_type"][-count:] == om.TRAVEL_TYPE_DEPOSITION
    deposit = np.zeros(count, bool)
    deposit[1:] = forward_deposit[::-1][:-1]

    result = nm.check(points, directions, deposit, SETTINGS)
    assert result.count > count // 3
