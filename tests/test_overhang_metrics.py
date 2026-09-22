"""Tests for the overhang metrics (build plan task P0.8).

Every case here is synthetic: hand-built toolpaths and face normals with known
answers, no Atomizer stage involved. These metrics decide whether the 5-axis
contribution worked, so they are checked against arithmetic rather than against
whatever the pipeline happens to produce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atom import overhang_metrics as om

TRAVEL_TYPE_DEPOSITION = om.TRAVEL_TYPE_DEPOSITION
TRAVEL_TYPE_NO_DEPOSITION = 1


class FakeToolpath:
    """The parts of `toolpath3.Toolpath` these metrics read.

    Tool orientations are given as Cartesian vectors for readability and
    converted to the spherical ``(N, 2)`` form the real class stores.
    """

    def __init__(self, points, directions, heights=0.45, travel_type=None):
        self.point = np.asarray(points, dtype=np.float32)
        count = len(self.point)

        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim == 1:
            directions = np.tile(directions, (count, 1))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.tool_orientation = np.column_stack(
            [
                np.arccos(np.clip(directions[:, 2], -1.0, 1.0)),
                np.arctan2(directions[:, 1], directions[:, 0]),
            ]
        ).astype(np.float32)

        self.height = np.full(count, heights, dtype=np.float32) if np.isscalar(
            heights
        ) else np.asarray(heights, dtype=np.float32)
        self.width = np.full(count, 0.9, dtype=np.float32)
        self.travel_type = (
            np.full(count, TRAVEL_TYPE_DEPOSITION, dtype=np.int32)
            if travel_type is None
            else np.asarray(travel_type, dtype=np.int32)
        )
        self.point_count = count


def direction_tilted_toward(tilt_deg: float, azimuth_deg: float = 0.0):
    """A build direction tilted `tilt_deg` from +Z, toward `azimuth_deg` in xy."""
    tilt, azimuth = math.radians(tilt_deg), math.radians(azimuth_deg)
    return np.array(
        [math.cos(azimuth) * math.sin(tilt), math.sin(azimuth) * math.sin(tilt), math.cos(tilt)]
    )


def normal_of_overhang(angle_deg: float, azimuth_deg: float = 0.0):
    """Outward normal of a surface overhanging at `angle_deg` from vertical.

    A face rising at ``angle_deg`` from +Z has downward normal
    ``(cos A, 0, -sin A)``, rotated to the requested azimuth.
    """
    angle, azimuth = math.radians(angle_deg), math.radians(azimuth_deg)
    return np.array(
        [math.cos(azimuth) * math.cos(angle), math.sin(azimuth) * math.cos(angle), -math.sin(angle)]
    )


# --------------------------------------------------------------------------
# Effective overhang angle
# --------------------------------------------------------------------------


def test_vertical_tool_over_a_60_degree_face_measures_60():
    angle = om.effective_overhang_angle_deg(
        normal_of_overhang(60.0), np.array([0.0, 0.0, 1.0])
    )
    assert angle[0] == pytest.approx(60.0)


def test_tilting_20_degrees_toward_the_overhang_lowers_it_to_40():
    """This is the whole mechanism: tilt buys overhang angle one-for-one."""
    angle = om.effective_overhang_angle_deg(
        normal_of_overhang(60.0), direction_tilted_toward(20.0, azimuth_deg=0.0)
    )
    assert angle[0] == pytest.approx(40.0)


def test_tilting_20_degrees_away_from_the_overhang_worsens_it_to_80():
    angle = om.effective_overhang_angle_deg(
        normal_of_overhang(60.0), direction_tilted_toward(20.0, azimuth_deg=180.0)
    )
    assert angle[0] == pytest.approx(80.0)


def test_flat_ceiling_with_a_vertical_tool_measures_90():
    angle = om.effective_overhang_angle_deg(
        np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    )
    assert angle[0] == pytest.approx(90.0)


def test_an_upward_face_is_not_an_overhang():
    angle = om.effective_overhang_angle_deg(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    )
    assert angle[0] <= 0.0


def test_a_vertical_wall_measures_zero():
    angle = om.effective_overhang_angle_deg(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    assert angle[0] == pytest.approx(0.0)


def test_geometric_angle_equals_effective_angle_for_a_planar_print():
    normals = np.array([normal_of_overhang(a) for a in (30.0, 45.0, 60.0, 90.0)])
    vertical = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))

    np.testing.assert_allclose(
        om.geometric_overhang_angle_deg(normals),
        om.effective_overhang_angle_deg(normals, vertical),
    )


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 180.0, 270.0])
def test_the_relationship_holds_at_any_azimuth(azimuth):
    """Tilting toward an overhang works regardless of which way it faces."""
    angle = om.effective_overhang_angle_deg(
        normal_of_overhang(70.0, azimuth_deg=azimuth),
        direction_tilted_toward(25.0, azimuth_deg=azimuth),
    )
    assert angle[0] == pytest.approx(45.0)


# --------------------------------------------------------------------------
# Spherical conversion, against the convention in atom.direction
# --------------------------------------------------------------------------


def test_spherical_conversion_matches_the_atomizer_convention():
    """theta is the polar angle from +Z, phi the azimuth from +X."""
    cartesian = om.spherical_to_cartesian(
        np.array([[0.0, 0.0], [math.pi / 2, 0.0], [math.pi / 2, math.pi / 2]])
    )
    np.testing.assert_allclose(cartesian[0], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(cartesian[1], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cartesian[2], [0.0, 1.0, 0.0], atol=1e-12)


def test_spherical_conversion_returns_unit_vectors():
    rng = np.random.default_rng(0)
    spherical = np.column_stack(
        [rng.uniform(0, math.pi, 100), rng.uniform(-math.pi, math.pi, 100)]
    )
    lengths = np.linalg.norm(om.spherical_to_cartesian(spherical), axis=1)
    np.testing.assert_allclose(lengths, 1.0)


# --------------------------------------------------------------------------
# Maximum tilt used
# --------------------------------------------------------------------------


def test_max_tool_tilt_of_a_planar_toolpath_is_zero():
    toolpath = FakeToolpath(
        points=[[0, 0, z] for z in (0.0, 1.0, 2.0)], directions=[0.0, 0.0, 1.0]
    )
    assert om.max_tool_tilt_deg(toolpath) == pytest.approx(0.0, abs=1e-6)


def test_max_tool_tilt_reports_the_steepest_direction_used():
    toolpath = FakeToolpath(
        points=[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
        directions=[
            direction_tilted_toward(0.0),
            direction_tilted_toward(12.0),
            direction_tilted_toward(7.0),
        ],
    )
    assert om.max_tool_tilt_deg(toolpath) == pytest.approx(12.0, abs=1e-4)


def test_max_tool_tilt_ignores_travel_moves():
    """Travel moves are not deposition; their orientation is not tilt 'used'."""
    toolpath = FakeToolpath(
        points=[[0, 0, 0], [0, 0, 1]],
        directions=[direction_tilted_toward(5.0), direction_tilted_toward(29.0)],
        travel_type=[TRAVEL_TYPE_DEPOSITION, TRAVEL_TYPE_NO_DEPOSITION],
    )
    assert om.max_tool_tilt_deg(toolpath) == pytest.approx(5.0, abs=1e-4)
    assert om.max_tool_tilt_deg(toolpath, deposition_only=False) == pytest.approx(
        29.0, abs=1e-4
    )


# --------------------------------------------------------------------------
# Unsupported deposition
# --------------------------------------------------------------------------


def test_a_straight_column_is_fully_supported():
    points = [[0.0, 0.0, 0.45 * i] for i in range(20)]
    result = om.unsupported_deposition(FakeToolpath(points, [0.0, 0.0, 1.0]))

    assert result.point_count == 20
    assert result.fraction == pytest.approx(0.0)
    assert not result.unsupported.any()


def test_a_point_floating_above_the_column_is_flagged():
    points = [[0.0, 0.0, 0.45 * i] for i in range(10)]
    points.append([0.0, 0.0, points[-1][2] + 5.0])
    result = om.unsupported_deposition(FakeToolpath(points, [0.0, 0.0, 1.0]))

    assert result.unsupported[-1], "the floating point should be unsupported"
    assert not result.unsupported[:-1].any()
    assert result.fraction == pytest.approx(1.0 / 11.0)


def test_points_on_the_bed_count_as_supported():
    """The bed holds up the first layer; nothing else needs to be beneath it."""
    points = [[float(i), 0.0, 0.45] for i in range(10)]
    result = om.unsupported_deposition(
        FakeToolpath(points, [0.0, 0.0, 1.0]), first_layer_height=0.45
    )

    assert not result.unsupported.any()
    assert result.fraction == pytest.approx(0.0)


def test_material_deposited_later_cannot_support_an_earlier_point():
    """Print order matters: a bridge printed before its pillar is unsupported."""
    points = [
        [0.0, 0.0, 5.0],   # printed first, in mid-air
        [0.0, 0.0, 4.6],   # printed after, directly beneath it
    ]
    result = om.unsupported_deposition(
        FakeToolpath(points, [0.0, 0.0, 1.0]), first_layer_height=0.45
    )

    assert result.unsupported[0], "the first point had nothing beneath it yet"


def test_a_point_outside_the_support_cone_is_unsupported():
    """Material to the side, beyond 65 degrees from straight down, cannot hold."""
    below = [0.0, 0.0, 0.45]
    # 80 degrees off vertical from the upper point, inside the search radius.
    offset = 0.6
    upper = [offset, 0.0, below[2] + offset / math.tan(math.radians(80.0))]

    result = om.unsupported_deposition(
        FakeToolpath([below, upper], [0.0, 0.0, 1.0], heights=1.0),
        first_layer_height=0.45,
    )
    assert result.unsupported[1]


def test_tilting_the_tool_changes_which_material_counts_as_support():
    """The support cone follows the build direction, not gravity."""
    lower = [0.0, 0.0, 2.0]
    upper = [1.0, 0.0, 2.3]          # up and to the +x side of `lower`

    # Vertical build direction: `lower` sits well off the cone axis.
    vertical = om.unsupported_deposition(
        FakeToolpath([lower, upper], [0.0, 0.0, 1.0], heights=1.0),
        first_layer_height=0.45,
    )
    # Tilted toward +x: the cone now leans over `lower`.
    tilted = om.unsupported_deposition(
        FakeToolpath([lower, upper], direction_tilted_toward(45.0), heights=1.0),
        first_layer_height=0.45,
    )

    assert vertical.unsupported[1]
    assert not tilted.unsupported[1]


def test_travel_moves_are_excluded_from_the_measurement():
    points = [[0.0, 0.0, 0.45], [0.0, 0.0, 0.9], [50.0, 50.0, 30.0]]
    result = om.unsupported_deposition(
        FakeToolpath(
            points,
            [0.0, 0.0, 1.0],
            travel_type=[TRAVEL_TYPE_DEPOSITION, TRAVEL_TYPE_DEPOSITION, TRAVEL_TYPE_NO_DEPOSITION],
        ),
        first_layer_height=0.45,
    )
    assert result.point_count == 2


def test_an_empty_toolpath_reports_nothing_rather_than_dividing_by_zero():
    result = om.unsupported_deposition(FakeToolpath([], np.zeros((0, 3))))
    assert result.point_count == 0
    assert result.fraction == 0.0


# --------------------------------------------------------------------------
# Per-surface effective angles
# --------------------------------------------------------------------------


def _single_face(angle_deg, centre=(0.0, 0.0, 10.0), area=4.0):
    return (
        np.array([normal_of_overhang(angle_deg)]),
        np.array([centre], dtype=float),
        np.array([area]),
    )


def test_a_face_printed_with_a_vertical_tool_keeps_its_geometric_angle():
    normals, centres, areas = _single_face(60.0)
    toolpath = FakeToolpath(
        [[0.0, 0.0, 10.0], [0.2, 0.0, 10.0]], [0.0, 0.0, 1.0]
    )

    groups = om.effective_overhang_angles(normals, centres, areas, toolpath, 0.9)

    assert len(groups) == 1
    assert groups[0].geometric_angle_deg == pytest.approx(60.0, abs=0.1)
    assert groups[0].max_effective_deg == pytest.approx(60.0, abs=1e-3)
    assert groups[0].max_tilt_used_deg == pytest.approx(0.0, abs=1e-3)
    assert groups[0].area_mm2 == pytest.approx(4.0)
    assert groups[0].measured


def test_a_face_printed_with_a_tilted_tool_reports_the_improvement():
    normals, centres, areas = _single_face(60.0)
    toolpath = FakeToolpath([[0.0, 0.0, 10.0]], direction_tilted_toward(20.0))

    groups = om.effective_overhang_angles(normals, centres, areas, toolpath, 0.9)

    assert groups[0].max_effective_deg == pytest.approx(40.0, abs=1e-3)
    assert groups[0].max_tilt_used_deg == pytest.approx(20.0, abs=1e-3)


def test_faces_are_grouped_by_geometric_angle():
    normals = np.array([normal_of_overhang(a) for a in (60.0, 60.0, 90.0)])
    centres = np.array([[0.0, 0.0, 10.0], [0.1, 0.0, 10.0], [5.0, 0.0, 10.0]])
    areas = np.array([1.0, 2.0, 4.0])
    toolpath = FakeToolpath([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]], [0.0, 0.0, 1.0])

    groups = om.effective_overhang_angles(normals, centres, areas, toolpath, 0.9)

    assert [g.geometric_angle_deg for g in groups] == [90.0, 60.0]
    assert groups[0].area_mm2 == pytest.approx(4.0)
    assert groups[1].area_mm2 == pytest.approx(3.0)
    assert groups[1].face_count == 2


def test_upward_and_bed_contact_faces_are_not_reported():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    centres = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])  # second is on the bed
    areas = np.array([1.0, 1.0])
    toolpath = FakeToolpath([[0.0, 0.0, 10.0]], [0.0, 0.0, 1.0])

    assert om.effective_overhang_angles(normals, centres, areas, toolpath, 0.9) == []


def test_a_face_with_no_deposition_nearby_is_reported_as_unmeasured():
    """Silence is not success: an unmeasured surface must not look like 0 degrees."""
    normals, centres, areas = _single_face(60.0)
    toolpath = FakeToolpath([[100.0, 100.0, 10.0]], [0.0, 0.0, 1.0])

    groups = om.effective_overhang_angles(normals, centres, areas, toolpath, 0.9)

    assert len(groups) == 1
    assert groups[0].sample_count == 0
    assert not groups[0].measured
    assert math.isnan(groups[0].max_effective_deg)


# --------------------------------------------------------------------------
# The bed-contact rule
#
# Atomizer's layers are conformal, so the first layer is a shell of finite
# thickness, not a plane. A threshold of one layer height cuts through the
# middle of it and reports the upper half as printing into air. These pin down
# the fix: anchor to the lowest deposition point.
# --------------------------------------------------------------------------


def test_a_thick_first_layer_is_all_treated_as_resting_on_the_bed():
    """Deposition spread through the first layer must not be called unsupported."""
    # A shell of points spanning most of a layer height, as a conformal first
    # layer does, with nothing above.
    points = [[float(i) * 0.4, 0.0, 0.42 + 0.01 * i] for i in range(12)]
    result = om.unsupported_deposition(FakeToolpath(points, [0.0, 0.0, 1.0]))

    assert result.fraction == pytest.approx(0.0), (
        "the whole first layer sits on the bed; none of it prints into air"
    )


def test_the_bed_threshold_follows_the_part_not_the_origin():
    """A part lifted off z=0 still has a first layer resting on its support."""
    lifted = [[float(i) * 0.4, 0.0, 5.0 + 0.01 * i] for i in range(12)]
    result = om.unsupported_deposition(FakeToolpath(lifted, [0.0, 0.0, 1.0]))

    assert result.fraction == pytest.approx(0.0)


def test_an_explicit_threshold_still_overrides_the_default():
    points = [[0.0, 0.0, 0.45], [0.0, 0.0, 6.0]]
    result = om.unsupported_deposition(
        FakeToolpath(points, [0.0, 0.0, 1.0]), first_layer_height=0.45
    )
    assert result.unsupported[1], "the floating point is still unsupported"


def test_material_above_the_first_layer_is_judged_normally():
    """Fixing the bed rule must not make everything look supported."""
    points = [[0.0, 0.0, 0.45], [0.0, 0.0, 0.9], [0.0, 0.0, 30.0]]
    result = om.unsupported_deposition(FakeToolpath(points, [0.0, 0.0, 1.0]))

    assert not result.unsupported[0]
    assert not result.unsupported[1]
    assert result.unsupported[2], "a point 29 mm above anything must be flagged"
