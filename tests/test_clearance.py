"""Tests for the clearance model (build plan P4.1).

Distances are checked against hand calculations on the reference machine:
nozzle cone half-angle 40 degrees, gantry 70 mm above the nozzle tip. The last
test checks the model against the kinematics themselves: the gantry
half-space must be the same one `kinematics3z.inverse` tests the bed corners
against, so the lift the IK asks for must equal how far the model says the
highest corner reaches into the gantry.

No `from __future__ import annotations` here; the kinematic test drives the
Taichi kernels in `atom.kinematics3z`.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from atom import bed_motion, clearance, contracts, machine_profile

SIN40 = math.sin(math.radians(40.0))
COS40 = math.cos(math.radians(40.0))
TAN40 = math.tan(math.radians(40.0))


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


@pytest.fixture(scope="module")
def model(reference):
    return clearance.ReferenceClearance(reference)


def test_cone_angle_is_atomizers_own():
    """The repeated constant must match what order_atoms plans with."""
    from atom import toolpath3

    assert clearance.NOZZLE_HALF_ANGLE_DEG == pytest.approx(
        math.degrees(toolpath3.NOZZLE_CONE_ANGLE / 2.0)
    )


def test_reference_model_is_sized_from_the_profile(model, reference):
    assert model.gantry_height_mm == reference.nozzle_to_gantry == 70
    assert model.half_angle_deg == 40.0
    assert model.bodies == ("nozzle", "gantry")
    assert isinstance(model, clearance.ClearanceModel)


# --------------------------------------------------------------------------
# Reference model against hand calculations
# --------------------------------------------------------------------------

# (point in the head frame, nozzle distance, gantry distance), all mm.
HAND_CASES = [
    # The nozzle tip itself: touching, not inside.
    ((0.0, 0.0, 0.0), 0.0, 70.0),
    # Level with the tip, 10 mm out: perpendicular to the slanted side.
    ((10.0, 0.0, 0.0), 10.0 * COS40, 70.0),
    # On the axis 10 mm up: inside, 10 sin 40 from the side.
    ((0.0, 0.0, 10.0), -10.0 * SIN40, 60.0),
    # Just under the gantry on the axis: the top cap is nearest.
    ((0.0, 0.0, 69.0), -1.0, 1.0),
    # Behind the apex: the apex itself is nearest.
    ((0.0, 0.0, -5.0), 5.0, 75.0),
    ((3.0, 4.0, -12.0), 13.0, 82.0),
    # Exactly on the slanted side at 35 mm.
    ((35.0 * TAN40, 0.0, 35.0), 0.0, 35.0),
    # Above the cap on the axis: 5 mm from the cone, 5 mm into the gantry.
    ((0.0, 0.0, 75.0), 5.0, -5.0),
    # Past the rim (r = 70 tan 40 = 58.737 at z = 70): the rim circle is nearest.
    ((70.0, 0.0, 80.0), math.hypot(70.0 - 70.0 * TAN40, 10.0), -10.0),
]


@pytest.mark.parametrize("point, nozzle, gantry", HAND_CASES)
def test_body_distances_match_hand_calculations(model, point, nozzle, gantry):
    distances = model.body_distances([point])
    np.testing.assert_allclose(distances[0], [nozzle, gantry], atol=1e-9)


def test_distance_is_rotationally_symmetric_about_the_nozzle(model):
    angles = np.radians(np.arange(0, 360, 30))
    points = np.column_stack([12 * np.cos(angles), 12 * np.sin(angles), np.full(12, 3.0)])
    distances = model.signed_distance(points)
    np.testing.assert_allclose(distances, distances[0], atol=1e-12)


def test_the_head_position_shifts_the_whole_model(model):
    """World frame: the nozzle tip is at (X, Y, 0)."""
    head = (150.0, 120.0)
    world = np.array([[160.0, 120.0, 0.0], [150.0, 120.0, 10.0], [0.0, 0.0, 75.0]])
    local = world - np.array([*head, 0.0])
    np.testing.assert_allclose(model.body_distances(world, head), model.body_distances(local))
    np.testing.assert_allclose(model.signed_distance(world, head)[:2], [10 * COS40, -10 * SIN40])


def test_each_point_can_carry_its_own_head_position(model):
    """The swept check scores many machine states in one call."""
    heads = np.array([[150.0, 120.0], [10.0, 20.0], [0.0, 0.0]])
    points = np.array([[160.0, 120.0, 0.0], [10.0, 20.0, 10.0], [0.0, 0.0, 75.0]])
    np.testing.assert_allclose(model.signed_distance(points, heads),
                               [10 * COS40, -10 * SIN40, -5.0])
    with pytest.raises(ValueError, match="head_xy"):
        model.signed_distance(points, heads[:2])


def test_bodies_can_be_chosen_and_ordered(model, boxes):
    points = [(0, 0, 75), (0, 0, 10)]
    np.testing.assert_allclose(model.body_distances(points, bodies=["gantry"]),
                               model.body_distances(points)[:, [1]])
    np.testing.assert_allclose(model.body_distances(points, bodies=["gantry", "nozzle"]),
                               model.body_distances(points)[:, [1, 0]])
    np.testing.assert_allclose(boxes.body_distances(points, bodies=["beam"]),
                               boxes.body_distances(points)[:, [2]])
    with pytest.raises(ValueError, match="unknown bodies"):
        model.body_distances(points, bodies=["hotend"])


def test_signed_distance_is_the_nearest_body(model):
    distances, body = model.nearest_body([(0, 0, 10), (0, 0, 75), (70, 0, 80)])
    np.testing.assert_allclose(distances, [-10 * SIN40, -5.0, -10.0])
    assert [model.bodies[i] for i in body] == ["nozzle", "gantry", "gantry"]


def test_violations_count_only_points_strictly_inside(model):
    points = [
        (0.0, 0.0, 0.0),        # touching the tip
        (0.0, 0.0, 10.0),       # inside the cone
        (35 * TAN40, 0, 35.0),  # on the side
        (100.0, 0.0, 0.0),      # well clear
        (100.0, 0.0, 70.5),     # into the gantry
    ]
    np.testing.assert_array_equal(model.violations(points), [1, 4])
    # A margin also flags points nearer than it.
    np.testing.assert_array_equal(model.violations(points, margin_mm=0.5), [0, 1, 2, 4])


def test_min_clearance(model):
    assert model.min_clearance([(10, 0, 0), (0, 0, -5)]) == pytest.approx(5.0)
    assert model.min_clearance([(0, 0, 75)]) == pytest.approx(-5.0)
    assert model.min_clearance(np.zeros((0, 3))) == math.inf
    assert model.violations(np.zeros((0, 3))).size == 0


def test_axial_form_agrees_with_the_upright_cone():
    """P4.3 uses the axial form along tilted tool directions."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-80, 90, size=(500, 3))
    radial = np.hypot(points[:, 0], points[:, 1])
    np.testing.assert_allclose(
        clearance.cone_signed_distance_axial(radial, points[:, 2], 40.0, 70.0),
        clearance.cone_signed_distance(points, 40.0, 70.0),
    )


def test_cone_distance_is_exact_against_brute_force():
    """Compare with the distance to a dense sampling of the cone's surface."""
    height, tan_half = 20.0, math.tan(math.radians(40.0))
    # The meridian outline of the cone: slanted side then top cap.
    side = np.column_stack([np.linspace(0, height * tan_half, 4001),
                            np.linspace(0, height, 4001)])
    cap = np.column_stack([np.linspace(0, height * tan_half, 4001), np.full(4001, height)])
    outline = np.vstack([side, cap])

    rng = np.random.default_rng(7)
    radial = rng.uniform(0, 30, 300)
    axial = rng.uniform(-10, 30, 300)
    brute = np.min(np.hypot(radial[:, None] - outline[None, :, 0],
                            axial[:, None] - outline[None, :, 1]), axis=1)
    exact = clearance.cone_signed_distance_axial(radial, axial, 40.0, height)

    np.testing.assert_allclose(np.abs(exact), brute, atol=0.01)


def test_invalid_reference_parameters(reference):
    with pytest.raises(ValueError, match="half_angle"):
        clearance.ReferenceClearance(reference, half_angle_deg=90.0)
    with pytest.raises(ValueError, match="nozzle_to_gantry"):
        clearance.ReferenceClearance(SimpleNamespace(name="x", nozzle_to_gantry=0))


def test_describe_records_the_parameters(model):
    assert model.describe() == {
        "model": "reference",
        "profile": "reference",
        "nozzle_half_angle_deg": 40.0,
        "gantry_height_mm": 70.0,
    }


# --------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------

BOXES = [
    # A hotend block 20 mm above the tip, following the head.
    {"name": "hotend", "moves_with": ["x", "y"], "min": [-10, -10, 20], "max": [10, 10, 40]},
    # A frame member, fixed, in machine coordinates.
    {"name": "frame", "moves_with": [], "min": [0, 290, 50], "max": [300, 300, 60]},
    # A beam riding on Y: machine X, tip-relative Y.
    {"name": "beam", "moves_with": ["y"], "min": [0, -5, 80], "max": [300, 5, 90]},
]


@pytest.fixture
def boxes():
    return clearance.BoxesClearance(BOXES, name="test")


def test_box_distances_match_hand_calculations(boxes):
    distances = boxes.body_distances([(0, 0, 30), (0, 0, 0), (13, 14, 30)])
    # Centre of the hotend: 10 mm to its nearest faces.
    assert distances[0, 0] == pytest.approx(-10.0)
    # Below it: 20 mm to its bottom face.
    assert distances[1, 0] == pytest.approx(20.0)
    # Off a vertical edge by (3, 4): 5 mm.
    assert distances[2, 0] == pytest.approx(5.0)
    # The frame member from the origin: 290 - 0 in y and 50 - 0 in z.
    assert distances[1, 1] == pytest.approx(math.hypot(290.0, 50.0))


def test_boxes_follow_the_axes_they_are_attached_to(boxes):
    head = (100.0, 200.0)
    # Hotend moves in X and Y.
    assert boxes.body_distances([(113, 214, 30)], head)[0, 0] == pytest.approx(5.0)
    # The frame member does not move at all.
    np.testing.assert_allclose(boxes.body_distances([(150, 295, 55)], head)[0, 1], -5.0)
    # The beam follows Y only: centred on y = 200 now, still x 0..300.
    assert boxes.body_distances([(150, 200, 85)], head)[0, 2] == pytest.approx(-5.0)
    assert boxes.body_distances([(150, 200, 85)], (0.0, 0.0))[0, 2] == pytest.approx(195.0)


def test_boxes_take_one_head_position_per_point(boxes):
    heads = np.array([[100.0, 200.0], [0.0, 0.0]])
    points = np.array([[113.0, 214.0, 30.0], [150.0, 200.0, 85.0]])
    distances = boxes.body_distances(points, heads)
    assert distances[0, 0] == pytest.approx(5.0)       # hotend, moved with the head
    assert distances[1, 2] == pytest.approx(195.0)     # beam, head at y = 0
    np.testing.assert_allclose(distances[0], boxes.body_distances(points[:1], heads[0])[0])


def test_box_violations_name_the_body(boxes):
    distance, body = boxes.nearest_body([(0, 0, 30)])
    assert boxes.bodies[body[0]] == "hotend"
    np.testing.assert_array_equal(boxes.violations([(0, 0, 30), (0, 0, 0)]), [0])


@pytest.mark.parametrize("bad, message", [
    ([], "non-empty"),
    ([{"name": "a", "min": [0, 0, 0]}], "missing key 'max'"),
    ([{"name": "a", "min": [0, 0, 0], "max": [1, 1]}], "three numbers"),
    ([{"name": "a", "min": [0, 0, 1], "max": [1, 1, 1]}], "below its 'max'"),
    ([{"name": "a", "min": [0, 0, 0], "max": [1, 1, 1], "moves_with": ["z"]}], "moves_with"),
    ([{"name": "a", "min": [0, 0, 0], "max": [1, 1, 1]},
      {"name": "a", "min": [0, 0, 0], "max": [1, 1, 1]}], "duplicate"),
])
def test_bad_boxes_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        clearance.BoxesClearance(bad)


def _write(path, status="verified", name="reference"):
    path.write_text(json.dumps({"name": name, "status": status, "boxes": BOXES}))
    return path


def test_boxes_round_trip_through_json(tmp_path):
    model = clearance.BoxesClearance.from_json(_write(tmp_path / "m.clearance.json"))
    assert model.bodies == ("hotend", "frame", "beam")
    described = model.describe()
    assert described["model"] == "boxes"
    assert described["boxes"][2] == {"name": "beam", "min": [0, -5, 80], "max": [300, 5, 90],
                                     "moves_with": ["y"]}


def test_a_placeholder_clearance_file_warns(tmp_path):
    path = _write(tmp_path / "m.clearance.json", status="PLACEHOLDER")
    with pytest.warns(machine_profile.PlaceholderProfileWarning, match="gate M3"):
        clearance.BoxesClearance.from_json(path)


def test_a_clearance_file_needs_its_keys(tmp_path):
    path = tmp_path / "m.clearance.json"
    path.write_text(json.dumps({"name": "reference", "boxes": BOXES}))
    with pytest.raises(ValueError, match="status"):
        clearance.BoxesClearance.from_json(path)


# --------------------------------------------------------------------------
# Choosing a model
# --------------------------------------------------------------------------


def test_no_machine_has_a_clearance_file_yet(repo_root):
    """Gate M3: nobody has measured an envelope, so none may exist."""
    assert not list((repo_root / "config" / "machines").glob("*" + clearance.CLEARANCE_SUFFIX))


def test_without_a_file_the_reference_proxy_is_used(reference):
    model = clearance.load_clearance(reference)
    assert isinstance(model, clearance.ReferenceClearance)
    assert clearance.clearance_path(reference).name == "reference.clearance.json"


def test_an_explicit_file_is_loaded_and_must_match_the_machine(tmp_path, reference):
    model = clearance.load_clearance(reference, _write(tmp_path / "a.clearance.json"))
    assert isinstance(model, clearance.BoxesClearance)

    other = _write(tmp_path / "b.clearance.json", name="ours")
    with pytest.raises(ValueError, match="'ours'"):
        clearance.load_clearance(reference, other)


# --------------------------------------------------------------------------
# Against the kinematics
# --------------------------------------------------------------------------


def test_gantry_matches_the_bed_corner_check_in_inverse(ti_cpu, reference, model):
    """`kinematics3z.inverse` asks for a lift of (highest corner - 70 mm) when
    a bed corner rises above the gantry. Put the bed corners in the world
    frame with `atom.bed_motion` and the model must say the same, to within
    the kinematics' float32 rounding. Where a screw would also pass its
    endstop, `inverse` asks for the larger of the two lifts."""
    import visualize_5ax as vt

    tilts = np.repeat([15.0, 25.0, 29.9], 8)
    azimuths = np.tile(np.arange(0, 360, 45.0), 3)
    count = len(tilts)
    toolpath = SimpleNamespace(
        point=np.tile([150.0, 145.0, 30.0], (count, 1)).astype(np.float32),
        travel_type=np.zeros(count, np.int32),
        tool_orientation=np.radians(np.column_stack([tilts, azimuths])).astype(np.float32),
        width=np.ones(count), height=np.ones(count), point_count=count,
    )
    solved = contracts.from_toolpath(toolpath)
    assert solved.valid.all()
    probed, _ = vt.machine_to_build_frame(bed_motion.probe_states(solved.machine))
    rotation, translation = bed_motion.poses_from_probes(solved.machine, probed)
    corners = bed_motion.bed_corners(reference)

    excess = np.empty(count)
    for index, state in enumerate(solved.machine):
        world = bed_motion.to_world(corners, rotation[index], translation[index])
        gantry = model.body_distances(world, state[:2])[:, 1]
        excess[index] = max(0.0, -gantry.min())
    endstop = np.maximum(0.0, -solved.machine[:, 2:].min(axis=1))

    np.testing.assert_allclose(solved.lift_mm, np.maximum(excess, endstop), atol=2e-3)
    # The comparison means something: some states do reach into the gantry.
    assert (excess > 1.0).sum() >= 4
