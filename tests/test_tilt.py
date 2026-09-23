"""Tests for the tilt geometry (build plan task P0.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atom import tilt


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", [-30.0, -7.5, 0.0, 12.0, 30.0])
@pytest.mark.parametrize("b", [-30.0, -7.5, 0.0, 12.0, 30.0])
def test_rotations_are_orthonormal_with_determinant_one(a, b):
    rotation = tilt.rotation_from_tilts(a, b)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_zero_tilt_is_the_identity():
    np.testing.assert_allclose(tilt.rotation_from_tilts(0.0, 0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        tilt.direction_from_tilts(0.0, 0.0), [0.0, 0.0, 1.0], atol=1e-12
    )


def test_direction_matches_the_documented_closed_form():
    """d = [sin(b)cos(a), -sin(a), cos(a)cos(b)]."""
    a, b = 17.0, -23.0
    ra, rb = math.radians(a), math.radians(b)

    np.testing.assert_allclose(
        tilt.direction_from_tilts(a, b),
        [math.sin(rb) * math.cos(ra), -math.sin(ra), math.cos(ra) * math.cos(rb)],
        atol=1e-12,
    )


@pytest.mark.parametrize("a", [-45.0, -20.0, -1.0, 0.0, 1.0, 20.0, 45.0])
@pytest.mark.parametrize("b", [-45.0, -20.0, -1.0, 0.0, 1.0, 20.0, 45.0])
def test_tilt_round_trip_over_a_grid(a, b):
    """tilts -> direction -> tilts recovers the original pair."""
    direction = tilt.direction_from_tilts(a, b)
    recovered_a, recovered_b = tilt.tilts_from_direction(direction)

    assert recovered_a == pytest.approx(a, abs=1e-9)
    assert recovered_b == pytest.approx(b, abs=1e-9)


def test_direction_round_trip_over_random_unit_vectors():
    rng = np.random.default_rng(20260921)
    for _ in range(200):
        vector = rng.normal(size=3)
        vector[2] = abs(vector[2]) + 0.2       # keep it in the upper hemisphere
        vector /= np.linalg.norm(vector)

        a, b = tilt.tilts_from_direction(vector)
        np.testing.assert_allclose(tilt.direction_from_tilts(a, b), vector, atol=1e-12)


# --------------------------------------------------------------------------
# Total tilt
# --------------------------------------------------------------------------


def test_total_tilt_of_vertical_is_zero():
    assert tilt.total_tilt_deg([0.0, 0.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize("angle", [5.0, 17.5, 30.0, 45.0, 89.0])
def test_total_tilt_recovers_a_single_axis_tilt(angle):
    assert tilt.total_tilt_deg(tilt.direction_from_tilts(angle, 0.0)) == pytest.approx(angle)
    assert tilt.total_tilt_deg(tilt.direction_from_tilts(0.0, angle)) == pytest.approx(angle)


def test_two_axis_tilts_do_not_simply_add():
    """cos(total) = cos(a)cos(b): 20 and 20 give 27.99 degrees, not 40.

    arccos(cos(20)^2) = arccos(0.883022) = 27.9909 degrees.
    """
    total = tilt.total_tilt_from_tilts_deg(20.0, 20.0)

    assert total == pytest.approx(27.9909, abs=1e-3)
    assert total < 40.0
    assert total > 20.0


def test_total_tilt_agrees_between_the_two_routes():
    for a, b in ((0.0, 0.0), (10.0, 0.0), (0.0, 25.0), (18.0, -14.0), (30.0, 30.0)):
        assert tilt.total_tilt_from_tilts_deg(a, b) == pytest.approx(
            tilt.total_tilt_deg(tilt.direction_from_tilts(a, b))
        )


def test_a_zero_length_direction_is_rejected():
    with pytest.raises(ValueError, match="zero-length"):
        tilt.total_tilt_deg([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero-length"):
        tilt.tilts_from_direction([0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


def test_cone_limit_uses_the_total_angle():
    assert tilt.within_limit(30.0, 0.0, 30.0, "cone")
    assert not tilt.within_limit(30.1, 0.0, 30.0, "cone")
    # 20 and 20 is 27.9 total, so inside a 30 degree cone.
    assert tilt.within_limit(20.0, 20.0, 30.0, "cone")
    # 25 and 25 is 34.78 total, so outside it.
    assert not tilt.within_limit(25.0, 25.0, 30.0, "cone")


def test_box_limit_treats_the_axes_independently():
    assert tilt.within_limit(30.0, 30.0, 30.0, "box")
    assert not tilt.within_limit(30.1, 0.0, 30.0, "box")


def test_a_box_permits_diagonals_a_cone_rejects():
    """The distinction is not academic: it is GATE M2 for our machine."""
    assert tilt.within_limit(25.0, 25.0, 30.0, "box")
    assert not tilt.within_limit(25.0, 25.0, 30.0, "cone")


def test_limits_are_symmetric_in_sign():
    for a, b in ((-20.0, 20.0), (20.0, -20.0), (-20.0, -20.0)):
        assert tilt.within_limit(a, b, 30.0, "cone")
        assert tilt.within_limit(a, b, 30.0, "box")


def test_an_unknown_limit_shape_is_rejected():
    with pytest.raises(ValueError, match="cone.*box"):
        tilt.within_limit(0.0, 0.0, 30.0, "sphere")


# --------------------------------------------------------------------------
# rotate_toward: the operation the overhang-aware field performs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [1.0, 7.0, 17.0, 30.0, 45.0])
def test_rotate_toward_moves_exactly_the_requested_angle(angle):
    start = np.array([0.0, 0.0, 1.0])
    rotated = tilt.rotate_toward(start, [1.0, 0.0], angle)

    assert tilt.total_tilt_deg(rotated) == pytest.approx(angle, abs=1e-9)
    assert np.linalg.norm(rotated) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "target,expected_axis",
    [([1.0, 0.0], 0), ([-1.0, 0.0], 0), ([0.0, 1.0], 1), ([0.0, -1.0], 1)],
)
def test_rotate_toward_leans_the_way_it_was_asked_to(target, expected_axis):
    rotated = tilt.rotate_toward([0.0, 0.0, 1.0], target, 20.0)

    assert np.sign(rotated[expected_axis]) == np.sign(target[expected_axis])
    assert rotated[1 - expected_axis] == pytest.approx(0.0, abs=1e-12)


def test_rotate_toward_ignores_any_vertical_component_of_the_target():
    """Tilt is a horizontal choice; a target's z must not change the result."""
    flat = tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 15.0)
    steep = tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 0.0, -5.0], 15.0)

    np.testing.assert_allclose(flat, steep, atol=1e-12)


def test_rotate_toward_by_a_negative_angle_leans_away():
    away = tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 0.0], -20.0)
    assert away[0] < 0.0
    assert tilt.total_tilt_deg(away) == pytest.approx(20.0)


def test_rotate_toward_composes():
    once = tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 0.0], 30.0)
    twice = tilt.rotate_toward(
        tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 0.0], 10.0), [1.0, 0.0], 20.0
    )
    np.testing.assert_allclose(once, twice, atol=1e-12)


def test_rotate_toward_with_no_usable_target_returns_the_input():
    start = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(tilt.rotate_toward(start, [0.0, 0.0], 20.0), start)
    # A target parallel to the direction defines no rotation plane.
    np.testing.assert_allclose(
        tilt.rotate_toward([1.0, 0.0, 0.0], [1.0, 0.0], 20.0), [1.0, 0.0, 0.0]
    )


def test_rotate_toward_normalises_a_non_unit_input():
    rotated = tilt.rotate_toward([0.0, 0.0, 5.0], [1.0, 0.0], 10.0)
    assert np.linalg.norm(rotated) == pytest.approx(1.0)
    assert tilt.total_tilt_deg(rotated) == pytest.approx(10.0)


def test_rotate_toward_rejects_a_bad_target_shape():
    with pytest.raises(ValueError, match="2 or 3 components"):
        tilt.rotate_toward([0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0], 10.0)
