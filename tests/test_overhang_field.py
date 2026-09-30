"""Tests for the overhang rule in numpy (build plan P2.2, `atom.overhang_field`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atom import overhang_field as of
from atom import overhang_metrics as om
from atom import tilt


def _face(angle_deg, azimuth_deg=0.0):
    """Outward normal of a surface overhanging at ``angle_deg`` toward ``azimuth_deg``."""
    a, p = math.radians(angle_deg), math.radians(azimuth_deg)
    return np.array([math.cos(a) * math.cos(p), math.cos(a) * math.sin(p), -math.sin(a)])


def test_a_60_degree_face_asks_for_17_degrees_toward_it():
    """The plan's example: 60 - 45 + 2 = 17, at the azimuth of the normal."""
    target = of.overhang_target(_face(60, 30), max_tilt_deg=30)
    assert target.tilt_deg == pytest.approx(17.0)
    assert target.azimuth_deg == pytest.approx(30.0)
    assert not target.capped


def test_the_target_agrees_with_rotate_toward():
    """Spherical [t, phi] is `tilt.rotate_toward(+Z, n_horizontal, t)` in Cartesian form."""
    normal = _face(60, 30)
    target = of.overhang_target(normal, max_tilt_deg=30)
    from_spherical = om.spherical_to_cartesian(np.array([target.spherical_rad]))[0]
    rotated = tilt.rotate_toward(np.array([0.0, 0.0, 1.0]), normal, target.tilt_deg)
    np.testing.assert_allclose(from_spherical, rotated, atol=1e-12)


def test_leaning_toward_the_overhang_lowers_it_by_the_tilt():
    normal = _face(60, 30)
    target = of.overhang_target(normal, max_tilt_deg=30)
    effective = of.effective_overhang_deg(normal, target.tilt_deg, target.azimuth_deg)
    assert effective == pytest.approx(60 - 17)
    # ...and leaning away raises it by the same: the stock field's direction.
    assert of.effective_overhang_deg(normal, 17, 30 + 180) == pytest.approx(60 + 17)


def test_a_flat_ceiling_at_30_degrees_is_capped_and_cannot_reach_45():
    normal = np.array([0.0, 0.0, -1.0])
    target = of.overhang_target(normal, max_tilt_deg=30)
    assert target.tilt_deg == 30
    assert target.wanted_deg == pytest.approx(47.0)
    assert target.capped
    assert of.effective_overhang_deg(normal, target.tilt_deg, target.azimuth_deg) == pytest.approx(60)


@pytest.mark.parametrize(
    "normal",
    [
        np.array([1.0, 0.0, 0.0]),  # a wall
        np.array([0.0, 0.0, 1.0]),  # a top surface
        _face(-30),  # facing up at 30 degrees
        _face(45),  # an overhang exactly at the threshold
        _face(30),  # a gentle overhang
    ],
)
def test_walls_tops_and_gentle_overhangs_are_left_alone(normal):
    assert of.overhang_target(normal, max_tilt_deg=30) is None


def test_the_geometric_angle_matches_the_metric():
    for angle in (10, 45, 60, 89):
        normal = _face(angle, 70)
        assert of.geometric_overhang_deg(normal) == pytest.approx(
            om.geometric_overhang_angle_deg(normal[None, :])[0]
        )


@pytest.mark.parametrize("field,value", [("max_overhang_deg", -1), ("max_overhang_deg", 91), ("margin_deg", -0.5)])
def test_settings_are_validated(field, value):
    with pytest.raises(ValueError, match=field):
        of.OverhangSettings(**{field: value})


# --------------------------------------------------------------------------
# The parameter-file keys
# --------------------------------------------------------------------------


def test_no_keys_means_upstreams_command():
    assert of.overhang_arguments() == ""
    assert of.overhang_arguments(False) == ""


def test_the_rule_is_written_out_in_full():
    assert of.overhang_arguments(True) == " --overhang_aware --max_overhang 45 --overhang_margin 2"
    assert of.overhang_arguments(True, 50, 1.5, True) == (
        " --overhang_aware --max_overhang 50 --overhang_margin 1.5 --hold_overhang"
    )


@pytest.mark.parametrize(
    "keys,message",
    [
        ({"max_overhang_deg": 40}, "need"),
        ({"hold_overhang": True}, "need"),
        ({"overhang_aware": "yes"}, "true or false"),
        ({"overhang_aware": True, "max_overhang_deg": 100}, "between 0 and 90"),
        ({"overhang_aware": True, "overhang_margin_deg": "2"}, "number of degrees"),
        ({"overhang_aware": True, "hold_overhang": 1}, "true or false"),
    ],
)
def test_bad_keys_are_refused_when_the_file_is_read(keys, message):
    with pytest.raises(ValueError, match=message):
        of.overhang_arguments(**keys)
