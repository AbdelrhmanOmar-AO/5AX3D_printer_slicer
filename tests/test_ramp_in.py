"""Tests for the tilt ramp-in (build plan P2.4, `atom.ramp_in`), numpy only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atom import ramp_in as ri

CELL = 0.2
LAYER = 0.45


def _grid(shape=(12, 3, 40), inside=None):
    """An all-solid grid (or ``inside``), no constraints yet."""
    sdf = np.full(shape, -1.0, dtype=np.float32)
    if inside is not None:
        sdf[~inside] = 1.0
    direction = np.zeros(shape + (2,), dtype=np.float32)
    state = np.zeros(shape, dtype=np.uint32)
    return sdf, direction, state


def _overhang(direction, state, cells, tilt_deg, phi_deg=0.0):
    mask = np.zeros(state.shape, dtype=bool)
    for c in cells:
        mask[c] = True
        direction[c] = (math.radians(tilt_deg), math.radians(phi_deg))
        state[c] |= 1
    return mask


def _cell_centre(index):
    return (np.asarray(index) + 0.5) * CELL


# --------------------------------------------------------------------------
# The maths
# --------------------------------------------------------------------------


def test_the_ramp_is_the_tilt_over_the_rate():
    """17 degrees at the placeholder 3 degrees per mm: 5.7 mm."""
    assert ri.ramp_length_mm(17.0, 3.0) == pytest.approx(17.0 / 3.0)
    assert ri.DEFAULT_MAX_TILT_RATE_DEG_PER_MM == 3.0


def test_the_tilt_falls_off_linearly_and_stops_at_zero():
    s = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(ri.ramp_tilt_deg(12.0, s, 3.0), [12, 9, 6, 0, 0])


def test_too_little_room_steepens_the_ramp_and_keeps_the_tilt():
    """The operator's decision, 2026-09-30: full tilt at the overhang, steeper below."""
    s = np.array([0.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(ri.ramp_tilt_deg(12.0, s, 3.0, room_mm=4.0), [12, 9, 6, 0])
    np.testing.assert_allclose(ri.ramp_tilt_deg(12.0, s, 3.0, room_mm=2.0), [12, 6, 0, 0])
    # No room limit where it is infinite.
    np.testing.assert_allclose(ri.ramp_tilt_deg(12.0, s, 3.0, room_mm=np.inf), [12, 9, 6, 0])


@pytest.mark.parametrize("rate", [0, -1, float("nan"), True, "3"])
def test_the_rate_must_be_positive(rate):
    with pytest.raises(ValueError, match="positive"):
        ri.RampSettings(rate)


# --------------------------------------------------------------------------
# On a grid
# --------------------------------------------------------------------------


def test_a_full_ramp_below_an_overhang_high_above_the_bed():
    sdf, direction, state = _grid()
    start = (6, 1, 35)  # 7.1 mm up
    mask = _overhang(direction, state, [start], 12.0)
    ramp, result = ri.apply_ramp_in(sdf, direction, state, mask, CELL, LAYER)

    assert result.cells > 0 and result.walks_used == 1 and result.walks_steepened == 0
    assert not ramp[start]
    cells = np.argwhere(ramp)
    # Every ramp cell is constrained, leans the overhang's way, and is below it.
    assert (state[ramp] & 1).all()
    np.testing.assert_allclose(direction[ramp][:, 1], 0.0)
    assert (cells[:, 2] < start[2]).all()
    # The tilt falls with the distance, at no more than the rate.
    distance = np.linalg.norm(_cell_centre(cells) - _cell_centre(start), axis=1)
    tilt = np.degrees(direction[ramp][:, 0])
    assert (tilt <= 12.0 - 3.0 * (distance - CELL) + 1e-4).all()
    assert (distance <= 12.0 / 3.0 + CELL).all()
    # The walk goes back along -d: down, and away from the lean (toward -x).
    assert cells[:, 0].min() < start[0]


def test_near_the_bed_the_ramp_is_steepened_not_the_tilt_reduced():
    sdf, direction, state = _grid()
    start = (6, 1, 12)  # 2.5 mm up: 12 degrees at 3 per mm would need 4 mm
    mask = _overhang(direction, state, [start], 12.0)
    _, result = ri.apply_ramp_in(sdf, direction, state, mask, CELL, LAYER)

    assert result.walks_steepened == 1
    assert result.steepest_rate_deg_per_mm > 3.0
    assert math.degrees(direction[start][0]) == pytest.approx(12.0)  # the overhang keeps its tilt


def test_a_walk_stops_where_the_part_ends():
    """Under an overhang there is air: nothing is printed before it there."""
    inside = np.ones((12, 3, 40), dtype=bool)
    inside[:, :, :30] = False  # air below z = 6 mm
    sdf, direction, state = _grid(inside=inside)
    mask = _overhang(direction, state, [(6, 1, 31)], 12.0)
    ramp, result = ri.apply_ramp_in(sdf, direction, state, mask, CELL, LAYER)

    assert (np.argwhere(ramp)[:, 2] >= 30).all()
    assert result.walks_steepened == 0


def test_existing_constraints_are_kept():
    sdf, direction, state = _grid()
    start = (6, 1, 35)
    mask = _overhang(direction, state, [start], 12.0)
    below = (6, 1, 33)
    direction[below] = (0.3, 2.0)
    state[below] |= 1
    ramp, _ = ri.apply_ramp_in(sdf, direction, state, mask, CELL, LAYER)
    assert not ramp[below]
    np.testing.assert_allclose(direction[below], (0.3, 2.0))


def test_where_walks_cross_the_larger_tilt_wins():
    """Two nearly vertical walks down one column of cells, at 1 degree per mm.

    At the cell 0.6 mm below the upper one (2 degrees) and 0.4 mm below the
    lower one (1 degree), they ask 1.4 and 0.6: the larger is kept, although
    the other walk is the closer.
    """
    sdf, direction, state = _grid()
    mask = _overhang(direction, state, [(6, 1, 35)], 2.0)
    mask |= _overhang(direction, state, [(6, 1, 34)], 1.0)
    ri.apply_ramp_in(sdf, direction, state, mask, CELL, LAYER, ri.RampSettings(1.0))
    assert math.degrees(direction[(6, 1, 32)][0]) == pytest.approx(1.4, abs=1e-4)


def test_no_overhang_cells_no_ramp():
    sdf, direction, state = _grid()
    ramp, result = ri.apply_ramp_in(sdf, direction, state, np.zeros(state.shape, bool), CELL, LAYER)
    assert not ramp.any() and result.cells == 0
