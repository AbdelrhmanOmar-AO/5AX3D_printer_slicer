"""Tests for carrying the overhang constraint to its edges (plan_corrections P2-19)."""

from __future__ import annotations

import numpy as np

from atom import overhang_edges as oe

CELL = 0.2
LAYER = 0.45


def _grid():
    """A slab 2 mm thick (z 2-4 mm) in a 20 x 10 x 30 grid, nothing constrained."""
    shape = (20, 10, 30)
    z = (np.arange(shape[2]) + 0.5) * CELL
    sdf = np.broadcast_to(np.maximum(2.0 - z, z - 4.0), shape).astype(np.float32).copy()
    return sdf, np.zeros(shape + (2,), np.float32), np.zeros(shape, np.uint32)


def test_boundary_cells_near_an_overhang_take_its_direction():
    sdf, direction, state = _grid()
    overhang = np.zeros(sdf.shape, bool)
    overhang[5:10, :, 10] = True  # part of the slab's underside, z 2.1 mm
    direction[overhang] = (0.3, 1.2)
    state[overhang] |= 1

    filled, _ = oe.fill_overhang_edges(sdf, direction, state, overhang, CELL, LAYER)

    assert filled[10, 5, 10] and filled[4, 5, 10]  # next to it along the underside
    np.testing.assert_allclose(direction[filled], np.tile([0.3, 1.2], (filled.sum(), 1)))
    assert (state[filled] & 1).all()
    assert not filled[15, 5, 10]  # 1.2 mm away: beyond two layer heights
    assert not filled[7, 5, 15]  # inside the slab, not a boundary cell


def test_constraints_and_the_first_layer_are_left_alone():
    sdf, direction, state = _grid()
    sdf[:, :, :3] = -0.1  # the bottom rows join the boundary band; z < 0.45 mm is the first layer
    overhang = np.zeros(sdf.shape, bool)
    overhang[5, 5, 3] = True
    direction[overhang] = (0.3, 1.2)
    state[overhang] |= 1
    direction[6, 5, 3] = (0.1, 0.0)
    state[6, 5, 3] |= 1  # an existing constraint next to it

    filled, _ = oe.fill_overhang_edges(sdf, direction, state, overhang, CELL, LAYER)

    assert not filled[6, 5, 3]
    np.testing.assert_allclose(direction[6, 5, 3], (0.1, 0.0))
    assert not filled[:, :, :2].any()  # cell centres 0.1 and 0.3 mm: the first layer
    assert filled[5, 5, 2]  # centre 0.5 mm: not the first layer


def test_the_material_behind_an_overhang_is_left_to_the_ramp_in():
    """Below an overhang cell along its build direction: printed before it."""
    sdf, direction, state = _grid()
    sdf[:] = -0.1  # every cell a boundary cell, to test the direction rule alone
    overhang = np.zeros(sdf.shape, bool)
    overhang[10, 5, 20] = True
    direction[overhang] = (0.0, 0.0)  # straight up
    state[overhang] |= 1

    filled, _ = oe.fill_overhang_edges(sdf, direction, state, overhang, CELL, LAYER)

    assert filled[10, 5, 22] and filled[12, 5, 20]  # above, and beside
    assert not filled[10, 5, 17]  # 0.6 mm below


def test_no_overhang_no_change():
    sdf, direction, state = _grid()
    filled, corrected = oe.fill_overhang_edges(sdf, direction, state, np.zeros(sdf.shape, bool), CELL, LAYER)
    assert not filled.any() and not corrected.any()


def test_overhang_cells_by_an_edge_take_the_core_direction():
    """The tip's blended normal asks for too little tilt; the core's direction replaces it."""
    sdf, direction, state = _grid()
    overhang = np.zeros(sdf.shape, bool)
    overhang[:, :, 10:12] = True  # the slab's underside band, z 2.1 and 2.3 mm, as the rule marks it
    direction[overhang] = (0.30, 0.0)
    direction[18:, :, 10:12] = (0.10, 0.5)  # by the slab's end at x = 4 mm: blended, too little tilt
    state[overhang] |= 1
    sdf[19, :, 10:20] = -0.05  # the end wall: another surface next to them

    _, corrected = oe.fill_overhang_edges(sdf, direction, state, overhang, CELL, LAYER)

    assert corrected[18, 5, 10]
    np.testing.assert_allclose(direction[18, 5, 10], (0.30, 0.0))
    assert not corrected[5, 5, 10]  # the core keeps its own
