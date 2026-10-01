"""Tests for one outward lean under a flat overhang (plan_corrections P2-20)."""

from __future__ import annotations

import numpy as np

from atom import analytic_sdf
from atom import overhang_azimuth as oa

CELL = 0.17
LAYER = 0.45


def _flat_ramp():
    """`ramp90`'s shape, 12 x 5.4 x 7.2 mm: a column (x < 8.4) and a flat slab beside it."""
    sdf, geometry = analytic_sdf.ramp_sdf(90, 12.0, 5.4, 7.2, CELL)
    normal = oa.surface_normals(sdf, CELL)
    z = (np.arange(sdf.shape[2]) + 0.5) * CELL
    overhang = (sdf < 0) & (sdf > -LAYER) & (normal[..., 2] < -0.9) & (z[None, None, :] > 1.0)
    return sdf, geometry, overhang


def test_scrambled_lean_directions_under_a_flat_underside_become_one_outward_direction():
    sdf, geometry, overhang = _flat_ramp()
    direction = np.zeros(sdf.shape + (2,), np.float32)
    rng = np.random.default_rng(0)
    direction[overhang, 0] = np.radians(30.0)
    direction[overhang, 1] = rng.uniform(-np.pi, np.pi, overhang.sum())  # the remeshed surface's noise

    aimed = oa.aim_flat_overhangs_outward(sdf, direction, overhang, CELL, LAYER)

    # Every overhang cell at least 80 degrees steep; the rest (the slab's edges) keep theirs.
    steep = np.degrees(np.arcsin(np.clip(-oa.surface_normals(sdf, CELL)[..., 2], -1, 1)))
    np.testing.assert_array_equal(aimed, overhang & (steep >= oa.DEFAULT_FLAT_DEG))
    assert aimed.sum() > 0.8 * overhang.sum()
    x = (np.argwhere(aimed)[:, 0] + 0.5) * CELL
    y = (np.argwhere(aimed)[:, 1] + 0.5) * CELL
    middle = (x > geometry["column_width"] + 1.0) & (np.abs(y - 2.7) < 1.0)
    phi = np.degrees(direction[aimed, 1])
    # Away from the column (+x) in the middle of the slab, never back toward it.
    assert np.abs(phi[middle]).max() < 30.0
    assert np.abs(phi).max() < 90.0 + 1e-6
    np.testing.assert_allclose(np.degrees(direction[aimed, 0]), 30.0, atol=1e-4)  # the tilt is the rule's


def test_steeper_overhangs_keep_their_own_azimuth():
    """A 60-degree underside has a real horizontal normal: nothing is re-aimed."""
    sdf, _ = analytic_sdf.ramp_sdf(60, 12.0, 5.4, 7.2, CELL)
    normal = oa.surface_normals(sdf, CELL)
    z = (np.arange(sdf.shape[2]) + 0.5) * CELL
    overhang = (sdf < 0) & (sdf > -LAYER) & (normal[..., 2] < -0.7) & (normal[..., 2] > -0.95) & (z[None, None, :] > 1.0)
    assert overhang.any()
    direction = np.zeros(sdf.shape + (2,), np.float32)
    before = direction.copy()
    assert not oa.aim_flat_overhangs_outward(sdf, direction, overhang, CELL, LAYER).any()
    np.testing.assert_array_equal(direction, before)
