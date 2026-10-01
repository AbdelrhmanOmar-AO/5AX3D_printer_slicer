"""Carry the overhang constraint to an overhang's edges (plan_corrections P2-19).

The overhang rule (`atom.orientation_field.init_overhang_aware`) constrains
only boundary cells whose own surface normal is steeper than
``max_overhang``, and, like upstream's ceiling rule, only where the
curvature is low. Along an overhang's edges, where it meets a side wall, its
tip or a corner, neither holds: the curvature is high and the normal is a
blend of the two faces. On a ramp, only a quarter of the boundary cells
within 0.5 mm of the side walls got the constraint, so the field there stayed
near vertical and the worst point of every part sat at an edge (the lab's
P2.5 matrix, P2-18).

This fills those cells: every boundary cell (inside the part, within one layer
height of the surface) that carries no constraint, is not in the first layer
and lies within ``reach`` of an overhang cell takes that nearest overhang
cell's direction, unless it lies behind that cell along its build direction.
Behind an overhang cell is the material printed before it, such as the wall
below a corner, and building the tilt up there is the ramp-in's job
(`atom.ramp_in`): copying the full tilt into it put 27 degrees a millimetre
above the bed on the test ramp. numpy and scipy only, on the arrays of the
field's initialisation; `atom.orientation_field` applies it before the
ramp-in.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: How far from an overhang cell its constraint is carried, in layer heights.
#: The unconstrained strip along a ramp's edges is about 0.5 mm wide (one
#: layer height plus a cell); two layer heights covers it with room.
DEFAULT_REACH_LAYERS = 2.0


def fill_overhang_edges(
    sdf: np.ndarray,
    direction: np.ndarray,
    state: np.ndarray,
    overhang: np.ndarray,
    cell_mm: float,
    layer_height_mm: float,
    reach_layers: float = DEFAULT_REACH_LAYERS,
):
    """Give an overhang's edges its constraint, in place.

    ``sdf`` ``(X, Y, Z)`` (inside < 0), ``direction`` ``(X, Y, Z, 2)``
    spherical radians, ``state`` (bit 0: constrained), ``overhang`` the cells
    the rule constrained. Grid-local coordinates (origin 0), as the
    initialisation uses.

    Two kinds of cell near the overhang's edge get the direction of the
    nearest **core** overhang cell (one further than ``reach`` from any other
    surface, whose normal is the overhang's own), when there is one within
    twice ``reach``:

    * boundary cells the rule skipped (unconstrained, not in the first layer),
      unless they lie behind the overhang cell nearest them along its build
      direction (the material printed before it: the ramp-in's);
    * overhang cells within ``reach`` of another surface, where the rule read
      a normal blended with that surface's and asked for too little tilt (at
      `ramp60_xs`'s tip, 9-13 degrees where the underside asks 17).

    Where an overhang has no core within reach (narrower than twice
    ``reach``), a skipped cell takes the nearest overhang cell's direction and
    an overhang cell keeps its own. Returns ``(filled, corrected)``: the masks
    of the skipped cells filled and of the overhang cells corrected.
    """
    filled = np.zeros(sdf.shape, dtype=bool)
    corrected = np.zeros(sdf.shape, dtype=bool)
    if not overhang.any():
        return filled, corrected
    reach = reach_layers * layer_height_mm
    z_centre = (np.arange(sdf.shape[2]) + 0.5) * cell_mm
    first_layer = np.broadcast_to(z_centre < layer_height_mm, sdf.shape)
    boundary = (sdf < 0.0) & (sdf > -layer_height_mm)

    # The overhang's core: its cells further than `reach` from any other surface.
    other_surfaces = boundary & ~overhang & ~first_layer
    if other_surfaces.any():
        to_other = ndimage.distance_transform_edt(~other_surfaces, sampling=cell_mm)
    else:
        to_other = np.full(sdf.shape, np.inf)
    core = overhang & (to_other > reach)

    # Skipped cells near the overhang, minus those behind it.
    candidates = boundary & ((state & 1) == 0) & ~first_layer & ~overhang
    to_overhang, nearest = ndimage.distance_transform_edt(~overhang, sampling=cell_mm, return_indices=True)
    filled = candidates & (to_overhang <= reach)
    if filled.any():
        cells = np.argwhere(filled)
        source = np.stack([index[filled] for index in nearest], axis=1)
        theta = direction[tuple(source.T)][:, 0].astype(np.float64)
        phi = direction[tuple(source.T)][:, 1].astype(np.float64)
        build = np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=1)
        behind = np.einsum("ij,ij->i", (cells - source) * cell_mm, build) < -cell_mm
        filled[tuple(cells[behind].T)] = False
        direction[filled] = direction[tuple(index[filled] for index in nearest)]
        state[filled] |= 1

    # Both kinds take the nearest core cell's direction, where there is one in reach.
    if core.any():
        to_core, nearest_core = ndimage.distance_transform_edt(~core, sampling=cell_mm, return_indices=True)
        # Twice `reach`: a cell in a corner is `reach` from two surfaces, so the
        # core is up to `reach` times the square root of two away diagonally.
        near_core = to_core <= 2.0 * reach
        corrected = overhang & ~core & near_core
        retarget = (filled | corrected) & near_core
        direction[retarget] = direction[tuple(index[retarget] for index in nearest_core)]
    return filled, corrected
