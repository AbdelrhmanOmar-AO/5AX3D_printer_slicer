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
    """Give the edge cells near an overhang its constraint, in place.

    ``sdf`` ``(X, Y, Z)`` (inside < 0), ``direction`` ``(X, Y, Z, 2)``
    spherical radians, ``state`` (bit 0: constrained), ``overhang`` the cells
    the rule constrained. Grid-local coordinates (origin 0), as the
    initialisation uses. Returns the mask of the cells filled.
    """
    filled = np.zeros(sdf.shape, dtype=bool)
    if not overhang.any():
        return filled
    z_centre = (np.arange(sdf.shape[2]) + 0.5) * cell_mm
    first_layer = np.broadcast_to(z_centre < layer_height_mm, sdf.shape)
    boundary = (sdf < 0.0) & (sdf > -layer_height_mm)
    candidates = boundary & ((state & 1) == 0) & ~first_layer & ~overhang
    if not candidates.any():
        return filled

    distance, nearest = ndimage.distance_transform_edt(
        ~overhang, sampling=cell_mm, return_indices=True
    )
    filled = candidates & (distance <= reach_layers * layer_height_mm)
    cells = np.argwhere(filled)
    source = tuple(index[filled] for index in nearest)
    theta, phi = direction[source][:, 0].astype(np.float64), direction[source][:, 1].astype(np.float64)
    build = np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=1)
    offset = (cells - np.stack(source, axis=1)) * cell_mm
    beside_or_above = np.einsum("ij,ij->i", offset, build) >= -cell_mm
    filled[tuple(cells[~beside_or_above].T)] = False
    keep = tuple(index[beside_or_above] for index in source)
    direction[filled] = direction[keep]
    state[filled] |= 1
    return filled
