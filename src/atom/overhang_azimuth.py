"""One lean direction for a flat overhang (plan_corrections P2-20).

The overhang rule leans the tool toward the overhang, at the azimuth of the
surface's horizontal normal, ``atan2(n_y, n_x)``. Under a flat or nearly flat
underside that normal has almost no horizontal part, and on the pipeline's
remeshed surface what is left is noise: on `ramp90_xs` the constraints under
the flat underside pointed every way about equally. Neighbours leaning 30
degrees in opposite directions average to vertical (the field a millimetre
above the underside tilted 2 degrees on average), and adjacent atoms pointing
apart can each wait for the other in `order_atoms`, which is how all four
P2.5 runs on flat undersides at 30 degrees failed (P2-18).

For a flat underside every lean lowers the effective angle by the same
amount, so the direction only has to be consistent. This makes it consistent
and outward: away from the material holding the overhang up, toward its free
edges. The horizontal normals of the walls and the overhang's own edges
around it point that way (the wall under the overhang's root, the tip, the
sides); smoothed over a few millimetres they give each flat overhang cell a
direction that varies smoothly. numpy and scipy only, on the arrays of the
field's initialisation; `atom.orientation_field` applies it after the rule.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Overhangs at least this steep (from vertical) take the smoothed direction:
#: their horizontal normal is at most cos(80) = 0.17, where the remeshed
#: surface's noise can turn it any way.
DEFAULT_FLAT_DEG = 80.0

#: How far the walls' and edges' directions are spread, mm (Gaussian sigma).
DEFAULT_SPREAD_MM = 3.0


def surface_normals(sdf: np.ndarray, cell_mm: float) -> np.ndarray:
    """``(X, Y, Z, 3)`` unit normals from central differences, as the rule's normals are."""
    gradient = np.stack(np.gradient(sdf.astype(np.float64), cell_mm), axis=-1)
    length = np.linalg.norm(gradient, axis=-1, keepdims=True)
    return gradient / np.maximum(length, 1e-12)


def aim_flat_overhangs_outward(
    sdf: np.ndarray,
    direction: np.ndarray,
    overhang: np.ndarray,
    cell_mm: float,
    layer_height_mm: float,
    flat_deg: float = DEFAULT_FLAT_DEG,
    spread_mm: float = DEFAULT_SPREAD_MM,
) -> np.ndarray:
    """Give the flat overhang cells one smooth, outward azimuth, in place.

    ``sdf`` ``(X, Y, Z)`` (inside < 0), ``direction`` ``(X, Y, Z, 2)``
    spherical radians, ``overhang`` the cells the rule constrained. Only the
    azimuth of the flat ones changes; their tilt stays the rule's. Returns
    the mask of the cells re-aimed.
    """
    aimed = np.zeros(sdf.shape, dtype=bool)
    if not overhang.any():
        return aimed
    normal = surface_normals(sdf, cell_mm)
    steepness = np.degrees(np.arcsin(np.clip(-normal[..., 2], -1.0, 1.0)))
    flat = overhang & (steepness >= flat_deg)
    if not flat.any():
        return aimed

    # The directions to spread: the walls in the boundary band, and the
    # overhang cells (whose edges blend outward).
    boundary = (sdf < 0.0) & (sdf > -layer_height_mm)
    sources = (boundary & (np.abs(normal[..., 2]) < 0.5)) | overhang
    sigma = spread_mm / cell_mm
    weight = ndimage.gaussian_filter(sources.astype(np.float64), sigma)
    nx = ndimage.gaussian_filter(np.where(sources, normal[..., 0], 0.0), sigma)
    ny = ndimage.gaussian_filter(np.where(sources, normal[..., 1], 0.0), sigma)
    nx, ny = nx / np.maximum(weight, 1e-12), ny / np.maximum(weight, 1e-12)

    direction[flat, 1] = np.arctan2(ny[flat], nx[flat])
    aimed[flat] = True
    return aimed
