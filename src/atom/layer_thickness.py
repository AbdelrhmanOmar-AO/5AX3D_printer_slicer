"""How thick the printed layers really are (build plan P3.2, measured on the toolpath).

The plan's P3.2 flags any deposition whose ``height`` lies outside
``[0.5, 1.5] x`` the nominal layer. But a toolpath's ``height`` is the
nominal layer height on every point (0.45 mm for 0.9 mm beads, in every
toolpath checked, stock and overhang-aware), not the real one: Atomizer's
layers are iso-surfaces of a phasor field, and their spacing varies where the
field bends (plan_corrections P2-25). So the check on ``height`` could never
fire.

This measures the thickness from the geometry instead. For each deposition
point, the nearest deposition point **below** it, along ``-d`` (its build
direction), within half a bead width sideways (so it is the bead it sits on,
not a neighbour); the distance along ``d`` is the layer's thickness there.
Points with nothing below within 2.5 nominal layers (the overhang's outer
beads, P2-22) are counted apart, and points on the bed (within half a layer
of the lowest point) are not judged.
Distances under a quarter of a layer count as the same layer, so overlaps
thinner than that are not seen.

numpy and scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .overhang_metrics import deposition_mask, tool_directions

#: P3.2's bounds on a layer, as fractions of the nominal (placeholders).
MIN_LAYER_FRACTION = 0.5
MAX_LAYER_FRACTION = 1.5
#: How far down the layer below is looked for, in nominal layers: past two,
#: so a bead over a one-layer gap (two layers down) is found, not lost to
#: rounding at the edge of the search.
SEARCH_LAYERS = 2.5
#: Points this close to the lowest one, in nominal layers, rest on the bed.
BED_LAYERS = 0.5
#: Closer than this along ``d``, in nominal layers, is the same layer.
SAME_LAYER_FRACTION = 0.25


@dataclass(frozen=True)
class LayerThickness:
    """The real layer thickness under each deposition point."""

    nominal_mm: float
    #: Per deposition point, in print order: the thickness, mm; NaN where
    #: there is no layer below (and on the bed).
    thickness_mm: np.ndarray = field(repr=False)
    #: Per deposition point: resting on the bed (not judged).
    on_bed: np.ndarray = field(repr=False)
    min_fraction: float = MIN_LAYER_FRACTION
    max_fraction: float = MAX_LAYER_FRACTION

    @property
    def judged(self) -> np.ndarray:
        return np.isfinite(self.thickness_mm)

    @property
    def thin(self) -> np.ndarray:
        return self.judged & (np.nan_to_num(self.thickness_mm, nan=np.inf) < self.min_fraction * self.nominal_mm)

    @property
    def thick(self) -> np.ndarray:
        return self.judged & (np.nan_to_num(self.thickness_mm, nan=-np.inf) > self.max_fraction * self.nominal_mm)

    @property
    def nothing_below(self) -> np.ndarray:
        return ~self.judged & ~self.on_bed


def layer_thickness(toolpath, nominal_mm: float, lateral_mm: float | None = None,
                    min_fraction: float = MIN_LAYER_FRACTION,
                    max_fraction: float = MAX_LAYER_FRACTION) -> LayerThickness:
    """The thickness of the layer under every deposition point.

    ``nominal_mm`` is the pipeline's layer height (half the deposition
    width); ``lateral_mm`` how far sideways the bead below may be, by default
    half the toolpath's median width.
    """
    count = int(np.asarray(toolpath.point_count).item())
    mask = deposition_mask(toolpath)
    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]
    directions = tool_directions(toolpath)[:count][mask]
    widths = np.asarray(toolpath.width[:count], dtype=np.float64)[mask]
    if lateral_mm is None:
        lateral_mm = 0.5 * float(np.median(widths)) if len(widths) else 0.5 * 2.0 * nominal_mm
    thickness = np.full(len(points), np.nan)
    on_bed = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return LayerThickness(nominal_mm, thickness, on_bed, min_fraction, max_fraction)
    on_bed = points[:, 2] <= points[:, 2].min() + BED_LAYERS * nominal_mm

    pairs = cKDTree(points).query_pairs(r=SEARCH_LAYERS * nominal_mm, output_type="ndarray")
    if len(pairs):
        source = np.concatenate([pairs[:, 0], pairs[:, 1]])
        other = np.concatenate([pairs[:, 1], pairs[:, 0]])
        offset = points[other] - points[source]
        down = -np.einsum("nk,nk->n", offset, directions[source])
        sideways = np.linalg.norm(offset + down[:, None] * directions[source], axis=1)
        below = (down > SAME_LAYER_FRACTION * nominal_mm) & (sideways <= lateral_mm)
        order = np.lexsort((down[below], source[below]))
        first = np.unique(source[below][order], return_index=True)
        thickness[first[0]] = down[below][order][first[1]]
    thickness[on_bed] = np.nan
    return LayerThickness(nominal_mm, thickness, on_bed, min_fraction, max_fraction)
