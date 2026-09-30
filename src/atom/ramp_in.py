"""Tilt ramp-in: build the tilt up before an overhang starts (build plan P2.4).

The overhang rule (`atom.overhang_field`) constrains only a thin band of
cells along an overhang's underside. Below it, the field is interpolated from
the flat first layer, so where a wall turns into an overhang the first strip
of overhang can be printed before the tilt has built up (plan_corrections
P2-13, `docs/orientation_field.md` section 7.4).

The ramp-in adds constraints in the material printed *before* each overhang
cell: from the cell, step backwards along its own build direction ``-d`` and
constrain each cell passed to the same azimuth, with a tilt that falls off
linearly with the distance ``s``:

    tilt(s) = t - rate * s,        s up to  t / rate

so the tilt changes by at most ``rate`` degrees per mm (``max_tilt_rate``, a
gate D3 placeholder, 3). A walk stops where the part ends (air: nothing is
printed before that point) or at the first layer. When the first layer comes
first, the ramp does not fit; by the operator's decision of 2026-09-30 the
overhang keeps its full tilt and the ramp is **steepened** to fit the room
there is,

    tilt(s) = t * (1 - s / room),

and the steepening is reported. (The plan capped ``t`` instead, which on
``ramp60_xs`` would give its corner 13 degrees instead of 17.)

Ramp cells never replace a constraint already set (first layer, ceiling,
overhang). Where several walks cross a cell, the largest tilt wins. They are
"soft", as the plan asks: constrained through the multigrid solve, but, unlike
held overhang cells, smoothed in the final passes like upstream's own
constraints.

numpy only, on the arrays of the field's initialisation; `atom.orientation_field`
applies it between the initialisation and the solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: GATE D3: placeholder, team decision pending (tuned on the printer, P7.4).
DEFAULT_MAX_TILT_RATE_DEG_PER_MM = 3.0

#: The walk's step, in cells: half a cell, so no cell on the path is skipped.
STEP_CELLS = 0.5


@dataclass(frozen=True)
class RampSettings:
    """The ramp-in's parameter."""

    #: How fast the tilt may change along the build direction, degrees per mm.
    max_tilt_rate_deg_per_mm: float = DEFAULT_MAX_TILT_RATE_DEG_PER_MM

    def __post_init__(self):
        rate = self.max_tilt_rate_deg_per_mm
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"max_tilt_rate_deg_per_mm must be a positive number, got {rate!r}")


def ramp_length_mm(tilt_deg: float, rate_deg_per_mm: float) -> float:
    """How far below an overhang the tilt starts to build up, mm."""
    return tilt_deg / rate_deg_per_mm


def ramp_tilt_deg(tilt_deg, s_mm, rate_deg_per_mm, room_mm=None):
    """The ramp's tilt ``s_mm`` below a cell asking for ``tilt_deg``.

    ``room_mm`` is the distance to the first layer when it is shorter than
    the ramp: the ramp is then steepened to reach 0 there. Never below 0.
    """
    tilt_deg = np.asarray(tilt_deg, dtype=np.float64)
    s_mm = np.asarray(s_mm, dtype=np.float64)
    if room_mm is None:
        tilt = tilt_deg - rate_deg_per_mm * s_mm
    else:
        room_mm = np.asarray(room_mm, dtype=np.float64)
        tilt = np.where(
            np.isfinite(room_mm),
            tilt_deg * (1.0 - s_mm / np.where(np.isfinite(room_mm), room_mm, 1.0)),
            tilt_deg - rate_deg_per_mm * s_mm,
        )
    return np.maximum(tilt, 0.0)


@dataclass
class RampResult:
    """What the ramp-in did."""

    #: New constraints set.
    cells: int
    #: Overhang cells whose walk passed through material and set at least one cell.
    walks_used: int
    #: Of those, walks that reached the first layer early and were steepened.
    walks_steepened: int
    #: The steepest rate a steepened walk needed, degrees per mm (0 if none).
    steepest_rate_deg_per_mm: float


def apply_ramp_in(
    sdf: np.ndarray,
    direction: np.ndarray,
    state: np.ndarray,
    overhang: np.ndarray,
    cell_mm: float,
    layer_height_mm: float,
    settings: RampSettings = RampSettings(),
):
    """Add ramp-in constraints to an initialised field, in place.

    ``sdf`` ``(X, Y, Z)``: the SDF the field was initialised on (inside < 0).
    ``direction`` ``(X, Y, Z, 2)`` spherical ``[theta, phi]`` radians and
    ``state`` ``(X, Y, Z)`` (bit 0: constrained), as the initialisation left
    them; ``overhang`` ``(X, Y, Z)`` bool, the overhang cells to ramp in
    below. Grid-local coordinates (origin 0), as the initialisation uses.

    Returns ``(ramp_mask, RampResult)``; ``direction`` and ``state`` are
    updated where ``ramp_mask`` is true.
    """
    rate = float(settings.max_tilt_rate_deg_per_mm)
    shape = np.array(sdf.shape)
    starts = np.argwhere(overhang)
    empty = np.zeros(sdf.shape, dtype=bool)
    if len(starts) == 0:
        return empty, RampResult(0, 0, 0, 0.0)

    theta = direction[overhang, 0].astype(np.float64)
    phi = direction[overhang, 1].astype(np.float64)
    tilt = np.degrees(theta)
    back = -np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=1
    )
    centre = (starts + 0.5) * cell_mm
    step = STEP_CELLS * cell_mm
    length = tilt / rate
    steps = int(math.ceil(length.max() / step)) if len(length) else 0

    # Pass 1: walk every overhang cell back along -d. A walk ends at air or at
    # the first layer; `room` is the distance to the first layer when that
    # came first, within the ramp's length.
    alive = np.ones(len(starts), dtype=bool)
    room = np.full(len(starts), np.inf)
    visits_walk, visits_cell, visits_s = [], [], []
    start_flat = np.ravel_multi_index(starts.T, sdf.shape)
    for k in range(1, steps + 1):
        s = k * step
        active = alive & (s <= length + 1e-9)
        if not active.any():
            break
        walk = np.flatnonzero(active)
        point = centre[walk] + s * back[walk]
        index = np.floor(point / cell_mm).astype(np.int64)
        inside_grid = np.all((index >= 0) & (index < shape), axis=1)
        # Leaving the grid is leaving the part.
        alive[walk[~inside_grid]] = False
        walk, point, index = walk[inside_grid], point[inside_grid], index[inside_grid]
        flat = np.ravel_multi_index(index.T, sdf.shape)
        in_part = sdf.ravel()[flat] < 0.0
        alive[walk[~in_part]] = False
        walk, point, flat = walk[in_part], point[in_part], flat[in_part]
        # The first layer, as the initialisation tests it: the cell centre's z.
        first_layer = (flat % shape[2] + 0.5) * cell_mm < layer_height_mm
        room[walk[first_layer]] = s
        alive[walk[first_layer]] = False
        keep = ~first_layer & (flat != start_flat[walk])
        visits_walk.append(walk[keep])
        visits_cell.append(flat[keep])
        visits_s.append(np.full(keep.sum(), s))

    if not visits_walk:
        return empty, RampResult(0, 0, 0, 0.0)
    walk = np.concatenate(visits_walk)
    flat = np.concatenate(visits_cell)
    s = np.concatenate(visits_s)

    # Pass 2: the tilt at each visit; never over an existing constraint.
    ramp = ramp_tilt_deg(tilt[walk], s, rate, room[walk])
    free = (state.ravel()[flat] & 1) == 0
    useful = free & (ramp > 0.0)
    walk, flat, s, ramp = walk[useful], flat[useful], s[useful], ramp[useful]
    if len(flat) == 0:
        return empty, RampResult(0, 0, 0, 0.0)

    # Where walks cross, the largest tilt wins.
    order = np.lexsort((ramp, flat))
    flat, walk, ramp = flat[order], walk[order], ramp[order]
    last = np.r_[flat[1:] != flat[:-1], True]
    flat, walk, ramp = flat[last], walk[last], ramp[last]

    d = direction.reshape(-1, 2)
    d[flat, 0] = np.radians(ramp)
    d[flat, 1] = phi[walk]
    st = state.reshape(-1)
    st[flat] |= 1
    mask = np.zeros(sdf.size, dtype=bool)
    mask[flat] = True

    used = np.unique(walk)
    steepened = used[np.isfinite(room[used])]
    steepest = float(np.max(tilt[steepened] / room[steepened])) if len(steepened) else 0.0
    return mask.reshape(sdf.shape), RampResult(
        cells=int(mask.sum()),
        walks_used=int(len(used)),
        walks_steepened=int(len(steepened)),
        steepest_rate_deg_per_mm=steepest,
    )
