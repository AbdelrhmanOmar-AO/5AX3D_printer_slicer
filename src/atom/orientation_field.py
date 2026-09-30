"""The tool-orientation field with the overhang-aware additions (build plan P2.2).

Pipeline stage 4, `tools/compute_tool_orientations.py`, computes the field the
way upstream does (`docs/orientation_field.md` section 2). When it is given
``--overhang_aware`` or ``--solid_sdf`` it runs `compute_direction_field`
here instead; without them it runs upstream's own code, untouched, so stock
output cannot change. With both options off, `compute_direction_field`
reproduces upstream's field bit for bit (`tests/test_orientation_field.py`).

What it adds:

1. **The overhang constraint** (`init_overhang_aware`). Upstream's
   initialisation, plus: a boundary cell whose surface faces down at
   ``theta_geo`` steeper than ``max_overhang`` is constrained to lean toward
   the overhang, ``[t, atan2(n_y, n_x)]`` with
   ``t = min(theta_geo - max_overhang + margin, max_tilt)``
   (`atom.overhang_field` has the rule on its own).
2. **The field on the solid part** (plan_corrections P2-8, option (a), the
   operator's decision). With infill, the pipeline's SDF is hollowed 1.8 mm
   under every surface, and the hollow's inner surface acts as a "ceiling"
   that tilts the field away from overhangs (P2-3). Given the SDF from
   before infill (``solid_sdf``), the field is computed on the solid part and
   then masked where the infilled SDF is outside, exactly the cells upstream
   masks, so no later stage sees a direction in an infill void.
3. **Holding the overhang constraints** through the 32 final smoothing passes
   (``hold_overhang``). Upstream smooths every constraint in those passes and
   puts back only the first layer (P2-2); holding re-applies the overhang
   constraints after each pass too.

The tilt budget is ``min(max_slope, (180 - nozzle cone) / 2)``, as upstream's
``fff3.MAX_SLOPE_ANGLE``; it is both the ceiling threshold and the overhang
constraint's cap until P2.3's reachability map gives a cap per position.

No `from __future__ import annotations`: this module defines Taichi kernels.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import taichi as ti

from . import direction, fff3, grid3, solid3, toolpath3
from .overhang_field import OverhangSettings

#: Smoothing passes after the multigrid solve, as upstream.
FINAL_SMOOTHING_PASSES = 32

#: ``mark`` values written by `init_overhang_aware`.
MARK_NONE = 0
MARK_OVERHANG = 1
#: An overhang cell whose tilt the cap stopped short of the rule's.
MARK_OVERHANG_CAPPED = 2

#: Passed as ``overhang_min_angle`` to switch the overhang rule off: no
#: surface is steeper than a flat ceiling.
_OVERHANG_OFF = 0.5 * math.pi + 1.0


@ti.kernel
def init_overhang_aware(
    sdf: ti.template(),
    cell_sides_length: float,
    ceil_max_angle: float,
    overhang_min_angle: float,
    overhang_margin: float,
    max_tilt: float,
    spherical_direction: ti.template(),
    state: ti.template(),
    mark: ti.template(),
):
    """Upstream's `fff3.init_spherical_direction_field_from_sdf`, plus overhangs.

    Angles in radians. ``ceil_max_angle`` plays upstream's ``CEIL_MAX_ANGLE``;
    ``overhang_min_angle`` is ``max_overhang`` (above pi/2 switches the rule
    off); ``max_tilt`` caps the overhang tilt. The first-layer, ceiling and
    outside branches are upstream's, line for line (without ``all_up``).
    """
    origin = ti.math.vec3(0.0, 0.0, 0.0)
    layer_height = fff3.layer_height_from_cell_sides_length(cell_sides_length)

    for i in ti.grouped(ti.ndrange(*state.shape)):
        cell_center_i = grid3.cell_center_point(i, origin, cell_sides_length)

        is_boundary_region_i = fff3.is_boundary_region(sdf[i], layer_height)
        is_outside_i = fff3.is_after_boundary(sdf[i])

        if is_boundary_region_i:
            n_i = solid3.sdf_compute_closest_normal_central(sdf, i, cell_sides_length)
            is_pointing_down = n_i.z < 0.0
            n_i_pointing_up = n_i
            if is_pointing_down:
                n_i_pointing_up = -n_i

            n_i_pointing_up_sph = direction.cartesian_to_spherical(n_i_pointing_up)
            closest_normal_angle = n_i_pointing_up_sph[0]
            is_ceiling = closest_normal_angle < ceil_max_angle and not is_pointing_down

            curvature = solid3.sdf_compute_closest_curvature_central(
                sdf, i, cell_sides_length
            )
            if cell_center_i.z < layer_height:
                spherical_direction[i] = ti.math.vec2(0.0, 0.0)
                state[i] = direction.constrain(state[i])
            elif curvature < fff3.CURVATURE_THRESHOLD:
                if is_ceiling:
                    spherical_direction[i] = n_i_pointing_up_sph
                    state[i] = direction.constrain(state[i])
                elif is_pointing_down:
                    # theta_geo: 0 for a wall, pi/2 for a flat ceiling.
                    theta_geo = 0.5 * ti.math.pi - closest_normal_angle
                    if theta_geo > overhang_min_angle:
                        wanted = theta_geo - overhang_min_angle + overhang_margin
                        tilt = ti.min(wanted, max_tilt)
                        spherical_direction[i] = ti.math.vec2(
                            tilt, ti.math.atan2(n_i.y, n_i.x)
                        )
                        state[i] = direction.constrain(state[i])
                        mark[i] = ti.u8(MARK_OVERHANG)
                        if wanted > max_tilt:
                            mark[i] = ti.u8(MARK_OVERHANG_CAPPED)
        elif is_outside_i:
            spherical_direction[i] = ti.math.vec2(ti.math.nan)


@ti.kernel
def restore_marked(spherical_direction: ti.template(), saved: ti.template(), mark: ti.template()):
    """Put every marked cell's saved direction back."""
    for i in ti.grouped(mark):
        if mark[i] != MARK_NONE:
            spherical_direction[i] = saved[i]


@ti.kernel
def mask_outside(sdf: ti.template(), spherical_direction: ti.template()):
    """NaN wherever ``sdf`` is outside: the cells upstream's initialisation masks."""
    for i in ti.grouped(sdf):
        if fff3.is_after_boundary(sdf[i]):
            spherical_direction[i] = ti.math.vec2(ti.math.nan)


def tilt_budget_rad(max_slope_deg: float) -> float:
    """Upstream's ``fff3.MAX_SLOPE_ANGLE``: ``max_slope`` capped by the nozzle."""
    return min(math.radians(max_slope_deg), (math.pi - toolpath3.NOZZLE_CONE_ANGLE) * 0.5)


@dataclass
class FieldResult:
    """The computed field and what the overhang rule did."""

    field: direction.SphericalField
    #: Cells given an overhang constraint.
    overhang_cells: int
    #: Of those, cells whose tilt the budget stopped short of the rule's.
    overhang_capped: int
    #: The field as initialised: the constraints, before any smoothing.
    initial: np.ndarray
    #: The level-0 field after the multigrid solve, before the final passes.
    after_multigrid: np.ndarray
    #: `MARK_*` per cell: which cells the overhang rule constrained.
    mark: np.ndarray


def compute_direction_field(
    sdf: solid3.SDF,
    max_slope_deg: float,
    overhang: Optional[OverhangSettings] = None,
    solid_sdf: Optional[solid3.SDF] = None,
    hold_overhang: bool = False,
) -> FieldResult:
    """Stage 4's field, with P2.2's options.

    ``sdf`` is the pipeline's SDF (infilled if the part has infill).
    ``overhang`` switches the overhang rule on. ``solid_sdf``, the SDF from
    before infill on the same grid, puts the field on the solid part, masked
    by ``sdf`` afterwards. ``hold_overhang`` re-applies the overhang
    constraints after each final smoothing pass. With all three off, this is
    upstream's stage 4 (without ``--ortho_to_wall`` and ``--allup``, which
    it does not support). Requires Taichi to be initialised.
    """
    source = sdf if solid_sdf is None else solid_sdf
    shape = tuple(int(n) for n in sdf.grid.cell_3dcount)
    if tuple(int(n) for n in source.grid.cell_3dcount) != shape:
        raise ValueError(
            f"the solid SDF's grid {tuple(source.grid.cell_3dcount)} is not the "
            f"pipeline SDF's {shape}"
        )
    cell = float(sdf.grid.cell_sides_length)
    budget = tilt_budget_rad(max_slope_deg)
    if overhang is None:
        min_angle, margin = _OVERHANG_OFF, 0.0
    else:
        min_angle = math.radians(overhang.max_overhang_deg)
        margin = math.radians(overhang.margin_deg)

    field = direction.SphericalField()
    field.grid = source.grid
    field.direction = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    field.state = ti.field(dtype=ti.u32, shape=shape)
    mark = ti.field(dtype=ti.u8, shape=shape)
    init_overhang_aware(
        source.sdf, cell, budget, min_angle, margin, budget,
        field.direction, field.state, mark,
    )
    initial = field.direction.to_numpy()
    saved = None
    if hold_overhang:
        saved = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        saved.copy_from(field.direction)

    aligner = direction.SphericalMultigridAligner()
    aligner.allocate_from_field(field)
    aligner.align()
    after_multigrid = aligner.direction[0][0].to_numpy()

    for _ in range(FINAL_SMOOTHING_PASSES):
        aligner.align_one_level_one_time(0, True)
        fff3.spherical_field_constrain_fisrt_layer_up(
            aligner.direction[0][0], aligner.multigrid.cell_sides_length[0]
        )
        if saved is not None:
            restore_marked(aligner.direction[0][0], saved, mark)

    final = aligner.direction[0][0]
    if final is not field.direction:
        field.direction.copy_from(final)
    if solid_sdf is not None:
        mask_outside(sdf.sdf, field.direction)

    marks = mark.to_numpy()
    return FieldResult(
        field=field,
        overhang_cells=int(np.count_nonzero(marks)),
        overhang_capped=int(np.count_nonzero(marks == MARK_OVERHANG_CAPPED)),
        initial=initial,
        after_multigrid=after_multigrid,
        mark=marks,
    )
