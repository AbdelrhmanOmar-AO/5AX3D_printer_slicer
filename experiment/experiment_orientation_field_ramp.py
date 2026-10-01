"""The orientation field next to a ramp's overhang, stock or overhang-aware (P2.1, P2.2).

Builds the exact SDF of a benchmark ramp (`atom.analytic_sdf`, no Blender),
optionally applies Atomizer's own infill kernel with the arguments
`tools/sdf_to_isdf.py` uses, and computes stage 4's field on the CPU with
`atom.orientation_field.compute_direction_field`: upstream's field when no
option is given (bit-identical to `tools/compute_tool_orientations.py`,
`tests/test_orientation_field.py`), or P2.2's with ``--overhang-aware``,
``--solid`` (the field on the pre-infill SDF, plan_corrections P2-8) and
``--hold`` / ``--no-hold`` (the overhang constraints held through the final
smoothing, on by default as in the pipeline) and ``--ramp`` / ``--no-ramp``
(P2.4's ramp-in, on by default, at ``--rate`` degrees per mm).

It reports which cells were constrained to a tilted direction and where, and
what the field does next to the underside: its tilt, the effective overhang
angle, and which way it leans (``lean_toward_overhang`` > 0 means toward).
`docs/orientation_field.md` sections 4 and 7 have results.

**Development only.** CPU, an analytic SDF instead of the remesh, field cells
instead of atoms: nothing from it is comparable with the lab baseline (plan
section 0 rule 10). About a minute for an ``xs`` ramp on four cores.

The machine profile is read when Atomizer's modules are imported, so choose it
with ``ATOM_MACHINE`` (``dev60`` for the 60-degree target), not afterwards.

Examples::

    python experiment/experiment_orientation_field_ramp.py 70 30 --infill
    python experiment/experiment_orientation_field_ramp.py 60 30 --infill --overhang-aware --solid
    ATOM_MACHINE=dev60 python experiment/experiment_orientation_field_ramp.py 80 60 --infill --overhang-aware --solid
    python experiment/experiment_orientation_field_ramp.py 70 30 --infill --overhang-aware --solid --no-hold
"""

# No `from __future__ import annotations`: this drives Taichi kernels.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import taichi as ti  # noqa: E402

ti.init(arch=ti.cpu, offline_cache_cleaning_policy="never")

import atom.fff3  # noqa: E402
import atom.solid3  # noqa: E402
from atom import analytic_sdf, grid3  # noqa: E402
from atom import benchmark_meshes as bm  # noqa: E402
from atom import orientation_field as of  # noqa: E402
from atom.overhang_field import OverhangSettings  # noqa: E402
from atom.ramp_in import RampSettings  # noqa: E402

#: Upstream's infill arguments in tools/sdf_to_isdf.py: offset 0, gyroid on.
INFILL_PERIOD = 8
SHELL_THICKNESS = 2


def as_sdf(array, cell):
    sdf = atom.solid3.SDF()
    sdf.grid = grid3.Grid()
    sdf.grid.cell_3dcount = np.array(array.shape)
    sdf.grid.origin = np.zeros(3)
    sdf.grid.cell_sides_length = cell
    sdf.sdf = ti.field(dtype=ti.f32, shape=array.shape)
    sdf.sdf.from_numpy(array)
    return sdf


def cartesian(spherical):
    theta, phi = spherical[..., 0], spherical[..., 1]
    return np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=-1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("angle", type=float, help="Ramp overhang angle, degrees from vertical.")
    parser.add_argument("max_slope", type=float, help="max_slope in degrees.")
    parser.add_argument("--infill", action="store_true", help="Apply the infill kernel first.")
    parser.add_argument("--overhang-aware", action="store_true", help="P2.2's overhang rule.")
    parser.add_argument("--solid", action="store_true", help="Compute the field on the pre-infill SDF.")
    parser.add_argument(
        "--hold", action=argparse.BooleanOptionalAction, default=True,
        help="Hold overhang constraints in the final passes (default on).",
    )
    parser.add_argument(
        "--ramp", action=argparse.BooleanOptionalAction, default=True,
        help="P2.4's ramp-in below overhangs (default on).",
    )
    parser.add_argument("--rate", type=float, default=3.0, help="Ramp-in rate, degrees per mm.")
    parser.add_argument(
        "--edges", action=argparse.BooleanOptionalAction, default=True,
        help="Carry the overhang constraint to its edges (plan_corrections P2-19, default on).",
    )
    parser.add_argument("--max-overhang", type=float, default=45.0)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--size", default="xs", choices=sorted(bm.SIZE_PRESETS))
    parser.add_argument("--save-sdf", type=Path, default=None)
    parser.add_argument("--save-direction", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="Also write the result here.")
    args = parser.parse_args()

    dims = bm.default_dimensions(bm.SIZE_PRESETS[args.size])
    length, depth, height = dims["length"], dims["depth"], dims["height"]
    cell = float(atom.fff3.cell_sides_length_from_deposition_width_kernel(0.9))
    solid, geometry = analytic_sdf.ramp_sdf(args.angle, length, depth, height, cell)

    sdf = as_sdf(solid, cell)
    if args.infill:
        source = ti.field(dtype=ti.f32, shape=solid.shape)
        source.from_numpy(solid)
        atom.fff3.sdf_generate_infill(
            source, sdf.sdf, cell, ti.math.vec3(0), INFILL_PERIOD, SHELL_THICKNESS, True
        )
    if args.save_sdf:
        sdf.save(str(args.save_sdf))

    result = of.compute_direction_field(
        sdf,
        args.max_slope,
        overhang=OverhangSettings(args.max_overhang, args.margin) if args.overhang_aware else None,
        solid_sdf=as_sdf(solid, cell) if args.solid else None,
        hold_overhang=args.hold,
        ramp=RampSettings(args.rate) if args.ramp else None,
        edges=args.edges,
    )
    final = result.field.direction.to_numpy()
    if args.save_direction:
        result.field.save(str(args.save_direction))

    # Where the tilted constraints are, and what the field does at the underside.
    shape = solid.shape
    xs = (np.arange(shape[0]) + 0.5) * cell
    ys = (np.arange(shape[1]) + 0.5) * cell
    zs = (np.arange(shape[2]) + 0.5) * cell
    Xc = np.broadcast_to(xs[:, None, None], shape)
    Yc = np.broadcast_to(ys[None, :, None], shape)
    Zc = np.broadcast_to(zs[None, None, :], shape)
    layer_height = float(atom.fff3.layer_height_from_cell_sides_length_kernel(cell))
    inside = solid < 0
    state = result.field.state.to_numpy()
    theta0 = np.degrees(result.initial[..., 0])
    phi0 = np.degrees(result.initial[..., 1])
    tilted = ((state & 1) == 1) & ~(Zc < layer_height) & (theta0 > 1.0)

    normal = geometry["normal"]
    above = -((Xc - geometry["column_width"]) * normal[0] + (Zc - geometry["base_height"]) * normal[2])
    along = (Xc - geometry["column_width"]) / geometry["run"]
    over_face = (along > 0.1) & (along < 0.9) & (Yc > 1.0) & (Yc < depth - 1.0)
    near = inside & over_face & (above > 0) & (above <= 0.9) & ~np.isnan(final[..., 0])
    tilted_above = tilted & over_face & (above > 0) & (above < 3.0)
    horizontal = np.array([normal[0], normal[1], 0.0])
    horizontal /= max(np.linalg.norm(horizontal), 1e-12)

    def at_underside(spherical):
        d = cartesian(spherical[near])
        tilt = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        effective = 90 - np.degrees(np.arccos(np.clip(d @ -normal, -1, 1)))
        return {
            "tilt_mean": float(tilt.mean()),
            "tilt_max": float(tilt.max()),
            "theta_eff_mean": float(effective.mean()),
            "theta_eff_max": float(effective.max()),
            "lean_toward_overhang": float((d @ horizontal).mean()),
        }

    output = {
        "ramp": args.angle,
        "size": args.size,
        "max_slope": args.max_slope,
        "tilt_budget": math.degrees(of.tilt_budget_rad(args.max_slope)),
        "infill": args.infill,
        "overhang_aware": args.overhang_aware,
        "solid": args.solid,
        "hold": bool(args.hold and args.overhang_aware),
        "edge_cells": result.edge_cells,
        "ramp_in": None if result.ramp is None else {
            "rate": args.rate,
            "cells": result.ramp.cells,
            "walks_used": result.ramp.walks_used,
            "walks_steepened": result.ramp.walks_steepened,
            "steepest_rate_deg_per_mm": result.ramp.steepest_rate_deg_per_mm,
        },
        "grid": list(shape),
        "overhang_cells": result.overhang_cells,
        "overhang_capped": result.overhang_capped,
        "tilted_constraints": int(tilted.sum()),
        "tilted_constraints_above_underside": {
            "count": int(tilted_above.sum()),
            "theta_mean": float(theta0[tilted_above].mean()) if tilted_above.any() else None,
            "phi_mean": float(phi0[tilted_above].mean()) if tilted_above.any() else None,
            "depth_mean_mm": float(-solid[tilted_above].mean()) if tilted_above.any() else None,
        },
        "cells_next_to_underside": int(near.sum()),
        "after_multigrid": at_underside(result.after_multigrid),
        "after_32_passes": at_underside(final),
        "max_tilt_anywhere": float(np.degrees(np.nanmax(final[..., 0][inside & ~np.isnan(final[..., 0])]))),
    }
    text = json.dumps(output, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
