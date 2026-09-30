"""Where stock Atomizer's tilt near an overhang comes from (build plan P2.1).

Builds the exact signed distance field of a benchmark ramp (no Blender, no
point cloud), optionally applies Atomizer's own infill kernel with the
arguments `tools/sdf_to_isdf.py` uses, then runs the steps of
`tools/compute_tool_orientations.py` on the CPU, keeping the constraints set
before any smoothing. It reports which cells were constrained to a tilted
direction, where they are, and what the finished field does next to the
underside. `docs/orientation_field.md` section 4 has the results and what
they mean; plan_corrections 7c P2-3 the finding.

**Development only.** CPU backend, an analytic SDF instead of the Blender
remesh, and field cells instead of extracted atoms: no number from it is
comparable with the lab baseline (plan section 0 rule 10). About a minute for
an `xs` ramp on four cores.

Examples::

    python experiment/experiment_orientation_field_ramp.py 70 30 --infill
    python experiment/experiment_orientation_field_ramp.py 70 30
    python experiment/experiment_orientation_field_ramp.py 70 30 --infill \\
        --save-sdf sdf.npz --save-direction direction.npz

``--save-sdf`` writes the SDF in the format of ``data/sdf/<part>.npz``, so
`tools/compute_tool_orientations.py` can be run on it and compared with
``--save-direction``.
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

import atom.direction  # noqa: E402
import atom.fff3  # noqa: E402
import atom.solid3  # noqa: E402
import atom.toolpath3  # noqa: E402
from atom import benchmark_meshes as bm  # noqa: E402
from atom import grid3  # noqa: E402

#: Upstream's infill arguments in tools/sdf_to_isdf.py: offset 0, gyroid on.
INFILL_PERIOD = 8
SHELL_THICKNESS = 2


def ramp_profile(angle_deg, length, height):
    """The xz profile of `benchmark_meshes.make_ramp`, with its geometry."""
    angle = math.radians(angle_deg)
    base_height = height * 0.25
    run = length * 0.30
    rise = run / math.tan(angle) if angle_deg < 90 else 0.0
    column_width = length - run
    profile = np.array(
        [
            [0.0, 0.0],
            [column_width, 0.0],
            [column_width, base_height],
            [length, base_height + rise],
            [length, height],
            [0.0, height],
        ]
    )
    return profile, column_width, base_height, run


def polygon_sdf(px, pz, vertices):
    """Signed distance to a closed polygon, negative inside (vectorised)."""
    d = (px - vertices[0, 0]) ** 2 + (pz - vertices[0, 1]) ** 2
    sign = np.ones_like(px)
    j = len(vertices) - 1
    for i in range(len(vertices)):
        ex, ez = vertices[j] - vertices[i]
        wx, wz = px - vertices[i, 0], pz - vertices[i, 1]
        t = np.clip((wx * ex + wz * ez) / (ex * ex + ez * ez), 0.0, 1.0)
        d = np.minimum(d, (wx - ex * t) ** 2 + (wz - ez * t) ** 2)
        above = pz >= vertices[i, 1]
        below = pz < vertices[j, 1]
        left = ex * wz > ez * wx
        sign = np.where((above & below & left) | (~above & ~below & ~left), -sign, sign)
        j = i
    return sign * np.sqrt(d)


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
    parser.add_argument("--size", default="xs", choices=sorted(bm.SIZE_PRESETS))
    parser.add_argument("--save-sdf", type=Path, default=None)
    parser.add_argument("--save-direction", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="Also write the result here.")
    args = parser.parse_args()

    dims = bm.default_dimensions(bm.SIZE_PRESETS[args.size])
    length, depth, height = dims["length"], dims["depth"], dims["height"]
    profile, column_width, base_height, run = ramp_profile(args.angle, length, height)

    cell = float(atom.fff3.cell_sides_length_from_deposition_width_kernel(0.9))
    layer_height = float(atom.fff3.layer_height_from_cell_sides_length_kernel(cell))
    count = np.ceil(np.array([length, depth, height]) / cell).astype(int)
    xs = (np.arange(count[0]) + 0.5) * cell
    ys = (np.arange(count[1]) + 0.5) * cell
    zs = (np.arange(count[2]) + 0.5) * cell

    # The extruded profile: a 2D polygon distance and the y slab, combined.
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    in_profile = polygon_sdf(X, Z, profile)[:, None, :]
    in_slab = (np.abs(ys - depth / 2) - depth / 2)[None, :, None]
    solid = np.minimum(np.maximum(in_profile, in_slab), 0.0) + np.sqrt(
        np.maximum(in_profile, 0.0) ** 2 + np.maximum(in_slab, 0.0) ** 2
    )
    solid = solid.astype(np.float32)

    sdf = atom.solid3.SDF()
    sdf.grid = grid3.Grid()
    sdf.grid.cell_3dcount = count
    sdf.grid.origin = np.zeros(3)
    sdf.grid.cell_sides_length = cell
    sdf.sdf = ti.field(dtype=ti.f32, shape=tuple(count))
    sdf.sdf.from_numpy(solid)
    if args.infill:
        source = ti.field(dtype=ti.f32, shape=tuple(count))
        source.from_numpy(solid)
        atom.fff3.sdf_generate_infill(
            source, sdf.sdf, cell, ti.math.vec3(0), INFILL_PERIOD, SHELL_THICKNESS, True
        )
    if args.save_sdf:
        sdf.save(str(args.save_sdf))

    # tools/compute_tool_orientations.py, the same steps in the same order.
    atom.fff3.MACHINE_MAX_SLOPE_ANGLE = args.max_slope * ti.math.pi / 180.0
    atom.fff3.MAX_SLOPE_ANGLE = min(
        atom.fff3.MACHINE_MAX_SLOPE_ANGLE, (math.pi - atom.toolpath3.NOZZLE_CONE_ANGLE) * 0.5
    )
    atom.fff3.CEIL_MAX_ANGLE = atom.fff3.MAX_SLOPE_ANGLE
    atom.fff3.FLOOR_MAX_ANGLE = 1.0 * ti.math.pi / 180.0

    field = atom.direction.SphericalField()
    sdf.init_spherical_direction_field(
        field, atom.fff3.init_spherical_direction_field_from_sdf, False
    )
    initial = field.direction.to_numpy()
    state = field.state.to_numpy()

    aligner = atom.direction.SphericalMultigridAligner()
    aligner.allocate_from_field(field)
    aligner.align()
    after_align = aligner.direction[0][0].to_numpy()
    for _ in range(32):
        aligner.align_one_level_one_time(0, True)
        atom.fff3.spherical_field_constrain_fisrt_layer_up(
            aligner.direction[0][0], aligner.multigrid.cell_sides_length[0]
        )
    final = aligner.direction[0][0].to_numpy()
    if args.save_direction:
        field.direction.from_numpy(final)
        field.save(str(args.save_direction))

    # Where the tilted constraints are, and what the field does at the underside.
    shape = solid.shape
    Xc = np.broadcast_to(xs[:, None, None], shape)
    Yc = np.broadcast_to(ys[None, :, None], shape)
    Zc = np.broadcast_to(zs[None, None, :], shape)
    inside = solid < 0
    theta0 = np.degrees(initial[..., 0])
    phi0 = np.degrees(initial[..., 1])
    tilted = ((state & 1) == 1) & ~(Zc < layer_height) & (theta0 > 1.0)

    angle = math.radians(args.angle)
    normal = np.array([math.cos(angle), 0.0, -math.sin(angle)])  # the underside's
    above = -((Xc - column_width) * normal[0] + (Zc - base_height) * normal[2])
    along = (Xc - column_width) / run
    over_face = (along > 0.1) & (along < 0.9) & (Yc > 1.0) & (Yc < depth - 1.0)
    near = inside & over_face & (above > 0) & (above <= 0.9)
    tilted_above = tilted & over_face & (above > 0) & (above < 3.0)

    def at_underside(spherical):
        d = cartesian(spherical[near])
        tilt = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        effective = 90 - np.degrees(np.arccos(np.clip(d @ -normal, -1, 1)))
        return {
            "tilt_mean": float(tilt.mean()),
            "tilt_max": float(tilt.max()),
            "theta_eff_mean": float(effective.mean()),
            "theta_eff_max": float(effective.max()),
        }

    result = {
        "ramp": args.angle,
        "size": args.size,
        "max_slope": args.max_slope,
        "infill": args.infill,
        "grid": count.tolist(),
        "tilted_constraints": int(tilted.sum()),
        "tilted_constraints_above_underside": {
            "count": int(tilted_above.sum()),
            "theta_mean": float(theta0[tilted_above].mean()) if tilted_above.any() else None,
            "phi_mean": float(phi0[tilted_above].mean()) if tilted_above.any() else None,
            "depth_mean_mm": float(-solid[tilted_above].mean()) if tilted_above.any() else None,
        },
        "cells_next_to_underside": int(near.sum()),
        "after_multigrid": at_underside(after_align),
        "after_32_passes": at_underside(final),
        "max_tilt_anywhere": float(np.degrees(np.nanmax(final[..., 0][inside]))),
    }
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
