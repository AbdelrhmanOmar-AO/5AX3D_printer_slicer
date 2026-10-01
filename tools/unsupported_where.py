"""Where does a part's unsupported deposition near its overhangs come from? (diagnostic, build plan P2.5)

`overhang_report.py` gives one number per run: the share of the deposition
points within two widths of an overhang that have neither the bed nor earlier
material in the support cone beneath them. In the first overhang-aware matrix
it was 0.8-4 % on most parts that otherwise passed (plan_corrections P2-18).
This sorts those points by cause and place:

* **Cause.** *Printed too early*: material does lie in the cone beneath the
  point, but the toolpath prints it later (an ordering problem). *Nothing
  beneath*: no deposition point lies in the cone at all, earlier or later (the
  bead hangs out further than the cone allows: a geometry or tilt problem).
* **Place.** How deep inside the part (distance to the nearest surface, in
  layer heights), which kind of surface is nearest (overhang, wall, top, bed),
  whether the point starts a bead run (the move into it is a travel), and its
  tilt.

Reads the toolpath the report measures (``data/toolpath/<part>_smoothed.npz``)
or one given with ``--toolpath``, e.g. an archived run in
``reports/toolpaths/``. Nothing is written. Examples::

    python tools/unsupported_where.py data/param/ramp50_xs.json --toolpath reports/toolpaths/ramp50_xs_ms30_aware.npz
    python tools/unsupported_where.py data/param/tshape_xs.json --toolpath WORKDIR/data/toolpath/tshape_xs_smoothed.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import overhang_report as orep  # noqa: E402
from atom import overhang_metrics as om  # noqa: E402
from atom.ti_env import init_taichi  # noqa: E402

#: The kinds of surface a point can be nearest, by the face's normal.
SURFACE_KINDS = ("overhang", "wall", "top", "bed")


def surface_kinds(triangles, bed_contact_height=0.5):
    """Each triangle's kind: overhang, wall, top (upward) or bed (the footprint)."""
    normals = om.unit_face_normals(triangles)
    centres = triangles.mean(axis=1)
    overhang = om.overhang_face_mask(normals, centres, bed_contact_height)
    bed = (normals[:, 2] < -1e-6) & ~overhang
    top = normals[:, 2] > 1e-6
    return np.where(overhang, 0, np.where(bed, 3, np.where(top, 2, 1)))


def classify(toolpath, triangles, sampled_normals, sampled_centres, deposition_width,
             cone_half_angle_deg=om.SUPPORT_CONE_HALF_ANGLE_DEG,
             search_heights=om.SUPPORT_SEARCH_HEIGHTS):
    """The unsupported points near overhangs, with their cause and place.

    ``triangles`` is the part's mesh; ``sampled_normals`` and
    ``sampled_centres`` its faces subdivided to one width, as the report
    samples them (`overhang_report.load_mesh_arrays`). Returns a dict of
    arrays, one entry per such point, in print order, plus the totals. Uses
    the report's own tests (`om.unsupported_deposition`,
    `om.points_near_overhangs`), so the points are exactly the ones counted.
    """
    count = int(np.asarray(toolpath.point_count).item())
    mask = om.deposition_mask(toolpath)
    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]
    directions = om.tool_directions(toolpath)[:count][mask]
    heights = np.asarray(toolpath.height[:count], dtype=np.float64)[mask]
    # The move into each deposition point: a travel when the point before it
    # in the full toolpath does not deposit, so the point starts a bead run.
    previous_deposits = np.r_[False, np.asarray(toolpath.travel_type[:count - 1]) == om.TRAVEL_TYPE_DEPOSITION]
    starts_run = ~previous_deposits[mask]

    overall = om.unsupported_deposition(toolpath, cone_half_angle_deg=cone_half_angle_deg, search_heights=search_heights)
    near = om.points_near_overhangs(points, sampled_normals, sampled_centres, orep.NEAR_OVERHANG_WIDTHS * deposition_width)
    chosen = np.flatnonzero(overall.unsupported & near)

    cos_half = np.cos(np.radians(cone_half_angle_deg))
    tree = cKDTree(points)
    later_support = np.zeros(len(chosen), dtype=bool)
    gap = np.full(len(chosen), -1, dtype=np.int64)
    for k, index in enumerate(chosen):
        neighbours = np.asarray(tree.query_ball_point(points[index], search_heights * heights[index]), dtype=np.int64)
        neighbours = neighbours[neighbours != index]
        if neighbours.size == 0:
            continue
        offsets = points[neighbours] - points[index]
        lengths = np.linalg.norm(offsets, axis=1)
        inside = (lengths > 1e-9) & ((offsets @ -directions[index]) >= cos_half * lengths)
        if inside.any():
            later_support[k] = True  # unsupported, so every point in its cone comes later
            gap[k] = int(neighbours[inside].min() - index)

    distances = om.triangle_distances(points[chosen], triangles) if len(chosen) else np.zeros((0, len(triangles)))
    kinds = surface_kinds(triangles)
    nearest = distances.argmin(axis=1) if len(chosen) else np.zeros(0, dtype=np.int64)
    return {
        "index": chosen,
        "point": points[chosen],
        "tilt_deg": om.tilt_from_vertical_deg(directions[chosen]) if len(chosen) else np.zeros(0),
        "printed_too_early": later_support,
        "order_gap": gap,
        "depth_mm": distances.min(axis=1) if len(chosen) else np.zeros(0),
        "nearest_kind": kinds[nearest] if len(chosen) else np.zeros(0, dtype=np.int64),
        "starts_run": starts_run[chosen],
        "deposition_count": len(points),
        "near_count": int(near.sum()),
        "unsupported_overall": int(overall.unsupported.sum()),
    }


def _share(mask, total):
    return f"{int(mask.sum())} ({100.0 * mask.sum() / total:.0f} %)" if total else "0"


def report(found, layer_height, worst=15):
    """The classification as text."""
    total = len(found["index"])
    lines = [
        f"{found['deposition_count']} deposition points, {found['near_count']} near overhangs, "
        f"{total} of those unsupported ({100.0 * total / max(found['near_count'], 1):.2f} %); "
        f"unsupported overall {found['unsupported_overall']} "
        f"({100.0 * found['unsupported_overall'] / max(found['deposition_count'], 1):.2f} %)",
    ]
    if not total:
        return "\n".join(lines)
    early = found["printed_too_early"]
    lines += [
        "",
        "Cause:",
        f"  printed too early (material in its cone, printed later): {_share(early, total)}",
        f"  nothing beneath (no material in its cone at all):        {_share(~early, total)}",
    ]
    if early.any():
        gaps = found["order_gap"][early]
        lines.append(
            f"  printed too early: the first point in its cone comes {int(np.median(gaps))} points later "
            f"(median; 10-90 %: {int(np.percentile(gaps, 10))}-{int(np.percentile(gaps, 90))})"
        )
    lines += ["", "Nearest surface:"]
    for code, name in enumerate(SURFACE_KINDS):
        m = found["nearest_kind"] == code
        if m.any():
            lines.append(f"  {name:8s} {_share(m, total)}: too early {int((m & early).sum())}, nothing beneath {int((m & ~early).sum())}")
    lines += ["", "Depth (distance to the nearest surface, in layer heights):"]
    depth = found["depth_mm"] / layer_height
    edges = [0.0, 1.0, 2.0, 3.0, 4.0, np.inf]
    for low, high in zip(edges[:-1], edges[1:]):
        m = (depth >= low) & (depth < high)
        if m.any():
            label = f"{low:g}-{high:g}" if np.isfinite(high) else f"{low:g}+"
            lines.append(f"  {label:5s} {_share(m, total)}: too early {int((m & early).sum())}, nothing beneath {int((m & ~early).sum())}")
    starts = found["starts_run"]
    lines += [
        "",
        f"Starting a bead run (a travel moves into it): {_share(starts, total)}",
        f"Tilt: median {np.median(found['tilt_deg']):.1f}, max {found['tilt_deg'].max():.1f} degrees",
        "",
        f"First {min(worst, total)} in print order:  x, y, z (mm) | tilt | depth (mm) | nearest | cause",
    ]
    for k in range(min(worst, total)):
        x, y, z = found["point"][k]
        cause = f"too early (+{found['order_gap'][k]})" if early[k] else "nothing beneath"
        lines.append(
            f"  {x:7.2f} {y:7.2f} {z:7.2f} | {found['tilt_deg'][k]:5.1f} | {found['depth_mm'][k]:4.2f} | "
            f"{SURFACE_KINDS[found['nearest_kind'][k]]:8s} | {cause}{' | run start' if starts[k] else ''}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("param_path", type=Path, help="The part's parameter file.")
    parser.add_argument("--toolpath", type=Path, default=None, help="The toolpath to read (default data/toolpath/<part>_smoothed.npz).")
    parser.add_argument("--worst", type=int, default=15, help="How many points to list.")
    args = parser.parse_args(argv)

    params = json.loads(args.param_path.read_text(encoding="utf-8"))
    part = params["solid_name"]
    width = float(params["deposition_width"])
    path = args.toolpath or REPO_ROOT / "data" / "toolpath" / f"{part}_smoothed.npz"
    if not Path(path).is_file():
        raise SystemExit(f"No toolpath at {path}.")
    init_taichi("cpu")
    from atom import toolpath3

    toolpath = toolpath3.Toolpath()
    toolpath.load(str(path))
    mesh, normals, centres, _ = orep.load_mesh_arrays(REPO_ROOT / "data" / "mesh" / f"{part}.stl", max_edge=width)
    found = classify(toolpath, np.asarray(mesh.triangles, dtype=np.float64), normals, centres, width)
    print(f"{part}: {path}")
    print(report(found, orep.LAYER_HEIGHT_WRT_DEPOSITION_WIDTH * width, args.worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
