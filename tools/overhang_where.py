"""Where on a part are the worst overhang points? (diagnostic, build plan P2)

`overhang_report.py` gives one worst effective overhang angle per part. This
says where it comes from: for each overhang surface of the mesh, the
deposition points near it that are out over the air (the ones the metric
counts, metrics version 3) and the ones printed onto material or the bed
(skipped), then the worst points with their position and tilt.

Reads the atoms of a field-only run (``data/frame/<part>.npz``) by default, or
a toolpath with ``--toolpath``. Nothing is written. Examples::

    python tools/overhang_where.py data/param/ramp60_xs.json
    python tools/overhang_where.py data/param/ramp60_xs.json --frame reports/frames/ramp60_xs_ms30_aware.npz
    python tools/overhang_where.py data/param/ramp60_xs.json --toolpath data/toolpath/ramp60_xs_smoothed.npz
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
from atom import frame_atoms  # noqa: E402
from atom import overhang_metrics as om  # noqa: E402
from atom.ti_env import init_taichi  # noqa: E402


def load_points(args, part):
    if args.toolpath:
        init_taichi("cpu")
        from atom import toolpath3

        toolpath = toolpath3.Toolpath()
        toolpath.load(str(args.toolpath))
        return toolpath
    frame = Path(args.frame or REPO_ROOT / "data" / "frame" / f"{part}.npz")
    if not frame.is_file():
        raise SystemExit(f"No atoms at {frame}. Run a field-only report first, or give --frame.")
    return frame_atoms.load_frame_atoms(frame)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("param_path", type=Path, help="The part's parameter file.")
    parser.add_argument("--frame", type=Path, default=None, help="Atoms to read (default data/frame/<part>.npz).")
    parser.add_argument("--toolpath", type=Path, default=None, help="A toolpath to read instead of atoms.")
    parser.add_argument("--worst", type=int, default=10, help="How many of the worst points to list.")
    args = parser.parse_args(argv)

    params = json.loads(args.param_path.read_text(encoding="utf-8"))
    part = params["solid_name"]
    width = float(params["deposition_width"])
    stl = REPO_ROOT / "data" / "mesh" / f"{part}.stl"
    mesh, normals, centres, _ = orep.load_mesh_arrays(stl, max_edge=width)
    source = load_points(args, part)

    count = int(np.asarray(source.point_count).item())
    keep = om.deposition_mask(source)
    points = np.asarray(source.point[:count], dtype=np.float64)[keep]
    directions = om.tool_directions(source)[:count][keep]
    tilts = om.tilt_from_vertical_deg(directions)
    radius = orep.SURFACE_SEARCH_WIDTHS * width
    tree = cKDTree(points)

    overhang = om.overhang_face_mask(normals, centres)
    geometric = np.round(om.geometric_overhang_angle_deg(normals), 1)
    print(f"{part}: {len(points)} deposition points, mesh bounds {np.round(mesh.bounds, 2).tolist()}")
    for angle in sorted(set(geometric[overhang]), reverse=True):
        faces = np.flatnonzero(overhang & (geometric == angle))
        samples = {}
        for f in faces:
            for i in tree.query_ball_point(centres[f], radius):
                e = float(om.effective_overhang_angle_deg(normals[f][None], directions[i:i + 1])[0])
                samples[i] = max(samples.get(i, -90.0), e)
        if not samples:
            print(f"\nsurface {angle:g} deg: no points near it")
            continue
        index = np.fromiter(samples, dtype=np.int64)
        effective = np.array([samples[i] for i in index])
        air = om.points_over_air(points[index], directions[index], mesh.triangles, width / 2.0)

        print(f"\nsurface {angle:g} deg ({len(faces)} faces)")
        for name, m in (("over air (counted)", air), ("onto material (skipped)", ~air)):
            if m.any():
                e = effective[m]
                print(
                    f"  {name}: {m.sum()} points, worst {e.max():.1f}, mean {e.mean():.1f}, "
                    f"p90 {np.percentile(e, 90):.1f}, over 45: {(e > 45).sum()}"
                )
            else:
                print(f"  {name}: none")
        if air.any():
            worst = np.argsort(effective[air])[::-1][: args.worst]
            chosen = index[air][worst]
            print(f"  worst {len(chosen)} over air:  theta_eff | x, y, z (mm) | tilt")
            for i, e in zip(chosen, effective[air][worst]):
                x, y, z = points[i]
                print(f"    {e:5.1f} | {x:7.2f} {y:7.2f} {z:7.2f} | {tilts[i]:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
