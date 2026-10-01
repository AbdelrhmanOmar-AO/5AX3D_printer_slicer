"""Build and view the reachable-tilt map (build plan P2.3).

Builds the map for the active machine profile (``ATOM_MACHINE``, default
``reference``) or loads it from ``data/reachability/<profile>.npz``, then
prints, for a few heights above the bed, how far the machine can tilt over
the bed: per grid point, the smallest over all directions of the largest
reachable tilt, with any platform and with none. With ``--part`` it places
that part on the bed as `toolpath_to_gcode` does and says how far every
point of its bounding box can tilt, toward each direction.

Not used by the field yet (the operator's decision, 2026-10-01: until the
real machine's geometry exists). About 20 s to build on a CPU. Examples::

    python tools/reachability_map.py
    python tools/reachability_map.py --part data/param/ramp60_s.json
    python tools/reachability_map.py --rebuild --heights 0 50 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom.ti_env import init_taichi  # noqa: E402


def _cell(value):
    return "  -" if np.isnan(value) else f"{value:3.0f}"


def bed_table(m, height_mm, without_lift, every=3):
    """Rows along x, columns along y: the worst direction's largest tilt."""
    k = int(np.argmin(np.abs(m.z_mm - height_mm)))
    worst = np.min(m.max_tilt_deg(without_lift)[:, :, k, :], axis=2)
    xs = range(0, len(m.x_mm), every)
    ys = range(0, len(m.y_mm), every)
    lines = [f"  z = {m.z_mm[k]:g} mm, {'no platform' if without_lift else 'any platform'}"
             f" (rows x, columns y, every {every * (m.x_mm[1] - m.x_mm[0]):g} mm; '-' reaches nothing)"]
    lines.append("     y " + "".join(f"{m.y_mm[j]:4.0f}" for j in ys))
    for i in xs:
        lines.append(f"  x{m.x_mm[i]:4.0f} " + "".join(" " + _cell(worst[i, j]) for j in ys))
    return lines


def part_summary(m, param_path):
    """How far each point of a part's bounding box can tilt, once placed on the bed."""
    import trimesh

    from atom import contracts, machine_profile

    params = json.loads(Path(param_path).read_text(encoding="utf-8"))
    mesh = trimesh.load_mesh(str(REPO_ROOT / "data" / "mesh" / f"{params['solid_name']}.stl"))
    mesh.apply_translation(-mesh.bounds[0])  # as process_for_atomizer.py does
    low, high = mesh.bounds
    box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    offset = contracts.bed_centering_offset(box, machine_profile.load_profile())
    lines = [f"{params['solid_name']}: bounding box {np.round(high - low, 1).tolist()} mm, "
             f"placed at {np.round(offset[:2], 1).tolist()} on the bed",
             "  direction (deg from +x): largest tilt every corner of the box reaches, any platform / no platform"]
    for azimuth in m.azimuths_deg:
        anywhere = np.nanmin(m.max_tilt_here(box, azimuth, offset_mm=offset))
        bare = np.nanmin(m.max_tilt_here(box, azimuth, without_lift=True, offset_mm=offset))
        lines.append(f"    {azimuth:5.1f}: {anywhere:4.1f} / {bare:4.1f}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="Build again even if a cached map exists.")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 50.0, 100.0],
                        help="Heights above the bed to show, mm (the nearest grid level).")
    parser.add_argument("--part", type=Path, default=None, help="A part's parameter file: show its bounding box's reach.")
    args = parser.parse_args(argv)

    init_taichi("cpu")
    from atom import reachability

    m = reachability.load_or_build(rebuild=args.rebuild)
    reach = m.reachable
    print(f"Reachable-tilt map, profile {m.profile_name} ({reachability.cache_path(m.profile_name)}): "
          f"{len(m.x_mm)} x {len(m.y_mm)} x {len(m.z_mm)} grid points, tilts 0-{m.tilts_deg[-1]:g} "
          f"in {m.tilts_deg[1] - m.tilts_deg[0]:g}-degree steps, {len(m.azimuths_deg)} directions; "
          f"clearance model {m.clearance_model}")
    print(f"  reachable (point, direction) pairs: {int(reach.sum())} of {reach.size}; "
          f"of them needing a platform: {int((reach & ~m.reachable_without_lift).sum())}; "
          f"largest platform asked: {np.nanmax(np.where(reach, m.lift_mm, np.nan)):.0f} mm")
    for height in args.heights:
        for without_lift in (False, True):
            print("\n".join(bed_table(m, height, without_lift)))
    if args.part is not None:
        print("\n".join(part_summary(m, args.part)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
