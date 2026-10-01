"""Run the real pipeline stages on the CPU from a Blender-free, pipeline-like SDF (development only).

The cloud session has no Blender and no GPU, so stage 1 (Blender's voxel
remesh) cannot run there, and an exact analytic SDF (`atom.analytic_sdf`) has
cleaner edges than the pipeline's. This builds what stages 2 and 3 would see:
the STL subdivided to 0.12 mm triangles, written as an OBJ with vertex
normals averaged across edges (as Blender's remesh and smooth shading round
them), then runs the real `obj_to_bpn`, `bpn_to_sdf`, `sdf_to_isdf` and stages
4 to 9 on the CPU. The grid is the laptop's (178 x 81 x 107 for the `xs`
parts).

It reproduced the laptop where the analytic SDF could not: the `ramp90_xs`
deadlock in `order_atoms` at 30 degrees (plan_corrections P2-20). It did not
reproduce the laptop's single tip-corner atom on `ramp60_xs` (P2-19), so the
laptop stays the reference. Numbers from it are development numbers, not
comparable with the lab baseline (plan section 0 rule 10).

Usage::

    python experiment/pipeline_like_run.py ramp90_xs WORKDIR --max-slope 30 --overhang-aware
    python experiment/pipeline_like_run.py ramp60_xs WORKDIR --max-slope 30 --overhang-aware --stop-after extract_explicit_atoms

Outputs go under ``WORKDIR/data`` (never the repository's ``data/``). Then, for
example: ``python tools/overhang_where.py data/param/ramp60_xs.json --frame
WORKDIR/data/frame/ramp60_xs.npz``. About 15 minutes on four cores for an
`xs` part to stage 8, plus 5-10 for `order_atoms`.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES_AFTER_SDF = (
    "compute_tool_orientations", "sdf_df_to_layers", "compute_tangents",
    "align_atoms", "extract_explicit_atoms", "order_atoms",
)


def write_pipeline_like_obj(stl_path, obj_path, max_edge=0.12):
    """The STL, dense, with vertex normals averaged across its edges."""
    import trimesh

    mesh = trimesh.load_mesh(str(stl_path))
    mesh.apply_translation(-mesh.bounds[0])  # as process_for_atomizer.py does
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
    dense = trimesh.Trimesh(vertices, faces, process=True)  # merging vertices averages the normals
    with open(obj_path, "w", encoding="utf-8") as handle:
        for p in dense.vertices:
            handle.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for n in dense.vertex_normals:
            handle.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for a, b, c in dense.faces + 1:
            handle.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    return len(dense.vertices)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", help="The part's name, as in data/mesh/<part>.stl.")
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--max-slope", type=float, default=30.0)
    parser.add_argument("--deposition-width", type=float, default=0.9)
    parser.add_argument("--overhang-aware", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES_AFTER_SDF, default="order_atoms")
    parser.add_argument(
        "--stage4", nargs=argparse.REMAINDER, default=[],
        help="Extra options for compute_tool_orientations.py (e.g. --no_overhang_edges); last.",
    )
    args = parser.parse_args(argv)

    data = args.workdir.resolve() / "data"
    for sub in ("mesh", "point_normal", "sdf", "direction", "phasor", "basis", "triphasor", "frame", "toolpath", "log"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    part, slope = args.part, f"{args.max_slope:g}"
    path = lambda sub, suffix="": str(data / sub / f"{part}{suffix}.npz")  # noqa: E731
    obj = data / "mesh" / f"{part}.obj"
    log = str(data / "log" / f"{part}.log")
    print(f"{write_pipeline_like_obj(REPO_ROOT / 'data' / 'mesh' / f'{part}.stl', obj)} vertices", flush=True)

    solid = path("sdf", "_solid") if args.overhang_aware else path("sdf")
    stage4 = ["--overhang_aware", "--solid_sdf", solid] if args.overhang_aware else []
    commands = [
        ("obj_to_bpn", [str(obj), path("point_normal")]),
        ("bpn_to_sdf", [path("point_normal"), solid, f"{args.deposition_width:.2f}"]),
        ("sdf_to_isdf", [path("point_normal"), solid, path("sdf"), "no_gui=True"]),
        ("compute_tool_orientations", [path("sdf"), path("direction"), "--maxslope", slope, *stage4, *args.stage4, "--logpath", log]),
        ("sdf_df_to_layers", [path("sdf"), path("direction"), path("phasor"), "--maxslope", slope, "--logpath", log]),
        ("compute_tangents", [path("sdf"), path("direction"), path("basis"), "--maxslope", slope, "--logpath", log]),
        ("align_atoms", [path("sdf"), path("phasor"), path("basis"), path("triphasor"), "--maxslope", slope, "--logpath", log]),
        ("extract_explicit_atoms", [path("sdf"), path("triphasor"), path("frame"), "--logpath", log]),
        ("order_atoms", [path("sdf"), path("frame"), path("toolpath"), "--logpath", log]),
    ]
    env = dict(os.environ, ATOM_TI_ARCH="cpu", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")]))
    for name, arguments in commands:
        started = time.time()
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / f"{name}.py"), *arguments],
            cwd=REPO_ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace",
        )
        last = (result.stdout + result.stderr).strip().splitlines()[-1:] if result.returncode else []
        print(f"{name}: exit {result.returncode}, {time.time() - started:.0f} s", *last, flush=True)
        if result.returncode or name == args.stop_after:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
