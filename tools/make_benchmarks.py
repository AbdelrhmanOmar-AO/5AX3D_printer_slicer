"""Generate the benchmark meshes and their parameter files (build plan P0.7).

Writes an STL into ``data/mesh/`` and a matching parameter JSON into
``data/param/`` for every part, at every requested size.

Each part is generated at several sizes so the cost of the pipeline can be
measured against part volume rather than guessed. `order_atoms` dominates
runtime (86 % of logged stage time on the calibration cube, see
``tests/golden/baseline.md``) and scales with atom count, which scales with
volume, so the size sweep is itself a result.

Usage
-----
    python tools/make_benchmarks.py --dry-run          # print the plan only
    python tools/make_benchmarks.py                    # every part, every size
    python tools/make_benchmarks.py --sizes s m --parts ramp60 tshape
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom import benchmark_meshes as bm  # noqa: E402

#: Measured on the calibration cube at deposition_width 0.9 mm; see
#: tests/golden/baseline.md. 31 630 atoms for an 8 400 mm^3 part, and
#: order_atoms took 331.6 s for them. Used only to print an estimate, so the
#: operator can see what a run will cost before starting it.
ATOMS_PER_MM3 = 31630 / 8400.0
SECONDS_PER_ATOM = 331.6 / 31630.0

#: Every part this generator knows how to build.
PART_NAMES: tuple[str, ...] = (
    ("box",) + tuple(f"ramp{angle}" for angle in bm.RAMP_ANGLES) + ("tshape", "twin_domes")
)


def build_part(name: str, principal_mm: float, underside_angle_deg: float):
    """Build one named part at the given principal dimension."""
    dims = bm.default_dimensions(principal_mm)

    if name == "box":
        return bm.make_box(**dims)
    if name.startswith("ramp"):
        return bm.make_ramp(float(name[len("ramp"):]), **dims)
    if name == "tshape":
        return bm.make_tshape(underside_angle_deg=underside_angle_deg, **dims)
    if name == "twin_domes":
        return bm.make_twin_domes(
            base_width=dims["length"],
            depth=dims["depth"],
            base_height=dims["height"] * 0.4,
        )
    raise ValueError(f"unknown part: {name}")


def format_duration(seconds: float) -> str:
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.0f}min"
    return f"{seconds / 3600.0:.1f}h"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parts", nargs="+", default=list(PART_NAMES), choices=list(PART_NAMES),
        help="Which parts to generate. Default: all.",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=list(bm.SIZE_PRESETS),
        choices=list(bm.SIZE_PRESETS),
        help=(
            "Size presets to generate. Principal dimension in mm: "
            + ", ".join(f"{k}={v:g}" for k, v in bm.SIZE_PRESETS.items())
        ),
    )
    parser.add_argument(
        "--deposition-width", type=float, default=0.9,
        help="Deposition width in mm written into each parameter file.",
    )
    parser.add_argument(
        "--max-slope", type=float, default=7.0,
        help=(
            "Default max_slope in degrees for the parameter files. The P0.8 "
            "matrix overrides this per run."
        ),
    )
    parser.add_argument(
        "--underside-angle", type=float, default=90.0,
        help=(
            "Overhang angle of the T-shape crossbar underside, in degrees from "
            "vertical. 90 is a flat underside. GATE D0: placeholder, team "
            "decision pending."
        ),
    )
    parser.add_argument(
        "--infill", action="store_true", default=True,
        help="Write infill: true into the parameter files (default).",
    )
    parser.add_argument(
        "--no-infill", dest="infill", action="store_false",
        help="Write infill: false instead.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data",
        help="Root of the data directory to write into.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and the runtime estimates without writing files.",
    )
    args = parser.parse_args(argv)

    mesh_dir = args.data_dir / "mesh"
    param_dir = args.data_dir / "param"
    if not args.dry_run:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        param_dir.mkdir(parents=True, exist_ok=True)

    header = f"{'part':<14}{'size':>5}{'volume mm3':>13}{'atoms (est)':>13}{'order_atoms (est)':>20}"
    print(header)
    print("-" * len(header))

    total_seconds = 0.0
    for size in args.sizes:
        principal = bm.SIZE_PRESETS[size]
        for name in args.parts:
            mesh = build_part(name, principal, args.underside_angle)
            solid_name = f"{name}_{size}"

            atoms = mesh.volume * ATOMS_PER_MM3
            seconds = atoms * SECONDS_PER_ATOM
            total_seconds += seconds
            print(
                f"{name:<14}{size:>5}{mesh.volume:>13,.0f}{atoms:>13,.0f}"
                f"{format_duration(seconds):>20}"
            )

            if args.dry_run:
                continue

            mesh.export(mesh_dir / f"{solid_name}.stl")
            params = {
                "solid_name": solid_name,
                "deposition_width": args.deposition_width,
                "max_slope": args.max_slope,
                "infill": args.infill,
            }
            (param_dir / f"{solid_name}.json").write_text(
                json.dumps(params, indent=4) + "\n", encoding="utf-8"
            )

    print("-" * len(header))
    print(
        f"{len(args.parts) * len(args.sizes)} parts; estimated order_atoms time "
        f"in total: {format_duration(total_seconds)}"
    )
    print(
        "\nEstimates extrapolate the calibration cube linearly "
        "(tests/golden/baseline.md);\nordering is a nearest-neighbour problem, "
        "so treat them as a floor, not a prediction."
    )

    if args.dry_run:
        print("\n--dry-run: no files written.")
    else:
        print(f"\nWrote meshes to {mesh_dir} and parameters to {param_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
