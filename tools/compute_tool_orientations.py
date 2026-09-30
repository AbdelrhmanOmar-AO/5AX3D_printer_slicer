import argparse
import time
from math import pi

import taichi as ti

from atom.ti_env import init_taichi

import atom.direction
import atom.fff3
import atom.orientation_field
import atom.solid3
import atom.toolpath3
from atom.overhang_field import (
    DEFAULT_MARGIN_DEG,
    DEFAULT_MAX_OVERHANG_DEG,
    OverhangSettings,
)

init_taichi("gpu", debug=False, offline_cache_cleaning_policy="never")


def compute_tool_orientation():
    parser = argparse.ArgumentParser(
        description="Compute the tool orientation field from the signed distance field and the user-defined parameters. See Chermain et al. 2025, Section 4.1, for more details."
    )
    parser.add_argument(
        "sdf_path",
        help="The path to the input signed distance field (SDF).",
    )
    parser.add_argument(
        "df_path",
        help="The path to store the output, i.e., the tool orientation field, also known as the direction field (DF).",
    )
    parser.add_argument(
        "--ortho_to_wall",
        action="store_true",
        help="Constrain layers to be orthogonal to the walls. This is described in Section 6, paragraph Strata orthogonal to wall, in Chermain et al. (2025). This is an experimental feature, and in general, it is better not to activate it to avoid over-constraining the system.",
    )
    parser.add_argument(
        "--allup",
        action="store_true",
        help="Force all the toolpath normals to point upwards (0, 0, 1). This is useful for testing the atomizer with a 3-axis filament printer.",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file.",
    )
    parser.add_argument(
        "--maxslope",
        type=float,
        help="Set the maximum angle (in degrees) that the bed is allowed to tilt. This parameter is also referred to as the maximum allowable tool inclination in Chermain et al. (2025); it is denoted by $\Theta$, and its domain is $[0, \pi / 2]$. For more details on how this parameter is used to constrain the tool orientation, please see Section 4.1, paragraph Tool orientation constraints.",
    )
    # Build plan P2.2 (this fork). Without these options the stage runs
    # upstream's code below, unchanged; with them, atom.orientation_field.
    parser.add_argument(
        "--overhang_aware",
        action="store_true",
        help="This fork, build plan P2.2: constrain downward-facing surfaces steeper than --max_overhang to lean toward the overhang (docs/orientation_field.md).",
    )
    parser.add_argument(
        "--max_overhang",
        type=float,
        default=None,
        help=f"With --overhang_aware: the steepest overhang left unconstrained, degrees from vertical (default {DEFAULT_MAX_OVERHANG_DEG:g}, a gate D1 placeholder).",
    )
    parser.add_argument(
        "--overhang_margin",
        type=float,
        default=None,
        help=f"With --overhang_aware: extra tilt, degrees (default {DEFAULT_MARGIN_DEG:g}, a gate D1 placeholder).",
    )
    parser.add_argument(
        "--hold_overhang",
        action="store_true",
        help="With --overhang_aware: re-apply the overhang constraints after each of the 32 final smoothing passes, as upstream does for the first layer.",
    )
    parser.add_argument(
        "--solid_sdf",
        type=str,
        default=None,
        help="This fork, plan_corrections P2-8 (a): the part's SDF from before infill. The field is computed on it, then masked where the main (infilled) SDF is outside, so the infill's inner surfaces do not act as ceilings.",
    )

    args = parser.parse_args()
    check_overhang_arguments(args)

    sdf_path = args.sdf_path
    ortho_to_wall = args.ortho_to_wall
    allup = args.allup
    df_path = args.df_path
    log_path = args.logpath

    atom.fff3.MACHINE_MAX_SLOPE_ANGLE = args.maxslope * ti.math.pi / 180.0
    atom.fff3.MAX_SLOPE_ANGLE = min(
        atom.fff3.MACHINE_MAX_SLOPE_ANGLE, (pi - atom.toolpath3.NOZZLE_CONE_ANGLE) * 0.5
    )
    atom.fff3.CEIL_MAX_ANGLE = atom.fff3.MAX_SLOPE_ANGLE
    # atom.fff3.FLOOR_MAX_ANGLE = atom.fff3.MAX_SLOPE_ANGLE
    atom.fff3.FLOOR_MAX_ANGLE = 1.0 * ti.math.pi / 180.0

    if log_path is not None:
        log_file = open(log_path, "a", encoding="utf-8")

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)

    str_to_print = f"Grid cell 3D count: {sdf.grid.cell_3dcount}"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    str_to_print = (
        f"Max slope angle: {atom.fff3.MAX_SLOPE_ANGLE / pi * 180:.1f} degrees"
    )
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    if args.overhang_aware or args.solid_sdf is not None:
        compute_overhang_aware(args, sdf, df_path, log_file if log_path is not None else None)
        return

    direction_field = atom.direction.SphericalField()
    init_func = atom.fff3.init_spherical_direction_field_from_sdf
    # Warmup
    sdf.init_spherical_direction_field(direction_field, init_func, allup)
    t0 = time.perf_counter()
    sdf.init_spherical_direction_field(direction_field, init_func, allup)

    if not allup:
        direction_multigrid = atom.direction.SphericalMultigridAligner()
        direction_multigrid.allocate_from_field(direction_field)
        direction_multigrid.align()

    if ortho_to_wall and not allup:
        atom.fff3.orthogonolize_direction_wall_region(
            sdf.sdf,
            sdf.grid.cell_sides_length,
            direction_multigrid.direction[0][0],
            direction_multigrid.state[0],
        )
        direction_multigrid.align()

    # Smooth everything - including constraints - except the first layer.
    # For more details, see Chermain et al. (2025), Section 6, paragraph titled Implementation details.
    for _ in range(32):
        direction_multigrid.align_one_level_one_time(0, True)
        atom.fff3.spherical_field_constrain_fisrt_layer_up(
            direction_multigrid.direction[0][0],
            direction_multigrid.multigrid.cell_sides_length[0],
        )

    t1 = time.perf_counter()
    duration = t1 - t0

    str_to_print = f"Direction computation took {duration:.1f} seconds"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()

    direction_field.save(df_path)

    ti.reset()


def check_overhang_arguments(args):
    """Refuse this fork's options in combinations they do not support.

    Checked before anything runs: without --overhang_aware and --solid_sdf the
    stage takes upstream's path, which would ignore the rule's options.
    """
    if not args.overhang_aware and (
        args.max_overhang is not None or args.overhang_margin is not None or args.hold_overhang
    ):
        raise SystemExit("--max_overhang, --overhang_margin and --hold_overhang need --overhang_aware.")
    if (args.overhang_aware or args.solid_sdf is not None) and (args.ortho_to_wall or args.allup):
        raise SystemExit(
            "--ortho_to_wall and --allup are not supported with --overhang_aware "
            "or --solid_sdf."
        )


def compute_overhang_aware(args, sdf, df_path, log_file):
    """Build plan P2.2: the field through `atom.orientation_field`.

    Its options off, `compute_direction_field` reproduces upstream's steps
    above bit for bit (tests/test_orientation_field.py); `--ortho_to_wall`
    and `--allup` are not supported there.
    """
    settings = None
    if args.overhang_aware:
        settings = OverhangSettings(
            DEFAULT_MAX_OVERHANG_DEG if args.max_overhang is None else args.max_overhang,
            DEFAULT_MARGIN_DEG if args.overhang_margin is None else args.overhang_margin,
        )
    solid = None
    if args.solid_sdf is not None:
        solid = atom.solid3.SDF()
        solid.load(args.solid_sdf)

    t0 = time.perf_counter()
    result = atom.orientation_field.compute_direction_field(
        sdf, args.maxslope, overhang=settings, solid_sdf=solid, hold_overhang=args.hold_overhang
    )
    duration = time.perf_counter() - t0

    lines = []
    if settings is not None:
        lines.append(
            f"Overhang constraints: {result.overhang_cells} cells, "
            f"{result.overhang_capped} of them capped by the tilt budget "
            f"(max overhang {settings.max_overhang_deg:g} degrees, margin "
            f"{settings.margin_deg:g}, hold {'on' if args.hold_overhang else 'off'})"
        )
    if solid is not None:
        lines.append(f"Field computed on the solid SDF {args.solid_sdf}, masked by the infilled one")
    lines.append(f"Direction computation took {duration:.1f} seconds")
    for line in lines:
        print(line)
        if log_file is not None:
            log_file.write(line + "\n")
    if log_file is not None:
        log_file.close()

    result.field.save(df_path)
    ti.reset()


if __name__ == "__main__":
    compute_tool_orientation()
