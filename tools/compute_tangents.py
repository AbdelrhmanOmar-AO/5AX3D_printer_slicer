import argparse
import time
from math import pi

import taichi as ti

from atom.ti_env import init_taichi

import atom.basis3
import atom.direction
import atom.fff3
import atom.line
import atom.solid3
import atom.toolpath3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def compute_tangents():
    parser = argparse.ArgumentParser(
        description="Compute the deposition tangent field from the signed distance field (SDF) and the tool orientation field. See Section 4.2 in Chermain et al. 2025 for more details."
    )
    parser.add_argument("sdf_path", help="The path to the input SDF.")
    parser.add_argument(
        "df_path",
        help="The path to the input direction field (DF), each direction representing a tool orientation.",
    )
    parser.add_argument(
        "bf_path",
        help="The path to the output basis field. A basis here is orthonormal, composed of the input tool orientation and the output tangent direction.",
    )
    parser.add_argument(
        "--top_lines",
        type=str,
        help="The path to the png containing the discretized line 2D field for the top layers. If not provided, no constraints are applied. Otherwise, the top layers will be constrained by the 2D line field.",
    )
    parser.add_argument(
        "--bottom_lines",
        type=str,
        help="The path to the png containing the discretized line 2D field for the bottom layers. If not provided, no constraints are applied. Otherwise, the bottom layers will be constrained by the 2D line field.",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file",
    )
    parser.add_argument(
        "--maxslope",
        type=float,
        help="Machine maximum slope, i.e., the maximum bed slope.",
    )

    args = parser.parse_args()

    sdf_path = args.sdf_path
    direction_field_path = args.df_path
    basis3_path = args.bf_path

    top_line_path = args.top_lines
    bottom_line_path = args.bottom_lines

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

    top_line = atom.line.Line2Field2()
    if top_line_path is not None:
        top_line.create_from_png_1channel_8bits(top_line_path)
    else:
        top_line.create_from_png_1channel_8bits("data/image/0.png")
        # To deactivate the constraints on the top layer
        top_line.angle[0, 0] = ti.math.nan

    bottom_line = atom.line.Line2Field2()
    if bottom_line_path is not None:
        bottom_line.create_from_png_1channel_8bits(bottom_line_path)
    else:
        bottom_line.create_from_png_1channel_8bits("data/image/0.png")
        # To deactivate the constraints on the bottom layer
        bottom_line.angle[0, 0] = ti.math.nan

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)
    print(f"Grid sides cell count: {sdf.grid.cell_3dcount}")

    direction_field = atom.direction.SphericalField()
    direction_field.load(direction_field_path)

    basis3_field = atom.basis3.Field()
    sdf.allocate_basis3_field(
        direction_field.direction,
        basis3_field,
    )

    # Warmup
    atom.fff3.init_deposition_tangent_field(
        sdf.sdf,
        basis3_field.normal,
        top_line.angle,
        bottom_line.angle,
        basis3_field.grid.cell_sides_length,
        basis3_field.phi_t,
        basis3_field.state,
    )

    t0 = time.perf_counter()
    atom.fff3.init_deposition_tangent_field(
        sdf.sdf,
        basis3_field.normal,
        top_line.angle,
        bottom_line.angle,
        basis3_field.grid.cell_sides_length,
        basis3_field.phi_t,
        basis3_field.state,
    )

    basis3_field_aligner = atom.basis3.FieldAligner()
    basis3_field_aligner.allocate_from_field(basis3_field)
    basis3_field_aligner.align(
        atom.fff3.basis3_field_restrict,
        atom.fff3.basis3_field_prolong,
        atom.fff3.basis3_field_align,
        atom.fff3.ALIGNMENT_ITERATION_COUNT,
    )

    t1 = time.perf_counter()
    duration = t1 - t0

    str_to_print = f"Tangent computation took {duration:.1f} seconds."
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()

    basis3_field.save(basis3_path)

    ti.reset()


if __name__ == "__main__":
    compute_tangents()
