import argparse
import time
from math import pi

import taichi as ti

from atom.ti_env import init_taichi

import atom.direction
import atom.fff3
import atom.phasor3
import atom.solid3
import atom.toolpath3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def sdf_df_to_layers():
    parser = argparse.ArgumentParser(
        description="Compute the implicit layers of the input 3D solid."
    )
    parser.add_argument(
        "sdf_path",
        help="The path to the input signed distance field (SDF) representing the solid.",
    )
    parser.add_argument(
        "df_path",
        help="The path to the input direction field (DF), i.e., the tool orientation field. The tool orientations are the layers' normals.",
    )
    parser.add_argument(
        "layer_path",
        help="The path to the output implicit layers, represented with a phasor field.",
    )
    parser.add_argument(
        "--allup",
        action="store_true",
        help="Force all the toolpath normal to be up (0, 0, 1).",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file.",
    )
    parser.add_argument(
        "--maxslope",
        type=float,
        help="Bed maximum slope.",
    )

    args = parser.parse_args()

    sdf_path = args.sdf_path
    df_path = args.df_path
    allup = args.allup
    layer_path = args.layer_path
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

    direction_field = atom.direction.SphericalField()
    direction_field.load(df_path)

    phasor3_field = atom.phasor3.Field()
    # Warmup
    sdf.init_phasor3_field(
        phasor3_field,
        direction_field.direction,
        atom.fff3.init_phasor3_field_from_sdf,
        allup,
    )
    t0 = time.perf_counter()
    sdf.init_phasor3_field(
        phasor3_field,
        direction_field.direction,
        atom.fff3.init_phasor3_field_from_sdf,
        allup,
    )
    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    )

    phasor3_multigrid = atom.phasor3.MultigridAligner()
    cos_period = atom.fff3.cos_period_along_normal_1extraction_from_layer_height_kernel(
        layer_height
    )
    phasor3_multigrid.allocate_from_field(phasor3_field, cos_period)
    phasor3_multigrid.align()

    t1 = time.perf_counter()
    duration = t1 - t0

    str_to_print = f"Implicit layer computation took {duration:.1f} seconds"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()

    phasor3_field.save(layer_path)

    ti.reset()


if __name__ == "__main__":
    sdf_df_to_layers()
