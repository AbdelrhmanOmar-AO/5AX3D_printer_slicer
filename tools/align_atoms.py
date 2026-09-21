import argparse
import time
from math import pi

import taichi as ti

from atom.ti_env import init_taichi

import atom.basis3
import atom.fff3
import atom.phasor3
import atom.solid3
import atom.toolpath3
import atom.triphasor3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def align_atoms():
    parser = argparse.ArgumentParser(
        description="Align atoms in a 3D solid, given a signed distance field (SDF), a phasor field representing the layers, and a basis field."
    )
    parser.add_argument("sdf_path", help="The path to the SDF")
    parser.add_argument(
        "phasor_path", help="The path to the phasor field representing the layers"
    )
    parser.add_argument("basis3_path", help="The path to the basis field")
    parser.add_argument(
        "triphasor_path", help="The triphasor will be saved at the given path"
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file",
    )
    parser.add_argument(
        "--maxslope",
        type=float,
        help="Machine maximum slope",
    )

    args = parser.parse_args()

    sdf_path = args.sdf_path
    phasor_path = args.phasor_path
    basis3_path = args.basis3_path
    triphasor_path = args.triphasor_path

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
    print(f"Grid sides cell count: {sdf.grid.cell_3dcount}")

    phasor3_field = atom.phasor3.Field()
    phasor3_field.load(phasor_path)

    basis3_field = atom.basis3.Field()
    basis3_field.load(basis3_path)

    triphasor3_field = atom.triphasor3.Field()

    # Warmup
    sdf.init_triphasor3_field_1ex(
        atom.fff3.init_triphasor3_field,
        basis3_field,
        phasor3_field.phase,
        triphasor3_field,
    )
    t0 = time.perf_counter()
    sdf.init_triphasor3_field_1ex(
        atom.fff3.init_triphasor3_field,
        basis3_field,
        phasor3_field.phase,
        triphasor3_field,
    )

    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    )
    print(f"Layer height: {layer_height:.2f}")
    cos_triperiod = atom.fff3.cos_triperiod_1ex_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    ).to_numpy()

    triphasor3_multigrid = atom.triphasor3.MultigridAligner()
    triphasor3_multigrid.allocate_from_field(triphasor3_field, cos_triperiod)
    triphasor3_multigrid.align()

    t1 = time.perf_counter()
    duration = t1 - t0

    sdf_memory_usage = sdf.compute_memory_usage()
    atom_aligner_memory_usage = triphasor3_multigrid.compute_memory_usage()
    total_memory_usage = sdf_memory_usage + atom_aligner_memory_usage

    str_to_print = f"Atom aligner memory usage: {total_memory_usage * 1e-9:.9f} Gb"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    str_to_print = f"Atoms alignment took {duration:.1f} seconds."
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()

    triphasor3_field.save(triphasor_path)


if __name__ == "__main__":
    align_atoms()
