import argparse
import time

import taichi as ti

from atom.ti_env import init_taichi

import atom.fff3
import atom.frame3
import atom.solid3
import atom.triphasor3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def extract_explicit_atoms():
    parser = argparse.ArgumentParser(
        description="Extract the explicit atoms from their implicit representation. This is similar to local maxima extraction of a 3D scalar field."
    )

    parser.add_argument(
        "sdf_path", help="The path to the input signed distance field (SDF)."
    )
    parser.add_argument(
        "triphasor_path",
        help="The path to the input triphasor field representing the atoms.",
    )
    parser.add_argument(
        "frame_path",
        help="The path to the input frame field, representating the tool orientations and deposition tangents.",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file.",
    )

    args = parser.parse_args()

    sdf_path = args.sdf_path
    triphasor_path = args.triphasor_path
    frame_path = args.frame_path

    log_path = args.logpath

    if log_path is not None:
        log_file = open(log_path, "a", encoding="utf-8")

    triphasor3_field = atom.triphasor3.Field()
    triphasor3_field.load(triphasor_path)

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)
    print(f"Sides cell count: {sdf.grid.cell_3dcount}")
    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    )

    print(f"Layer height: {layer_height:.2f}")
    cos_triperiod = atom.fff3.cos_triperiod_1ex_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    )

    triphasor3_field = atom.triphasor3.Field()
    triphasor3_field.load(triphasor_path)

    t0 = time.perf_counter()

    frame_field = atom.frame3.Field()
    triphasor3_field_cell_point_val = triphasor3_field.allocate_frame_field(frame_field)
    triphasor3_field.extract_frame_field(
        frame_field, triphasor3_field_cell_point_val, cos_triperiod, 1
    )
    atom.fff3.frame_field_filter_point_too_close_to_boundary(
        sdf.sdf, sdf.grid.cell_sides_length, frame_field.point
    )
    frame_field.save_active_frame_set(frame_path)

    t1 = time.perf_counter()
    duration = t1 - t0

    sdf_memory_usage = sdf.compute_memory_usage()
    triphasor3_memory_usage = triphasor3_field.compute_memory_usage()
    frame_memory_usage = frame_field.compute_memory_usage()

    total_memory_usage = sdf_memory_usage + triphasor3_memory_usage + frame_memory_usage

    str_to_print = f"Atom extractor memory usage: {total_memory_usage * 1e-9:.9f} Gb"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    str_to_print = f"Extracting explicit atoms took {duration:.1f} seconds."
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()


if __name__ == "__main__":
    extract_explicit_atoms()
