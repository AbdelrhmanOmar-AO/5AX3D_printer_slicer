import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.fff3

init_taichi("cpu")


def cl_fr_dw():
    parser = argparse.ArgumentParser(
        description="Compute the cell sides length from the deposition width. All units are in millimeter."
    )
    parser.add_argument("dw", help="The deposition width in millimeter.")

    args = parser.parse_args()

    nozzle_width = float(args.dw)

    print(atom.fff3.cell_sides_length_from_deposition_width_kernel(nozzle_width))


if __name__ == "__main__":
    cl_fr_dw()
