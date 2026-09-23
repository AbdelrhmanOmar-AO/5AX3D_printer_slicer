import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.toolpath3
from numpy import pi

init_taichi("cpu")


def tesselate_toolpath_orientations():
    parser = argparse.ArgumentParser(
        description="Tesselate the toolpath to obtain a maximum angle deviation between adjacent points less than an input angle."
    )
    parser.add_argument("input_path", help="Path to the input toolpath.")
    parser.add_argument("output_path", help="Path to the output toolpath.")
    parser.add_argument("max_angle", help="The input maximum angle in degrees.")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    max_angle = float(args.max_angle)

    # DEBUG
    # input_path = "data/toolpath/triangle_24.npz"
    # output_path = "data/toolpath/triangle_24_tesselated.npz"
    # max_angle = 1.0

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Maximum angle: {max_angle} degrees")

    input = atom.toolpath3.Toolpath()
    input.load(input_path)
    max_diff, max_index = input.max_diff_orientation()
    max_diff = max_diff / 3.14 * 180.0
    print("\nBEFORE")
    print(
        f"- Maximum difference angle between two consecutive points: {max_diff:.2f} degrees. Point index: {max_index}"
    )
    print(f"- Toolpath point count: {input.point_count}")

    input.tesselate_orientation(max_angle)
    max_diff, max_index = input.max_diff_orientation()
    max_diff = max_diff / pi * 180.0
    print("\nAFTER:")
    print(
        f"- Maximum difference angle between two consecutive points: {max_diff:.2f} degrees. Point index: {max_index}"
    )
    print(f"- Toolpath point count: {input.point_count}")
    input.save(output_path)


if __name__ == "__main__":
    tesselate_toolpath_orientations()
