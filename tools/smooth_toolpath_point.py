import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.toolpath3

init_taichi("cpu")


def smooth_toolpath_orientation():
    parser = argparse.ArgumentParser(description="Smooth the toolpath given as input.")
    parser.add_argument("input_path", help="Path to the input toolpath.")
    parser.add_argument("output_path", help="Path to the output smooth toolpath.")
    parser.add_argument("iter_count", help="The number of smoothing operations.")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    iter_count = int(args.iter_count)

    # DEBUG
    # input_path = "data/toolpath/ankle_25.npz"
    # output_path = "data/toolpath/ankle_25_smoothed.npz"
    # iter_count = 8

    input = atom.toolpath3.Toolpath()
    input.load(input_path)
    input.smooth_points(iter_count)
    input.save(output_path)


if __name__ == "__main__":
    smooth_toolpath_orientation()
