import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.fff3
import atom.solid3
from atom.bvh import BVH

init_taichi("gpu", debug=False, offline_cache_cleaning_policy="never")


def bpn_to_sdf():
    parser = argparse.ArgumentParser(
        description="Convert boundary point normals (BPN), i.e., a set of 3D points, each associated with a normal pointing outside the solid, to a signed distance field (SDF)."
    )
    parser.add_argument("bpn_path", help="The path to the BPN.")
    parser.add_argument("sdf_path", help="The path to the output SDF.")
    parser.add_argument(
        "deposition_width",
        help="The target deposition width determines the cell sides length of the 3D grid.",
    )

    args = parser.parse_args()

    bpn_path = args.bpn_path
    sdf_path = args.sdf_path
    deposition_width = float(args.deposition_width)
    cell_sides_length = atom.fff3.cell_sides_length_from_deposition_width_kernel(
        deposition_width
    )
    print(f"Cell sides length: {cell_sides_length:.3f}")

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)
    print(f"Domain size: {bpn.get_size()}")
    print(f"Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]: {bpn.bounding_box}")
    print(f"Point count: {bpn.point.shape[0]}")

    sdf = atom.solid3.SDF()
    bvh = BVH()
    bvh.from_bpn(bpn)
    sdf.create_from_bpn(bpn, cell_sides_length, bvh, False)
    sdf.save(sdf_path)
    # print(f"SDF cell 3D count: {sdf.grid.cell_3dcount}")

    ti.reset()


if __name__ == "__main__":
    bpn_to_sdf()
