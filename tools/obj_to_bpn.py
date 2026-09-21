import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.solid3

init_taichi("cpu", debug=False, offline_cache_cleaning_policy="never")


def obj_to_bpn():
    parser = argparse.ArgumentParser(
        description="Extract the boundary point normals (BPN) of the OBJ mesh.The OBJ mesh has to be obtained with the tool `process_for_atomizer.py`."
    )
    parser.add_argument("obj_path", help="The path to the input OBJ mesh.")
    parser.add_argument("bpn_path", help="The path to the output BPN.")

    args = parser.parse_args()

    obj_path = args.obj_path
    bpn_path = args.bpn_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.create_from_file(obj_path)
    print(f"Point count: {bpn.point.shape[0]}")
    print(f"bounding box [xmin, xmax, ymin, ymax, zmin, zmax]: {bpn.bounding_box}")
    bpn.save(bpn_path)
    ti.reset()


if __name__ == "__main__":
    obj_to_bpn()
