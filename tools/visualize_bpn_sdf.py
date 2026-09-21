import argparse

import numpy as np
import taichi as ti

from atom.ti_env import init_taichi

import atom.drawer3
import atom.solid3

init_taichi("gpu")


def visualize_pn():
    parser = argparse.ArgumentParser(
        description="Visualize signed distance field and boundary point normal."
    )
    parser.add_argument(
        "bpn_path", help="The path to the input boundary point normals."
    )
    parser.add_argument("sdf_path", help="The path to the signed distance field.")

    args = parser.parse_args()

    bpn_path = args.bpn_path
    sdf_path = args.sdf_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)

    normal_scale = 0.1
    see_boundary = False
    only_see_one_cell = False

    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    sdf_drawer = atom.drawer3.SDFDrawer()
    sdf_drawer.init_from_sdf(sdf)

    cell_3dindex_to_view_x = 0
    cell_3dindex_to_view_y = 0
    cell_3dindex_to_view_z = 0

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )

    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()

    camera = ti.ui.Camera()
    camera.position(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 2.0, bpn.bounding_box[3] * 0.5
    )
    camera.lookat(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 0.5, -bpn.bounding_box[3] * 0.5
    )
    camera.up(0, 1, 0)

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "b":
                if see_boundary:
                    see_boundary = False
                else:
                    see_boundary = True
            if window.event.key == "c":
                if only_see_one_cell:
                    only_see_one_cell = False
                else:
                    only_see_one_cell = True
        with gui.sub_window("Parameters", 0.05, 0.05, 0.28, 0.26) as w:
            cell_3dindex_to_view_x = w.slider_int(
                "Cell 3D index to view X",
                cell_3dindex_to_view_x,
                0,
                sdf.grid.cell_3dcount[0] - 1,
            )
            cell_3dindex_to_view_y = w.slider_int(
                "Cell 3D index to view Y",
                cell_3dindex_to_view_y,
                0,
                sdf.grid.cell_3dcount[1] - 1,
            )
            cell_3dindex_to_view_z = w.slider_int(
                "Cell 3D index to view Z",
                cell_3dindex_to_view_z,
                0,
                sdf.grid.cell_3dcount[2] - 1,
            )
        cell_3dindex_to_view = np.array(
            [cell_3dindex_to_view_x, cell_3dindex_to_view_y, cell_3dindex_to_view_z]
        )
        sdf_drawer.update(sdf, cell_3dindex_to_view, only_see_one_cell)

        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((1.0, 1.0, 1.0))

        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )
        scene.particles(
            sdf_drawer.vertex,
            sdf_drawer.radius,
            per_vertex_color=sdf_drawer.per_vertex_color,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_pn()
