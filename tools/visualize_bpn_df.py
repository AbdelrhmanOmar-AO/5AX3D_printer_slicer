import argparse

import numpy as np
import taichi as ti

from atom.ti_env import init_taichi

import atom.direction
import atom.drawer3
import atom.solid3

init_taichi("gpu")


def visualize_direction_field():
    parser = argparse.ArgumentParser(description="Visualize direction field.")
    parser.add_argument("bpn_path", help="The path to the input boundary point normal.")
    parser.add_argument("df_path", help="The path to the input direction field.")

    args = parser.parse_args()

    bpn_path = args.bpn_path
    df_path = args.df_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    direction_field = atom.direction.SphericalField()
    direction_field.load(df_path)

    normal_scale = 0.1
    see_boundary = False
    only_see_one_cell = False
    cell_3dindex_to_view = np.array([0, 0, 0])

    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    dir_field_drawer = atom.drawer3.SphericalDirectionFieldDrawer()
    dir_field_drawer.init_from_field(direction_field)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )

    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 2.0, bpn.bounding_box[3] * 0.5
    )
    camera.lookat(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 0.5, -bpn.bounding_box[3] * 0.5
    )
    camera.up(0, 1, 0)
    # camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    camera.fov(45 * 0.5)
    gui = window.get_gui()

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
            cell_3dindex_to_view[0] = w.slider_int(
                "Cell 3D index to view X",
                cell_3dindex_to_view[0],
                0,
                direction_field.grid.cell_3dcount[0] - 1,
            )
            cell_3dindex_to_view[1] = w.slider_int(
                "Cell 3D index to view Y",
                cell_3dindex_to_view[1],
                0,
                direction_field.grid.cell_3dcount[1] - 1,
            )
            cell_3dindex_to_view[2] = w.slider_int(
                "Cell 3D index to view Z",
                cell_3dindex_to_view[2],
                0,
                direction_field.grid.cell_3dcount[2] - 1,
            )

        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((1.0, 1.0, 1.0))

        dir_field_drawer.update(
            direction_field, cell_3dindex_to_view, only_see_one_cell
        )

        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )
        scene.lines(
            dir_field_drawer.line_vertex,
            width=2,
            per_vertex_color=dir_field_drawer.per_vertex_color,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_direction_field()
