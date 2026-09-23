import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.basis3
import atom.direction
import atom.drawer3
import atom.solid3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def visualize_bases():
    parser = argparse.ArgumentParser(
        description="Visualize bases, and boundary point normals."
    )
    parser.add_argument("bpn_path", help="The path to the input boundary point normal.")
    parser.add_argument("bf_path", help="The path to the input basis field.")

    args = parser.parse_args()

    bpn_path = args.bpn_path
    bf_path = args.bf_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    basis3_field = atom.basis3.Field()
    basis3_field.load(bf_path)

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)
    print(f"Domain size: {bpn.get_size()}")
    print(f"bounding box [xmin, xmax, ymin, ymax, zmin, zmax]: {bpn.bounding_box}")

    normal_scale = 0.1
    see_boundary = False
    only_see_one_cell = False
    cell_3dindex_to_view = basis3_field.grid.cell_3dcount // 2

    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    basis3_field_drawer = atom.drawer3.BasisFieldDrawer()
    basis3_field_drawer.init_from_field(basis3_field)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 1.5, bpn.bounding_box[3] * 0.5
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

        with gui.sub_window("Parameters", 0.05, 0.05, 0.28, 0.26) as w:
            cell_3dindex_to_view[2] = w.slider_int(
                "Layer number",
                cell_3dindex_to_view[2],
                0,
                basis3_field.grid.cell_3dcount[2] - 1,
            )

        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        basis3_field_drawer.update(
            basis3_field, cell_3dindex_to_view, only_see_one_cell
        )

        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )

        scene.lines(
            basis3_field_drawer.line_vertex,
            width=2,
            per_vertex_color=basis3_field_drawer.per_vertex_color,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_bases()
