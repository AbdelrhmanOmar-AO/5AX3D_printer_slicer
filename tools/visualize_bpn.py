import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.drawer3
import atom.solid3

init_taichi("cpu")


def visualize_pn():
    parser = argparse.ArgumentParser(description="Visualize boundary point normals.")
    parser.add_argument(
        "bpn_path",
        help="The path to the input boundary point normal. Represents a 3D solid.",
    )

    args = parser.parse_args()

    bpn_path = args.bpn_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    normal_scale = 0.1
    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

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

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((1.0, 1.0, 1.0))

        scene.lines(
            bpn_drawer.line_vertex,
            width=2,
            per_vertex_color=bpn_drawer.per_vertex_color,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_pn()
