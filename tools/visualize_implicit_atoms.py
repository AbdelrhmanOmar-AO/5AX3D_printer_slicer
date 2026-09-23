import argparse

import numpy as np
import taichi as ti

from atom.ti_env import init_taichi

import atom.drawer3
import atom.solid3
import atom.triphasor3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def visualize_implicit_atoms():
    parser = argparse.ArgumentParser(description="Visualize implicit atoms.")
    parser.add_argument(
        "bpn_path",
        help="The path to the boundary point normals representing the 3D solid's surface.",
    )
    parser.add_argument(
        "triphasor_path",
        help="The path to the triphasor field (i.e., implicit atoms) to visualize.",
    )

    args = parser.parse_args()

    bpn_path = args.bpn_path
    triphasor_path = args.triphasor_path

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    triphasor3_field = atom.triphasor3.Field()
    triphasor3_field.load(triphasor_path)

    subdivisions = np.array([1024, 1024])
    grid_mesh2 = atom.drawer3.GridMesh2()
    grid_mesh2.create(subdivisions)
    grid_mesh2.origin = np.array(
        [
            bpn.bounding_box[1] * 0.5,
            bpn.bounding_box[3] * 0.5,
            bpn.bounding_box[5] * 0.5,
        ]
    )
    grid_mesh2.size = max(bpn.bounding_box) * 1.5
    grid_mesh2.orientation[0] = 0.0
    grid_mesh2.orientation[1] = 0.0

    normal_scale = 0.1
    see_boundary = True
    draw_grid = False
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

            if window.event.key == "g":
                if draw_grid:
                    draw_grid = False
                else:
                    draw_grid = True

        with gui.sub_window("Parameters", 0.05, 0.05, 0.9, 0.26) as w:
            grid_mesh2.size = w.slider_float(
                "Grid mesh size",
                grid_mesh2.size,
                min(bpn.bounding_box),
                max(bpn.bounding_box),
            )
            grid_mesh2.orientation[0] = w.slider_float(
                "Theta", grid_mesh2.orientation[0], 0.0, ti.math.pi
            )
            grid_mesh2.orientation[1] = w.slider_float(
                "Phi", grid_mesh2.orientation[1], 0.0, ti.math.pi * 2.0
            )
            grid_mesh2.origin[0] = w.slider_float(
                "Origin X", grid_mesh2.origin[0], 0.0, bpn.bounding_box[1]
            )
            grid_mesh2.origin[1] = w.slider_float(
                "Origin Y", grid_mesh2.origin[1], 0.0, bpn.bounding_box[3]
            )
            grid_mesh2.origin[2] = w.slider_float(
                "Origin Z", grid_mesh2.origin[2], 0.0, bpn.bounding_box[5]
            )

        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        grid_mesh2.update_vertex_normal()
        grid_mesh2.update_per_vertex_color_with_triphasor3_field_1ex(
            triphasor3_field, True, draw_grid
        )

        scene.mesh(
            grid_mesh2.vertex,
            grid_mesh2.index,
            grid_mesh2.normal,
            show_wireframe=False,
            per_vertex_color=grid_mesh2.per_vertex_color,
        )
        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )
        canvas.scene(scene)
        window.show()

    ti.reset()


if __name__ == "__main__":
    visualize_implicit_atoms()
