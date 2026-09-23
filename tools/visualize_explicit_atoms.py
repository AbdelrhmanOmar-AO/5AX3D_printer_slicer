import argparse

import numpy as np
import taichi as ti

from atom.ti_env import init_taichi

import atom.color
import atom.drawer3
import atom.frame3

init_taichi("gpu", offline_cache_cleaning_policy="never")


def visualize_explicit_atoms():
    parser = argparse.ArgumentParser(description="Visualize all explicit atoms.")
    parser.add_argument(
        "frame_path", help="The path to the frames (i.e., atoms) to visualize."
    )
    parser.add_argument(
        "layer_height",
        help="The layer height, i.e., the deposition height, is used to determine the frame vector lengths.",
    )

    args = parser.parse_args()

    frame_path = args.frame_path
    layer_height = float(args.layer_height)

    frame_set = atom.frame3.Set()
    frame_set.load(frame_path)
    p_min, p_max = frame_set.get_aabb()
    side_length = p_max - p_min

    atom_min = 0
    atom_max = frame_set.phi_t.shape[0] - 1

    frame_drawer = atom.drawer3.FrameSetDrawer()
    frame_drawer.init_from_field(frame_set)

    window = ti.ui.Window(
        name="Atoms Visualizer",
        res=(1280, 720),
        fps_limit=200,
        pos=(150, 150),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(side_length[0] * 0.5, side_length[2] * 1.5, side_length[1] * 0.5)
    camera.lookat(side_length[0] * 0.5, side_length[2] * 0.5, -side_length[1] * 0.5)
    camera.up(0, 1, 0)
    camera.fov(45 * 0.5)

    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            atom_min = w.slider_int(
                "Atom min", atom_min, 0, frame_set.phi_t.shape[0] - 1
            )
            atom_max = w.slider_int(
                "Atom max", atom_max, 0, frame_set.phi_t.shape[0] - 1
            )
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        frame_drawer.update(frame_set, np.array([atom_min, atom_max]), layer_height)

        scene.particles(
            frame_drawer.point,
            0.01,
            color=tuple(atom.color.class3dark20),
        )
        scene.lines(
            frame_drawer.line_vertex,
            width=2,
            per_vertex_color=frame_drawer.line_per_vertex_color,
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_explicit_atoms()
