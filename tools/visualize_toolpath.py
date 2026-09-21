import argparse

import taichi as ti

from atom.ti_env import init_taichi

import atom.color
import atom.drawer3
import atom.toolpath3

init_taichi("cpu", offline_cache_cleaning_policy="never")


def visualize_toolpath():
    parser = argparse.ArgumentParser(
        description="Visualize the toolpath. The Turbo colormap represents the relative length along the toolpath. Use the sliders to select the first and last toolpath vertices (do not affect the relative length). Press `h` to hide or show the travel moves (green paths)."
    )
    parser.add_argument("toolpath_path", help="Path to the toolpath to be visualized.")

    args = parser.parse_args()

    toolpath_path = args.toolpath_path

    toolpath = atom.toolpath3.Toolpath()
    toolpath.load(toolpath_path)
    toolpath.compute_length_from_start()

    print(f"Toolpath point count: {toolpath.point_count}")
    print(f"Toolpath length: {toolpath.length_from_start[-1]:.1f}")

    _, p_max = toolpath.get_aabb()

    animation_forward = False
    animation_backward = False

    toolpath_drawer_start = 1
    toolpath_drawer_end = 1
    hide_travels = False
    toolpath_drawer = atom.drawer3.ToolpathDrawer()
    toolpath_drawer.allocate(toolpath.point_count)

    cone_mesh = atom.drawer3.ConeMesh()
    p_0 = toolpath.point[toolpath_drawer_end - 1]
    cone_mesh.create(
        64,
        atom.toolpath3.NOZZLE_CONE_ANGLE,
        16.0,
        p_0,
        toolpath.tool_orientation[toolpath_drawer_end - 1],
    )

    window = ti.ui.Window(
        name="Toolpath Visualizer",
        res=(1280, 720),
        fps_limit=200,
        pos=(150, 150),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(p_0[0] + 10.0, p_0[2] + 10.0, -p_0[1] + 10.0)
    camera.lookat(p_0[0], p_0[2], -p_0[1])
    camera.up(0, 1, 0)
    camera.fov(45 * 0.5)

    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            toolpath_drawer_start = w.slider_int(
                "Toolpath start point", toolpath_drawer_start, 1, toolpath.point_count
            )
            toolpath_drawer_end = w.slider_int(
                "Toolpath end point", toolpath_drawer_end, 1, toolpath.point_count
            )
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "n":
                toolpath_drawer_end += 1
                if toolpath_drawer_end > toolpath.point_count:
                    toolpath_drawer_end = toolpath.point_count
            if window.event.key == "p":
                toolpath_drawer_end -= 1
                if toolpath_drawer_end <= 0:
                    toolpath_drawer_end = 1
            if window.event.key == "x":
                if animation_forward:
                    animation_forward = False
                else:
                    animation_forward = True
            if window.event.key == "c":
                if animation_backward:
                    animation_backward = False
                else:
                    animation_backward = True
            if window.event.key == "h":
                if hide_travels:
                    hide_travels = False
                else:
                    hide_travels = True

        if animation_forward:
            toolpath_drawer_end = min(toolpath_drawer_end + 2, toolpath.point_count)
        if animation_backward:
            toolpath_drawer_end = max(toolpath_drawer_end - 1, 1)

        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=p_max * 2.0, color=(1.0, 1.0, 1.0))

        toolpath_drawer.update(
            toolpath, toolpath_drawer_start, toolpath_drawer_end, hide_travels
        )

        cone_mesh.origin = toolpath.point[toolpath_drawer_end - 1]
        cone_mesh.orientation = toolpath.tool_orientation[toolpath_drawer_end - 1]
        cone_mesh.update_vertex_normal()

        scene.mesh(
            cone_mesh.vertex,
            cone_mesh.index,
            cone_mesh.normal,
            show_wireframe=False,
            color=(0.5, 0.5, 0.5),
        )
        scene.particles(
            toolpath_drawer.vertex,
            0.02,
            color=atom.color.class3set21,
        )
        scene.lines(
            toolpath_drawer.line_vertex,
            width=2,
            per_vertex_color=toolpath_drawer.per_vertex_color,
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    visualize_toolpath()
