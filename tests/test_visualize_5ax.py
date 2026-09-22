"""Tests for the toolpath viewer tool (build plan P5.4a): G-code reading and loading.

The round trip is the test that matters: real G-code written by
`tools/toolpath_to_gcode.py`, read back and run through Atomizer's forward
kinematics, must land on the toolpath it was written from.

No `from __future__ import annotations` here; reading G-code drives the
Taichi kernels in `atom.kinematics3z`.
"""

import os
import sys

import numpy as np
import pytest

import visualize_5ax as vt
from atom import contracts, machine_profile
from atom import toolpath_view as tv

FIXTURE = "tests/fixtures/gcode/calibration_cube_head.gcode"
GOLDEN = "tests/golden/calibration_cube.toolpath.npz"

SMALL = """\
G21
M82 ; absolute extrusion for the purge
M98 P"/macros/enable3Z.g"
G1 X0.1 Y20 Z75.3 U75.3 V75.3 F1000.0 ; purge, carries all five axes
G1 X0.1 Y200.0 Z75.3 U75.3 V75.3 F1000.0 E15
M83 ; relative extrusion
G1 X10 Y10 Z75.5 U75.5 V75.5 E0.000000 F3000
G1 E2.00 F2700 ; prime
G1 X11 Y10 Z75.5 U75.6 V75.4 E0.08 F608
M106 S127 ; fan on
G1 X12 Y10 Z75.5 U75.7 V75.3 E0.08 F610
M82 ; absolute extrusion
G1 X0 Y0 Z80 U80 V80 F3000 ; after the footer starts: not part of the body
"""


# --------------------------------------------------------------------------
# Reading G-code
# --------------------------------------------------------------------------


def test_only_the_body_moves_are_read():
    moves = vt.read_gcode_moves(SMALL.splitlines())

    np.testing.assert_allclose(moves.machine[:, 0], [10, 11, 12])
    np.testing.assert_allclose(moves.machine[1], [11, 10, 75.5, 75.6, 75.4])
    np.testing.assert_array_equal(moves.line, [7, 9, 11])


def test_extrusion_and_feed_belong_to_each_move():
    moves = vt.read_gcode_moves(SMALL.splitlines())
    np.testing.assert_allclose(moves.extrusion, [0.0, 0.08, 0.08])
    np.testing.assert_allclose(moves.feed, [3000, 608, 610])


def test_quoted_macro_paths_are_not_read_as_words():
    """Hazard 2: M98 P"/macros/enable3Z.g" would otherwise yield E3."""
    text = SMALL.replace("M83 ; relative extrusion", 'M83\nM98 P"/macros/e3.g"')
    moves = vt.read_gcode_moves(text.splitlines())
    assert len(moves.line) == 3


def test_gcode_from_elsewhere_is_refused_rather_than_misread():
    with pytest.raises(ValueError, match="M83"):
        vt.read_gcode_moves(["G1 X1 Y1 Z1 U1 V1 E1 F600"])


def test_fixture_has_one_move_per_toolpath_point(repo_root):
    with open(repo_root / FIXTURE) as handle:
        moves = vt.read_gcode_moves(handle)
    assert len(moves.line) == 400
    # The first move is the travel to the start; the rest print.
    assert moves.extrusion[0] == 0.0
    assert np.all(moves.extrusion[1:5] > 0)


# --------------------------------------------------------------------------
# Machine axes back to the part
# --------------------------------------------------------------------------


def test_real_gcode_maps_back_onto_the_toolpath_it_came_from(ti_cpu, repo_root):
    """Forward kinematics undo the inverse kinematics the G-code was written with.

    The G-code frame is the part frame shifted by the bed re-centring
    `toolpath_to_gcode` applies, which `contracts.bed_centering_offset`
    reproduces from the whole toolpath's bounding box.
    """
    with open(repo_root / FIXTURE) as handle:
        moves = vt.read_gcode_moves(handle)
    points, orientation = vt.machine_to_build_frame(moves.machine)

    golden = tv.load_toolpath_arrays(repo_root / GOLDEN)
    offset = contracts.bed_centering_offset(golden["point"], machine_profile.load_profile())
    count = len(points)

    np.testing.assert_allclose(points - offset, golden["point"][:count], atol=1e-3)

    recovered = om_tilt(orientation)
    expected = om_tilt(golden["tool_orientation"][:count])
    np.testing.assert_allclose(recovered, expected, atol=0.01)


def om_tilt(orientation):
    from atom import overhang_metrics as om

    return om.tilt_from_vertical_deg(om.spherical_to_cartesian(orientation))


def test_gcode_that_does_not_match_the_toolpath_is_not_paired(ti_cpu, repo_root):
    """The fixture holds 400 moves; the golden toolpath has 46 773 points."""
    notes = []
    view = vt.load_gcode_view(repo_root / FIXTURE, repo_root / GOLDEN, notes)

    assert view.count == 400
    assert view.height is None
    assert view.mesh_offset is None
    assert "bed-centred" in view.frame
    assert any("not the same run" in note for note in notes)
    assert view.feed_mm_min is not None and view.machine is not None


# --------------------------------------------------------------------------
# Loading toolpaths
# --------------------------------------------------------------------------


def _write_toolpath(path, points, tilt_deg=0.0):
    points = np.asarray(points, dtype=np.float32)
    count = len(points)
    np.savez(
        path,
        point=points,
        travel_type=np.zeros(count, np.int32),
        tool_orientation=np.column_stack(
            [np.full(count, np.radians(tilt_deg)), np.zeros(count)]
        ).astype(np.float32),
        width=np.full(count, 0.9, np.float32),
        height=np.full(count, 0.45, np.float32),
        point_count=np.array(count),
        platform_height=np.array(0),
    )


def test_a_smoothed_toolpath_is_in_the_part_frame(tmp_path):
    path = tmp_path / "ramp_xs_smoothed.npz"
    _write_toolpath(path, [[1, 1, 0.45], [2, 1, 0.45]], tilt_deg=5)

    view = vt.load_npz_view(path, [])

    assert view.frame == "part frame"
    np.testing.assert_allclose(view.mesh_offset, 0.0)
    np.testing.assert_allclose(view.tilt_deg, 5.0, atol=1e-4)
    assert not view.is_platform.any()


def test_a_platform_toolpath_uses_its_input_to_find_the_platform(tmp_path):
    part = np.array([[1, 1, 0.45], [2, 1, 0.45]], float)
    platform = np.array([[0, 0, 0.45], [4, 0, 0.45], [4, 4, 0.45]], float)
    _write_toolpath(tmp_path / "ramp_xs_smoothed_tesselated.npz", part)
    _write_toolpath(tmp_path / "ramp_xs_platform.npz", np.vstack([platform, part + [0, 0, 1.35]]))

    view = vt.load_npz_view(tmp_path / "ramp_xs_platform.npz", [])

    np.testing.assert_array_equal(view.is_platform, [True, True, True, False, False])
    np.testing.assert_allclose(view.mesh_offset, [0, 0, 1.35], atol=1e-5)


def test_a_platform_toolpath_alone_does_not_pretend_to_know_the_stl_position(tmp_path):
    _write_toolpath(tmp_path / "ramp_xs_platform.npz", [[0, 0, 0.45], [1, 0, 0.45]])
    notes = []
    view = vt.load_npz_view(tmp_path / "ramp_xs_platform.npz", notes)
    assert view.mesh_offset is None
    assert notes


def test_the_stl_is_found_beside_the_input_first(tmp_path):
    (tmp_path / "data" / "mesh").mkdir(parents=True)
    (tmp_path / "data" / "toolpath").mkdir()
    stl = tmp_path / "data" / "mesh" / "widget_s.stl"
    stl.write_text("solid widget\nendsolid widget\n")

    found = vt.find_stl("widget_s", tmp_path / "data" / "toolpath" / "widget_s_smoothed.npz")

    assert found == stl


def test_the_stl_falls_back_to_this_repository(tmp_path, repo_root):
    found = vt.find_stl("ramp60_xs", tmp_path / "reports" / "toolpaths" / "ramp60_xs_ms30.npz")
    assert found == repo_root / "data" / "mesh" / "ramp60_xs.stl"


def test_stl_is_moved_to_the_origin_like_the_blender_stage(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(2, 2, 2))  # centred on the origin
    path = tmp_path / "box.stl"
    mesh.export(path)

    vertices, faces = vt.load_mesh(path, np.array([0, 0, 1.5]))

    np.testing.assert_allclose(vertices.min(axis=0), [0, 0, 1.5], atol=1e-6)
    np.testing.assert_allclose(vertices.max(axis=0), [2, 2, 3.5], atol=1e-6)
    assert len(faces) == 12


@pytest.mark.parametrize("text, expected", [(None, 99), ("50%", 50), ("0%", 0),
                                            ("100%", 99), ("10", 9), ("1000", 99)])
def test_end_accepts_a_point_number_or_a_percentage(text, expected):
    assert vt.parse_end(text, 100) == expected


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
    reason="needs a display (run under xvfb-run on Linux)",
)
@pytest.mark.parametrize("machine_view", [False, True])
def test_off_screen_render_writes_a_picture(tmp_path, machine_view):
    """Smoke test: builds the whole window, controls included, off screen.

    A small synthetic toolpath keeps it near the unit-test time budget; most
    of the time is pyvista starting up.
    """
    pytest.importorskip("pyvista")
    trimesh = pytest.importorskip("trimesh")
    angle = np.linspace(0, 12 * np.pi, 400)
    spiral = np.column_stack([5 + 4 * np.cos(angle), 5 + 4 * np.sin(angle), angle / 4])
    _write_toolpath(tmp_path / "spiral_smoothed.npz", spiral, tilt_deg=10)
    trimesh.creation.box(extents=(10, 10, 10)).export(tmp_path / "box.stl")
    output = tmp_path / "view.png"
    extra = ["--machine-view"] if machine_view else []

    assert vt.main([str(tmp_path / "spiral_smoothed.npz"), "--screenshot", str(output),
                    "--mode", "shell", "--end", "80%", "--no-nozzle",
                    "--stl", str(tmp_path / "box.stl"), *extra]) == 0

    assert output.stat().st_size > 10_000


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
    reason="needs a display (run under xvfb-run on Linux)",
)
def test_play_button_advances_the_print_at_the_chosen_speed(tmp_path, monkeypatch):
    pytest.importorskip("pyvista")
    import time

    _write_toolpath(tmp_path / "line_smoothed.npz", [[i * 0.5, 0, 0.45] for i in range(1000)])
    view = vt.load_npz_view(tmp_path / "line_smoothed.npz", [])
    viewer = vt.Viewer(view, end=0)
    viewer.build(off_screen=True)
    clock = [100.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])

    viewer._set_speed(200.0)
    viewer._set_playing(True)
    clock[0] += 0.5
    viewer._tick(0)
    assert viewer.end == 100
    assert viewer._play_button.GetRepresentation().GetState() == 1

    viewer._set_playing(False)
    clock[0] += 0.5
    viewer._tick(0)
    assert viewer.end == 100  # paused
    viewer.plotter.close()


# --------------------------------------------------------------------------
# Machine view
# --------------------------------------------------------------------------


def test_machine_state_of_a_toolpath_is_solved_on_the_bed(ti_cpu, tmp_path):
    """A toolpath has no screw values, so they are solved after re-centring."""
    _write_toolpath(tmp_path / "part_smoothed.npz", [[1, 1, 0.45], [5, 1, 0.45], [5, 5, 0.9]],
                    tilt_deg=10)
    view = vt.load_npz_view(tmp_path / "part_smoothed.npz", [])

    state = vt.solve_machine_state(view)

    assert state.solved
    assert state.valid.all()
    np.testing.assert_allclose(
        state.bed_offset,
        contracts.bed_centering_offset(view.point, machine_profile.load_profile()))
    # Each bed point under the nozzle lands at the nozzle tip (X, Y, 0).
    for index in range(view.count):
        world = state.rotation[index] @ (view.point[index] + state.bed_offset)
        world += state.translation[index]
        np.testing.assert_allclose(world, [*state.machine[index, :2], 0.0], atol=2e-3)


def test_machine_state_of_gcode_uses_the_written_screw_values(ti_cpu, repo_root):
    view = vt.load_gcode_view(repo_root / FIXTURE, repo_root / GOLDEN, [])

    state = vt.solve_machine_state(view)

    assert not state.solved
    np.testing.assert_array_equal(state.machine, view.machine)
    np.testing.assert_allclose(state.bed_offset, 0.0)
