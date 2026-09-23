"""Tests for the Qt toolpath viewer window (build plan P5.4, UI).

Skipped unless a Qt binding, pyvistaqt and a display are available (on Linux,
run under ``xvfb-run``). CI installs neither, so these run on the laptop and
in development sessions.

No `from __future__ import annotations` here; the window drives
visualize_5ax, which imports Taichi kernels for G-code.
"""

import os
import sys

import numpy as np
import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("needs a display (run under xvfb-run on Linux)", allow_module_level=True)
pytest.importorskip("pyvistaqt")
qtpy = pytest.importorskip("qtpy")

from qtpy import QtCore, QtWidgets  # noqa: E402

import viewer_qt  # noqa: E402
import visualize_5ax as vt  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setStyle("Fusion")
    application.setPalette(viewer_qt.dark_palette())
    application.setStyleSheet(viewer_qt.stylesheet())
    yield application


def _toolpath(tmp_path, count=2000):
    angle = np.linspace(0, 20 * np.pi, count)
    points = np.column_stack([5 + 4 * np.cos(angle), 5 + 4 * np.sin(angle), 0.45 + angle / 8])
    path = tmp_path / "spiral_smoothed.npz"
    np.savez(
        path,
        point=points.astype(np.float32),
        travel_type=np.zeros(count, np.int32),
        tool_orientation=np.column_stack(
            [np.full(count, np.radians(8.0)), np.zeros(count)]).astype(np.float32),
        width=np.full(count, 0.9, np.float32),
        height=np.full(count, 0.45, np.float32),
        point_count=np.array(count),
        platform_height=np.array(0),
    )
    return vt.load_npz_view(path, [])


@pytest.fixture
def window(app, tmp_path):
    viewer = vt.Viewer(_toolpath(tmp_path), mode="tilt", end=0)
    win = viewer_qt.ViewerWindow(viewer)
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    yield win
    win.close()


def wait(app, seconds):
    deadline = QtCore.QDeadlineTimer(int(seconds * 1000))
    while not deadline.hasExpired():
        app.processEvents()


def test_the_scene_is_drawn_into_the_qt_view_without_overlay_controls(window):
    viewer = window.viewer
    assert viewer.plotter is window.plotter
    assert not viewer.overlay
    assert "deposit" in viewer._actors
    assert not viewer._mode_buttons  # the classic window's widgets are not built


def test_every_colour_mode_is_listed_and_unavailable_ones_are_disabled(window):
    combo = window.mode_combo
    assert combo.count() == len(vt.tv.COLOUR_MODES)
    model = combo.model()
    enabled = {combo.itemData(i): model.item(i).isEnabled() for i in range(combo.count())}
    assert enabled["tilt"] and enabled["progress"]
    assert not enabled["feed"]  # a toolpath has no feed rate
    feed = [combo.itemData(i) for i in range(combo.count())].index("feed")
    assert "G-code" in combo.itemData(feed, QtCore.Qt.ItemDataRole.ToolTipRole)


def test_choosing_a_colour_mode_recolours_the_scene(window):
    index = [window.mode_combo.itemData(i) for i in range(window.mode_combo.count())]
    window.mode_combo.setCurrentIndex(index.index("progress"))
    assert window.viewer.mode == "progress"


def test_the_collision_mode_names_what_the_p4_checks_found(ti_cpu, app, tmp_path):
    """Pick "Collisions (P4)" from the dropdown on a travel through a block:
    the card names the collision at that point, the notes give the totals."""
    from test_visualize_5ax import _write_travel_through_block

    crossing = _write_travel_through_block(tmp_path / "block_smoothed.npz")
    viewer = vt.Viewer(vt.load_npz_view(tmp_path / "block_smoothed.npz", []), mode="tilt",
                       end=crossing)
    win = viewer_qt.ViewerWindow(viewer)
    win.show()
    app.processEvents()
    try:
        keys = [win.mode_combo.itemData(i) for i in range(win.mode_combo.count())]
        win.mode_combo.setCurrentIndex(keys.index("collision"))
        app.processEvents()

        assert viewer.mode == "collision"
        assert "nozzle vs material" in win.info.text()
        assert "P4.2 swept check between points: 1 moves" in win.notes.text()
    finally:
        win.close()


def test_play_advances_on_the_qt_timer_and_pause_stops_it(app, window):
    window.speed_slider.setValue(window._speed_to_slider(1000.0))
    window.toggle_play()
    wait(app, 0.6)
    moved = window.viewer.end
    assert moved > 150
    assert window.timeline.value() == moved  # the timeline follows playback

    window.toggle_play()
    wait(app, 0.2)
    assert window.viewer.end == moved


def test_transport_buttons_jump_and_step(window):
    window._jump(window.count - 1)
    assert window.viewer.end == window.count - 1
    window._step(-1)
    assert window.viewer.end == window.count - 2
    window._jump(0)
    assert window.viewer.end == 0


def test_dragging_the_timeline_moves_the_print(window):
    window.timeline.setValue(1234)
    assert window.viewer.end == 1234
    assert window.counter.text().startswith("1,235 / 2,000")
    assert "1,235 / 2,000" in window.info.text()


def test_z_clip_slider_sets_the_height(window):
    low, high = window.viewer.z_range
    window.z_slider.setValue(500)
    assert window.viewer.z_max == pytest.approx(low + (high - low) / 2, rel=1e-3)
    assert window.z_label.text().endswith("mm")


def test_machine_view_switch(ti_cpu, window):
    window.switches["machine_view"].setChecked(True)
    assert window.viewer.machine_view
    assert "Machine view" in window.frame_label.text()
    assert any(label == "Gantry gap (mm)" for label, _ in window.viewer.point_fields())


def test_stl_switch_is_disabled_without_an_stl(window):
    assert not window.switches["show_mesh"].isEnabled()


def test_toggle_switch_animates_to_its_new_state(app):
    switch = viewer_qt.ToggleSwitch(False)
    switch.show()
    switch.setChecked(True)
    wait(app, 0.3)
    assert switch._position == pytest.approx(1.0)
    switch.close()


def test_icons_are_drawn(app):
    for kind in ("play", "pause", "step_forward", "step_back", "to_end", "to_start"):
        assert not viewer_qt.icon(kind).isNull()


def test_classic_window_is_used_when_qt_is_missing(tmp_path, monkeypatch, capsys):
    """visualize_5ax falls back rather than failing without PySide6/pyvistaqt."""
    view = _toolpath(tmp_path)
    shown = []
    monkeypatch.setitem(sys.modules, "viewer_qt", None)  # import raises ImportError
    monkeypatch.setattr(vt.Viewer, "show", lambda self: shown.append(True))

    assert vt.main([view.source, "--no-stl"]) == 0

    assert shown
    assert "conda install -c conda-forge pyside6 pyvistaqt" in capsys.readouterr().out
