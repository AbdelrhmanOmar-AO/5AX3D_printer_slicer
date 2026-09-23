"""The toolpath viewer as a Qt desktop window (build plan P5.4, UI).

The 3D scene is `visualize_5ax.Viewer`, the same engine as the classic
window, drawn into a `pyvistaqt.QtInteractor`. Around it:

* a side panel: file, colour mode (dropdown), show/hide switches, Z clip,
  camera presets and screenshot, the current point, notes;
* a timeline bar: jump to start / step back / play-pause / step forward /
  jump to end, the progress slider and the playback speed.

Playback runs on a Qt timer, so it does not depend on VTK timers at all
(`docs/plan_corrections.md` 4.13).

Needs PySide6 (or another Qt binding) and pyvistaqt. Install them once into
the conda environment::

    conda install -c conda-forge pyside6 pyvistaqt

`tools/visualize_5ax.py` opens this window when they are installed and falls
back to the classic window otherwise.
"""

# No `from __future__ import annotations`: this module drives
# visualize_5ax, which imports Taichi kernels for G-code (plan_corrections 3.2).

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtGui, QtWidgets

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import visualize_5ax as vt  # noqa: E402

tv = vt.tv

Qt = QtCore.Qt

# --------------------------------------------------------------------------
# Look
# --------------------------------------------------------------------------

WINDOW_BG = "#16181d"
PANEL_BG = "#1d2027"
CARD_BG = "#252932"
BORDER = "#323743"
TEXT = "#e6e8eb"
TEXT_DIM = "#8b919c"
ACCENT = "#4c9aff"
ACCENT_HOVER = "#6aabff"
WARN = "#e8a33d"

PANEL_WIDTH = 330


def _chevron_file():
    """A small down-chevron SVG for the dropdown (QSS can only use files)."""
    path = Path(tempfile.gettempdir()) / "atom_viewer_chevron.svg"
    if not path.is_file():
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
            f'<path d="M2.5 4.5 L6 8 L9.5 4.5" fill="none" stroke="{TEXT_DIM}" '
            'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
            encoding="utf-8",
        )
    return path.as_posix()


def stylesheet():
    return f"""
    QMainWindow, QWidget#root {{ background: {WINDOW_BG}; }}
    QWidget {{ color: {TEXT}; font-size: 10pt; }}
    QToolTip {{ background: {CARD_BG}; color: {TEXT}; border: 1px solid {BORDER};
                padding: 5px 8px; border-radius: 6px; }}

    QFrame#side {{ background: {PANEL_BG}; border-right: 1px solid {BORDER}; }}
    QScrollArea#sideScroll, QWidget#sideInner {{ background: {PANEL_BG}; border: none; }}
    QFrame#timeline {{ background: {PANEL_BG}; border-top: 1px solid {BORDER}; }}

    QLabel#title {{ font-size: 14pt; font-weight: 600; }}
    QLabel#subtitle, QLabel#dim {{ color: {TEXT_DIM}; }}
    QLabel#section {{ color: {TEXT_DIM}; font-size: 8pt; font-weight: 600;
                      letter-spacing: 1.2px; padding-top: 6px; }}
    QLabel#counter {{ font-size: 10pt; color: {TEXT}; }}
    QLabel#notes {{ color: {TEXT_DIM}; background: {CARD_BG}; border-radius: 8px;
                    padding: 8px 10px; }}
    QLabel#info {{ background: {CARD_BG}; border-radius: 8px; padding: 8px 8px; font-size: 9.5pt; }}

    QComboBox {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 7px;
                 padding: 6px 10px; min-height: 18px; }}
    QComboBox:hover {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 26px; }}
    QComboBox::down-arrow {{ image: url({_chevron_file()}); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{ background: {CARD_BG}; border: 1px solid {BORDER};
                                   border-radius: 7px; padding: 4px; outline: 0;
                                   selection-background-color: {ACCENT}; }}

    QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px;
                                  margin: -5px 0; border-radius: 7px; }}
    QSlider::handle:horizontal:hover {{ background: #ffffff; }}
    QSlider::handle:horizontal:pressed {{ background: {ACCENT_HOVER}; }}

    QToolButton {{ background: transparent; border: none; border-radius: 8px; padding: 6px; }}
    QToolButton:hover {{ background: {CARD_BG}; }}
    QToolButton:pressed {{ background: {BORDER}; }}
    QToolButton#play {{ background: {ACCENT}; border-radius: 20px; }}
    QToolButton#play:hover {{ background: {ACCENT_HOVER}; }}

    QPushButton {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 7px;
                   padding: 6px 10px; }}
    QPushButton:hover {{ border-color: {ACCENT}; }}
    QPushButton:pressed {{ background: {BORDER}; }}

    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """


def dark_palette():
    palette = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, colour in (
        (roles.Window, WINDOW_BG), (roles.WindowText, TEXT), (roles.Base, CARD_BG),
        (roles.AlternateBase, PANEL_BG), (roles.Text, TEXT), (roles.Button, CARD_BG),
        (roles.ButtonText, TEXT), (roles.Highlight, ACCENT), (roles.HighlightedText, "#ffffff"),
        (roles.ToolTipBase, CARD_BG), (roles.ToolTipText, TEXT),
    ):
        palette.setColor(role, QtGui.QColor(colour))
    disabled = QtGui.QPalette.ColorGroup.Disabled
    for role in (roles.Text, roles.WindowText, roles.ButtonText):
        palette.setColor(disabled, role, QtGui.QColor("#5c626d"))
    return palette


def icon(kind, colour=TEXT, size=20):
    """Crisp vector icons for the transport buttons, drawn with QPainter."""
    ratio = 2
    pixmap = QtGui.QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(colour))
    s = float(size)

    def triangle(x0, x1, pointing_right=True):
        top, bottom, middle = s * 0.22, s * 0.78, s * 0.5
        points = ([QtCore.QPointF(x0, top), QtCore.QPointF(x1, middle), QtCore.QPointF(x0, bottom)]
                  if pointing_right else
                  [QtCore.QPointF(x1, top), QtCore.QPointF(x0, middle), QtCore.QPointF(x1, bottom)])
        painter.drawPolygon(QtGui.QPolygonF(points))

    def bar(x, width=s * 0.12):
        painter.drawRoundedRect(QtCore.QRectF(x, s * 0.22, width, s * 0.56), 1.2, 1.2)

    if kind == "play":
        triangle(s * 0.32, s * 0.78)
    elif kind == "pause":
        bar(s * 0.28, s * 0.16)
        bar(s * 0.56, s * 0.16)
    elif kind == "step_forward":
        triangle(s * 0.26, s * 0.64)
        bar(s * 0.66)
    elif kind == "step_back":
        bar(s * 0.22)
        triangle(s * 0.36, s * 0.74, pointing_right=False)
    elif kind == "to_end":
        triangle(s * 0.16, s * 0.46)
        triangle(s * 0.42, s * 0.72)
        bar(s * 0.74)
    elif kind == "to_start":
        bar(s * 0.14)
        triangle(s * 0.28, s * 0.58, pointing_right=False)
        triangle(s * 0.54, s * 0.84, pointing_right=False)
    painter.end()
    return QtGui.QIcon(pixmap)


class ToggleSwitch(QtWidgets.QAbstractButton):
    """A pill-shaped on/off switch whose knob slides when toggled."""

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._position = 1.0 if checked else 0.0
        self._animation = QtCore.QVariantAnimation(self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._move_knob)
        self.toggled.connect(self._animate)

    def sizeHint(self):
        return QtCore.QSize(38, 22)

    def _animate(self, checked):
        self._animation.stop()
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def _move_knob(self, value):
        self._position = float(value)
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(1, 2, 36, 18)
        off, on = QtGui.QColor(BORDER), QtGui.QColor(ACCENT)
        track = QtGui.QColor(
            int(off.red() + (on.red() - off.red()) * self._position),
            int(off.green() + (on.green() - off.green()) * self._position),
            int(off.blue() + (on.blue() - off.blue()) * self._position),
        )
        if not self.isEnabled():
            track = QtGui.QColor("#2a2e36")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(rect, 9, 9)
        knob = QtGui.QColor("#ffffff" if self.isEnabled() else "#5c626d")
        painter.setBrush(knob)
        x = rect.left() + 2 + (rect.width() - 18) * self._position
        painter.drawEllipse(QtCore.QRectF(x, rect.top() + 2, 14, 14))
        painter.end()


def _section(text):
    label = QtWidgets.QLabel(text.upper())
    label.setObjectName("section")
    return label


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


class ViewerWindow(QtWidgets.QMainWindow):
    """Side panel + 3D view + timeline around a `visualize_5ax.Viewer`."""

    TICK_MS = 16
    SPEED_STEPS = 1000

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        view = viewer.view
        self.count = view.count
        self.playing = False
        self._play_position = float(viewer.end)
        self._last_tick = None
        self._syncing = False

        name = tv.part_name_from_path(view.source)
        self.setWindowTitle(f"Toolpath viewer \N{EM DASH} {name}")
        self.resize(1560, 960)

        root = QtWidgets.QWidget(objectName="root")
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_side_panel(name))

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        self.plotter = QtInteractor(root, auto_update=False, multi_samples=8)
        right.addWidget(self.plotter.interactor, stretch=1)
        right.addWidget(self._build_timeline())
        layout.addLayout(right, stretch=1)
        self.setCentralWidget(root)

        viewer.build(plotter=self.plotter)
        viewer.on_frame = self._sync
        self._refresh_panel()
        self._sync()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._add_shortcuts()

    # -- building -------------------------------------------------------------

    def _build_side_panel(self, name):
        view = self.viewer.view
        side = QtWidgets.QFrame(objectName="side")
        side.setFixedWidth(PANEL_WIDTH)
        outer = QtWidgets.QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(objectName="sideScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget(objectName="sideInner")
        panel = QtWidgets.QVBoxLayout(inner)
        panel.setContentsMargins(18, 18, 18, 18)
        panel.setSpacing(8)
        inner.setMinimumWidth(0)
        inner.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                            QtWidgets.QSizePolicy.Policy.Preferred)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        title = QtWidgets.QLabel(name, objectName="title")
        source = QtWidgets.QLabel(Path(view.source).name, objectName="subtitle")
        source.setToolTip(str(view.source))
        printing = int(np.count_nonzero(view.deposit))
        stats = QtWidgets.QLabel(f"{view.count:,} points \N{MIDDLE DOT} {printing:,} printing",
                                 objectName="dim")
        self.frame_label = QtWidgets.QLabel(objectName="dim")
        self.frame_label.setWordWrap(True)
        for widget in (title, source, stats, self.frame_label):
            panel.addWidget(widget)

        panel.addWidget(_section("Colour by"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for index, mode in enumerate(tv.COLOUR_MODES):
            self.mode_combo.addItem(mode.label, mode.key)
            reason = self.viewer.unavailable(mode.key)
            if reason:
                item = self.mode_combo.model().item(index)
                item.setEnabled(False)
                self.mode_combo.setItemData(index, f"Not available: {reason}",
                                            Qt.ItemDataRole.ToolTipRole)
            else:
                self.mode_combo.setItemData(index, mode.title, Qt.ItemDataRole.ToolTipRole)
        self.mode_combo.setCurrentIndex(
            [m.key for m in tv.COLOUR_MODES].index(self.viewer.mode))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        panel.addWidget(self.mode_combo)

        panel.addWidget(_section("Show"))
        self.switches = {}
        for label, attribute, tip in (
            ("Travel moves", "show_travel", "Non-printing moves, as thin grey lines"),
            ("STL overlay", "show_mesh", "The part's STL, semi-transparent"),
            ("Nozzle", "show_nozzle", "The nozzle cone at the current point"),
            ("Machine view", "machine_view",
             "The nozzle stays still and vertical; the bed tilts beneath it, "
             "posed from the three screws Z, U, V"),
        ):
            row = QtWidgets.QHBoxLayout()
            text = QtWidgets.QLabel(label)
            switch = ToggleSwitch(getattr(self.viewer, attribute))
            if attribute == "show_mesh" and self.viewer.mesh is None:
                switch.setEnabled(False)
                tip = "No STL found for this part (use --stl)"
                text.setObjectName("dim")
            text.setToolTip(tip)
            switch.setToolTip(tip)
            switch.toggled.connect(lambda state, a=attribute: self._option_changed(a, state))
            row.addWidget(text)
            row.addStretch(1)
            row.addWidget(switch)
            panel.addLayout(row)
            self.switches[attribute] = switch

        panel.addWidget(_section("Z clip"))
        low, high = self.viewer.z_range
        self._z_low, self._z_high = low, max(high, low + 1e-6)
        zrow = QtWidgets.QHBoxLayout()
        self.z_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 1000)
        self.z_slider.setValue(self._z_to_slider(self.viewer.z_max))
        self.z_slider.setToolTip("Hide everything above this height")
        self.z_slider.valueChanged.connect(self._z_changed)
        self.z_label = QtWidgets.QLabel(objectName="dim")
        self.z_label.setFixedWidth(62)
        self.z_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        zrow.addWidget(self.z_slider, stretch=1)
        zrow.addWidget(self.z_label)
        panel.addLayout(zrow)

        panel.addWidget(_section("Camera"))
        camera = QtWidgets.QHBoxLayout()
        camera.setSpacing(6)
        for text, method, tip in (
            ("Iso", "view_isometric", "Isometric view"),
            ("Top", "view_xy", "Look down the Z axis"),
            ("Front", "view_xz", "Look along the Y axis"),
            ("Side", "view_yz", "Look along the X axis"),
        ):
            button = QtWidgets.QPushButton(text)
            # Let the four share the row rather than set the panel's width.
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
            button.setToolTip(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, m=method: self._camera(m))
            camera.addWidget(button)
        panel.addLayout(camera)
        save = QtWidgets.QPushButton("Save image\N{HORIZONTAL ELLIPSIS}")
        save.setToolTip("Save the 3D view as a PNG")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.clicked.connect(self._save_image)
        panel.addWidget(save)

        panel.addWidget(_section("Current point"))
        self.info = QtWidgets.QLabel(objectName="info")
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        panel.addWidget(self.info)

        panel.addWidget(_section("Notes"))
        self.notes = QtWidgets.QLabel(objectName="notes")
        self.notes.setWordWrap(True)
        panel.addWidget(self.notes)
        panel.addStretch(1)
        return side

    def _build_timeline(self):
        bar = QtWidgets.QFrame(objectName="timeline")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(6)

        def transport(kind, tip, handler, name=None, size=20):
            button = QtWidgets.QToolButton()
            if name:
                button.setObjectName(name)
            button.setIcon(icon(kind, "#ffffff" if name == "play" else TEXT, size))
            button.setIconSize(QtCore.QSize(size, size))
            button.setToolTip(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(handler)
            row.addWidget(button)
            return button

        transport("to_start", "Jump to the start (Home)", lambda: self._jump(0))
        transport("step_back", "Back one point (\N{LEFTWARDS ARROW})", lambda: self._step(-1))
        self.play_button = transport("play", "Play / pause (Space)", self.toggle_play,
                                     name="play", size=22)
        self.play_button.setFixedSize(40, 40)
        transport("step_forward", "Forward one point (\N{RIGHTWARDS ARROW})",
                  lambda: self._step(1))
        transport("to_end", "Jump to the end (End)", lambda: self._jump(self.count - 1))
        row.addSpacing(10)

        self.timeline = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, max(self.count - 1, 0))
        self.timeline.setToolTip("Scrub through the print in the order the printer runs it")
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline.sliderPressed.connect(lambda: self.set_playing(False))
        row.addWidget(self.timeline, stretch=1)

        self.counter = QtWidgets.QLabel(objectName="counter")
        self.counter.setMinimumWidth(190)
        self.counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.counter)
        row.addSpacing(18)

        speed_title = QtWidgets.QLabel("Speed", objectName="dim")
        row.addWidget(speed_title)
        low, high = tv.speed_limits(self.count)
        self._log_speed = (math.log10(low), math.log10(high))
        self.speed_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, self.SPEED_STEPS)
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.setValue(self._speed_to_slider(self.viewer.speed))
        self.speed_slider.valueChanged.connect(self._speed_changed)
        row.addWidget(self.speed_slider)
        self.speed_label = QtWidgets.QLabel(objectName="dim")
        self.speed_label.setMinimumWidth(150)
        row.addWidget(self.speed_label)
        self._speed_changed(self.speed_slider.value())
        return bar

    def _add_shortcuts(self):
        step = max(1, self.count // 100)
        for keys, handler in (
            (Qt.Key.Key_Space, self.toggle_play),
            (Qt.Key.Key_Left, lambda: self._step(-1)),
            (Qt.Key.Key_Right, lambda: self._step(1)),
            (Qt.Key.Key_Comma, lambda: self._step(-step)),
            (Qt.Key.Key_Period, lambda: self._step(step)),
            (Qt.Key.Key_Home, lambda: self._jump(0)),
            (Qt.Key.Key_End, lambda: self._jump(self.count - 1)),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)

    # -- controls -> viewer --------------------------------------------------

    def _busy(self):
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()

    def _done(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def _mode_changed(self, index):
        key = self.mode_combo.itemData(index)
        self._busy()
        try:
            reason = self.viewer.set_mode(key)
        finally:
            self._done()
        if reason:  # unreachable through the UI (items are disabled), but keep the combo honest
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(
                [m.key for m in tv.COLOUR_MODES].index(self.viewer.mode))
            self.mode_combo.blockSignals(False)
        self._refresh_panel()

    def _option_changed(self, attribute, state):
        self._busy()
        try:
            value = self.viewer.set_option(attribute, state)
        finally:
            self._done()
        if value != state:
            self.switches[attribute].blockSignals(True)
            self.switches[attribute].setChecked(value)
            self.switches[attribute].blockSignals(False)
        self._refresh_panel()

    def _z_to_slider(self, z):
        return int(round(1000 * (z - self._z_low) / (self._z_high - self._z_low)))

    def _z_changed(self, value):
        z = self._z_low + (self._z_high - self._z_low) * value / 1000.0
        self.z_label.setText(f"{z:.1f} mm")
        self.viewer.set_z_max(z + (1e-6 if value == 1000 else 0.0))

    def _camera(self, method):
        getattr(self.plotter, method)()
        self.plotter.reset_camera()
        self.plotter.render()

    def _save_image(self):
        name = tv.part_name_from_path(self.viewer.view.source)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", f"{name}_{self.viewer.end + 1}.png", "PNG image (*.png)")
        if path:
            self.plotter.screenshot(path)

    def _timeline_changed(self, value):
        if self._syncing:
            return
        self._play_position = float(value)
        self.viewer.go_to(value)

    def _speed_to_slider(self, speed):
        low, high = self._log_speed
        return int(round(self.SPEED_STEPS * (math.log10(speed) - low) / (high - low)))

    def _speed_changed(self, value):
        low, high = self._log_speed
        self.viewer.speed = 10.0 ** (low + (high - low) * value / self.SPEED_STEPS)
        text = tv.describe_speed(self.viewer.speed, self.count)
        # "Speed: 780 points/s (whole print in 60 s)" -> "780 pts/s · 60 s"
        rate, whole = text.removeprefix("Speed: ").split(" (whole print in ")
        self.speed_label.setText(f"{rate.replace('points', 'pts')} \N{MIDDLE DOT} "
                                 f"{whole.rstrip(')')}")
        self.speed_label.setToolTip(text)

    # -- playback ----------------------------------------------------------------

    def toggle_play(self):
        self.set_playing(not self.playing)

    def set_playing(self, playing):
        playing = bool(playing)
        if playing and self.viewer.end >= self.count - 1:
            self.viewer.go_to(0)
        self.playing = playing
        self._play_position = float(self.viewer.end)
        self._last_tick = time.perf_counter()
        self.play_button.setIcon(icon("pause" if playing else "play", "#ffffff", 22))
        if playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        now = time.perf_counter()
        elapsed = now - (self._last_tick or now)
        self._last_tick = now
        self._play_position, finished = tv.playback_advance(
            self._play_position, self.viewer.speed, elapsed, self.count)
        self.viewer.go_to(int(self._play_position))
        if finished:
            self.set_playing(False)

    def _step(self, delta):
        self.set_playing(False)
        self._jump(self.viewer.end + delta)

    def _jump(self, index):
        self.set_playing(False)
        self.viewer.go_to(index)
        self._play_position = float(self.viewer.end)

    # -- viewer -> controls --------------------------------------------------

    def _sync(self):
        """After every frame: timeline, counter and the current-point card."""
        end = self.viewer.end
        self._syncing = True
        self.timeline.setValue(end)
        self._syncing = False
        percent = 100.0 * end / max(self.count - 1, 1)
        self.counter.setText(f"{end + 1:,} / {self.count:,}  \N{MIDDLE DOT}  {percent:.1f} %")
        rows = "".join(
            f"<tr><td style='color:{TEXT_DIM}; padding-right:8px'>{label}</td>"
            f"<td>{value}</td></tr>"
            for label, value in self.viewer.point_fields()
        )
        self.info.setText(f"<table cellspacing='0' cellpadding='2'>{rows}</table>")

    def _refresh_panel(self):
        """After a rebuild: frame line and notes."""
        lines = self.viewer.header_lines()
        self.frame_label.setText(self.viewer.frame_text())
        # The classic window splits long notes over lines ending in ";" to fit
        # its overlay; here they wrap, so join them back.
        notes = []
        for line in lines[2:]:
            if notes and notes[-1].endswith(";"):
                notes[-1] = f"{notes[-1]} {line}"
            else:
                notes.append(line)
        self.notes.setText("\n\n".join(notes) if notes else "Nothing to note.")
        self._z_changed(self.z_slider.value())

    def closeEvent(self, event):
        self._timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def run(viewer):
    """Open the Qt window for ``viewer`` and run until it is closed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(stylesheet())
    window = ViewerWindow(viewer)
    window.show()
    return app.exec()
