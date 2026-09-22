"""See what Atomizer produced, the way a slicer's preview shows it (build plan P5.4a).

Opens a 3D window with the toolpath drawn as lines, a slider to scrub through
the print in the order the printer will do it, a Z-height clip to look inside,
and buttons to colour the lines by tilt, tilt direction, bead size, feed rate,
unsupported points or shell/infill. The part's STL can be shown over it, and a
cone marks the nozzle and its tilt at the current point.

Usage
-----
    python tools/visualize_5ax.py data/toolpath/ramp60_xs_smoothed.npz
    python tools/visualize_5ax.py reports/toolpaths/ramp60_s_ms30.npz --mode azimuth
    python tools/visualize_5ax.py data/gcode/ramp60_xs.gcode
    python tools/visualize_5ax.py <file> --screenshot out.png --mode tilt --end 50%

Which file to open
------------------
``<part>_smoothed.npz``
    What the P0.8 overhang metrics measure. Part frame, lines up with the STL.
``reports/toolpaths/<part>_ms<deg>.npz``
    The archived copy of the above for one matrix run. Use these while a matrix
    is running: the ``data/`` files are overwritten by every run.
``<part>_platform.npz``
    What the G-code is written from: the part plus the sacrificial platform.
``data/gcode/<part>.gcode``
    What the printer receives. Each move's nozzle position and tilt are
    recovered from the X, Y, Z, U, V values with Atomizer's own forward
    kinematics. If ``data/toolpath/<part>_platform.npz`` is found, the two are
    paired: bead sizes come from it and the view is moved back into the part
    frame so the STL lines up.

G-code is read with the active machine profile (``ATOM_MACHINE``, default
``reference``); it must be the one the G-code was written with.

Controls
--------
Mouse: left-drag rotates, right-drag or scroll zooms, shift+left-drag pans.
Left / Right arrow: one point back / forward.   , / . : 1 % back / forward.
Space: play / pause.   Buttons on the left: colour mode and what is shown.
v: isometric view.   q: quit.
"""

# No `from __future__ import annotations`: reading G-code imports
# atom.kinematics3z, which defines Taichi kernels. See
# docs/plan_corrections.md 3.2.

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
# Put this checkout's src/ first, so a second checkout (a git worktree) runs
# its own code rather than whichever one `pip install -e` pointed at.
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gcode_stats  # noqa: E402
from atom import toolpath_view as tv  # noqa: E402

#: Largest acceptable gap between a G-code move, mapped back to the part
#: frame, and the toolpath point it was written from. The G-code prints six
#: decimals and the kinematics run in float32, so a genuine pair agrees to a
#: few thousandths of a millimetre; anything past this is a different run.
PAIRING_TOLERANCE_MM = 0.05


# --------------------------------------------------------------------------
# Reading G-code
# --------------------------------------------------------------------------


@dataclass
class GcodeMoves:
    """The body moves of a G-code file written by `toolpath_to_gcode.py`."""

    #: ``(N, 5)`` X, Y, Z, U, V.
    machine: np.ndarray
    #: ``(N,)`` E word of each move (relative extrusion, mm of filament).
    extrusion: np.ndarray
    #: ``(N,)`` feed rate in effect, mm/min.
    feed: np.ndarray
    #: ``(N,)`` 1-based line number.
    line: np.ndarray


def read_gcode_moves(lines) -> GcodeMoves:
    """Extract one row per toolpath point from G-code text.

    `tools/toolpath_to_gcode.py` writes the header, then exactly one
    ``G1 X Y Z U V E F`` line per toolpath point, with prime/retract lines (E
    only) between them, then the footer. The header ends by switching to
    relative extrusion (``M83``) and the footer starts by switching back
    (``M82``), so the body is the moves carrying all five axes between those
    two. The purge lines in the header also carry all five axes, which is why
    the ``M83`` marker is needed rather than just the axis words.

    Words are parsed with `gcode_stats`, which strips quoted strings first
    (build plan hazard 2).
    """
    rows, extrusion, feed, numbers = [], [], [], []
    in_body = False
    current_feed = np.nan

    for number, raw in enumerate(lines, start=1):
        code = gcode_stats.strip_comment(raw)
        if not code:
            continue
        words = gcode_stats.parse_words(code)
        m_code = words.get("M")
        if m_code == 83 and not in_body:
            in_body = True
            continue
        if not in_body:
            continue
        if m_code == 82:
            break

        if "F" in words:
            current_feed = words["F"]
        if words.get("G") in (0.0, 1.0) and all(axis in words for axis in "XYZUV"):
            rows.append([words[axis] for axis in "XYZUV"])
            extrusion.append(words.get("E", 0.0))
            feed.append(current_feed)
            numbers.append(number)

    if not in_body:
        raise ValueError(
            "no 'M83' found: this does not look like G-code written by "
            "tools/toolpath_to_gcode.py, so the part's moves cannot be told "
            "apart from the header's"
        )

    return GcodeMoves(
        machine=np.asarray(rows, dtype=np.float64).reshape(-1, 5),
        extrusion=np.asarray(extrusion, dtype=np.float64),
        feed=np.asarray(feed, dtype=np.float64),
        line=np.asarray(numbers, dtype=np.int64),
    )


def machine_to_build_frame(machine: np.ndarray):
    """Nozzle positions and build directions for machine states X, Y, Z, U, V.

    Runs `kinematics3z.forward` through the vendored
    `toolpath_to_cartesian_toolpath` kernel. The result is in the frame the
    G-code was solved in: the part frame shifted by the bed re-centring
    `toolpath_to_gcode` applies. Returns ``(points (N, 3), orientation (N, 2)
    spherical radians)``. Requires Taichi to be initialised.
    """
    from atom import kinematics3z

    machine = np.ascontiguousarray(np.asarray(machine, dtype=np.float32))
    points = np.zeros((len(machine), 3), dtype=np.float32)
    orientation = np.zeros((len(machine), 2), dtype=np.float32)
    if len(machine):
        kinematics3z.toolpath_to_cartesian_toolpath(machine, points, orientation)
    return points.astype(np.float64), orientation.astype(np.float64)


# --------------------------------------------------------------------------
# Loading a view
# --------------------------------------------------------------------------


def _platform_info(platform_path: Path):
    """``(platform_count, mesh_offset)`` for a ``_platform`` toolpath, or None."""
    part = tv.part_name_from_path(platform_path)
    sibling = platform_path.with_name(f"{part}_smoothed_tesselated.npz")
    if not sibling.is_file():
        return None
    split = tv.platform_split(
        tv.load_toolpath_arrays(platform_path)["point"],
        tv.load_toolpath_arrays(sibling)["point"],
    )
    if split is None:
        return None
    count, lift = split
    return count, np.array([0.0, 0.0, lift])


def load_npz_view(path: Path, notes: list) -> tv.ViewData:
    """A view of a toolpath ``.npz``, in the part frame."""
    arrays = tv.load_toolpath_arrays(path)
    platform_count, mesh_offset = 0, np.zeros(3)

    if path.stem.endswith("_platform"):
        info = _platform_info(path)
        if info is None:
            mesh_offset = None
            notes.append(
                "Platform file without its _smoothed_tesselated.npz next to it: "
                "the platform cannot be told apart and the STL is not shown."
            )
        else:
            platform_count, mesh_offset = info

    return tv.from_arrays(
        arrays["point"],
        arrays["travel_type"],
        arrays["tool_orientation"],
        arrays["width"],
        arrays["height"],
        platform_count=platform_count,
        frame="part frame",
        source=str(path),
        mesh_offset=mesh_offset,
    )


def load_gcode_view(path: Path, toolpath_path: Path | None, notes: list) -> tv.ViewData:
    """A view of a G-code file, paired with its toolpath when one is found."""
    from atom import contracts, machine_profile
    from atom.ti_env import init_taichi

    init_taichi("cpu")

    with open(path, encoding="utf-8", errors="replace") as handle:
        moves = read_gcode_moves(handle)
    if len(moves.line) == 0:
        raise SystemExit(f"{path}: no moves found after the header.")

    points, orientation = machine_to_build_frame(moves.machine)
    view = tv.from_arrays(
        points,
        np.where(moves.extrusion > 0, tv.TRAVEL_TYPE_DEPOSITION, 1),
        orientation,
        frame="bed-centred frame (as sent to the printer)",
        source=str(path),
        mesh_offset=None,
    )
    view.feed_mm_min = moves.feed
    view.gcode_line = moves.line
    view.machine = moves.machine
    view.bed_offset = np.zeros(3)

    if toolpath_path is None:
        part = tv.part_name_from_path(path)
        candidate = path.parent.parent / "toolpath" / f"{part}_platform.npz"
        toolpath_path = candidate if candidate.is_file() else None
    if toolpath_path is None:
        notes.append(
            "No matching _platform.npz found: shown as sent to the printer, "
            "without the STL or bead sizes. Pass --toolpath to pair them."
        )
        return view

    arrays = tv.load_toolpath_arrays(toolpath_path)
    if arrays["point_count"] != view.count:
        notes.append(
            f"{toolpath_path.name} has {arrays['point_count']:,} points but the "
            f"G-code has {view.count:,} moves: not the same run, not paired."
        )
        return view

    offset = contracts.bed_centering_offset(arrays["point"], machine_profile.load_profile())
    part_points = view.point - offset
    gap = float(np.max(np.linalg.norm(part_points - arrays["point"], axis=1)))
    if not gap <= PAIRING_TOLERANCE_MM:
        notes.append(
            f"G-code and {toolpath_path.name} differ by up to {gap:.3f} mm: not "
            "the same run (or a different ATOM_MACHINE), not paired."
        )
        return view

    notes.append(
        f"Paired with {toolpath_path.name}: G-code reproduces it within {gap:.4f} mm."
    )
    view.point = part_points
    view.bed_offset = offset
    view.width = np.asarray(arrays["width"], dtype=np.float64)
    view.height = np.asarray(arrays["height"], dtype=np.float64)
    view.frame = "part frame (recovered from the G-code)"
    info = _platform_info(toolpath_path)
    if info is None:
        notes.append("Platform points cannot be identified; the STL is not shown.")
    else:
        count, view.mesh_offset = info
        view.is_platform[:count] = True
    return view


def find_stl(part: str, near: Path) -> Path | None:
    """The part's STL: ``data/mesh/<part>.stl`` beside the input, else in this repo."""
    for parent in list(near.resolve().parents)[:4]:
        for candidate in (
            parent / "mesh" / f"{part}.stl",
            parent / "data" / "mesh" / f"{part}.stl",
        ):
            if candidate.is_file():
                return candidate
    candidate = REPO_ROOT / "data" / "mesh" / f"{part}.stl"
    return candidate if candidate.is_file() else None


def load_mesh(stl_path: Path, offset: np.ndarray):
    """The STL in the view's frame, as ``(vertices, faces)``.

    `tools/process_for_atomizer.py` moves the mesh so its bounding-box minimum
    sits at the origin before slicing; the same shift is applied here, then
    ``offset`` (the platform lift, if any).
    """
    import trimesh

    mesh = trimesh.load_mesh(str(stl_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices - vertices.min(axis=0) + np.asarray(offset, dtype=np.float64)
    return vertices, np.asarray(mesh.faces, dtype=np.int64)


def surface_samples(vertices, faces, spacing: float) -> np.ndarray:
    """Points covering the mesh surface at most ``spacing`` apart (hazard 16)."""
    import trimesh

    fine_vertices, fine_faces = trimesh.remesh.subdivide_to_size(
        vertices, faces, max_edge=spacing
    )
    centres = fine_vertices[fine_faces].mean(axis=1)
    return np.vstack([fine_vertices, centres])


def parse_end(text: str | None, count: int) -> int:
    """``--end`` as a point index: ``"1234"`` or ``"50%"``."""
    if text is None:
        return count - 1
    text = text.strip()
    if text.endswith("%"):
        fraction = float(text[:-1]) / 100.0
        return int(round(np.clip(fraction, 0.0, 1.0) * (count - 1)))
    return int(np.clip(int(text) - 1, 0, count - 1))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

_BUTTON = 22
_ROW = 30

#: Dark slate gradient: easier on the eye than white for long sessions, and
#: the colour maps stay readable on it.
BACKGROUND_BOTTOM = "#1c1f25"
BACKGROUND_TOP = "#3a404b"
TEXT = "#e6e8eb"
TEXT_DIM = "#8b919c"
BUTTON_OFF = "#d0d4da"

_TAICHI_READY = False


def _ensure_taichi():
    """Initialise Taichi on the CPU once (it resets its runtime if re-initialised)."""
    global _TAICHI_READY
    if not _TAICHI_READY:
        from atom.ti_env import init_taichi

        init_taichi("cpu")
        _TAICHI_READY = True


@dataclass
class MachineState:
    """The bed pose at every point, for the machine view (P5.4b)."""

    #: ``(N, 5)`` X, Y, Z, U, V.
    machine: np.ndarray
    #: ``(N,)`` False where the inverse kinematics found the point unreachable.
    valid: np.ndarray
    #: ``(N, 3, 3)`` and ``(N, 3)``: bed frame to world frame, `atom.bed_motion`.
    rotation: np.ndarray
    translation: np.ndarray
    #: View frame to bed frame, mm.
    bed_offset: np.ndarray
    #: True when the screws were solved here rather than read from G-code.
    solved: bool


def solve_machine_state(view) -> MachineState:
    """Screw values and bed poses for every point of ``view``.

    G-code carries the screw values the printer will get. A toolpath does not,
    so they are solved with `contracts.from_toolpath` after re-centring on the
    bed as `toolpath_to_gcode` does; for a ``_smoothed`` toolpath (no platform
    yet) they can differ from the final G-code's by the platform lift.
    """
    from types import SimpleNamespace

    from atom import bed_motion, contracts, machine_profile

    _ensure_taichi()
    profile = machine_profile.load_profile()

    if view.machine is not None:
        machine = np.asarray(view.machine, dtype=np.float64)
        valid = np.isfinite(machine).all(axis=1)
        offset = np.zeros(3) if view.bed_offset is None else np.asarray(view.bed_offset)
        solved = False
    else:
        spherical = np.column_stack([
            np.arccos(np.clip(view.direction[:, 2], -1.0, 1.0)),
            np.arctan2(view.direction[:, 1], view.direction[:, 0]),
        ])
        toolpath = SimpleNamespace(
            point=view.point.astype(np.float32),
            travel_type=np.where(view.deposit, tv.TRAVEL_TYPE_DEPOSITION, 1),
            tool_orientation=spherical.astype(np.float32),
            width=np.ones(view.count) if view.width is None else view.width,
            height=np.ones(view.count) if view.height is None else view.height,
            point_count=view.count,
        )
        result = contracts.from_toolpath(toolpath, profile, center_on_bed=True)
        machine, valid = result.machine, result.valid & np.isfinite(result.machine).all(axis=1)
        offset = contracts.bed_centering_offset(view.point, profile)
        solved = True

    count = len(machine)
    rotation = np.tile(np.eye(3), (count, 1, 1))
    translation = np.zeros((count, 3))
    if np.any(valid):
        good = machine[valid]
        probed, _ = machine_to_build_frame(bed_motion.probe_states(good))
        rotation[valid], translation[valid] = bed_motion.poses_from_probes(good, probed)
    return MachineState(machine, valid, rotation, translation, np.asarray(offset, float), solved)


class Viewer:
    """The pyvista window. All state changes go through `refresh`."""

    def __init__(self, view, mesh=None, mode="progress", end=None, z_max=None,
                 show_travel=True, show_mesh=True, show_nozzle=True, machine_view=False,
                 notes=()):
        self.view = view
        self.mesh = mesh  # (vertices, faces) in the view's frame, or None
        self.notes = list(notes)
        self.mode = mode
        self.end = view.count - 1 if end is None else int(end)
        z = view.point[:, 2]
        self.z_range = (float(z.min()), float(z.max())) if view.count else (0.0, 1.0)
        self.z_max = self.z_range[1] if z_max is None else float(z_max)
        self.show_travel = show_travel
        self.show_mesh = show_mesh and mesh is not None
        self.show_nozzle = show_nozzle
        self.machine_view = machine_view
        self.playing = False
        self.speed = tv.default_speed(view.count)
        self._play_position = float(self.end)
        self._last_tick = None
        self.message = ""
        self._shell = None
        self._machine = None
        self._profile = None
        self._mode_buttons = {}
        self._toggle_buttons = {}
        self._play_button = None
        self._progress_slider = None
        self.plotter = None

    # -- data ---------------------------------------------------------------

    def unavailable(self, key):
        return tv.mode_unavailable_reason(self.view, key, self.mesh is not None)

    def shell(self):
        if self._shell is None:
            vertices, faces = self.mesh
            width = tv.typical_deposition_width(self.view)
            print(f"Classifying shell and infill (shell = {tv.SHELL_THICKNESS_WIDTHS:g} x "
                  f"{width:.2f} mm from the surface)...")
            samples = surface_samples(vertices, faces, 0.5 * width)
            self._shell = tv.shell_mask(self.view, samples, width)
        return self._shell

    def scalars(self):
        if self.mode == "unsupported" and "unsupported" not in self.view._cache:
            print("Computing unsupported points (P0.8 metric); large parts take a while...")
        shell = self.shell() if self.mode == "shell" else None
        return tv.point_scalars(self.view, self.mode, shell=shell)

    def machine_state(self):
        if self._machine is None:
            from atom import machine_profile

            print("Solving the bed pose at every point...")
            self._machine = solve_machine_state(self.view)
            self._profile = machine_profile.load_profile()
            unreachable = int(np.count_nonzero(~self._machine.valid))
            if unreachable:
                self.notes.append(f"{unreachable:,} points are unreachable (IK): "
                                  "the bed turns red there.")
        return self._machine

    # -- building the scene -------------------------------------------------

    def build(self, off_screen=False, window_size=(1500, 950)):
        import pyvista as pv

        self.pv = pv
        self.window_size = window_size
        plotter = pv.Plotter(off_screen=off_screen, window_size=window_size,
                             title=f"Toolpath viewer: {Path(self.view.source).name}")
        self.plotter = plotter
        plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        plotter.add_axes(color=TEXT, viewport=(0.84, 0.0, 0.98, 0.16))

        self._add_controls()
        self.refresh(reset_camera=True)
        return plotter

    def _label(self, text, position, name, size=9, color=TEXT):
        self.plotter.add_text(text, position=position, font_size=size, color=color, name=name)

    def _add_controls(self):
        plotter = self.plotter
        count = self.view.count
        width, height = self.window_size

        self._progress_slider = plotter.add_slider_widget(
            lambda value: self._set_end(int(round(value)) - 1),
            [1, max(count, 2)],
            value=self.end + 1,
            title="Print progress (point number)",
            pointa=(0.30, 0.07),
            pointb=(0.78, 0.07),
            style="modern",
            title_height=0.02,
            interaction_event="always",
            fmt="%.0f",
            color=TEXT,
        )
        low, high = self.z_range
        plotter.add_slider_widget(
            self._set_z_max,
            [low, high + 1e-6],
            value=self.z_max,
            title="",
            pointa=(0.955, 0.30),
            pointb=(0.955, 0.85),
            style="modern",
            title_height=0.02,
            interaction_event="always",
            fmt="%.1f",
            color=TEXT,
        )
        plotter.add_text("Z clip (mm)", position=(0.925, 0.88), viewport=True,
                         font_size=10, color=TEXT, name="z_clip_label")

        # Playback: a play/pause button and a speed slider, bottom left.
        self._play_button = plotter.add_checkbox_button_widget(
            self._set_playing, value=False, position=(16, int(0.07 * height) - 14),
            size=28, border_size=2, color_on="#e8a33d", color_off=BUTTON_OFF,
            background_color="grey",
        )
        self._label("Play", (52, int(0.07 * height) - 7), "play_label", size=11)
        speed_low, speed_high = tv.speed_limits(count)
        self._speed_label_position = (int(0.10 * width), int(0.07 * height) + 18)
        speed_slider = plotter.add_slider_widget(
            lambda value: self._set_speed(10.0 ** value),
            [np.log10(speed_low), np.log10(speed_high)],
            value=np.log10(self.speed),
            title="",
            pointa=(0.10, 0.07),
            pointb=(0.25, 0.07),
            style="modern",
            interaction_event="always",
            color=TEXT,
        )
        speed_slider.GetRepresentation().SetShowSliderLabel(False)
        self._draw_speed()

        # Colour modes as radio buttons, then the show/hide toggles. Anchored
        # below the header text in the upper left.
        top = height - 200
        self._label("Colour by", (12, top + 8), "colour_header", size=11)
        for row, mode in enumerate(tv.COLOUR_MODES):
            y = top - (row + 1) * _ROW
            reason = self.unavailable(mode.key)
            widget = plotter.add_checkbox_button_widget(
                lambda state, key=mode.key: self._pick_mode(key, state),
                value=mode.key == self.mode,
                position=(12, y), size=_BUTTON, border_size=2,
                color_on="#3b8fd9", color_off=BUTTON_OFF, background_color="grey",
            )
            self._mode_buttons[mode.key] = widget
            self._label(mode.label, (12 + _BUTTON + 8, y + 3), f"mode_label_{mode.key}",
                        color=TEXT_DIM if reason else TEXT)

        toggles = (
            ("Travel moves", "show_travel"),
            ("STL overlay", "show_mesh"),
            ("Nozzle", "show_nozzle"),
            ("Machine view (bed moves)", "machine_view"),
        )
        base = top - (len(tv.COLOUR_MODES) + 2) * _ROW
        self._label("Show", (12, base + 8), "show_header", size=11)
        for row, (label, attribute) in enumerate(toggles):
            y = base - (row + 1) * _ROW
            disabled = attribute == "show_mesh" and self.mesh is None
            self._toggle_buttons[attribute] = plotter.add_checkbox_button_widget(
                lambda state, a=attribute: self._toggle(a, state),
                value=getattr(self, attribute),
                position=(12, y), size=_BUTTON, border_size=2,
                color_on="#3fae5a", color_off=BUTTON_OFF, background_color="grey",
            )
            self._label(label + (" (no STL)" if disabled else ""), (12 + _BUTTON + 8, y + 3),
                        f"toggle_label_{attribute}", color=TEXT_DIM if disabled else TEXT)

        plotter.add_key_event("Left", lambda: self._step(-1))
        plotter.add_key_event("Right", lambda: self._step(1))
        plotter.add_key_event("comma", lambda: self._step(-max(1, count // 100)))
        plotter.add_key_event("period", lambda: self._step(max(1, count // 100)))
        plotter.add_key_event("space", lambda: self._set_playing(not self.playing))

    def start_timer(self):
        """Create the playback timer, bound to the on-screen window.

        pyvista only attaches the interactor to its window inside ``show()``.
        On Windows, VTK attaches a timer to the window's handle, so a timer
        created before that (as `add_timer_event` does if called while the
        scene is built) belongs to no window and never fires: Play did
        nothing. Initialising the interactor first creates the window and
        binds it. On Linux (X11), VTK keeps its own list of timers, which is
        why the first version worked in testing there.
        """
        self.plotter.iren.initialize()
        self.plotter.add_timer_event(max_steps=10**9, duration=30, callback=self._tick)

    def show(self):
        """Open the window and hand control to it until it is closed."""
        self.start_timer()
        self.plotter.show()

    # -- interaction ----------------------------------------------------------

    def _pick_mode(self, key, state):
        reason = self.unavailable(key)
        if reason:
            self.message = f"{tv.MODES_BY_KEY[key].label}: {reason}"
            key = self.mode
        elif not state and key == self.mode:
            pass  # clicking the active mode keeps it on
        else:
            self.mode = key
            self.message = ""
        for other, widget in self._mode_buttons.items():
            widget.GetRepresentation().SetState(int(other == self.mode))
        self.refresh()

    def _toggle(self, attribute, state):
        if attribute == "show_mesh" and self.mesh is None:
            state = False
            self._toggle_buttons[attribute].GetRepresentation().SetState(0)
        setattr(self, attribute, bool(state))
        self.refresh(reset_camera=attribute == "machine_view")

    def _set_end(self, index):
        self.end = int(np.clip(index, 0, self.view.count - 1))
        self._play_position = float(self.end)
        self.refresh()

    def _set_z_max(self, value):
        self.z_max = float(value)
        self.refresh()

    def _set_speed(self, speed):
        self.speed = float(speed)
        self._draw_speed()

    def _set_playing(self, playing):
        import time

        playing = bool(playing)
        if playing and self.end >= self.view.count - 1:
            self.end = 0  # play again from the start
        self.playing = playing
        self._play_position = float(self.end)
        self._last_tick = time.perf_counter()
        if self._play_button is not None:
            self._play_button.GetRepresentation().SetState(int(playing))
            self._label("Pause" if playing else "Play",
                        (52, int(0.07 * self.window_size[1]) - 7), "play_label", size=11)
        self.refresh()

    def _step(self, delta):
        if self.playing:
            self._set_playing(False)
        self._play_position = float(self.end + delta)
        self._move_to(self.end + delta)

    def _move_to(self, index):
        self.end = int(np.clip(index, 0, self.view.count - 1))
        if self._progress_slider is not None:
            self._progress_slider.GetRepresentation().SetValue(self.end + 1)
        self.refresh()

    def _tick(self, _step):
        import time

        if not self.playing:
            return
        now = time.perf_counter()
        elapsed = now - (self._last_tick or now)
        self._last_tick = now
        self._play_position, finished = tv.playback_advance(
            self._play_position, self.speed, elapsed, self.view.count)
        self._move_to(int(self._play_position))
        if finished:
            self._set_playing(False)

    # -- drawing ----------------------------------------------------------------

    def refresh(self, reset_camera=False):
        plotter, pv, view = self.plotter, self.pv, self.view
        mode = tv.MODES_BY_KEY[self.mode]

        # In the machine view everything attached to the bed is drawn in the
        # bed frame and moved by the bed's pose (a VTK user matrix), so only
        # the pose changes as the print plays.
        pose = None
        points = view.point
        if self.machine_view:
            state = self.machine_state()
            points = view.point + state.bed_offset
            pose = self._pose_index()

        def attach(actor):
            if actor is not None and pose is not None:
                actor.user_matrix = self._pose_matrix(pose)
            return actor

        deposit = tv.visible_segments(view, self.end, z_max=self.z_max, deposit=True)
        cells = pv.PolyData(points, lines=tv.line_cells(deposit))
        cells.cell_data["value"] = self.scalars()[deposit]

        plotter.remove_actor("deposit", reset_camera=False, render=False)
        for title in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(title, render=False)
        if len(deposit):
            options = dict(line_width=3, render_lines_as_tubes=True, name="deposit",
                           scalars="value", reset_camera=False, render=False)
            if mode.categories:
                from matplotlib.colors import ListedColormap

                # Leave trailing categories nothing uses (e.g. "platform" on
                # a part without one) out of the legend.
                used = int(self.scalars()[view.deposit].max(initial=0)) + 1
                count = max(2, min(used, len(mode.categories)))
                options.update(
                    cmap=ListedColormap(list(mode.category_colours[:count])),
                    clim=[-0.5, count - 0.5],
                    annotations={float(i): name
                                 for i, name in enumerate(mode.categories[:count])},
                    scalar_bar_args=dict(title=mode.title, n_labels=0, **self._bar_position()),
                )
            else:
                options.update(
                    cmap=mode.cmap,
                    clim=tv.colour_range(self.scalars()[view.deposit], mode.key),
                    nan_color="#bdbdbd",
                    scalar_bar_args=dict(title=mode.title, fmt="%.1f", **self._bar_position()),
                )
            attach(plotter.add_mesh(cells, **options))

        # A few red segments are easy to miss among thousands, so the
        # unsupported points are also drawn as dots.
        plotter.remove_actor("unsupported_dots", reset_camera=False, render=False)
        if self.mode == "unsupported" and len(deposit):
            flagged = deposit[tv.unsupported_mask(view)[deposit]]
            if len(flagged):
                attach(plotter.add_mesh(pv.PolyData(points[flagged]), color="#ff3b30",
                                        point_size=9, render_points_as_spheres=True,
                                        name="unsupported_dots", reset_camera=False,
                                        render=False))

        plotter.remove_actor("travel", reset_camera=False, render=False)
        if self.show_travel:
            travel = tv.visible_segments(view, self.end, z_max=self.z_max, deposit=False)
            if len(travel):
                attach(plotter.add_mesh(pv.PolyData(points, lines=tv.line_cells(travel)),
                                        color="#5fd35f" if self.mode == "type" else "#8f96a3",
                                        opacity=0.6, line_width=1, name="travel",
                                        reset_camera=False, render=False))

        plotter.remove_actor("mesh", reset_camera=False, render=False)
        if self.show_mesh and self.mesh is not None:
            vertices, faces = self.mesh
            if self.machine_view:
                vertices = vertices + self.machine_state().bed_offset
            surface = pv.PolyData(vertices, np.column_stack(
                [np.full(len(faces), 3), faces]).ravel())
            attach(plotter.add_mesh(surface, color="#b0c4de", opacity=0.18, name="mesh",
                                    reset_camera=False, render=False, show_edges=False))

        for name in ("nozzle", "bed", "bed_outline", "balls", "gantry", "gantry_outline"):
            plotter.remove_actor(name, reset_camera=False, render=False)
        if self.machine_view:
            self._draw_machine(pose)
        elif self.show_nozzle and view.count and view.point[self.end, 2] <= self.z_max:
            self._draw_nozzle()

        self._draw_text()
        if reset_camera:
            plotter.view_isometric()
            plotter.reset_camera()
        plotter.render()

    def _pose_index(self):
        from atom import bed_motion

        return bed_motion.last_valid_index(self.machine_state().valid, self.end)

    def _pose_matrix(self, index):
        from atom import bed_motion

        state = self.machine_state()
        return bed_motion.pose_matrix(state.rotation[index], state.translation[index])

    def _bed_clearance(self, index):
        """Gantry clearance of the highest bed corner at pose ``index``, mm."""
        from atom import bed_motion

        state = self.machine_state()
        corners = bed_motion.bed_corners(self._profile)
        heights = bed_motion.corner_heights(state.rotation[index], state.translation[index],
                                            corners)
        return float(self._profile.nozzle_to_gantry - heights.max())

    def _draw_machine(self, pose):
        """The bed (tilting), the ball joints, the nozzle and the gantry level."""
        from math import degrees

        from atom import bed_motion, toolpath3

        pv, plotter, profile = self.pv, self.plotter, self._profile
        state = self.machine_state()
        if pose is None:
            return

        clash = (not state.valid[self.end]) or self._bed_clearance(pose) < 0
        corners = bed_motion.bed_corners(profile)
        plate = pv.PolyData(corners, faces=[4, 0, 1, 2, 3])
        matrix = self._pose_matrix(pose)
        bed = plotter.add_mesh(plate, color="#c0392b" if clash else "#5b6676", opacity=0.55,
                               name="bed", reset_camera=False, render=False)
        bed.user_matrix = matrix
        outline = plotter.add_mesh(pv.PolyData(corners, lines=[5, 0, 1, 2, 3, 0]),
                                   color="#ff6b5e" if clash else "#aab4c3", line_width=2,
                                   name="bed_outline", reset_camera=False, render=False)
        outline.user_matrix = matrix
        balls = plotter.add_mesh(pv.PolyData(bed_motion.ball_positions(profile)),
                                 color="#e8a33d", point_size=16, render_points_as_spheres=True,
                                 name="balls", reset_camera=False, render=False)
        balls.user_matrix = matrix

        # Fixed to the machine: the nozzle at (X, Y, 0), vertical, and the
        # gantry level nozzle_to_gantry above it (the reference proxy P4.1
        # also uses).
        x, y = state.machine[pose, 0], state.machine[pose, 1]
        if self.show_nozzle:
            height = 20.0
            plotter.add_mesh(pv.Cone(center=(x, y, height / 2.0), direction=(0, 0, -1),
                                     height=height,
                                     angle=degrees(toolpath3.NOZZLE_CONE_ANGLE / 2.0),
                                     resolution=48),
                             color="#d9dde3", opacity=0.8, name="nozzle",
                             reset_camera=False, render=False)
        level = float(profile.nozzle_to_gantry)
        span = np.array([[0, 0, level], [profile.max_x_axis, 0, level],
                         [profile.max_x_axis, profile.max_y_axis, level],
                         [0, profile.max_y_axis, level]], dtype=float)
        plotter.add_mesh(pv.PolyData(span, faces=[4, 0, 1, 2, 3]), color="#e8a33d",
                         opacity=0.06, name="gantry", reset_camera=False, render=False)
        plotter.add_mesh(pv.PolyData(span, lines=[5, 0, 1, 2, 3, 0]), color="#e8a33d",
                         opacity=0.5, line_width=1, name="gantry_outline",
                         reset_camera=False, render=False)

    def _bar_position(self):
        return dict(position_x=0.80, position_y=0.30, width=0.05, height=0.45,
                    vertical=True, title_font_size=12, label_font_size=11, color=TEXT)

    def _draw_speed(self):
        self._label(tv.describe_speed(self.speed, self.view.count),
                    self._speed_label_position, "speed_label", size=9)

    def _draw_nozzle(self):
        from math import degrees

        from atom import toolpath3  # imports Taichi; only needed for the constant

        point = self.view.point[self.end]
        direction = self.view.direction[self.end]
        extent = float(np.ptp(self.view.point, axis=0).max()) if self.view.count > 1 else 10.0
        height = max(3.0, 0.12 * extent)
        cone = self.pv.Cone(center=point + direction * height / 2.0, direction=-direction,
                            height=height, angle=degrees(toolpath3.NOZZLE_CONE_ANGLE / 2.0),
                            resolution=48)
        self.plotter.add_mesh(cone, color="#d9dde3", opacity=0.55, name="nozzle",
                              reset_camera=False, render=False)

    def _draw_text(self):
        view = self.view
        mode = tv.MODES_BY_KEY[self.mode]
        frame = "machine view: nozzle fixed, bed moves" if self.machine_view else view.frame
        header = [
            Path(view.source).name,
            f"{frame}   colour: {mode.label}",
        ]
        if self.mode in tv.ROBUST_RANGE_MODES:
            header.append("Colour range: 0.5th to 99.5th percentile; "
                          "values outside take the end colours.")
        if self.mode == "shell":
            header.append("Shell/infill is a guess from distance to the surface.")
        if self.mode == "unsupported" and view.count:
            share = float(np.mean(tv.unsupported_mask(view)[view.deposit])) * 100.0
            header.append(f"Unsupported: {share:.2f} % of deposition points (P0.8 metric)")
        header.extend(self.notes)
        if self.machine_view:
            header.append("Grey plate: bed (red on a clash)   orange dots: ball joints")
            header.append(f"Orange frame: gantry level, {self._profile.nozzle_to_gantry} mm "
                          "above the nozzle tip")
            if self.machine_state().solved:
                header.append("Screw values solved from this toolpath, re-centred on the bed;")
                header.append("open the G-code for the exact values the printer gets.")
        if self.message:
            header.append(self.message)
        self.plotter.add_text("\n".join(header), position="upper_left", font_size=9,
                              color=TEXT, name="header")

        status = tv.describe_point(view, self.end)
        if self.machine_view:
            state = self.machine_state()
            pose = self._pose_index()
            if not state.valid[self.end]:
                status += "\nUNREACHABLE: the inverse kinematics reject this point"
            if pose is not None:
                if view.machine is None:
                    _, _, z0, z1, z2 = state.machine[pose]
                    status += f"\nscrews Z {z0:.2f} U {z1:.2f} V {z2:.2f}"
                clearance = self._bed_clearance(pose)
                verdict = "CLASH" if clearance < 0 else "ok"
                status += f"\nbed-gantry clearance {clearance:.1f} mm ({verdict})"
        self.plotter.add_text(status, position="upper_right", font_size=9, color=TEXT,
                              name="status")
        self.plotter.add_text(
            "Left/Right: 1 point   , / . : 1 %   Space: play/pause   v: iso view   q: quit",
            position="lower_left", font_size=8, color=TEXT_DIM, name="help")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_view(args, notes):
    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"No such file: {path}")
    if path.suffix.lower() == ".npz":
        return load_npz_view(path, notes)
    toolpath = Path(args.toolpath) if args.toolpath else None
    return load_gcode_view(path, toolpath, notes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview an Atomizer toolpath (.npz) or G-code in 3D, like a slicer preview.",
    )
    parser.add_argument("path", help="A toolpath .npz or a .gcode file.")
    parser.add_argument("--toolpath", help="For G-code: the _platform.npz it was written from "
                        "(found automatically under data/toolpath/ when omitted).")
    parser.add_argument("--stl", help="The part's STL (default: data/mesh/<part>.stl).")
    parser.add_argument("--no-stl", action="store_true", help="Do not load the STL.")
    parser.add_argument("--mode", default="progress", choices=[m.key for m in tv.COLOUR_MODES],
                        help="Colour mode to start in.")
    parser.add_argument("--end", help="Show the print up to this point: a number or a "
                        "percentage such as 50%%. Default: the whole print.")
    parser.add_argument("--z-max", type=float, help="Hide everything above this height, mm.")
    parser.add_argument("--hide-travel", action="store_true", help="Start with travel moves hidden.")
    parser.add_argument("--no-nozzle", action="store_true", help="Do not draw the nozzle cone.")
    parser.add_argument("--machine-view", action="store_true",
                        help="Start in the machine view: nozzle fixed, the bed tilting beneath it.")
    parser.add_argument("--screenshot", help="Render to this PNG and exit, without a window.")
    args = parser.parse_args(argv)

    notes = []
    view = build_view(args, notes)
    print(f"{view.count:,} points ({int(view.deposit.sum()):,} printing) from {view.source}")

    mesh = None
    if not args.no_stl:
        stl = Path(args.stl) if args.stl else find_stl(tv.part_name_from_path(args.path),
                                                       Path(args.path))
        if stl is None:
            notes.append("No STL found for this part (use --stl).")
        elif view.mesh_offset is None:
            pass  # the reason is already in the notes
        else:
            mesh = load_mesh(stl, view.mesh_offset)

    reason = tv.mode_unavailable_reason(view, args.mode, mesh is not None)
    if reason:
        raise SystemExit(f"--mode {args.mode}: {reason}")

    for note in notes:
        print(note)

    viewer = Viewer(view, mesh, mode=args.mode, end=parse_end(args.end, view.count),
                    z_max=args.z_max, show_travel=not args.hide_travel,
                    show_nozzle=not args.no_nozzle, machine_view=args.machine_view,
                    notes=notes)
    if args.screenshot:
        plotter = viewer.build(off_screen=True)
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f"Wrote {args.screenshot}")
        return 0

    viewer.build()
    viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
