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


class Viewer:
    """The pyvista window. All state changes go through `refresh`."""

    def __init__(self, view, mesh=None, mode="progress", end=None, z_max=None,
                 show_travel=True, show_mesh=True, show_nozzle=True, notes=()):
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
        self.playing = False
        self.message = ""
        self._shell = None
        self._mode_buttons = {}
        self._toggle_buttons = {}
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

    # -- building the scene -------------------------------------------------

    def build(self, off_screen=False, window_size=(1500, 950)):
        import pyvista as pv

        self.pv = pv
        plotter = pv.Plotter(off_screen=off_screen, window_size=window_size,
                             title=f"Toolpath viewer: {Path(self.view.source).name}")
        self.plotter = plotter
        plotter.set_background("white")
        plotter.add_axes()

        self._point_cloud = pv.PolyData(self.view.point)

        self._add_controls()
        self.refresh(reset_camera=True)
        return plotter

    def _add_controls(self):
        plotter = self.plotter
        count = self.view.count

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
        )

        plotter.add_text("Z clip (mm)", position=(0.925, 0.88), viewport=True,
                         font_size=10, color="black", name="z_clip_label")

        # Colour modes as radio buttons, then the show/hide toggles.
        top = 800
        plotter.add_text("Colour by", position=(12, top + 8), font_size=11,
                         color="black", name="colour_header")
        for row, mode in enumerate(tv.COLOUR_MODES):
            y = top - (row + 1) * _ROW
            reason = self.unavailable(mode.key)
            widget = plotter.add_checkbox_button_widget(
                lambda state, key=mode.key: self._pick_mode(key, state),
                value=mode.key == self.mode,
                position=(12, y), size=_BUTTON, border_size=2,
                color_on="#1f77b4", color_off="#e0e0e0", background_color="grey",
            )
            self._mode_buttons[mode.key] = widget
            plotter.add_text(mode.label, position=(12 + _BUTTON + 8, y + 3), font_size=9,
                             color="#a0a0a0" if reason else "black",
                             name=f"mode_label_{mode.key}")

        toggles = (
            ("Travel moves", "show_travel"),
            ("STL overlay", "show_mesh"),
            ("Nozzle cone", "show_nozzle"),
        )
        base = top - (len(tv.COLOUR_MODES) + 2) * _ROW
        plotter.add_text("Show", position=(12, base + 8), font_size=11, color="black",
                         name="show_header")
        for row, (label, attribute) in enumerate(toggles):
            y = base - (row + 1) * _ROW
            disabled = attribute == "show_mesh" and self.mesh is None
            self._toggle_buttons[attribute] = plotter.add_checkbox_button_widget(
                lambda state, a=attribute: self._toggle(a, state),
                value=getattr(self, attribute),
                position=(12, y), size=_BUTTON, border_size=2,
                color_on="#2ca02c", color_off="#e0e0e0", background_color="grey",
            )
            plotter.add_text(label + (" (no STL)" if disabled else ""),
                             position=(12 + _BUTTON + 8, y + 3), font_size=9,
                             color="#a0a0a0" if disabled else "black",
                             name=f"toggle_label_{attribute}")

        plotter.add_key_event("Left", lambda: self._step(-1))
        plotter.add_key_event("Right", lambda: self._step(1))
        plotter.add_key_event("comma", lambda: self._step(-max(1, count // 100)))
        plotter.add_key_event("period", lambda: self._step(max(1, count // 100)))
        plotter.add_key_event("space", self._toggle_play)
        plotter.add_timer_event(max_steps=10**9, duration=40, callback=self._tick)

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
        self.refresh()

    def _set_end(self, index):
        self.end = int(np.clip(index, 0, self.view.count - 1))
        self.refresh()

    def _set_z_max(self, value):
        self.z_max = float(value)
        self.refresh()

    def _step(self, delta):
        self.playing = False
        self._move_to(self.end + delta)

    def _move_to(self, index):
        self.end = int(np.clip(index, 0, self.view.count - 1))
        if self._progress_slider is not None:
            self._progress_slider.GetRepresentation().SetValue(self.end + 1)
        self.refresh()

    def _toggle_play(self):
        if not self.playing and self.end >= self.view.count - 1:
            self.end = 0
        self.playing = not self.playing

    def _tick(self, _step):
        if not self.playing:
            return
        if self.end >= self.view.count - 1:
            self.playing = False
            return
        self._move_to(self.end + max(1, self.view.count // 600))

    # -- drawing ----------------------------------------------------------------

    def refresh(self, reset_camera=False):
        plotter, pv, view = self.plotter, self.pv, self.view
        mode = tv.MODES_BY_KEY[self.mode]

        deposit = tv.visible_segments(view, self.end, z_max=self.z_max, deposit=True)
        cells = pv.PolyData(view.point, lines=tv.line_cells(deposit))
        values = self.scalars()[deposit]
        cells.cell_data["value"] = values

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
            plotter.add_mesh(cells, **options)

        # A few red segments are easy to miss among thousands, so the
        # unsupported points are also drawn as dots.
        plotter.remove_actor("unsupported_dots", reset_camera=False, render=False)
        if self.mode == "unsupported" and len(deposit):
            flagged = deposit[tv.unsupported_mask(view)[deposit]]
            if len(flagged):
                plotter.add_mesh(pv.PolyData(view.point[flagged]), color="#e31a1c",
                                 point_size=9, render_points_as_spheres=True,
                                 name="unsupported_dots", reset_camera=False, render=False)

        plotter.remove_actor("travel", reset_camera=False, render=False)
        if self.show_travel:
            travel = tv.visible_segments(view, self.end, z_max=self.z_max, deposit=False)
            if len(travel):
                plotter.add_mesh(pv.PolyData(view.point, lines=tv.line_cells(travel)),
                                 color="#2ca02c" if self.mode == "type" else "#9e9e9e",
                                 opacity=0.6, line_width=1, name="travel",
                                 reset_camera=False, render=False)

        plotter.remove_actor("mesh", reset_camera=False, render=False)
        if self.show_mesh and self.mesh is not None:
            vertices, faces = self.mesh
            surface = pv.PolyData(vertices, np.column_stack(
                [np.full(len(faces), 3), faces]).ravel())
            plotter.add_mesh(surface, color="#b0c4de", opacity=0.18, name="mesh",
                             reset_camera=False, render=False, show_edges=False)

        plotter.remove_actor("nozzle", reset_camera=False, render=False)
        if self.show_nozzle and view.count and view.point[self.end, 2] <= self.z_max:
            self._draw_nozzle()

        self._draw_text()
        if reset_camera:
            plotter.view_isometric()
            plotter.reset_camera()
        plotter.render()

    def _bar_position(self):
        return dict(position_x=0.80, position_y=0.30, width=0.05, height=0.45,
                    vertical=True, title_font_size=12, label_font_size=11, color="black")

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
        self.plotter.add_mesh(cone, color="#555555", opacity=0.55, name="nozzle",
                              reset_camera=False, render=False)

    def _draw_text(self):
        view = self.view
        mode = tv.MODES_BY_KEY[self.mode]
        header = [
            Path(view.source).name,
            f"{view.frame}   colour: {mode.label}",
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
        if self.message:
            header.append(self.message)
        self.plotter.add_text("\n".join(header), position="upper_left", font_size=9,
                              color="black", name="header")
        self.plotter.add_text(tv.describe_point(view, self.end), position="upper_right",
                              font_size=9, color="black", name="status")
        self.plotter.add_text(
            "Left/Right: 1 point   , / . : 1 %   Space: play   v: iso view   q: quit",
            position="lower_left", font_size=8, color="#555555", name="help")


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
                    show_nozzle=not args.no_nozzle, notes=notes)
    if args.screenshot:
        plotter = viewer.build(off_screen=True)
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f"Wrote {args.screenshot}")
        return 0

    viewer.build().show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
