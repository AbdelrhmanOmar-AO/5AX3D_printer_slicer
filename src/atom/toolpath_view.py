"""What the toolpath viewer draws, independent of how it is drawn (build plan P5.4a).

`tools/visualize_5ax.py` is the window; this module is everything behind it
that can be tested without a display: turning a toolpath into per-point
arrays, choosing which segments are visible, and the quantity each colour mode
shows.

numpy and scipy only: no Taichi, no pyvista, no trimesh. Reading G-code and
running the forward kinematics happen in the tool, which hands the results in
as plain arrays.

Segments
--------
Segment ``i`` runs from point ``i - 1`` to point ``i`` and takes point ``i``'s
attributes, matching how `tools/toolpath_to_gcode.py` writes one ``G1`` per
point: the move *to* point ``i`` deposits if point ``i`` is a deposition point.

Frames
------
Toolpath files are in the **part frame**, the frame the STL was moved into by
`tools/process_for_atomizer.py` (bounding-box minimum at the origin), except
``<part>_platform.npz``, which `tools/add_platform.py` lifts by the platform
height. G-code is in the bed-centred frame `toolpath_to_gcode` re-centres into
(`docs/conventions.md` section 1). `ViewData.mesh_offset` is the translation
that puts the STL into whichever frame the view is in, or ``None`` when that is
not known.

Units: millimetres and degrees throughout.
"""

from __future__ import annotations  # no Taichi kernels in this module

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import overhang_metrics as om

#: `toolpath3.TRAVEL_TYPE_DEPOSITION`, repeated so this module stays free of
#: Taichi (as `overhang_metrics` does).
TRAVEL_TYPE_DEPOSITION = om.TRAVEL_TYPE_DEPOSITION

#: Solid shell kept around the gyroid infill, in deposition widths.
#: `tools/sdf_to_isdf.py` hard-codes ``shell_thickness = 2`` and
#: `fff3.sdf_generate_infill` turns it into ``wall_width = shell *
#: deposition_width``. Build plan P1.3 will make it a parameter; pass the
#: real value to `shell_mask` once it is one.
SHELL_THICKNESS_WIDTHS = 2.0

#: Below this tilt the azimuth is meaningless (the tool is effectively
#: vertical), so the tilt-direction mode leaves those points uncoloured.
AZIMUTH_MIN_TILT_DEG = 0.1

#: File-name endings the pipeline and the P0.8 archive add to a part's name.
_SUFFIXES = ("_smoothed_tesselated", "_smoothed", "_platform", "_craftware")
_ARCHIVE_SUFFIX = re.compile(r"_ms\d+(?:\.\d+)?$")


def part_name_from_path(path) -> str:
    """The part name a toolpath or G-code file belongs to.

    ``data/toolpath/ramp60_xs_smoothed.npz``, ``data/gcode/ramp60_xs.gcode`` and
    the P0.8 archive copy ``reports/toolpaths/ramp60_xs_ms30.npz`` all give
    ``ramp60_xs``, which is how the matching STL is found.
    """
    # Everything before the first dot: the golden archive is
    # ``calibration_cube.toolpath.npz``.
    name = Path(path).name.split(".", 1)[0]
    changed = True
    while changed:
        changed = False
        stripped = _ARCHIVE_SUFFIX.sub("", name)
        if stripped != name:
            name, changed = stripped, True
        for suffix in _SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name, changed = name[: -len(suffix)], True
    return name


# --------------------------------------------------------------------------
# Per-point data
# --------------------------------------------------------------------------


@dataclass
class ViewData:
    """Per-point arrays for one toolpath, in print order.

    ``width``, ``height``, ``feed_mm_min``, ``gcode_line`` and ``machine`` are
    ``None`` when the source does not carry them: a toolpath file has no feed
    rate, and G-code has no bead width or height.
    """

    #: ``(N, 3)`` nozzle positions, mm.
    point: np.ndarray
    #: ``(N,)`` True where the move to this point deposits material.
    deposit: np.ndarray
    #: ``(N, 3)`` unit build directions (the tool orientation, Cartesian).
    direction: np.ndarray
    #: ``(N,)`` True for points of the sacrificial platform `add_platform` adds.
    is_platform: np.ndarray
    #: Human-readable name of the coordinate frame, shown in the window.
    frame: str
    #: Where the data came from, shown in the window.
    source: str
    #: Translation that puts the STL into this frame, mm; None if unknown.
    mesh_offset: np.ndarray | None = None
    width: np.ndarray | None = None
    height: np.ndarray | None = None
    #: ``(N,)`` feed rate as written in the G-code, mm/min. This is the
    #: machine-axis feed after `calculate_feedrate`'s ``F * d / l``
    #: compensation, not the speed of the nozzle over the part.
    feed_mm_min: np.ndarray | None = None
    #: ``(N,)`` 1-based G-code line number of each point's move.
    gcode_line: np.ndarray | None = None
    #: ``(N, 5)`` machine axes X, Y, Z, U, V as written in the G-code.
    machine: np.ndarray | None = None
    #: Translation from this view's frame to the bed frame the kinematics use
    #: (`atom.bed_motion`), mm; None until known. For G-code it is the bed
    #: re-centring `toolpath_to_gcode` applied (zero if the view is already
    #: in that frame).
    bed_offset: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def count(self) -> int:
        return len(self.point)

    @property
    def tilt_deg(self) -> np.ndarray:
        """Tilt of the build direction from +Z per point, degrees."""
        if "tilt" not in self._cache:
            self._cache["tilt"] = om.tilt_from_vertical_deg(self.direction)
        return self._cache["tilt"]

    @property
    def azimuth_deg(self) -> np.ndarray:
        """Direction the tool leans toward, degrees from +X in [0, 360).

        NaN where the tilt is below `AZIMUTH_MIN_TILT_DEG`, since a vertical
        tool leans nowhere.
        """
        if "azimuth" not in self._cache:
            azimuth = np.degrees(np.arctan2(self.direction[:, 1], self.direction[:, 0]))
            azimuth = np.mod(azimuth, 360.0)
            azimuth[self.tilt_deg < AZIMUTH_MIN_TILT_DEG] = np.nan
            self._cache["azimuth"] = azimuth
        return self._cache["azimuth"]


def from_arrays(
    point,
    travel_type,
    tool_orientation,
    width=None,
    height=None,
    platform_count: int = 0,
    frame: str = "part frame",
    source: str = "",
    mesh_offset=(0.0, 0.0, 0.0),
) -> ViewData:
    """Build a `ViewData` from the arrays a `toolpath3.Toolpath` stores.

    ``tool_orientation`` is ``(N, 2)`` spherical ``[theta, phi]`` in radians,
    as Atomizer stores it (`docs/conventions.md` section 2). The first
    ``platform_count`` points are marked as platform.
    """
    point = np.asarray(point, dtype=np.float64)
    count = len(point)
    is_platform = np.zeros(count, dtype=bool)
    is_platform[: max(0, min(int(platform_count), count))] = True

    return ViewData(
        point=point,
        deposit=np.asarray(travel_type) == TRAVEL_TYPE_DEPOSITION,
        direction=om.spherical_to_cartesian(tool_orientation)
        if count
        else np.zeros((0, 3)),
        is_platform=is_platform,
        frame=frame,
        source=source,
        mesh_offset=None if mesh_offset is None else np.asarray(mesh_offset, float),
        width=None if width is None else np.asarray(width, dtype=np.float64),
        height=None if height is None else np.asarray(height, dtype=np.float64),
    )


def load_toolpath_arrays(path) -> dict:
    """The arrays of a toolpath ``.npz``, cut to its ``point_count``.

    Reads the file with numpy alone, so viewing a toolpath does not start
    Taichi. `toolpath3.Toolpath.allocate` pads with NaN beyond the count.
    """
    with np.load(path) as data:
        count = int(np.asarray(data["point_count"]).item())
        arrays = {
            key: np.asarray(data[key])[:count]
            for key in ("point", "travel_type", "tool_orientation", "width", "height")
        }
    arrays["point_count"] = count
    return arrays


def platform_split(platform_points, tesselated_points):
    """Platform point count and lift, from a ``_platform`` toolpath and its input.

    `tools/add_platform.py` writes the platform first, then every point of
    ``<part>_smoothed_tesselated.npz`` lifted by the platform height. Returns
    ``(platform_count, lift_mm)``, or ``None`` if the two files do not fit that
    pattern (e.g. one is from a different run).
    """
    platform_points = np.asarray(platform_points, dtype=np.float64)
    tesselated_points = np.asarray(tesselated_points, dtype=np.float64)
    extra = len(platform_points) - len(tesselated_points)
    if extra < 0 or len(tesselated_points) == 0:
        return None

    lifted = platform_points[extra:]
    lift = lifted[:, 2] - tesselated_points[:, 2]
    same_xy = np.allclose(lifted[:, :2], tesselated_points[:, :2], atol=1e-4)
    if not same_xy or np.ptp(lift) > 1e-3:
        return None
    return extra, float(np.median(lift))


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------


def visible_segments(
    view: ViewData,
    end: int,
    start: int = 1,
    z_max: float = np.inf,
    deposit: bool = True,
) -> np.ndarray:
    """Indices ``i`` of the segments ``(i - 1, i)`` to draw.

    ``end`` and ``start`` are point indices (inclusive): scrubbing to ``end``
    shows the print up to and including the move to that point. A segment is
    clipped when either end lies above ``z_max``. ``deposit`` selects printing
    moves (True) or travel moves (False).
    """
    count = view.count
    if count < 2:
        return np.zeros(0, dtype=np.int64)

    start = max(1, int(start))
    end = min(count - 1, int(end))
    if end < start:
        return np.zeros(0, dtype=np.int64)

    index = np.arange(start, end + 1)
    kind = view.deposit[index] if deposit else ~view.deposit[index]
    z = view.point[:, 2]
    below = np.maximum(z[index], z[index - 1]) <= z_max
    return index[kind & below]


def line_cells(indices: np.ndarray) -> np.ndarray:
    """VTK ``lines`` connectivity for segments ``(i - 1, i)``: ``[2, i-1, i, ...]``."""
    indices = np.asarray(indices, dtype=np.int64)
    return np.column_stack(
        [np.full(len(indices), 2, dtype=np.int64), indices - 1, indices]
    ).ravel()


# --------------------------------------------------------------------------
# Derived per-point quantities
# --------------------------------------------------------------------------


def unsupported_mask(view: ViewData) -> np.ndarray:
    """True where a deposition point prints into air, by the P0.8 metric.

    Exactly `overhang_metrics.unsupported_deposition`: supported by the bed
    within one layer of the lowest deposition, or by earlier material inside
    the 65-degree cone beneath the point. Needs bead heights, so it is only
    available when the view has them. Travel points are always False.
    """
    if view.height is None:
        raise ValueError("unsupported points need bead heights (a toolpath .npz)")
    if "unsupported" not in view._cache:
        toolpath = SimpleNamespace(
            point=view.point,
            travel_type=np.where(view.deposit, TRAVEL_TYPE_DEPOSITION, 1),
            tool_orientation=_spherical_from_directions(view.direction),
            height=view.height,
            point_count=view.count,
        )
        result = om.unsupported_deposition(toolpath)
        mask = np.zeros(view.count, dtype=bool)
        mask[view.deposit] = result.unsupported
        view._cache["unsupported"] = mask
    return view._cache["unsupported"]


def surface_distance(points: np.ndarray, surface_samples: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest surface sample, mm.

    ``surface_samples`` must be dense (build plan hazard 16): the tool
    subdivides the STL to half a deposition width, so the error is at most a
    quarter width.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0)
    distance, _ = cKDTree(np.asarray(surface_samples, dtype=np.float64)).query(
        points, workers=-1
    )
    return distance


def shell_mask(
    view: ViewData,
    surface_samples: np.ndarray,
    deposition_width: float,
    shell_widths: float = SHELL_THICKNESS_WIDTHS,
) -> np.ndarray:
    """Best guess at which deposition points belong to the solid shell.

    **A heuristic.** Atomizer does not record whether a bead is shell or
    infill. This uses the rule its infill stage applies: material within
    ``shell_widths`` deposition widths of the surface is solid shell, the rest
    is gyroid infill. It can misjudge beads near thin features and where the
    infill meets the shell. ``surface_samples`` are in the view's frame.
    Platform and travel points are always False.
    """
    mask = np.zeros(view.count, dtype=bool)
    candidates = view.deposit & ~view.is_platform
    if np.any(candidates):
        distance = surface_distance(view.point[candidates], surface_samples)
        mask[candidates] = distance <= shell_widths * deposition_width + 1e-9
    return mask


def _spherical_from_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    return np.column_stack([theta, phi])


def typical_deposition_width(view: ViewData, fallback: float | None = None) -> float | None:
    """Median bead width of the part's deposition points, mm."""
    if view.width is None:
        return fallback
    widths = view.width[view.deposit & ~view.is_platform]
    widths = widths[np.isfinite(widths) & (widths > 0)]
    return float(np.median(widths)) if len(widths) else fallback


# --------------------------------------------------------------------------
# Colour modes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourMode:
    """One way of colouring the printing moves."""

    key: str
    label: str
    #: Scalar-bar title, with units.
    title: str
    cmap: str
    #: Names for a categorical mode's values 0, 1, ...; empty if continuous.
    categories: tuple = ()
    #: Colours for the categories, same order.
    category_colours: tuple = ()


COLOUR_MODES = (
    ColourMode("progress", "Print order", "Print order (%)", "turbo"),
    ColourMode(
        "type",
        "Part / platform",
        "Line type",
        "",
        ("part", "platform"),
        ("#f28e2b", "#9c9c9c"),
    ),
    ColourMode("tilt", "Tilt", "Tilt from vertical (deg)", "viridis"),
    ColourMode("azimuth", "Tilt direction", "Tool leans toward (deg from +X)", "hsv"),
    ColourMode("width", "Bead width", "Bead width (mm)", "cividis"),
    ColourMode("height", "Bead height", "Bead height (mm)", "cividis"),
    ColourMode("feed", "Feed rate", "Feed as written (mm/min)", "plasma"),
    ColourMode(
        "unsupported",
        "Unsupported",
        "Printed into air",
        "",
        ("supported", "unsupported"),
        ("#c8c8c8", "#e31a1c"),
    ),
    ColourMode(
        "shell",
        "Shell / infill (guess)",
        "Shell or infill (heuristic)",
        "",
        ("infill", "shell", "platform"),
        ("#4e79a7", "#f28e2b", "#9c9c9c"),
    ),
)

MODES_BY_KEY = {mode.key: mode for mode in COLOUR_MODES}


def mode_unavailable_reason(view: ViewData, key: str, has_mesh: bool) -> str | None:
    """Why a colour mode cannot be shown for this view, or None if it can."""
    if key in ("width", "height", "unsupported") and view.height is None:
        return "needs a toolpath .npz (G-code has no bead sizes)"
    if key == "feed" and view.feed_mm_min is None:
        return "needs G-code (toolpath files have no feed rate)"
    if key == "shell":
        if not has_mesh or view.mesh_offset is None:
            return "needs the part's STL, aligned with this toolpath"
        if typical_deposition_width(view) is None:
            return "needs bead widths (a toolpath .npz)"
    return None


def point_scalars(view: ViewData, key: str, shell: np.ndarray | None = None) -> np.ndarray:
    """The value each point is coloured by in mode ``key``, ``(N,)``.

    Categorical modes return integer codes indexing `ColourMode.categories`.
    ``shell`` is the output of `shell_mask`, required for the ``shell`` mode.
    """
    count = view.count
    if key == "progress":
        return np.linspace(0.0, 100.0, count) if count > 1 else np.zeros(count)
    if key == "type":
        return view.is_platform.astype(np.int64)
    if key == "tilt":
        return view.tilt_deg
    if key == "azimuth":
        return view.azimuth_deg
    if key == "width":
        return _require(view.width, key)
    if key == "height":
        return _require(view.height, key)
    if key == "feed":
        return _require(view.feed_mm_min, key)
    if key == "unsupported":
        return unsupported_mask(view).astype(np.int64)
    if key == "shell":
        if shell is None:
            raise ValueError("the shell mode needs the result of shell_mask")
        codes = np.where(shell, 1, 0)
        codes[view.is_platform] = 2
        return codes.astype(np.int64)
    raise KeyError(f"unknown colour mode {key!r}")


def _require(values, key):
    if values is None:
        raise ValueError(f"colour mode {key!r} is not available for this view")
    return values


#: Modes whose colour range ignores the extreme 0.5 % at each end. A handful
#: of feed-rate spikes (where the tilt changes quickly, ``F * d / l`` grows
#: large) otherwise squeeze every other move into one colour. Values beyond
#: the range take the end colours.
ROBUST_RANGE_MODES = ("width", "height", "feed")
ROBUST_PERCENTILES = (0.5, 99.5)


def colour_range(values: np.ndarray, key: str) -> tuple[float, float]:
    """Colour-bar limits for a continuous mode."""
    if key == "progress":
        return 0.0, 100.0
    if key == "azimuth":
        return 0.0, 360.0
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    if key in ROBUST_RANGE_MODES:
        low, high = (float(v) for v in np.percentile(finite, ROBUST_PERCENTILES))
    else:
        low, high = float(finite.min()), float(finite.max())
    if key == "tilt":
        low = 0.0
    if high - low < 1e-9:
        high = low + (1.0 if key == "tilt" else max(abs(low) * 0.01, 1e-3))
    return low, high


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------

#: Default playback speed: the whole print in this many seconds.
DEFAULT_PLAYBACK_S = 60.0
#: Fastest selectable speed: the whole print in this many seconds.
FASTEST_PLAYBACK_S = 5.0


def speed_limits(count: int) -> tuple[float, float]:
    """Slowest and fastest playback speeds offered, points per second."""
    return 1.0, max(10.0, count / FASTEST_PLAYBACK_S)


def default_speed(count: int) -> float:
    """Playback speed that shows the whole print in `DEFAULT_PLAYBACK_S`."""
    low, high = speed_limits(count)
    return float(np.clip(count / DEFAULT_PLAYBACK_S, low, high))


def playback_advance(position: float, speed: float, elapsed_s: float, count: int):
    """Advance playback by wall-clock time, so the speed holds however fast
    the window redraws.

    ``position`` is a fractional point index. Returns ``(new_position,
    finished)``; playback finishes on the last point.
    """
    last = max(count - 1, 0)
    position = min(float(position) + max(speed, 0.0) * max(elapsed_s, 0.0), float(last))
    return position, position >= last


def describe_speed(speed: float, count: int) -> str:
    """Speed as shown next to its slider."""
    seconds = count / speed if speed > 0 else float("inf")
    if seconds >= 120:
        whole = f"{seconds / 60:.1f} min"
    else:
        whole = f"{seconds:.0f} s"
    return f"Speed: {speed:,.0f} points/s (whole print in {whole})"


# --------------------------------------------------------------------------
# Text shown in the window
# --------------------------------------------------------------------------


def point_fields(view: ViewData, index: int) -> list[tuple[str, str]]:
    """The current point as ``(label, value)`` rows, for a side panel."""
    if view.count == 0:
        return [("Point", "empty toolpath")]
    index = int(np.clip(index, 0, view.count - 1))
    x, y, z = view.point[index]
    kind = "print" if view.deposit[index] else "travel"
    if view.is_platform[index]:
        kind += " (platform)"
    azimuth = view.azimuth_deg[index]
    fields = [
        ("Point", f"{index + 1:,} / {view.count:,}"),
        ("Move", kind),
        ("Position (mm)", f"{x:.2f}, {y:.2f}, {z:.2f}"),
        ("Tilt", f"{view.tilt_deg[index]:.1f}\N{DEGREE SIGN}"),
        ("Leans toward", "-" if np.isnan(azimuth) else f"{azimuth:.0f}\N{DEGREE SIGN} from +X"),
    ]
    if view.width is not None:
        fields.append(("Bead (mm)", f"{view.width[index]:.2f} x {view.height[index]:.2f}"))
    if view.feed_mm_min is not None:
        fields.append(("Feed (mm/min)", f"{view.feed_mm_min[index]:.0f}"))
    if view.gcode_line is not None:
        fields.append(("G-code line", f"{view.gcode_line[index]:,}"))
    if view.machine is not None:
        _, _, z0, z1, z2 = view.machine[index]
        fields.append(("Screws Z, U, V", f"{z0:.2f}, {z1:.2f}, {z2:.2f}"))
    return fields


def describe_point(view: ViewData, index: int) -> str:
    """One point's state, for the window's status line."""
    index = int(np.clip(index, 0, max(view.count - 1, 0)))
    if view.count == 0:
        return "empty toolpath"

    x, y, z = view.point[index]
    kind = "print" if view.deposit[index] else "travel"
    if view.is_platform[index]:
        kind += " (platform)"
    tilt = view.tilt_deg[index]
    azimuth = view.azimuth_deg[index]
    toward = "" if np.isnan(azimuth) else f" toward {azimuth:.0f} deg"
    percent = 100.0 * index / max(view.count - 1, 1)

    lines = [
        f"Point {index + 1:,} of {view.count:,} ({percent:.1f} %)  {kind}",
        f"x {x:.2f}  y {y:.2f}  z {z:.2f} mm   tilt {tilt:.1f} deg{toward}",
    ]
    extras = []
    if view.width is not None:
        extras.append(f"width {view.width[index]:.2f} mm")
    if view.height is not None:
        extras.append(f"height {view.height[index]:.2f} mm")
    if view.feed_mm_min is not None:
        extras.append(f"F {view.feed_mm_min[index]:.0f} mm/min")
    if view.gcode_line is not None:
        extras.append(f"G-code line {view.gcode_line[index]:,}")
    if view.machine is not None:
        _, _, z0, z1, z2 = view.machine[index]
        extras.append(f"screws Z {z0:.2f} U {z1:.2f} V {z2:.2f}")
    if extras:
        lines.append("   ".join(extras))
    return "\n".join(lines)
