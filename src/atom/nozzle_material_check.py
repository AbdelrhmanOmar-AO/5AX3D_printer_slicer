"""Nozzle against material already printed, at any tilt (build plan P4.3).

Atomizer's planner (`order_atoms`) avoids nozzle collisions with a cone in each
atom's frame, but it was tuned for small tilts. This checks the finished
toolpath independently: at every position the nozzle visits, is any material
printed earlier inside the nozzle?

numpy and scipy only; no Taichi. The part frame is used throughout, so no
kinematics are needed: the nozzle is placed at each toolpath point along that
point's own tool orientation.

The nozzle
----------
The nozzle body is the cone of `atom.clearance`: apex at the toolpath point,
axis along the tool orientation ``d`` (the build direction, pointing away from
the layer being printed, i.e. up the nozzle), half-angle
``clearance.NOZZLE_HALF_ANGLE_DEG`` (Atomizer's own), and capped at
``nozzle_to_gantry`` along the axis. Beyond that cap is the gantry, whose
check is the swept one (P4.2).

As in Atomizer (`toolpath3.toolpath_planner_init_unaccessibility`), the apex
is moved ``APEX_OFFSET_MM`` (0.001 mm) along ``d``, so a material point that
coincides with the nozzle position is not counted as inside it.

Which material counts
---------------------
`toolpath3.Toolpath.travel_type[i]` describes the move *to* point ``i``: it
deposits when ``travel_type[i] == TRAVEL_TYPE_DEPOSITION``. A deposition move
``j`` lays material from ``p[j-1]`` to ``p[j]``, so both endpoints are
material from move ``j`` onward. With the nozzle at ``p[i]``, the moves up to
``i`` are done, but move ``i``'s own bead ends under the nozzle, so only
material from moves up to ``i - 1`` is tested. Every toolpath point is
checked, printing or travelling: a travel that ends inside material is as
much a collision as a print move. Travel *between* points is the swept check
(P4.2).

Speed
-----
The cone is covered by a chain of spheres along its axis, each covering one
slab of it, so a KD-tree query returns only material near the cone (and,
beyond the first slab, only material above the nozzle tip's plane, which is
rare in a layer-by-layer print). The tree is rebuilt every ``block_size``
positions from the material printed before the block; material printed within
the block is compared pairwise. ``subsample_mm`` optionally keeps one point
per voxel, the earliest printed, to trade exactness for speed on large parts.

Units: millimetres and degrees at every public boundary.
"""

from __future__ import annotations  # no Taichi kernels in this module

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import clearance
from . import overhang_metrics as om

#: Atomizer's own apex offset along the tool direction, mm
#: (``p_i + n_i * 0.001`` in `toolpath3`).
APEX_OFFSET_MM = 0.001

#: The covering spheres: the first slab reaches this far up the axis, mm, and
#: each later slab ends at most this many times as far up as it starts.
#:
#: A sphere covering the slab from ``a`` to ``q a`` stays above the nozzle
#: tip's plane only while ``q <= 1 / tan(half_angle)^2``: 1.42 for the
#: 40-degree nozzle, so 1.4 keeps every sphere past the first clear of the
#: layer being printed on. `cone_slabs` lowers the ratio for wider cones. At
#: 45 degrees and beyond no ratio works, the spheres reach below the tip, and
#: the check slows to comparing most point pairs (still correct).
FIRST_SLAB_MM = 1.0
SLAB_RATIO = 1.4

#: Nozzle positions per KD-tree query, to bound memory when a wide cone makes
#: the spheres return many points.
_QUERY_CHUNK = 512

#: Not material: earlier than every move, or never deposited.
_NEVER = np.iinfo(np.int64).max


@dataclass
class NozzleCheckSettings:
    """Parameters of the check. Lengths in mm, angles in degrees."""

    #: Cone height along the tool axis: the gantry's height above the tip.
    height_mm: float
    half_angle_deg: float = clearance.NOZZLE_HALF_ANGLE_DEG
    apex_offset_mm: float = APEX_OFFSET_MM
    #: Material must reach at least this far inside the cone to count.
    tolerance_mm: float = 0.0
    #: Voxel size for thinning the material, or None to use every point.
    subsample_mm: float | None = None
    #: Nozzle positions per KD-tree rebuild.
    block_size: int = 512

    @classmethod
    def for_profile(cls, profile, **overrides) -> "NozzleCheckSettings":
        """Settings sized from a machine profile's ``nozzle_to_gantry``."""
        return cls(height_mm=float(profile.nozzle_to_gantry), **overrides)


@dataclass
class NozzleCollisions:
    """Result of `check`. One entry per colliding nozzle position.

    Attributes
    ----------
    checked
        How many nozzle positions were checked (every toolpath point).
    index
        ``(K,)`` toolpath indices where material is inside the nozzle, in
        print order.
    deposit
        ``(K,)`` True where the move to that point prints, False for travel.
    blocker
        ``(K,)`` toolpath index of the material point deepest inside.
    depth_mm
        ``(K,)`` how far inside the cone that point is (positive).
    axial_mm
        ``(K,)`` its height along the tool axis above the nozzle tip.
    blocker_count
        ``(K,)`` how many earlier material points are inside.
    tilt_deg
        ``(K,)`` the tool's tilt from vertical at that position.
    """

    checked: int
    index: np.ndarray
    deposit: np.ndarray
    blocker: np.ndarray
    depth_mm: np.ndarray
    axial_mm: np.ndarray
    blocker_count: np.ndarray
    tilt_deg: np.ndarray
    settings: NozzleCheckSettings = field(repr=False, default=None)

    @property
    def count(self) -> int:
        return int(len(self.index))

    @property
    def ok(self) -> bool:
        return self.count == 0

    def to_dict(self, limit: int | None = 100) -> dict:
        """A JSON-ready summary; ``limit`` caps how many collisions are listed."""
        order = np.argsort(-self.depth_mm, kind="stable")
        listed = order if limit is None else order[:limit]
        return {
            "check": "nozzle_vs_material",
            "ok": self.ok,
            "checked_positions": self.checked,
            "collisions": self.count,
            "collisions_while_printing": int(np.count_nonzero(self.deposit)),
            "collisions_while_travelling": int(np.count_nonzero(~self.deposit)),
            "max_depth_mm": float(self.depth_mm.max()) if self.count else 0.0,
            "settings": asdict(self.settings) if self.settings else None,
            "deepest": [
                {
                    "index": int(self.index[k]),
                    "kind": "print" if self.deposit[k] else "travel",
                    "blocker_index": int(self.blocker[k]),
                    "depth_mm": round(float(self.depth_mm[k]), 4),
                    "height_above_tip_mm": round(float(self.axial_mm[k]), 4),
                    "blockers": int(self.blocker_count[k]),
                    "tilt_deg": round(float(self.tilt_deg[k]), 3),
                }
                for k in listed
            ],
        }


# --------------------------------------------------------------------------
# Material and its timing
# --------------------------------------------------------------------------


def material_time(deposit: np.ndarray) -> np.ndarray:
    """The move from which each point is material, or a huge value if never.

    ``deposit[j]`` means the move to point ``j`` prints, laying material from
    ``p[j-1]`` to ``p[j]``. Point ``k`` is therefore material from move ``k``
    if that move prints, else from move ``k + 1`` if that one does (``k``
    starts a bead). ``deposit[0]`` describes no move and is ignored.
    """
    deposit = np.asarray(deposit, dtype=bool).copy()
    count = len(deposit)
    if count:
        deposit[0] = False
    time = np.full(count, _NEVER, dtype=np.int64)
    starts = np.flatnonzero(deposit[1:])  # k = j - 1 for every printing move j
    time[starts] = starts + 1
    ends = np.flatnonzero(deposit)
    time[ends] = ends
    return time


def _subsample(points: np.ndarray, time: np.ndarray, voxel_mm: float):
    """Keep the earliest-printed material point in each voxel."""
    keys = np.floor(points / voxel_mm).astype(np.int64)
    order = np.lexsort((time, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    return np.sort(order[first])


# --------------------------------------------------------------------------
# The cone test
# --------------------------------------------------------------------------


def cone_slabs(height_mm: float, half_angle_deg: float,
               first_mm: float = FIRST_SLAB_MM, ratio: float = SLAB_RATIO):
    """Spheres that together cover the cone: ``(centres along the axis, radii)``.

    Each covers one frustum slab of the cone. A frustum is the convex hull of
    its two rim circles, so a sphere containing both rims contains the slab.
    """
    tan_half = np.tan(np.radians(half_angle_deg))
    # Keep the spheres above the tip's plane where the cone allows it.
    limit = 0.98 / tan_half**2
    if limit > 1.05:
        ratio = min(ratio, limit)
    edges = [0.0, min(first_mm, height_mm)]
    while edges[-1] < height_mm:
        edges.append(min(edges[-1] * ratio, height_mm))
    edges = np.asarray(edges)
    lower, upper = edges[:-1], edges[1:]
    centre = 0.5 * (lower + upper)
    radius = np.maximum(
        np.hypot(upper - centre, upper * tan_half),
        np.hypot(centre - lower, lower * tan_half),
    )
    # A hair of padding so a point exactly on a rim is never lost to rounding.
    return centre, radius * (1.0 + 1e-9) + 1e-9


def cone_coordinates(apex, axis, material) -> tuple:
    """``(radial, axial)`` of ``material`` relative to cones, pairwise, mm.

    ``apex``, ``axis`` and ``material`` are ``(P, 3)``: one cone per row.
    ``axial`` is the height along the axis above the apex, ``radial`` the
    distance from the axis.
    """
    offset = material - apex
    axial = np.einsum("ij,ij->i", offset, axis)
    radial = np.sqrt(np.maximum(np.einsum("ij,ij->i", offset, offset) - axial**2, 0.0))
    return radial, axial


def cone_hits(apex, axis, nozzle_time, material_points, material_time_of,
              settings: NozzleCheckSettings):
    """Every (nozzle, material) pair with the material inside the nozzle.

    The general form of the check, shared with the swept check (P4.2), whose
    nozzle positions lie between toolpath points.

    Parameters
    ----------
    apex, axis
        ``(P, 3)``: each nozzle's cone apex (already offset) and unit axis.
    nozzle_time
        ``(P,)`` non-decreasing: nozzle ``p`` sees material whose time is at
        most ``nozzle_time[p]``.
    material_points, material_time_of
        ``(M, 3)`` and ``(M,)``: candidate material and the move from which
        each point exists.

    Returns ``(nozzle index, material index, depth mm, axial mm)``, one row
    per pair found, unordered.
    """
    apex = np.asarray(apex, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(axis, dtype=np.float64).reshape(-1, 3)
    nozzle_time = np.asarray(nozzle_time, dtype=np.int64).ravel()
    material_points = np.asarray(material_points, dtype=np.float64).reshape(-1, 3)
    material_time_of = np.asarray(material_time_of, dtype=np.int64).ravel()
    count = len(apex)
    if len(axis) != count or len(nozzle_time) != count:
        raise ValueError("apex, axis and nozzle_time must have the same length")
    if len(material_time_of) != len(material_points):
        raise ValueError("material points and times must have the same length")
    if count > 1 and np.any(np.diff(nozzle_time) < 0):
        raise ValueError("nozzle_time must be non-decreasing")
    if settings.tolerance_mm < 0:
        raise ValueError("tolerance_mm must be >= 0: it is how far inside material must be")

    material = np.flatnonzero(material_time_of != _NEVER)
    if settings.subsample_mm:
        material = material[_subsample(material_points[material], material_time_of[material],
                                       settings.subsample_mm)]
    material = material[np.argsort(material_time_of[material], kind="stable")]
    material_time_sorted = material_time_of[material]

    slab_centre, slab_radius = cone_slabs(settings.height_mm, settings.half_angle_deg)
    threshold = -float(settings.tolerance_mm)
    tan_half = np.tan(np.radians(settings.half_angle_deg))
    found = ([], [], [], [])

    def test_pairs(i, k):
        if not len(i):
            return
        radial, axial = cone_coordinates(apex[i], axis[i], material_points[k])
        # Cheap inside test first; the exact depth only for what is inside.
        inside = (axial > 0.0) & (axial <= settings.height_mm) & (radial <= axial * tan_half)
        if not np.any(inside):
            return
        i, k, radial, axial = i[inside], k[inside], radial[inside], axial[inside]
        distance = clearance.cone_signed_distance_axial(
            radial, axial, settings.half_angle_deg, settings.height_mm)
        hit = distance < threshold
        if np.any(hit):
            for store, values in zip(found, (i[hit], k[hit], -distance[hit], axial[hit])):
                store.append(values)

    block = max(1, int(settings.block_size))
    size = np.int64(max(len(material_points), 1))
    for start in range(0, count, block):
        stop = min(start + block, count)

        # Material every nozzle in the block can see: into the tree.
        before = np.searchsorted(material_time_sorted, nozzle_time[start], side="right")
        if before:
            tree_members = material[:before]
            tree = cKDTree(material_points[tree_members])
            for chunk_start in range(start, stop, _QUERY_CHUNK):
                chunk = np.arange(chunk_start, min(chunk_start + _QUERY_CHUNK, stop))
                centres = (apex[chunk, None, :]
                           + axis[chunk, None, :] * slab_centre[None, :, None])
                radii = np.broadcast_to(slab_radius, (len(chunk), len(slab_radius)))
                hits = tree.query_ball_point(centres.reshape(-1, 3), radii.ravel(),
                                             workers=1, return_sorted=False)
                lengths = np.fromiter((len(h) for h in hits), dtype=np.int64,
                                      count=len(hits))
                if not lengths.any():
                    continue
                owner = np.repeat(np.repeat(chunk, len(slab_radius)), lengths)
                member = tree_members[np.fromiter(itertools.chain.from_iterable(hits),
                                                  dtype=np.int64, count=int(lengths.sum()))]
                # Overlapping spheres return a point more than once.
                pairs = np.unique(owner * size + member)
                test_pairs(pairs // size, pairs % size)

        # Material laid during the block: pairwise, respecting print order.
        late = material[before:np.searchsorted(material_time_sorted, nozzle_time[stop - 1],
                                               side="right")]
        if len(late):
            i, k = np.meshgrid(np.arange(start, stop), late, indexing="ij")
            visible = material_time_of[k] <= nozzle_time[i]
            test_pairs(i[visible], k[visible])

    if not found[0]:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0)
    return tuple(np.concatenate(values) for values in found)


def check(points, directions, deposit, settings: NozzleCheckSettings) -> NozzleCollisions:
    """Test every nozzle position against the material printed before it.

    Parameters
    ----------
    points
        ``(N, 3)`` toolpath points in print order, part frame, mm.
    directions
        ``(N, 3)`` unit build directions (up the nozzle) at those points.
    deposit
        ``(N,)`` True where the move *to* that point prints.
    settings
        See `NozzleCheckSettings`.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    deposit = np.asarray(deposit, dtype=bool).ravel()
    count = len(points)
    if not (len(directions) == len(deposit) == count):
        raise ValueError("points, directions and deposit must have the same length")

    # The nozzle at p[i] sees material from moves up to i - 1.
    found = cone_hits(points + settings.apex_offset_mm * directions, directions,
                      np.arange(count) - 1, points, material_time(deposit), settings)
    return _collect(*found, deposit, directions, count, settings)


def _collect(nozzle, blocker, depth, axial, deposit, directions, count, settings):
    """One `NozzleCollisions` row per nozzle position, keeping its deepest pair."""
    if not len(nozzle):
        empty_int = np.zeros(0, dtype=np.int64)
        empty = np.zeros(0)
        return NozzleCollisions(count, empty_int, np.zeros(0, bool), empty_int, empty, empty,
                                empty_int, empty, settings)

    # Deepest blocker per nozzle position.
    order = np.lexsort((-depth, nozzle))
    nozzle, blocker, depth, axial = nozzle[order], blocker[order], depth[order], axial[order]
    first = np.ones(len(nozzle), dtype=bool)
    first[1:] = nozzle[1:] != nozzle[:-1]
    positions = nozzle[first]
    counts = np.diff(np.append(np.flatnonzero(first), len(nozzle)))
    tilt = om.tilt_from_vertical_deg(directions[positions])

    return NozzleCollisions(
        checked=count,
        index=positions,
        deposit=deposit[positions],
        blocker=blocker[first],
        depth_mm=depth[first],
        axial_mm=axial[first],
        blocker_count=counts,
        tilt_deg=tilt,
        settings=settings,
    )


def check_toolpath(toolpath, profile=None, **overrides) -> NozzleCollisions:
    """`check` on a `toolpath3.Toolpath`, or anything with the same arrays.

    ``profile`` sizes the cone (``nozzle_to_gantry``); it defaults to the
    active machine profile. Keyword arguments override `NozzleCheckSettings`.
    """
    from . import machine_profile

    profile = machine_profile.load_profile() if profile is None else profile
    count = int(np.asarray(toolpath.point_count).item())
    settings = NozzleCheckSettings.for_profile(profile, **overrides)
    return check(
        np.asarray(toolpath.point)[:count],
        om.spherical_to_cartesian(np.asarray(toolpath.tool_orientation)[:count]),
        np.asarray(toolpath.travel_type)[:count] == om.TRAVEL_TYPE_DEPOSITION,
        settings,
    )
