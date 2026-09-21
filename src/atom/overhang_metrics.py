"""Measuring how well a toolpath handles overhangs (build plan task P0.8).

These are the numbers the whole 5-axis contribution is judged by. They are
computed identically for stock Atomizer (the P0.8 baseline) and for the
overhang-aware field (P2.5), so the two are directly comparable, and again on
real prints in P8.

numpy and scipy only: no Taichi, no trimesh, no Atomizer stage. Mesh geometry
arrives as plain arrays so the module can be exercised on synthetic data, which
is how every unit test here works.

Definitions
-----------
Build direction
    The tool orientation ``d`` at a deposition point, a unit vector. A plain
    3-axis print has ``d = +Z`` everywhere.

Effective overhang angle
    For a downward-facing surface with outward normal ``n``, seen from build
    direction ``d``::

        theta_eff = 90 - degrees(arccos(n . -d))

    A wall parallel to ``d`` gives 0 degrees, a ceiling facing ``-d`` gives 90,
    and anything at or below 0 is not an overhang. With ``d = +Z`` this equals
    the geometric overhang angle, so tilting the tool toward an overhang lowers
    it one-for-one. That is exactly what the overhang-aware field is meant to
    do.

Unsupported deposition
    A deposition point is supported when the bed, or material deposited earlier
    in the print, lies in the cone beneath it: apex at the point, axis ``-d``,
    half-angle ``SUPPORTING_REGION_CONE_ANGLE / 2`` (65 degrees, from
    `atom.toolpath3`), within ``1.5 x height``. Otherwise it is printing into
    air.

Units
-----
Millimetres and degrees at every public boundary. Radians appear only inside
functions.

Note on tool_orientation
------------------------
`toolpath3.Toolpath.tool_orientation` is ``(N, 2)`` **spherical**
``[theta, phi]``, not ``(N, 3)`` Cartesian as the build plan's facts table
states. See `toolpath3.Toolpath.allocate` and the conversion in
`kinematics3z.get_vertical_offset_kernel`. `tool_directions` does the
conversion, mirroring `atom.direction.spherical_to_cartesian`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

#: Half-angle of the support cone, in degrees. `toolpath3` defines the full
#: cone as 130 degrees; this is the half-angle the geometry test uses.
SUPPORT_CONE_HALF_ANGLE_DEG = 65.0

#: Multiple of a deposition's own height searched for supporting material.
SUPPORT_SEARCH_HEIGHTS = 1.5

#: `toolpath3.TRAVEL_TYPE_DEPOSITION`; repeated so this module stays free of
#: Taichi, which `toolpath3` imports at module level.
TRAVEL_TYPE_DEPOSITION = 0


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """Convert ``(N, 2)`` ``[theta, phi]`` radians to ``(N, 3)`` unit vectors.

    Mirrors `atom.direction.spherical_to_cartesian`: ``theta`` is the polar
    angle from +Z, ``phi`` the azimuth from +X in the xy-plane.
    """
    spherical = np.atleast_2d(np.asarray(spherical, dtype=np.float64))
    theta, phi = spherical[:, 0], spherical[:, 1]
    sin_theta = np.sin(theta)
    return np.column_stack(
        [np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, np.cos(theta)]
    )


def tool_directions(toolpath) -> np.ndarray:
    """Cartesian build direction per toolpath point, ``(N, 3)``."""
    return spherical_to_cartesian(toolpath.tool_orientation)


def tilt_from_vertical_deg(directions: np.ndarray) -> np.ndarray:
    """Angle of each build direction from +Z, in degrees."""
    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    return np.degrees(np.arccos(np.clip(directions[:, 2], -1.0, 1.0)))


def effective_overhang_angle_deg(
    normals: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """``theta_eff`` for each (surface normal, build direction) pair, in degrees.

    Both arrays are ``(N, 3)`` and must be unit length. Values at or below zero
    mean the surface is not an overhang from that build direction.
    """
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))

    cosine = np.sum(normals * -directions, axis=1)
    return 90.0 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def geometric_overhang_angle_deg(normals: np.ndarray) -> np.ndarray:
    """``theta_eff`` measured from a vertical build direction: the planar case."""
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    vertical = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))
    return effective_overhang_angle_deg(normals, vertical)


def deposition_mask(toolpath) -> np.ndarray:
    """Boolean mask of points that actually deposit material."""
    count = int(np.asarray(toolpath.point_count).item())
    return np.asarray(toolpath.travel_type[:count]) == TRAVEL_TYPE_DEPOSITION


def max_tool_tilt_deg(toolpath, deposition_only: bool = True) -> float:
    """The largest tilt the toolpath actually uses, in degrees from +Z.

    Compared against the part's ``max_slope`` this says whether Atomizer spends
    its tilt budget at all, which is the first thing the P0.8 baseline needs to
    establish.
    """
    count = int(np.asarray(toolpath.point_count).item())
    directions = tool_directions(toolpath)[:count]

    if deposition_only:
        directions = directions[deposition_mask(toolpath)]
    if len(directions) == 0:
        return 0.0

    return float(np.max(tilt_from_vertical_deg(directions)))


@dataclass(frozen=True)
class UnsupportedResult:
    """Outcome of the unsupported-deposition test."""

    #: One entry per deposition point, in print order.
    unsupported: np.ndarray = field(repr=False)
    #: Fraction of deposition points printing into air, 0.0 to 1.0.
    fraction: float
    #: Number of deposition points considered.
    point_count: int

    @property
    def percent(self) -> float:
        return self.fraction * 100.0


def unsupported_deposition(
    toolpath,
    first_layer_height: float | None = None,
    cone_half_angle_deg: float = SUPPORT_CONE_HALF_ANGLE_DEG,
    search_heights: float = SUPPORT_SEARCH_HEIGHTS,
) -> UnsupportedResult:
    """Flag deposition points with neither the bed nor earlier material beneath.

    A point is supported when the bed lies within ``first_layer_height``, or
    when some point deposited **earlier in the print** falls inside the cone
    below it. Print order matters: material laid down later cannot hold up
    something printed before it.

    Parameters
    ----------
    first_layer_height
        Points at or below this height rest on the bed. Defaults to the
        greatest deposition height in the toolpath, which is the layer height
        the first layer was laid at.
    """
    count = int(np.asarray(toolpath.point_count).item())
    mask = deposition_mask(toolpath)

    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]
    directions = tool_directions(toolpath)[:count][mask]
    heights = np.asarray(toolpath.height[:count], dtype=np.float64)[mask]

    n_points = len(points)
    if n_points == 0:
        return UnsupportedResult(np.zeros(0, dtype=bool), 0.0, 0)

    if first_layer_height is None:
        first_layer_height = float(np.max(heights)) if len(heights) else 0.0

    # On the bed: supported, whatever is or is not around it.
    on_bed = points[:, 2] <= first_layer_height + 1e-6

    cos_half_angle = np.cos(np.radians(cone_half_angle_deg))
    radii = search_heights * heights

    tree = cKDTree(points)
    neighbourhoods = tree.query_ball_point(points, r=radii, workers=-1)

    unsupported = np.zeros(n_points, dtype=bool)
    for index, neighbours in enumerate(neighbourhoods):
        if on_bed[index]:
            continue

        # Only material already deposited can support this point.
        earlier = np.array([j for j in neighbours if j < index], dtype=int)
        if earlier.size == 0:
            unsupported[index] = True
            continue

        offsets = points[earlier] - points[index]
        lengths = np.linalg.norm(offsets, axis=1)
        valid = lengths > 1e-9
        if not np.any(valid):
            # Coincident points count as supporting material.
            continue

        cosines = (offsets[valid] @ -directions[index]) / lengths[valid]
        unsupported[index] = not np.any(cosines >= cos_half_angle)

    return UnsupportedResult(
        unsupported=unsupported,
        fraction=float(np.count_nonzero(unsupported)) / n_points,
        point_count=n_points,
    )


@dataclass(frozen=True)
class FaceGroupMetrics:
    """How one group of equally-sloped overhang faces was actually printed."""

    geometric_angle_deg: float
    area_mm2: float
    face_count: int
    #: Deposition points found near the group. Zero means nothing was measured
    #: and every angle below is NaN.
    sample_count: int
    max_effective_deg: float
    mean_effective_deg: float
    max_tilt_used_deg: float

    @property
    def measured(self) -> bool:
        return self.sample_count > 0


def effective_overhang_angles(
    face_normals: np.ndarray,
    face_centres: np.ndarray,
    face_areas: np.ndarray,
    toolpath,
    search_radius: float,
    bed_contact_height: float = 0.5,
    angle_decimals: int = 1,
) -> list[FaceGroupMetrics]:
    """Per overhang surface, the effective angle the toolpath actually achieved.

    Downward-facing mesh faces are grouped by their geometric overhang angle.
    For each group, the deposition points within ``search_radius`` of a face
    centre supply the build directions, and ``theta_eff`` is computed against
    that face's normal.

    Parameters
    ----------
    face_normals, face_centres, face_areas
        ``(F, 3)``, ``(F, 3)`` and ``(F,)``. Plain arrays rather than a mesh
        object, so this module needs no mesh library.
    search_radius
        How far from a face centre to look for deposition points; one
        deposition width is the intended value.
    bed_contact_height
        Faces whose centre sits below this are resting on the bed. They are
        downward-facing but supported, so they are not overhangs.
    """
    face_normals = np.asarray(face_normals, dtype=np.float64)
    face_centres = np.asarray(face_centres, dtype=np.float64)
    face_areas = np.asarray(face_areas, dtype=np.float64)

    geometric = geometric_overhang_angle_deg(face_normals)
    is_overhang = (
        (face_normals[:, 2] < -1e-6)
        & (geometric > 1e-6)
        & (face_centres[:, 2] > bed_contact_height)
    )
    if not np.any(is_overhang):
        return []

    count = int(np.asarray(toolpath.point_count).item())
    mask = deposition_mask(toolpath)
    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]
    directions = tool_directions(toolpath)[:count][mask]

    tree = cKDTree(points) if len(points) else None

    results: list[FaceGroupMetrics] = []
    rounded = np.round(geometric, angle_decimals)
    for angle in sorted(set(rounded[is_overhang]), reverse=True):
        in_group = is_overhang & (rounded == angle)
        indices = np.flatnonzero(in_group)

        effective: list[float] = []
        tilts: list[float] = []
        if tree is not None:
            for face_index in indices:
                nearby = tree.query_ball_point(face_centres[face_index], search_radius)
                if not nearby:
                    continue
                nearby_directions = directions[nearby]
                normal = np.tile(face_normals[face_index], (len(nearby), 1))
                effective.extend(
                    effective_overhang_angle_deg(normal, nearby_directions)
                )
                tilts.extend(tilt_from_vertical_deg(nearby_directions))

        results.append(
            FaceGroupMetrics(
                geometric_angle_deg=float(angle),
                area_mm2=float(np.sum(face_areas[in_group])),
                face_count=int(in_group.sum()),
                sample_count=len(effective),
                max_effective_deg=float(np.max(effective)) if effective else float("nan"),
                mean_effective_deg=float(np.mean(effective)) if effective else float("nan"),
                max_tilt_used_deg=float(np.max(tilts)) if tilts else float("nan"),
            )
        )

    return results


def overhang_face_mask(
    face_normals: np.ndarray,
    face_centres: np.ndarray,
    bed_contact_height: float = 0.5,
) -> np.ndarray:
    """Boolean mask of mesh faces that are genuine overhangs.

    Downward-facing, actually sloped, and not resting on the bed. The bed
    exclusion matters: a part's own footprint faces straight down but is fully
    supported, and counting it would report every part as having a 90-degree
    overhang.
    """
    face_normals = np.asarray(face_normals, dtype=np.float64)
    face_centres = np.asarray(face_centres, dtype=np.float64)

    return (
        (face_normals[:, 2] < -1e-6)
        & (geometric_overhang_angle_deg(face_normals) > 1e-6)
        & (face_centres[:, 2] > bed_contact_height)
    )


def points_near_overhangs(
    points: np.ndarray,
    face_normals: np.ndarray,
    face_centres: np.ndarray,
    radius: float,
    bed_contact_height: float = 0.5,
) -> np.ndarray:
    """Boolean mask of points lying within ``radius`` of an overhang face.

    The overall unsupported fraction is dominated by whatever the part is
    mostly made of, so it barely moves even when an overhang prints badly.
    Restricting the measurement to the neighbourhood of the overhangs is what
    makes it sensitive to the thing under study (build plan P0.8 step 2, which
    uses two deposition widths).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    overhangs = overhang_face_mask(face_normals, face_centres, bed_contact_height)
    if not np.any(overhangs):
        return np.zeros(len(points), dtype=bool)

    centres = np.asarray(face_centres, dtype=np.float64)[overhangs]
    near = np.zeros(len(points), dtype=bool)
    for index in cKDTree(points).query_ball_point(centres, radius):
        near[index] = True
    return near


def unsupported_near_overhangs(
    toolpath,
    face_normals: np.ndarray,
    face_centres: np.ndarray,
    radius: float,
    bed_contact_height: float = 0.5,
    **kwargs,
) -> tuple[UnsupportedResult, float, int]:
    """Unsupported deposition overall, and restricted to overhang neighbourhoods.

    Returns ``(overall_result, near_overhang_fraction, near_overhang_count)``.
    The fraction is NaN when no deposition point lies near an overhang, which
    is not the same as zero: it means nothing was measured.
    """
    overall = unsupported_deposition(toolpath, **kwargs)

    count = int(np.asarray(toolpath.point_count).item())
    mask = deposition_mask(toolpath)
    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]

    near = points_near_overhangs(
        points, face_normals, face_centres, radius, bed_contact_height
    )
    near_count = int(np.count_nonzero(near))
    if near_count == 0:
        return overall, float("nan"), 0

    fraction = float(np.count_nonzero(overall.unsupported & near)) / near_count
    return overall, fraction, near_count
