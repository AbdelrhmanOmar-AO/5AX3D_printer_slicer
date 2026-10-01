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

    Measured on the deposition points near each overhang face whose
    **nearest surface is an overhang** (`nearest_surface_is_overhang`). A
    point nearer a wall, a top or the bed is printed against that surface, not
    out over the overhang; where a wall turns into an overhang such points sit
    within the search radius and used to be counted (metrics version 4,
    plan_corrections P2-14 and P2-17).

Unsupported deposition
    A deposition point is supported when the bed, or material deposited earlier
    in the print, lies in the cone beneath it: apex at the point, axis ``-d``,
    half-angle ``SUPPORTING_REGION_CONE_ANGLE / 2`` (65 degrees, from
    `atom.toolpath3`), within ``SUPPORT_SEARCH_HEIGHTS x height`` (2.5; the
    search radius only bounds the neighbour search, and the cone decides).
    Otherwise it is printing into air.

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
#:
#: The build plan specifies 1.5, but that value silently overrides the cone it
#: is meant to work alongside. On a surface at angle t from vertical,
#: successive layers step ``h*tan(t)`` sideways, so the nearest earlier bead
#: lies at ``h/cos(t)``. A radius of 1.5h therefore stops reaching at
#: ``arccos(1/1.5) = 48.2`` degrees, well inside the 65-degree cone, and the
#: cone never gets to decide anything.
#:
#: 2.5h stops binding at 66.4 degrees, just past the cone, so the cone governs
#: as intended. Two independent checks support the cone being the right
#: criterion: it is Atomizer's own ``SUPPORTING_REGION_CONE_ANGLE``, and the
#: classic bead-overlap limit for fused filament (supported while
#: ``h*tan(t) < w``) gives 63.4 degrees at this part's 0.45/0.9 geometry.
#:
#: Measured consequence: with 1.5h a 45-degree ramp, which any 3-axis printer
#: manages, reported 16.35 % of its deposition as printing into air, and no
#: part past ~48 degrees could pass the threshold at all.
SUPPORT_SEARCH_HEIGHTS = 2.5

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
        Points at or below this height rest on the bed. Defaults to the lowest
        deposition point plus one layer height, which covers the whole first
        layer.

        The obvious choice, the layer height alone, is wrong: Atomizer's layers
        are conformal, so the first layer is a shell of finite thickness rather
        than a plane. On the calibration cube its deposition centres span
        z = 0.415 to beyond 0.45, and a threshold of 0.45 cuts through the
        middle of it, leaving the upper half with nothing beneath but bed it is
        not credited for. That alone reported 5.86 % of a plain cube as
        printing into air; anchoring to the lowest point gives 2.78 %, which is
        the gyroid infill genuinely bridging its own voids.
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
        first_layer_height = (
            float(points[:, 2].min() + np.max(heights)) if len(heights) else 0.0
        )

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
    #: Samples left out because their nearest surface is not an overhang
    #: (``part_triangles`` given); 0 otherwise.
    skipped_samples: int = 0

    @property
    def measured(self) -> bool:
        return self.sample_count > 0


def _segment_distance_sq(p, a, b):
    ab = b - a
    t = np.einsum("ptk,ptk->pt", p - a, ab) / np.maximum(np.einsum("ptk,ptk->pt", ab, ab), 1e-30)
    closest = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    d = p - closest
    return np.einsum("ptk,ptk->pt", d, d)


def triangle_distances(points: np.ndarray, triangles: np.ndarray, chunk_elements: int = 1_000_000) -> np.ndarray:
    """``(P, T)`` exact distances from each point to each triangle.

    The distance to the plane where the point projects inside the triangle,
    else to the nearest edge. Costs ``P x T``; the points are processed in
    chunks of ``chunk_elements``.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    result = np.empty((len(points), len(triangles)))
    if len(points) == 0 or len(triangles) == 0:
        return result
    a, b, c = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    normal = np.cross(b - a, c - a)
    normal_sq = np.maximum(np.einsum("tk,tk->t", normal, normal), 1e-30)
    step = max(1, chunk_elements // len(triangles))
    for start in range(0, len(points), step):
        p = points[start:start + step, None, :]
        shape = (p.shape[0], len(triangles), 3)
        A, B, C = (np.broadcast_to(v[None], shape) for v in (a, b, c))
        height = np.einsum("ptk,tk->pt", p - A, normal)
        projected = p - (height / normal_sq)[..., None] * normal[None]

        def on_inner_side(u, v):
            return np.einsum("ptk,tk->pt", np.cross(v - u, projected - u), normal) >= 0.0

        inside = on_inner_side(A, B) & on_inner_side(B, C) & on_inner_side(C, A)
        to_plane = height * height / normal_sq
        to_edges = np.minimum(
            np.minimum(_segment_distance_sq(p, A, B), _segment_distance_sq(p, B, C)),
            _segment_distance_sq(p, C, A),
        )
        result[start:start + step] = np.sqrt(np.where(inside, to_plane, to_edges))
    return result


#: How much nearer an overhang face than any other face a point must be to
#: count, mm: where a wall meets an overhang the nearest point is their shared
#: edge, at the same distance from both, and such a point does not count.
NEAREST_SURFACE_MARGIN_MM = 0.05


def nearest_surface_is_overhang(
    points: np.ndarray,
    part_triangles: np.ndarray,
    bed_contact_height: float = 0.5,
    margin: float = NEAREST_SURFACE_MARGIN_MM,
) -> np.ndarray:
    """Which points have an overhang face as their nearest surface.

    ``part_triangles`` ``(T, 3, 3)``, the part's mesh, outward winding. A
    triangle is an overhang by the same test as `effective_overhang_angles`
    uses for faces. A point counts when it is nearer an overhang triangle, by
    ``margin``, than any other triangle.
    """
    triangles = np.asarray(part_triangles, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    centres = triangles.mean(axis=1)
    overhang = (
        (normals[:, 2] < -1e-6)
        & (geometric_overhang_angle_deg(normals) > 1e-6)
        & (centres[:, 2] > bed_contact_height)
    )
    result = np.zeros(len(points), dtype=bool)
    if not overhang.any() or len(points) == 0:
        return result
    distances = triangle_distances(points, triangles)
    to_overhang = distances[:, overhang].min(axis=1)
    if overhang.all():
        return np.ones(len(points), dtype=bool)
    to_other = distances[:, ~overhang].min(axis=1)
    return to_overhang + margin < to_other


def effective_overhang_angles(
    face_normals: np.ndarray,
    face_centres: np.ndarray,
    face_areas: np.ndarray,
    toolpath,
    search_radius: float,
    bed_contact_height: float = 0.5,
    angle_decimals: int = 1,
    part_triangles: np.ndarray | None = None,
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
    part_triangles
        The part's mesh ``(T, 3, 3)``. Given, only points whose nearest
        surface is an overhang count (`nearest_surface_is_overhang`); the
        others are skipped and counted in ``skipped_samples``. Omitted, every
        point near a face counts (metrics version 2 and earlier).
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

    # Which of the points near any overhang face count, tested once.
    counts = None
    if tree is not None and part_triangles is not None:
        near = sorted({
            i
            for face_index in np.flatnonzero(is_overhang)
            for i in tree.query_ball_point(face_centres[face_index], search_radius)
        })
        counts = np.zeros(len(points), dtype=bool)
        if near:
            near = np.asarray(near)
            counts[near] = nearest_surface_is_overhang(
                points[near], part_triangles, bed_contact_height
            )

    results: list[FaceGroupMetrics] = []
    rounded = np.round(geometric, angle_decimals)
    for angle in sorted(set(rounded[is_overhang]), reverse=True):
        in_group = is_overhang & (rounded == angle)
        indices = np.flatnonzero(in_group)

        effective: list[float] = []
        tilts: list[float] = []
        skipped = 0
        if tree is not None:
            for face_index in indices:
                nearby = tree.query_ball_point(face_centres[face_index], search_radius)
                if counts is not None and nearby:
                    kept = [i for i in nearby if counts[i]]
                    skipped += len(nearby) - len(kept)
                    nearby = kept
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
                skipped_samples=skipped,
            )
        )

    return results


def closest_points_on_triangles(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The closest point of each triangle to the point paired with it.

    ``points`` ``(N, 3)`` and ``triangles`` ``(N, 3, 3)``, taken in pairs.
    Returns ``(closest, barycentric)``, ``(N, 3)`` each: the closest point and
    its barycentric coordinates on the triangle. Ericson, *Real-Time Collision
    Detection* (2005), 5.1.5, region by region.
    """
    p = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    dot = lambda u, v: np.einsum("nk,nk->n", u, v)  # noqa: E731
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def ratio(numerator, denominator):
        return numerator / np.where(np.abs(denominator) > 1e-300, denominator, 1.0)

    v_ab = ratio(d1, d1 - d3)
    w_ac = ratio(d2, d2 - d6)
    w_bc = ratio(d4 - d3, (d4 - d3) + (d5 - d6))
    total = va + vb + vc
    v_in, w_in = ratio(vb, total), ratio(vc, total)
    ones, zeros = np.ones(len(p)), np.zeros(len(p))
    regions = [
        (d1 <= 0) & (d2 <= 0),  # vertex a
        (d3 >= 0) & (d4 <= d3),  # vertex b
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),  # edge ab
        (d6 >= 0) & (d5 <= d6),  # vertex c
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),  # edge ac
        (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),  # edge bc
    ]
    v = np.select(regions, [zeros, ones, v_ab, zeros, zeros, 1.0 - w_bc], default=v_in)
    w = np.select(regions, [zeros, zeros, zeros, ones, w_ac, w_bc], default=w_in)
    barycentric = np.column_stack([1.0 - v - w, v, w])
    closest = a + v[:, None] * ab + w[:, None] * ac
    return closest, barycentric


def _nearest_within(points: np.ndarray, triangles: np.ndarray, reach: float):
    """For each point, its nearest triangle among those within ``reach``.

    Returns ``(distance, triangle, barycentric)``: ``inf``, -1 and NaN where no
    triangle is that close. Only point and triangle pairs that can be within
    ``reach`` are measured (a k-d tree on the points, queried from each
    triangle's centroid out to its farthest vertex plus ``reach``), so large
    meshes cost what their surface near the points costs.
    """
    count = len(points)
    distance = np.full(count, np.inf)
    nearest = np.full(count, -1, dtype=np.int64)
    barycentric = np.full((count, 3), np.nan)
    if count == 0 or len(triangles) == 0:
        return distance, nearest, barycentric
    centres = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centres[:, None, :], axis=2).max(axis=1)
    found = cKDTree(points).query_ball_point(centres, r=radii + reach)
    sizes = np.fromiter((len(f) for f in found), dtype=np.int64, count=len(found))
    if not sizes.any():
        return distance, nearest, barycentric
    pair_point = np.concatenate([np.asarray(f, dtype=np.int64) for f in found if f])
    pair_triangle = np.repeat(np.arange(len(triangles)), sizes)
    closest, pair_bary = closest_points_on_triangles(points[pair_point], triangles[pair_triangle])
    pair_distance = np.linalg.norm(points[pair_point] - closest, axis=1)
    # The nearest pair per point: sort by point, then distance; keep the first.
    order = np.lexsort((pair_distance, pair_point))
    first = order[np.unique(pair_point[order], return_index=True)[1]]
    keep = first[pair_distance[first] <= reach]
    distance[pair_point[keep]] = pair_distance[keep]
    nearest[pair_point[keep]] = pair_triangle[keep]
    barycentric[pair_point[keep]] = pair_bary[keep]
    return distance, nearest, barycentric


def unit_face_normals(triangles: np.ndarray) -> np.ndarray:
    """``(T, 3)`` unit normals from the winding, as the other mesh tests use."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)


#: Faces meeting at a vertex at less than this share their normals there
#: (`corner_normals`); sharper edges, like a top meeting a wall, stay sharp.
CREASE_ANGLE_DEG = 30.0


def corner_normals(triangles: np.ndarray, crease_deg: float = CREASE_ANGLE_DEG) -> np.ndarray:
    """``(T, 3, 3)`` smooth normals at each triangle's corners.

    At each corner, the area-weighted mean of the normals of the faces that
    share the vertex and lie within ``crease_deg`` of this face, as a
    renderer's auto-smooth does. Interpolated across a face, they stand for
    the smooth surface a faceted mesh approximates: a dome's facets are about
    10 degrees apart (`twin_domes`), five times the top-surface tolerance, so
    the facet normal itself is no target; a flat face keeps its own normal.
    Vertices are matched by position, so an STL's unshared vertices work.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    count = len(triangles)
    normals = unit_face_normals(triangles)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    keys = np.round(triangles.reshape(-1, 3) * 1e6).astype(np.int64)
    _, vertex = np.unique(keys, axis=0, return_inverse=True)
    vertex = vertex.reshape(-1)
    corner_face = np.repeat(np.arange(count), 3)
    order = np.argsort(vertex, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(vertex[order]) != 0])
    ends = np.r_[starts[1:], len(order)]
    cos_crease = np.cos(np.radians(crease_deg))
    result = np.empty((count * 3, 3))
    for start, end in zip(starts, ends):
        corners = order[start:end]
        faces = corner_face[corners]
        n = normals[faces]
        shared = (n @ n.T) >= cos_crease
        summed = shared @ (n * areas[faces, None])
        result[corners] = summed / np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-30)
    return result.reshape(count, 3, 3)


#: Top-surface quality (build plan P2.5): a top-layer point is on target when
#: its build direction is within this of the top surface's normal.
TOP_SURFACE_TOLERANCE_DEG = 2.0


@dataclass(frozen=True)
class TopSurfaceResult:
    """How the top surfaces were printed (build plan P2.5, the operator's
    toolpath form of the plan's "ceiling cells within 2 degrees of their
    target", 2026-10-01)."""

    #: Deposition points counted: in the top layer of a top surface, and
    #: nearer it than any other surface. Zero: nothing measured, NaN below.
    sample_count: int
    #: How many of them are within the tolerance of the surface's normal.
    on_target_count: int
    #: ``on_target_count / sample_count``.
    fraction_on_target: float
    mean_deviation_deg: float
    max_deviation_deg: float
    #: Points in a top layer that are nearer another surface (a wall), left out.
    skipped_samples: int
    #: The faces that count as top surfaces: their normal within this of +Z.
    top_max_angle_deg: float
    tolerance_deg: float

    @property
    def measured(self) -> bool:
        return self.sample_count > 0


def top_surface_quality(
    toolpath,
    part_triangles: np.ndarray,
    top_max_angle_deg: float,
    layer_height: float,
    tolerance_deg: float = TOP_SURFACE_TOLERANCE_DEG,
    margin: float = NEAREST_SURFACE_MARGIN_MM,
    crease_deg: float = CREASE_ANGLE_DEG,
) -> TopSurfaceResult:
    """The share of top-layer points printed along the top surface's normal.

    A **top surface** is an upward face whose normal is within
    ``top_max_angle_deg`` of +Z: upstream's ceiling (`fff3.CEIL_MAX_ANGLE`,
    the slicer's ``max_slope`` capped by the nozzle cone), which the field
    is asked to follow exactly, so a print whose layers there are parallel
    to the surface has a smooth top rather than a staircase.

    A deposition point counts when it lies within ``layer_height`` of a top
    face (the top layer, as upstream's ceiling cells lie within one layer
    height of the surface) and that face is nearer it, by ``margin``, than
    any other face (as the effective overhang angle's
    `nearest_surface_is_overhang`): a top-layer bead against a wall is
    printed against the wall. Its target is the smooth surface normal at the
    nearest point (`corner_normals`), and its deviation the angle between
    that and its build direction.

    Upstream's ceiling rule also needs low curvature (``CURVATURE_THRESHOLD``
    on the SDF); this measure has no such condition, so a small dome's top
    counts here although the field may leave it unconstrained.
    """
    triangles = np.asarray(part_triangles, dtype=np.float64)
    normals = unit_face_normals(triangles)
    top = (normals[:, 2] > 0.0) & (
        np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0))) < top_max_angle_deg
    )

    count = int(np.asarray(toolpath.point_count).item())
    mask = deposition_mask(toolpath)
    points = np.asarray(toolpath.point[:count], dtype=np.float64)[mask]
    directions = tool_directions(toolpath)[:count][mask]

    def result(samples, on_target, deviations, skipped):
        return TopSurfaceResult(
            sample_count=samples,
            on_target_count=on_target,
            fraction_on_target=on_target / samples if samples else float("nan"),
            mean_deviation_deg=float(np.mean(deviations)) if samples else float("nan"),
            max_deviation_deg=float(np.max(deviations)) if samples else float("nan"),
            skipped_samples=skipped,
            top_max_angle_deg=float(top_max_angle_deg),
            tolerance_deg=float(tolerance_deg),
        )

    if not top.any() or len(points) == 0:
        return result(0, 0, [], 0)

    top_index = np.flatnonzero(top)
    to_top, nearest, barycentric = _nearest_within(points, triangles[top], layer_height)
    in_top_layer = np.flatnonzero(np.isfinite(to_top))
    if len(in_top_layer) == 0:
        return result(0, 0, [], 0)

    to_other, _, _ = _nearest_within(points[in_top_layer], triangles[~top], layer_height + margin)
    counted = in_top_layer[to_top[in_top_layer] + margin < to_other]
    skipped = len(in_top_layer) - len(counted)
    if len(counted) == 0:
        return result(0, 0, [], skipped)

    corners = corner_normals(triangles, crease_deg)[top_index[nearest[counted]]]
    target = np.einsum("nk,nkj->nj", barycentric[counted], corners)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-30)
    cosine = np.clip(np.einsum("nk,nk->n", target, directions[counted]), -1.0, 1.0)
    deviations = np.degrees(np.arccos(cosine))
    on_target = int(np.count_nonzero(deviations <= tolerance_deg))
    return result(len(counted), on_target, deviations, skipped)


@dataclass(frozen=True)
class ToolpathSegments:
    """Each move of a toolpath: from point ``i - 1`` to point ``i``."""

    #: Its length, mm.
    length_mm: np.ndarray = field(repr=False)
    #: How far the build direction turns over it, degrees.
    turn_deg: np.ndarray = field(repr=False)
    #: Whether it deposits (`toolpath3`'s ``travel_type`` of its end point).
    deposits: np.ndarray = field(repr=False)


def toolpath_segments(toolpath) -> ToolpathSegments:
    """The moves between consecutive toolpath points, for the tilt rate."""
    count = int(np.asarray(toolpath.point_count).item())
    points = np.asarray(toolpath.point[:count], dtype=np.float64)
    directions = tool_directions(toolpath)[:count]
    if count < 2:
        empty = np.zeros(0)
        return ToolpathSegments(empty, empty, np.zeros(0, dtype=bool))
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cosine = np.clip(np.einsum("nk,nk->n", directions[1:], directions[:-1]), -1.0, 1.0)
    deposits = np.asarray(toolpath.travel_type[1:count]) == TRAVEL_TYPE_DEPOSITION
    return ToolpathSegments(length, np.degrees(np.arccos(cosine)), deposits)


#: The length of printing the tilt rate is measured over, mm (build plan P2.5,
#: the operator's choice, 2026-10-01).
TILT_RATE_STRETCH_MM = 1.0


@dataclass(frozen=True)
class TiltRateResult:
    """How fast the tool turns while printing (build plan P2.5)."""

    #: The largest turn within any ``stretch_mm`` of continuous printing,
    #: divided by ``stretch_mm``: degrees per mm. NaN with no printing move.
    max_deg_per_mm: float
    #: Where that stretch starts, mm (part frame).
    max_at_mm: tuple
    #: The share of printing moves where the ``stretch_mm`` of printing
    #: starting there (near a run's end, the run's last ``stretch_mm``) turns
    #: faster than ``limit_deg_per_mm``.
    fraction_over_limit: float
    limit_deg_per_mm: float
    stretch_mm: float
    printing_moves: int


def tilt_rate(toolpath, limit_deg_per_mm: float, stretch_mm: float = TILT_RATE_STRETCH_MM) -> TiltRateResult:
    """The tilt rate along the printing moves, over stretches of ``stretch_mm``.

    The turn is the angle between consecutive build directions, summed along
    each run of consecutive printing moves (travel moves end a run: the bed
    may turn freely between beads). Per move, the rate would depend on how
    finely the path is cut: smoothing the points alone took a T-shape's
    largest per-move rate from 70 to 111 degrees per mm with no more turning
    (plan_corrections P2-21). Over a stretch it measures how far the bed
    turns per mm of printing, which is what a tilt-rate limit (gate D3)
    bounds. A run shorter than the stretch counts its whole turn.

    Turning accumulates linearly along each move, so the largest stretch
    starts or ends at a point of the toolpath; both are tried.
    """
    segments = toolpath_segments(toolpath)
    points = np.asarray(toolpath.point[: int(np.asarray(toolpath.point_count).item())], dtype=np.float64)
    deposits = segments.deposits
    best, best_at = float("nan"), (float("nan"),) * 3
    over = moves = 0
    starts = np.flatnonzero(deposits & ~np.r_[False, deposits[:-1]])
    ends = np.flatnonzero(deposits & ~np.r_[deposits[1:], False]) + 1
    for first, last in zip(starts, ends):
        length = np.r_[0.0, np.cumsum(segments.length_mm[first:last])]
        turn = np.r_[0.0, np.cumsum(segments.turn_deg[first:last])]
        room = max(length[-1] - stretch_mm, 0.0)
        candidates = np.clip(np.r_[length, length - stretch_mm], 0.0, room)
        windows = np.interp(np.minimum(candidates + stretch_mm, length[-1]), length, turn) - np.interp(candidates, length, turn)
        k = int(np.argmax(windows))
        if np.isnan(best) or windows[k] / stretch_mm > best:
            best = float(windows[k] / stretch_mm)
            position = np.interp(candidates[k], length, np.arange(len(length)))
            i = int(min(np.floor(position), len(length) - 2))
            best_at = tuple(float(v) for v in points[first + i] + (position - i) * (points[first + i + 1] - points[first + i]))
        ahead = np.clip(length[:-1], 0.0, room)
        rates = (np.interp(np.minimum(ahead + stretch_mm, length[-1]), length, turn) - np.interp(ahead, length, turn)) / stretch_mm
        over += int(np.count_nonzero(rates > limit_deg_per_mm))
        moves += int(last - first)
    return TiltRateResult(
        max_deg_per_mm=best,
        max_at_mm=best_at,
        fraction_over_limit=over / moves if moves else float("nan"),
        limit_deg_per_mm=float(limit_deg_per_mm),
        stretch_mm=float(stretch_mm),
        printing_moves=moves,
    )


@dataclass(frozen=True)
class OverhangStart:
    """The tilt where an overhang starts (build plan P2.4's pipeline test)."""

    #: The points counted, ``(N, 3)``, and their tilts, degrees.
    points: np.ndarray = field(repr=False)
    tilts_deg: np.ndarray = field(repr=False)
    #: How far past the start they were taken, mm.
    strip_mm: float

    @property
    def count(self) -> int:
        return len(self.tilts_deg)


def overhang_start_tilts(
    points: np.ndarray,
    directions: np.ndarray,
    part_triangles: np.ndarray,
    overhang_normal,
    strip_mm: float,
    search_radius: float,
    bed_contact_height: float = 0.5,
    angle_tolerance_deg: float = 1.0,
) -> OverhangStart:
    """The tilts of the points by one overhang, in the first ``strip_mm`` past its start.

    The overhang is the part's overhang faces whose normal is within
    ``angle_tolerance_deg`` of ``overhang_normal``. It **starts** at its
    lowest point, where the wall below turns into it (on a ramp, the column's
    edge), and the strip is measured from there horizontally, toward the
    overhang (its normal's azimuth): out over the air. A point counts when it
    is within ``search_radius`` of one of those faces and its nearest surface
    is an overhang (`nearest_surface_is_overhang`, the effective angle's own
    choice of points), so material on top of the wall below does not.
    """
    triangles = np.asarray(part_triangles, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    target = np.asarray(overhang_normal, dtype=np.float64)
    target = target / np.linalg.norm(target)
    normals = unit_face_normals(triangles)
    faces = overhang_face_mask(normals, triangles.mean(axis=1), bed_contact_height) & (
        normals @ target >= np.cos(np.radians(angle_tolerance_deg))
    )
    empty = OverhangStart(np.zeros((0, 3)), np.zeros(0), float(strip_mm))
    horizontal = np.array([target[0], target[1], 0.0])
    if not faces.any() or len(points) == 0 or np.linalg.norm(horizontal) < 1e-9:
        return empty  # a flat underside has no direction to measure the strip in
    outward = horizontal / np.linalg.norm(horizontal)
    vertices = triangles[faces].reshape(-1, 3)
    start = vertices[np.argmin(vertices[:, 2])]
    past = (points - start) @ outward
    candidates = np.flatnonzero((past > 0.0) & (past <= strip_mm))
    if len(candidates) == 0:
        return empty
    close = triangle_distances(points[candidates], triangles[faces]).min(axis=1) <= search_radius
    candidates = candidates[close]
    if len(candidates) == 0:
        return empty
    counted = candidates[nearest_surface_is_overhang(points[candidates], triangles, bed_contact_height)]
    return OverhangStart(points[counted], tilt_from_vertical_deg(directions[counted]), float(strip_mm))


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
