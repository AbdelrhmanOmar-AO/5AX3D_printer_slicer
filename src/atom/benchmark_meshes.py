"""Parametric benchmark meshes for the 5-axis overhang study (build plan P0.7).

Pure geometry: every function returns a watertight `trimesh.Trimesh` sitting on
the bed (minimum z exactly 0). The CLI wrapper is `tools/make_benchmarks.py`.

Nothing here is imported by a vendored Atomizer stage; trimesh is a dev-only
dependency.

Overhang convention
-------------------
The overhang angle of a downward-facing surface is measured **from vertical**,
matching the build plan and `overhang_metrics`:

* 0 degrees  - the surface is a vertical wall; not an overhang at all.
* 45 degrees - the usual limit for unsupported planar printing.
* 90 degrees - a flat horizontal ledge, the hardest case.

A face whose direction makes angle ``A`` with +Z advances ``sin(A)`` outward for
every ``cos(A)`` it rises, so its downward normal is ``(cos A, 0, -sin A)``.
Seen from a vertical tool direction ``d = +Z`` that gives an effective overhang
angle of exactly ``A``:

    theta_eff = 90 - degrees(arccos(n . -d)) = 90 - (90 - A) = A

Units are millimetres throughout.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

#: Named size presets. The value is the part's principal dimension in mm
#: (length in x for ramps and the T-shape, base width for the twin domes).
#: A spread of sizes is generated so the cost of `order_atoms`, which dominates
#: pipeline runtime and scales with volume, can be measured rather than guessed.
SIZE_PRESETS: dict[str, float] = {
    "xs": 30.0,
    "s": 50.0,
    "m": 75.0,
    "l": 100.0,
}

#: Ramp angles the study sweeps, in degrees from vertical.
RAMP_ANGLES: tuple[int, ...] = (45, 50, 60, 70, 80, 90)


def _cross_2d(u: np.ndarray, v: np.ndarray) -> float:
    """z-component of the 3D cross product of two 2D vectors.

    Written out because numpy 2 deprecated `np.cross` on 2-vectors.
    """
    return float(u[0] * v[1] - u[1] * v[0])


def _is_convex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """True when b is a convex corner of a counter-clockwise polygon."""
    return _cross_2d(b - a, c - b) > 0.0


def _point_in_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    """True when p lies inside (or on) triangle abc, which is counter-clockwise."""
    return (
        _cross_2d(b - a, p - a) >= 0.0
        and _cross_2d(c - b, p - b) >= 0.0
        and _cross_2d(a - c, p - c) >= 0.0
    )


def _triangulate_simple_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple (non-self-intersecting) polygon.

    ``points`` must be counter-clockwise and free of duplicate or collinear
    consecutive vertices. Returns index triples into ``points``.

    Written out rather than delegated to shapely: the profiles here are simple
    polygons of under ten vertices, and shapely is not a trimesh core
    dependency, so this keeps the install to what `pip install -e ".[dev]"`
    already provides.
    """
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []

    guard = 0
    max_iterations = len(points) ** 2
    while len(remaining) > 3:
        guard += 1
        if guard > max_iterations:
            raise ValueError(
                "ear clipping failed to terminate; the profile is probably "
                "self-intersecting or has collinear vertices"
            )

        for position in range(len(remaining)):
            i_prev = remaining[position - 1]
            i_curr = remaining[position]
            i_next = remaining[(position + 1) % len(remaining)]

            a, b, c = points[i_prev], points[i_curr], points[i_next]
            if not _is_convex(a, b, c):
                continue
            if any(
                _point_in_triangle(points[other], a, b, c)
                for other in remaining
                if other not in (i_prev, i_curr, i_next)
            ):
                continue

            triangles.append((i_prev, i_curr, i_next))
            remaining.pop(position)
            break
        else:
            raise ValueError("no ear found; the profile is not a simple polygon")

    triangles.append(tuple(remaining))  # type: ignore[arg-type]
    return triangles


def _extrude_profile(profile_xz: np.ndarray, depth: float) -> trimesh.Trimesh:
    """Extrude a closed, counter-clockwise 2D profile in xz along +y.

    Builds the side walls as quads between corresponding profile vertices and
    caps both ends with the triangulated profile, then lets trimesh make the
    winding consistent and outward-facing.
    """
    if depth <= 0.0:
        raise ValueError(f"depth must be positive, got {depth}")

    count = len(profile_xz)
    cap_triangles = _triangulate_simple_polygon(profile_xz)

    # Vertices 0..count-1 at y=0, count..2*count-1 at y=depth.
    vertices = np.vstack(
        [
            np.column_stack(
                [profile_xz[:, 0], np.zeros(count), profile_xz[:, 1]]
            ),
            np.column_stack(
                [profile_xz[:, 0], np.full(count, depth), profile_xz[:, 1]]
            ),
        ]
    )

    faces: list[tuple[int, int, int]] = []
    for i, j, k in cap_triangles:
        faces.append((i, k, j))                                  # y=0 cap
        faces.append((i + count, j + count, k + count))           # y=depth cap
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, j + count))                           # side wall
        faces.append((i, j + count, i + count))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    if not mesh.is_watertight:
        raise ValueError("the extruded profile is not watertight")
    return _ensure_outward(mesh)


def _ensure_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make face winding outward-facing, judged by the sign of the volume.

    Used instead of trimesh's own normal repair, which pulls in networkx.
    Every mesh here is built explicitly with a known winding, so the signed
    volume is enough: it is positive exactly when the faces face outward.
    """
    if not mesh.is_watertight:
        raise ValueError("mesh is not watertight, so its winding cannot be judged")
    if not mesh.is_winding_consistent:
        raise ValueError(
            "mesh has inconsistent face winding; some faces point inward. "
            "The signed volume is meaningless in that state, so this cannot be "
            "repaired by flipping the whole mesh."
        )
    if mesh.volume < 0.0:
        mesh.invert()
    if mesh.volume <= 0.0:
        raise ValueError(f"mesh has non-positive volume: {mesh.volume}")
    return mesh


def _settle(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Move a mesh so its bounding box starts at x=0, y=0, z=0."""
    mesh.apply_translation(-mesh.bounds[0])
    mesh.merge_vertices()
    return _ensure_outward(mesh)


def make_box(length: float, depth: float, height: float) -> trimesh.Trimesh:
    """A plain rectangular block: the no-overhang regression part."""
    mesh = trimesh.creation.box(extents=(length, depth, height))
    return _settle(mesh)


def make_ramp(
    angle_deg: float,
    length: float,
    depth: float,
    height: float,
    base_fraction: float = 0.25,
    run_fraction: float = 0.30,
) -> trimesh.Trimesh:
    """A block with exactly one overhanging face at ``angle_deg`` from vertical.

    Parameters
    ----------
    angle_deg
        Overhang angle in degrees from vertical. 0 is a vertical wall (no
        overhang), 90 a flat horizontal ledge.
    length, depth, height
        Overall bounding-box dimensions in x, y and z.
    base_fraction
        Height of the supporting column below the overhang, as a fraction of
        ``height``. The overhang starts here, so this must leave room for both
        the overhanging face and some wall above it.
    run_fraction
        Horizontal reach of the overhang, as a fraction of ``length``. Held
        constant across angles so the ramps are comparable: every ramp reaches
        the same distance sideways, and only the steepness of the underside
        differs. The face rises ``run / tan(angle)``, which is largest at 45
        degrees and zero at 90.

    The profile, counter-clockwise in the xz-plane, is a supporting column, the
    overhanging face rising outward from its top corner, a vertical wall to the
    top, and the flat top surface back to the origin.
    """
    if not 0.0 < angle_deg <= 90.0:
        raise ValueError(f"angle_deg must be in (0, 90], got {angle_deg}")
    if not 0.0 < base_fraction < 1.0:
        raise ValueError(f"base_fraction must be in (0, 1), got {base_fraction}")
    if not 0.0 < run_fraction < 1.0:
        raise ValueError(f"run_fraction must be in (0, 1), got {run_fraction}")

    angle = math.radians(angle_deg)
    base_height = height * base_fraction
    run = length * run_fraction
    rise = run / math.tan(angle)

    column_width = length - run
    top_of_face = base_height + rise
    if top_of_face >= height:
        raise ValueError(
            f"a {angle_deg} deg overhang reaching {run:.1f} mm rises "
            f"{rise:.1f} mm from a {base_height:.1f} mm base, needing height > "
            f"{top_of_face:.1f} mm (got {height:.1f}). Raise height, or lower "
            "base_fraction or run_fraction."
        )

    profile = np.array(
        [
            [0.0, 0.0],                      # bottom-left
            [column_width, 0.0],             # bottom-right of the column
            [column_width, base_height],     # where the overhang begins
            [length, top_of_face],           # end of the overhanging face
            [length, height],                # outer wall up to the top
            [0.0, height],                   # top surface back to the left
        ]
    )
    return _settle(_extrude_profile(profile, depth))


def make_tshape(
    length: float,
    depth: float,
    height: float,
    underside_angle_deg: float = 90.0,
    stem_fraction: float = 0.3,
) -> trimesh.Trimesh:
    """A stem carrying a crossbar, with overhangs on both sides.

    Parameters
    ----------
    underside_angle_deg
        Overhang angle of the crossbar's underside, from vertical. 90 gives a
        flat underside (the hardest case); smaller values angle it. Gate D0
        picks the value used for the benchmark.
    stem_fraction
        Stem width as a fraction of ``length``.
    """
    if not 0.0 < underside_angle_deg <= 90.0:
        raise ValueError(
            f"underside_angle_deg must be in (0, 90], got {underside_angle_deg}"
        )

    angle = math.radians(underside_angle_deg)
    stem_width = length * stem_fraction
    overhang_run = (length - stem_width) * 0.5
    rise = 0.0 if math.tan(angle) > 1e9 else overhang_run / math.tan(angle)

    # A shallow underside rises a long way (rise = run / tan(angle), so 35 mm of
    # reach at 45 degrees rises 35 mm). Lower the crossbar as needed rather than
    # refusing the angle, keeping a minimum thickness of solid bar above it, so
    # the whole D0 range of underside angles is available.
    minimum_bar_thickness = height * 0.15
    bar_bottom = min(height * 0.5, height - rise - minimum_bar_thickness)
    if bar_bottom <= 0.0:
        raise ValueError(
            f"an underside at {underside_angle_deg} deg reaching "
            f"{overhang_run:.1f} mm rises {rise:.1f} mm, which needs height > "
            f"{rise + minimum_bar_thickness:.1f} mm (got {height:.1f})"
        )

    left = overhang_run
    right = overhang_run + stem_width
    profile = np.array(
        [
            [left, 0.0],                     # stem, bottom-left
            [right, 0.0],                    # stem, bottom-right
            [right, bar_bottom],             # right overhang starts
            [length, bar_bottom + rise],     # right underside
            [length, height],                # right outer wall
            [0.0, height],                   # top surface
            [0.0, bar_bottom + rise],        # left outer wall
            [left, bar_bottom],              # left underside
        ]
    )
    return _settle(_extrude_profile(profile, depth))


def _heightfield_solid(
    xs: np.ndarray, ys: np.ndarray, top_z: np.ndarray
) -> trimesh.Trimesh:
    """A watertight solid between a flat z=0 base and a height field.

    ``top_z`` is indexed ``[ix, iy]``. Built directly rather than by a boolean
    union: constructive solid geometry in trimesh needs a backend (shapely or
    manifold3d) that is not a core dependency, and a height field cannot
    produce a non-watertight result.
    """
    nx, ny = len(xs), len(ys)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

    top = np.column_stack([grid_x.ravel(), grid_y.ravel(), top_z.ravel()])
    bottom = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(nx * ny)]
    )
    vertices = np.vstack([top, bottom])
    offset = nx * ny

    def index(ix: int, iy: int) -> int:
        return ix * ny + iy

    faces: list[tuple[int, int, int]] = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            a, b = index(ix, iy), index(ix + 1, iy)
            c, d = index(ix + 1, iy + 1), index(ix, iy + 1)
            faces.append((a, b, c))          # top surface, upward
            faces.append((a, c, d))
            faces.append((a + offset, c + offset, b + offset))  # base, downward
            faces.append((a + offset, d + offset, c + offset))

    # Side walls around the four edges of the grid.
    #
    # `wall(i0, i1)` produces the normal
    #     (i0_top -> i0_bottom) x (i0_bottom -> i1_bottom)
    #   = (0, 0, -1) x (dx, dy, 0) = (dy, -dx, 0)
    # so the outward normal is the i0 -> i1 direction turned -90 degrees in xy.
    # Each edge below is therefore walked in the direction that points its
    # normal away from the solid.
    def wall(i0: int, i1: int) -> None:
        faces.append((i0, i0 + offset, i1 + offset))
        faces.append((i0, i1 + offset, i1))

    for ix in range(nx - 1):
        wall(index(ix, 0), index(ix + 1, 0))                  # y=0, outward -y
        wall(index(ix + 1, ny - 1), index(ix, ny - 1))         # y=max, outward +y
    for iy in range(ny - 1):
        wall(index(0, iy + 1), index(0, iy))                   # x=0, outward -x
        wall(index(nx - 1, iy), index(nx - 1, iy + 1))         # x=max, outward +x

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    if not mesh.is_watertight:
        raise ValueError("the height-field solid is not watertight")
    return _ensure_outward(mesh)


def make_twin_domes(
    base_width: float,
    depth: float,
    base_height: float,
    radius_fraction: float = 0.20,
    samples_per_mm: float = 1.2,
) -> trimesh.Trimesh:
    """Two hemispherical domes on a shared rectangular base.

    The curved tops are what planar slicing turns into staircases, so this is
    the part that shows conformal layers off. It has no overhang worth the
    name, which makes it the check that an overhang-aware field does not make
    smooth surfaces worse.

    Parameters
    ----------
    radius_fraction
        Dome radius as a fraction of ``base_width``. The domes are centred at
        27 % and 73 % of the width, so anything up to 0.23 keeps them apart.
    samples_per_mm
        Height-field resolution. The domes are only as smooth as this grid, so
        it should stay comfortably finer than the deposition width.
    """
    if radius_fraction <= 0.0 or radius_fraction > 0.23:
        raise ValueError(
            f"radius_fraction must be in (0, 0.23], got {radius_fraction}"
        )

    radius = base_width * radius_fraction
    if 2.0 * radius >= depth:
        raise ValueError(
            f"domes of radius {radius:.1f} mm do not fit in a {depth:.1f} mm depth"
        )

    nx = max(24, int(round(base_width * samples_per_mm)) + 1)
    ny = max(12, int(round(depth * samples_per_mm)) + 1)
    xs = np.linspace(0.0, base_width, nx)
    ys = np.linspace(0.0, depth, ny)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    top_z = np.full(grid_x.shape, base_height)
    for centre_x in (base_width * 0.27, base_width * 0.73):
        squared = radius**2 - (grid_x - centre_x) ** 2 - (grid_y - depth * 0.5) ** 2
        dome = np.sqrt(np.maximum(squared, 0.0))
        top_z = np.maximum(top_z, base_height + dome)

    return _settle(_heightfield_solid(xs, ys, top_z))


def default_dimensions(principal_mm: float) -> dict[str, float]:
    """Proportions used for every part at a given principal dimension.

    The parts are deliberately not cubic. `order_atoms` costs roughly linear
    time in atom count and atom count scales with volume, so a part that is
    100 mm in every axis is ~11x the work of one that is 100 mm long but
    slimmer, for no extra information about overhang behaviour.
    """
    return {
        "length": principal_mm,
        "depth": principal_mm * 0.45,
        "height": principal_mm * 0.60,
    }
