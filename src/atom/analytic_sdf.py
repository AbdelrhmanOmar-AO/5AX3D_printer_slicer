"""Exact signed distance fields of simple parts, in numpy (build plans P2.1, P2.2).

The pipeline builds its SDF from a Blender remesh and a point cloud
(`tools/bpn_to_sdf.py`), which needs Blender and a GPU to be quick. For
testing and iterating on the orientation field, an exact SDF of the same
part is enough and costs a second: the field only reads the SDF.

The grid follows `solid3.SDF`: origin at 0, ``count = ceil(size / cell)``
cells, cell centres at ``(i + 0.5) * cell``; negative inside, in mm.

numpy only.
"""

from __future__ import annotations

import math

import numpy as np


def polygon_sdf(px, pz, vertices) -> np.ndarray:
    """Signed distance from points ``(px, pz)`` to a closed polygon, negative inside."""
    vertices = np.asarray(vertices, dtype=np.float64)
    d = (px - vertices[0, 0]) ** 2 + (pz - vertices[0, 1]) ** 2
    sign = np.ones_like(px, dtype=np.float64)
    j = len(vertices) - 1
    for i in range(len(vertices)):
        ex, ez = vertices[j] - vertices[i]
        wx, wz = px - vertices[i, 0], pz - vertices[i, 1]
        t = np.clip((wx * ex + wz * ez) / (ex * ex + ez * ez), 0.0, 1.0)
        d = np.minimum(d, (wx - ex * t) ** 2 + (wz - ez * t) ** 2)
        above = pz >= vertices[i, 1]
        below = pz < vertices[j, 1]
        left = ex * wz > ez * wx
        sign = np.where((above & below & left) | (~above & ~below & ~left), -sign, sign)
        j = i
    return sign * np.sqrt(d)


def grid_count(size_mm, cell_mm) -> np.ndarray:
    """Cells per axis for a part of ``size_mm``, as `solid3.SDF.create_from_bpn` sizes them."""
    return np.ceil(np.asarray(size_mm, dtype=np.float64) / cell_mm).astype(int)


def extruded_profile_sdf(profile_xz, depth_mm, cell_mm, size_mm=None) -> np.ndarray:
    """The SDF of an xz profile extruded along y from 0 to ``depth_mm``, float32.

    ``size_mm`` is the grid's extent; by default the profile's bounding box
    and the depth, which is what the pipeline's grid would cover.
    """
    profile_xz = np.asarray(profile_xz, dtype=np.float64)
    if size_mm is None:
        size_mm = (profile_xz[:, 0].max(), depth_mm, profile_xz[:, 1].max())
    count = grid_count(size_mm, cell_mm)
    xs = (np.arange(count[0]) + 0.5) * cell_mm
    ys = (np.arange(count[1]) + 0.5) * cell_mm
    zs = (np.arange(count[2]) + 0.5) * cell_mm
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    in_profile = polygon_sdf(X, Z, profile_xz)[:, None, :]
    in_slab = (np.abs(ys - depth_mm / 2) - depth_mm / 2)[None, :, None]
    sdf = np.minimum(np.maximum(in_profile, in_slab), 0.0) + np.sqrt(
        np.maximum(in_profile, 0.0) ** 2 + np.maximum(in_slab, 0.0) ** 2
    )
    return sdf.astype(np.float32)


def ramp_geometry(angle_deg, length, height, base_fraction=0.25, run_fraction=0.30) -> dict:
    """The profile and key points of `benchmark_meshes.make_ramp`.

    Returns ``profile`` (the xz polygon), ``column_width``, ``base_height``,
    ``run`` and ``normal``: the underside's outward unit normal,
    ``(cos A, 0, -sin A)``.
    """
    angle = math.radians(angle_deg)
    base_height = height * base_fraction
    run = length * run_fraction
    rise = run / math.tan(angle) if angle_deg < 90 else 0.0
    column_width = length - run
    profile = np.array(
        [
            [0.0, 0.0],
            [column_width, 0.0],
            [column_width, base_height],
            [length, base_height + rise],
            [length, height],
            [0.0, height],
        ]
    )
    return {
        "profile": profile,
        "column_width": column_width,
        "base_height": base_height,
        "run": run,
        "normal": np.array([math.cos(angle), 0.0, -math.sin(angle)]),
    }


def ramp_sdf(angle_deg, length, depth, height, cell_mm):
    """``(sdf, geometry)``: the exact SDF of a benchmark ramp and its `ramp_geometry`."""
    geometry = ramp_geometry(angle_deg, length, height)
    return extruded_profile_sdf(geometry["profile"], depth, cell_mm, (length, depth, height)), geometry
