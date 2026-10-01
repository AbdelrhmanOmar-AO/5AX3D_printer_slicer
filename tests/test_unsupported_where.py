"""`tools/unsupported_where.py`: why deposition near an overhang is unsupported (diagnostic)."""

from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_report as orep
import unsupported_where
from atom import benchmark_meshes as bm


class _Toolpath:
    def __init__(self, points):
        self.point = np.asarray(points, dtype=np.float32)
        n = len(points)
        self.tool_orientation = np.zeros((n, 2), dtype=np.float32)  # all vertical
        self.travel_type = np.zeros(n, dtype=np.int32)
        self.height = np.full(n, 0.45, dtype=np.float32)
        self.width = np.full(n, 0.9, dtype=np.float32)
        self.point_count = n


def _underside_z(x, angle=60.0):
    return 4.5 + (x - 21.0) / math.tan(math.radians(angle))


def test_it_tells_printed_too_early_from_nothing_beneath(tmp_path):
    stl = tmp_path / "ramp60.stl"
    bm.make_ramp(60, length=30.0, depth=13.5, height=18.0).export(stl)
    mesh, normals, centres, _ = orep.load_mesh_arrays(stl, max_edge=0.9)
    za, zc = _underside_z(25.0) + 0.2, _underside_z(28.0) + 0.2
    points = [
        [5.0, 6.0, 0.225],  # on the bed
        [25.0, 6.0, za],  # A: the bead below it comes later
        [28.0, 6.0, zc],  # C: nothing below it at all
        [25.0, 6.0, za - 0.45],  # B: below A, printed after it
    ]
    found = unsupported_where.classify(_Toolpath(points), np.asarray(mesh.triangles), normals, centres, 0.9)

    by_index = dict(zip(found["index"].tolist(), found["printed_too_early"].tolist()))
    assert by_index[1] is True and by_index[2] is False
    assert found["order_gap"][found["index"].tolist().index(1)] == 2
    text = unsupported_where.report(found, 0.45)
    assert "printed too early (material in its cone, printed later): 1" in text
