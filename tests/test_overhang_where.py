"""`tools/overhang_where.py`: where a part's worst overhang points are (diagnostic)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_where


def test_it_splits_points_over_air_from_points_on_material(repo_root, tmp_path, capsys):
    """Two atoms by `ramp60_xs`'s underside: one hanging over it, one on the column below the corner."""
    under = 4.5 + 4.0 / math.tan(math.radians(60))
    points = np.array([[25.0, 6.0, under + 0.2], [20.7, 6.0, 4.2]], dtype=np.float32)
    frame = tmp_path / "atoms.npz"
    np.savez(frame, point=points, normal=np.zeros((2, 2), dtype=np.float32), phi_t=np.zeros(2, dtype=np.float32))

    assert overhang_where.main([str(repo_root / "data" / "param" / "ramp60_xs.json"), "--frame", str(frame)]) == 0
    out = capsys.readouterr().out
    assert "surface 60 deg" in out
    assert "over air (counted): 1 points, worst 60.0" in out
    assert "onto material (skipped): 1 points" in out
    assert "25.00    6.00" in out
