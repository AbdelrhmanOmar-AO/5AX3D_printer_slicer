"""`atom.layer_thickness`: the real layer thickness under each bead (build plan P3.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atom import layer_thickness as lt

NOMINAL = 0.45


class _Toolpath:
    def __init__(self, points, direction=(0.0, 0.0, 1.0), travel=None):
        self.point = np.asarray(points, dtype=np.float32)
        n = len(self.point)
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        self.tool_orientation = np.tile([math.acos(d[2]), math.atan2(d[1], d[0])], (n, 1)).astype(np.float32)
        self.travel_type = np.zeros(n, dtype=np.int32) if travel is None else np.asarray(travel, dtype=np.int32)
        self.width = np.full(n, 0.9, dtype=np.float32)
        self.height = np.full(n, NOMINAL, dtype=np.float32)
        self.point_count = n


def _columns(zs, xs=(0.0, 0.9, 1.8)):
    return [[x, 0.0, z] for z in zs for x in xs]


def test_evenly_stacked_layers_measure_the_nominal():
    zs = 0.225 + NOMINAL * np.arange(8)
    result = lt.layer_thickness(_Toolpath(_columns(zs)), NOMINAL)
    assert result.on_bed.sum() == 3  # the first layer is not judged
    np.testing.assert_allclose(result.thickness_mm[result.judged], NOMINAL, atol=1e-5)
    assert not result.thin.any() and not result.thick.any()


def test_a_missing_bead_below_reads_as_a_thick_layer():
    """The bead over a one-layer gap sits two layers above the next one down."""
    zs = 0.225 + NOMINAL * np.arange(6)
    points = [p for p in _columns(zs) if not (p[0] == 0.9 and abs(p[2] - zs[3]) < 1e-9)]
    result = lt.layer_thickness(_Toolpath(points), NOMINAL)
    above_gap = [i for i, p in enumerate(points) if p[0] == 0.9 and abs(p[2] - zs[4]) < 1e-9][0]
    assert result.thickness_mm[above_gap] == pytest.approx(2 * NOMINAL, abs=1e-5)
    assert result.thick.sum() == 1


def test_a_squeezed_layer_reads_as_thin():
    zs = [0.225, 0.675, 1.125, 1.275, 1.725]  # 0.15 between the third and fourth layers
    result = lt.layer_thickness(_Toolpath(_columns(zs)), NOMINAL)
    assert result.thin.sum() == 3
    np.testing.assert_allclose(result.thickness_mm[result.thin], 0.15, atol=1e-5)


def test_the_layer_below_is_found_along_a_tilted_build_direction():
    d = np.array([math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))])
    base = np.array([[x, 0.0, 0.225] for x in (0.0, 0.9, 1.8)])
    points = np.vstack([base + k * NOMINAL * d for k in range(6)])
    result = lt.layer_thickness(_Toolpath(points, d), NOMINAL)
    np.testing.assert_allclose(result.thickness_mm[result.judged], NOMINAL, atol=1e-5)


def test_a_bead_with_nothing_under_it_is_counted_apart():
    points = _columns([0.225, 0.675, 1.125]) + [[3.0, 0.0, 1.575]]  # out past the edge
    result = lt.layer_thickness(_Toolpath(points), NOMINAL)
    assert result.nothing_below.tolist() == [False] * 9 + [True]


def test_travel_moves_are_not_beads():
    zs = 0.225 + NOMINAL * np.arange(4)
    points = _columns(zs, xs=(0.0,))
    travel = [0, 0, 1, 0]  # the third point is a travel move: the fourth sits over a gap
    result = lt.layer_thickness(_Toolpath(points, travel=travel), NOMINAL)
    assert len(result.thickness_mm) == 3
    assert result.thickness_mm[-1] == pytest.approx(2 * NOMINAL, abs=1e-5)
