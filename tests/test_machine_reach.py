"""`atom.machine_reach`: the platform `add_platform` prints, and points out of reach (build plan P2.5).

On the reference profile, on the CPU. The vendored `get_plaftorm_size` is the
oracle where it does not abort.
"""

import math

import numpy as np
import pytest

from atom import machine_reach as mr


def _toolpath(points, tilt_deg, azimuth_deg=0.0):
    from atom import toolpath3

    points = np.asarray(points, dtype=np.float32)
    count = len(points)
    tp = toolpath3.Toolpath()
    tp.from_numpy({
        "point": points,
        "travel_type": np.zeros(count, dtype=np.int32),
        "tool_orientation": np.tile([math.radians(tilt_deg), math.radians(azimuth_deg)], (count, 1)).astype(np.float32),
        "width": np.full(count, 0.9, dtype=np.float32),
        "height": np.full(count, 0.45, dtype=np.float32),
        "point_count": np.array(count),
    })
    return tp


def _column(height=6.0):
    """A small column of points from the bed up, 5 x 5 mm."""
    return [[x, y, z] for x in (0.0, 5.0) for y in (0.0, 5.0) for z in np.arange(0.2, height, 0.45)]


@pytest.fixture
def reference(ti_cpu, monkeypatch):
    monkeypatch.delenv("ATOM_MACHINE", raising=False)


def test_an_upright_toolpath_needs_no_platform(reference):
    reach = mr.platform_and_reach(_toolpath(_column(), 0.0), 0.9, 0.45)
    assert reach.unreachable_points == 0 and reach.platform_mm == 0.0
    assert reach.profile == "reference"


def test_the_platform_is_the_one_add_platform_prints(reference):
    """Tilted 30 degrees near the bed the corners hit the gantry: a platform, as the vendored code sizes it."""
    from atom import kinematics3z

    tilted = _toolpath(_column(), 30.0, 45.0)
    reach = mr.platform_and_reach(tilted, 0.9, 0.45, tesselate_deg=None)
    vendored = kinematics3z.get_plaftorm_size(_toolpath(_column(), 30.0, 45.0), 0.9, 0.45)
    assert reach.unreachable_points == 0
    assert reach.platform_mm > 10.0
    assert reach.platform_mm == pytest.approx(float(vendored[2]))
    assert reach.platform_mm == pytest.approx(math.ceil(reach.lift_mm / 0.45) * 0.45)


def test_points_out_of_reach_are_counted_where_the_pipeline_would_stop(reference):
    """Over the machine's 30-degree limit: the vendored code asserts; this counts."""
    points = _column()
    tp = _toolpath(points, 35.0)
    reach = mr.platform_and_reach(tp, 0.9, 0.45, tesselate_deg=None)
    assert reach.unreachable_points == len(points)
    assert math.isnan(reach.platform_mm)


def test_the_toolpath_given_is_left_as_it_was(reference):
    tp = _toolpath(_column(), 0.0)
    before = np.array(tp.point).copy()
    mr.platform_and_reach(tp, 0.9, 0.45)
    assert int(tp.point_count) == len(before)
    np.testing.assert_array_equal(tp.point, before)


def test_the_tesselated_count_matches_the_kernel(reference):
    from atom import toolpath3

    tp = _toolpath(_column(3.0), 0.0)
    orientation = np.array(tp.tool_orientation)
    orientation[::10, 0] = math.radians(4.5)  # one point in ten 4.5 degrees over: within the kernel's room
    tp.tool_orientation = np.ascontiguousarray(orientation)
    expected = mr.tesselated_count(tp.tool_orientation, 1.0)
    assert len(orientation) < expected < 3 * len(orientation)
    tp.tesselate_orientation(1.0)
    assert int(tp.point_count) == expected


def test_turns_too_large_for_the_tesselation_are_checked_as_given(reference):
    """The vendored kernel would write past its array (a crash, not an error)."""
    points = _column()
    tp = _toolpath(points, 0.0)
    orientation = np.array(tp.tool_orientation)
    orientation[::2, 0] = math.radians(25.0)  # 25 one-degree steps per move: far over three times
    tp.tool_orientation = np.ascontiguousarray(orientation)
    reach = mr.platform_and_reach(tp, 0.9, 0.45)
    assert reach.tesselated is False and reach.points_checked == len(points)
