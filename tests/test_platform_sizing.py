"""The platform is sized in the frame the G-code is written in (plan_corrections 7b, P4-4).

`tools/add_platform.py` draws platform layers from x = 0, y = 0, and
`toolpath_to_gcode` re-centres the part *plus* platform on the bed. A lift
found with the part's own box re-centred could fall short in that frame and
abort the whole G-code ("Fatal Error: collision found!"). These tests build
what `add_platform` builds and solve it the way `toolpath_to_gcode` does.

No `from __future__ import annotations` here; these tests drive the Taichi
kernels in `atom.kinematics3z`.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atom import contracts, machine_profile
from atom import toolpath_view as tv

WIDTH, LAYER = 0.9, 0.45


@pytest.fixture(scope="module")
def kinematics(ti_cpu):
    from atom import kinematics3z

    return kinematics3z


class Toolpath:
    """The two attributes and one method `get_plaftorm_size` uses."""

    def __init__(self, point, tool_orientation):
        self.point = np.ascontiguousarray(point, dtype=np.float32)
        self.tool_orientation = np.ascontiguousarray(tool_orientation, dtype=np.float32)

    def get_aabb(self):
        return self.point.min(axis=0), self.point.max(axis=0)


def slab(x0, tilt_deg, azimuth_deg):
    """A one-layer part 30 x 10 mm, x from ``x0``, every point tilted alike."""
    xs, ys = np.meshgrid(np.arange(x0, x0 + 30.1, 1.0), np.arange(5.0, 15.1, 1.0))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.45)])
    orientation = np.column_stack([np.full(len(points), np.radians(tilt_deg)),
                                   np.full(len(points), np.radians(azimuth_deg))])
    return Toolpath(points, orientation)


def lift_left_in_the_gcode(toolpath, size):
    """Build what add_platform builds for ``size`` and solve it re-centred as
    toolpath_to_gcode does: the largest lift any point still asks for.
    toolpath_to_gcode aborts the file if it is not zero."""
    height = size[2]
    layers = [
        np.array([[0, 0, z], [0, size[1], z], [size[0], size[1], z], [size[0], 0, z]])
        for z in ((i + 1) * LAYER for i in range(int(height / LAYER) - 1))
    ]
    part = toolpath.point + np.array([0.0, 0.0, height])
    points = np.vstack(layers + [part]) if layers else part
    orientation = np.vstack([np.zeros((len(points) - len(part), 2)), toolpath.tool_orientation])
    count = len(points)
    solved = contracts.from_toolpath(SimpleNamespace(
        point=points.astype(np.float32), tool_orientation=orientation.astype(np.float32),
        travel_type=np.zeros(count, np.int32), width=np.ones(count), height=np.ones(count),
        point_count=count), machine_profile.load_profile("reference"), center_on_bed=True)
    assert solved.valid.all()
    return solved.max_lift_mm


def test_a_platform_sized_in_the_parts_frame_could_abort_the_gcode(kinematics):
    """The failure, reproduced: 25 degrees toward 135 near the bed, the part
    40 mm from the origin. Sized in the part's own frame, the platform leaves
    about 5 mm of lift in the G-code's frame."""
    toolpath = slab(40.0, 25.0, 135.0)
    low, high = toolpath.get_aabb()
    height = np.ceil(kinematics._lift_needed(toolpath, low, high, 0.0) / LAYER) * LAYER
    old = (np.ceil(high[0] / WIDTH) * WIDTH, np.ceil(high[1] / WIDTH) * WIDTH, height)

    assert lift_left_in_the_gcode(toolpath, old) > 1.0


@pytest.mark.parametrize("x0, tilt, azimuth", [(40.0, 25.0, 135.0), (40.0, 28.0, 225.0)])
def test_the_platform_is_sized_in_the_gcodes_frame(kinematics, x0, tilt, azimuth):
    toolpath = slab(x0, tilt, azimuth)
    size = kinematics.get_plaftorm_size(toolpath, WIDTH, LAYER)

    assert int(size[2] / LAYER) - 1 > 0  # a drawn platform: the case that mattered
    assert lift_left_in_the_gcode(toolpath, size) == 0.0


def test_without_a_drawn_platform_nothing_changes(kinematics, repo_root):
    """The golden cube needs no lift, so no platform layers are drawn, the
    two frames are the same, and its sizing is what it always was: the
    golden G-code is untouched."""
    arrays = tv.load_toolpath_arrays(repo_root / "tests/golden/calibration_cube.toolpath.npz")
    toolpath = Toolpath(arrays["point"], arrays["tool_orientation"])
    high = toolpath.point.max(axis=0)

    size = kinematics.get_plaftorm_size(toolpath, WIDTH, LAYER)

    assert size[2] == 0.0
    assert size[0] == pytest.approx(np.ceil(high[0] / WIDTH) * WIDTH)
    assert size[1] == pytest.approx(np.ceil(high[1] / WIDTH) * WIDTH)
