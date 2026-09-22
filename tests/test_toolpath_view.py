"""Tests for the toolpath viewer's data layer (build plan P5.4a).

Everything the window shows that can be decided without a display: which
segments are visible, what each colour mode shows, and the frames involved.
All synthetic except where noted.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from atom import overhang_metrics as om
from atom import toolpath_view as tv


def make_view(points, deposit=None, tilts_deg=None, azimuths_deg=None, height=0.45,
              width=0.9, platform_count=0):
    points = np.asarray(points, dtype=float)
    count = len(points)
    deposit = np.ones(count, bool) if deposit is None else np.asarray(deposit, bool)
    tilts = np.zeros(count) if tilts_deg is None else np.asarray(tilts_deg, float)
    azimuths = np.zeros(count) if azimuths_deg is None else np.asarray(azimuths_deg, float)
    return tv.from_arrays(
        points,
        np.where(deposit, tv.TRAVEL_TYPE_DEPOSITION, 1),
        np.column_stack([np.radians(tilts), np.radians(azimuths)]),
        width=np.full(count, width),
        height=np.full(count, height),
        platform_count=platform_count,
    )


# --------------------------------------------------------------------------
# Names and files
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, part",
    [
        ("data/toolpath/ramp60_xs_smoothed.npz", "ramp60_xs"),
        ("data/toolpath/ramp60_xs_smoothed_tesselated.npz", "ramp60_xs"),
        ("data/toolpath/ramp60_xs_platform.npz", "ramp60_xs"),
        ("data/toolpath/ramp60_xs.npz", "ramp60_xs"),
        ("data/gcode/ramp60_xs.gcode", "ramp60_xs"),
        ("data/gcode/ramp60_xs_craftware.gcode", "ramp60_xs"),
        ("reports/toolpaths/ramp60_s_ms30.npz", "ramp60_s"),
        ("reports/toolpaths/twin_domes_xs_ms7.npz", "twin_domes_xs"),
        ("tests/golden/calibration_cube.toolpath.npz", "calibration_cube"),
    ],
)
def test_part_name_is_recovered_from_every_file_the_pipeline_writes(path, part):
    assert tv.part_name_from_path(path) == part


def test_load_toolpath_arrays_cuts_the_nan_padding(tmp_path):
    """`toolpath3.Toolpath.allocate` pads with NaN beyond point_count."""
    point = np.full((5, 3), np.nan, dtype=np.float32)
    point[:3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    path = tmp_path / "part_smoothed.npz"
    np.savez(
        path,
        point=point,
        travel_type=np.array([1, 0, 0, -1, -1], np.int32),
        tool_orientation=np.zeros((5, 2), np.float32),
        width=np.full(5, 0.9, np.float32),
        height=np.full(5, 0.45, np.float32),
        point_count=np.array(3),
        platform_height=np.array(0),
    )

    arrays = tv.load_toolpath_arrays(path)

    assert arrays["point_count"] == 3
    assert arrays["point"].shape == (3, 3)
    assert np.isfinite(arrays["point"]).all()


def test_golden_toolpath_loads_with_every_point(repo_root):
    arrays = tv.load_toolpath_arrays(repo_root / "tests/golden/calibration_cube.toolpath.npz")
    assert arrays["point_count"] == 46773
    assert len(arrays["point"]) == 46773


# --------------------------------------------------------------------------
# Tilt and tilt direction
# --------------------------------------------------------------------------


def test_tilt_and_azimuth_come_straight_from_the_spherical_orientation():
    """theta is the tilt and phi the azimuth (docs/conventions.md section 2)."""
    view = make_view([[0, 0, 0], [1, 0, 0], [2, 0, 0]], tilts_deg=[0, 20, 30],
                     azimuths_deg=[0, 90, 225])

    np.testing.assert_allclose(view.tilt_deg, [0, 20, 30], atol=1e-9)
    assert math.isnan(view.azimuth_deg[0])  # vertical: leans nowhere
    np.testing.assert_allclose(view.azimuth_deg[1:], [90, 225], atol=1e-9)


def test_azimuth_is_in_zero_to_360():
    view = make_view([[0, 0, 0], [1, 0, 0]], tilts_deg=[10, 10], azimuths_deg=[-90, 359])
    np.testing.assert_allclose(view.azimuth_deg, [270, 359], atol=1e-9)


# --------------------------------------------------------------------------
# Visible segments
# --------------------------------------------------------------------------


def test_scrubbing_shows_segments_up_to_and_including_the_end_point():
    view = make_view([[i, 0, 0] for i in range(6)])

    np.testing.assert_array_equal(tv.visible_segments(view, end=3), [1, 2, 3])
    np.testing.assert_array_equal(tv.visible_segments(view, end=0), [])
    np.testing.assert_array_equal(tv.visible_segments(view, end=99), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(tv.visible_segments(view, end=4, start=3), [3, 4])


def test_travel_and_printing_moves_are_separated_by_the_arriving_point():
    """Segment i takes point i's type, as the G-code move to point i does."""
    view = make_view([[i, 0, 0] for i in range(5)], deposit=[True, True, False, True, True])

    np.testing.assert_array_equal(tv.visible_segments(view, end=4, deposit=True), [1, 3, 4])
    np.testing.assert_array_equal(tv.visible_segments(view, end=4, deposit=False), [2])


def test_z_clip_hides_a_segment_when_either_end_is_above_it():
    view = make_view([[0, 0, 0], [1, 0, 0], [1, 0, 2], [2, 0, 2]])

    np.testing.assert_array_equal(tv.visible_segments(view, end=3, z_max=1.0), [1])
    np.testing.assert_array_equal(tv.visible_segments(view, end=3, z_max=2.0), [1, 2, 3])


def test_line_cells_are_vtk_two_point_lines():
    np.testing.assert_array_equal(tv.line_cells(np.array([1, 4])), [2, 0, 1, 2, 3, 4])
    assert tv.line_cells(np.array([], dtype=int)).size == 0


# --------------------------------------------------------------------------
# Platform
# --------------------------------------------------------------------------


def test_platform_split_recovers_count_and_lift():
    """add_platform writes the platform, then every input point lifted."""
    part = np.array([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
    platform = np.array([[0.0, 0.0, 0.45], [5.0, 0.0, 0.45]])
    lifted = part + [0.0, 0.0, 0.9]

    assert tv.platform_split(np.vstack([platform, lifted]), part) == (2, pytest.approx(0.9))


def test_platform_split_refuses_files_from_different_runs():
    part = np.array([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5]])
    other = np.array([[0.0, 0.0, 0.45], [1.5, 1.0, 1.4], [2.0, 1.0, 1.4]])
    assert tv.platform_split(other, part) is None
    assert tv.platform_split(part[:1], part) is None  # fewer points than the input


# --------------------------------------------------------------------------
# Colour modes
# --------------------------------------------------------------------------


def test_every_mode_has_a_distinct_key_and_categorical_modes_have_colours():
    keys = [mode.key for mode in tv.COLOUR_MODES]
    assert len(keys) == len(set(keys))
    for mode in tv.COLOUR_MODES:
        assert len(mode.categories) == len(mode.category_colours)
        assert mode.categories or mode.cmap


def test_modes_a_source_cannot_supply_are_reported_not_drawn():
    view = make_view([[0, 0, 0], [1, 0, 0]])
    assert tv.mode_unavailable_reason(view, "tilt", has_mesh=False) is None
    assert "G-code" in tv.mode_unavailable_reason(view, "feed", has_mesh=False)
    assert "STL" in tv.mode_unavailable_reason(view, "shell", has_mesh=False)

    view.height = None  # G-code alone has no bead sizes
    for key in ("width", "height", "unsupported"):
        assert tv.mode_unavailable_reason(view, key, has_mesh=True) is not None


def test_progress_runs_from_zero_to_one_hundred():
    view = make_view([[i, 0, 0] for i in range(5)])
    np.testing.assert_allclose(tv.point_scalars(view, "progress"), [0, 25, 50, 75, 100])


def test_type_mode_codes_platform_points():
    view = make_view([[i, 0, 0] for i in range(4)], platform_count=2)
    np.testing.assert_array_equal(tv.point_scalars(view, "type"), [1, 1, 0, 0])


def test_feed_range_ignores_a_few_spikes():
    """A handful of F*d/l spikes must not flatten every other move's colour."""
    feed = np.concatenate([np.full(995, 600.0), np.full(5, 10000.0)])
    low, high = tv.colour_range(feed, "feed")
    assert low == pytest.approx(600.0)
    assert high < 10000.0


def test_tilt_range_starts_at_zero_and_is_never_empty():
    assert tv.colour_range(np.array([5.0, 12.0]), "tilt") == (0.0, 12.0)
    assert tv.colour_range(np.zeros(3), "tilt") == (0.0, 1.0)
    assert tv.colour_range(np.array([np.nan]), "azimuth") == (0.0, 360.0)


# --------------------------------------------------------------------------
# Unsupported points
# --------------------------------------------------------------------------


def test_unsupported_mask_is_exactly_the_p08_metric():
    """A column, then a point hanging 3 mm out to the side at the top."""
    points = [[0, 0, 0.45], [0, 0, 0.9], [0, 0, 1.35], [3.0, 0, 1.35]]
    view = make_view(points, deposit=[True, True, True, True])

    mask = tv.unsupported_mask(view)

    np.testing.assert_array_equal(mask, [False, False, False, True])
    toolpath = SimpleNamespace(
        point=np.asarray(points, float), travel_type=np.zeros(4, int),
        tool_orientation=np.zeros((4, 2)), height=np.full(4, 0.45), point_count=4)
    np.testing.assert_array_equal(mask[view.deposit], om.unsupported_deposition(toolpath).unsupported)


def test_unsupported_mask_leaves_travel_points_false():
    view = make_view([[0, 0, 0.45], [5, 0, 3.0], [9, 0, 3.0]], deposit=[True, False, True])
    mask = tv.unsupported_mask(view)
    assert not mask[1]
    assert mask[2]


def test_unsupported_mask_needs_bead_heights():
    view = make_view([[0, 0, 0], [1, 0, 0]])
    view.height = None
    with pytest.raises(ValueError, match="heights"):
        tv.unsupported_mask(view)


# --------------------------------------------------------------------------
# Shell / infill guess
# --------------------------------------------------------------------------


def _box_surface_samples(size=10.0, spacing=0.25):
    """Points covering the surface of a cube [0, size]^3."""
    grid = np.arange(0.0, size + 1e-9, spacing)
    u, v = np.meshgrid(grid, grid)
    u, v = u.ravel(), v.ravel()
    zero, full = np.zeros_like(u), np.full_like(u, size)
    faces = [
        np.column_stack([zero, u, v]), np.column_stack([full, u, v]),
        np.column_stack([u, zero, v]), np.column_stack([u, full, v]),
        np.column_stack([u, v, zero]), np.column_stack([u, v, full]),
    ]
    return np.vstack(faces)


def test_shell_is_within_two_deposition_widths_of_the_surface():
    """sdf_to_isdf keeps shell_thickness = 2 widths of solid around the infill."""
    width = 0.9
    points = [
        [0.45, 5, 5],   # outer perimeter
        [1.7, 5, 5],    # second perimeter, 1.7 mm in
        [2.5, 5, 5],    # past 1.8 mm: infill
        [5, 5, 5],      # centre: infill
    ]
    view = make_view(points, width=width)

    mask = tv.shell_mask(view, _box_surface_samples(), width)

    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_platform_and_travel_are_never_shell():
    view = make_view([[0.45, 5, 5], [0.45, 5, 6], [0.45, 5, 7]], deposit=[True, False, True],
                     platform_count=1)
    mask = tv.shell_mask(view, _box_surface_samples(), 0.9)
    np.testing.assert_array_equal(mask, [False, False, True])

    codes = tv.point_scalars(view, "shell", shell=mask)
    assert codes[0] == 2  # platform
    assert codes[2] == 1  # shell


def test_typical_width_ignores_platform_and_travel():
    view = make_view([[i, 0, 0] for i in range(4)], deposit=[True, False, True, True],
                     platform_count=1)
    view.width = np.array([2.0, 5.0, 0.9, 0.9])
    assert tv.typical_deposition_width(view) == pytest.approx(0.9)


# --------------------------------------------------------------------------
# Status line
# --------------------------------------------------------------------------


def test_status_line_reports_what_the_source_has():
    view = make_view([[0, 0, 0], [1, 2, 3]], tilts_deg=[0, 12], azimuths_deg=[0, 45])
    text = tv.describe_point(view, 1)
    assert "Point 2 of 2" in text
    assert "tilt 12.0 deg toward 45 deg" in text
    assert "width 0.90 mm" in text
    assert "G-code line" not in text

    view.gcode_line = np.array([30, 31])
    view.machine = np.array([[0, 0, 75, 75, 75], [1, 2, 80, 70, 76]], float)
    view.feed_mm_min = np.array([3000.0, 612.0])
    text = tv.describe_point(view, 1)
    assert "G-code line 31" in text
    assert "screws Z 80.00 U 70.00 V 76.00" in text
    assert "F 612 mm/min" in text


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------


def test_default_speed_plays_the_whole_print_in_a_minute():
    assert tv.default_speed(60_000) == pytest.approx(1000.0)
    low, high = tv.speed_limits(60_000)
    assert low <= tv.default_speed(60_000) <= high
    assert high == pytest.approx(60_000 / tv.FASTEST_PLAYBACK_S)


def test_tiny_toolpaths_still_get_a_usable_speed_range():
    low, high = tv.speed_limits(10)
    assert low < high
    assert low <= tv.default_speed(10) <= high


def test_playback_moves_by_speed_times_elapsed_time():
    position, finished = tv.playback_advance(100.0, speed=500.0, elapsed_s=0.1, count=10_000)
    assert position == pytest.approx(150.0)
    assert not finished


def test_playback_stops_on_the_last_point():
    position, finished = tv.playback_advance(9_990.0, speed=500.0, elapsed_s=1.0, count=10_000)
    assert position == 9_999
    assert finished


def test_playback_never_runs_backwards():
    position, _ = tv.playback_advance(10.0, speed=-5.0, elapsed_s=-1.0, count=100)
    assert position == 10.0


def test_speed_text_gives_the_time_for_the_whole_print():
    assert tv.describe_speed(1000.0, 60_000) == "Speed: 1,000 points/s (whole print in 60 s)"
    assert "10.0 min" in tv.describe_speed(100.0, 60_000)
