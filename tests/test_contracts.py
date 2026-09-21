"""Tests for the MachineToolpath contract (build plan task P0.6).

The final test is the one that matters: solving the committed golden toolpath
and checking the machine axis values against the G-code the pipeline actually
wrote. Everything else is synthetic.

No `from __future__ import annotations` here; `atom.contracts` compiles a
Taichi kernel and this module drives it.
"""

import json
import math

import numpy as np
import pytest

from atom import contracts, machine_profile


class FakeToolpath:
    """The attributes `from_toolpath` reads."""

    def __init__(self, points, tilts_deg=None, azimuths_deg=None):
        self.point = np.asarray(points, dtype=np.float32)
        count = len(self.point)

        tilts = np.zeros(count) if tilts_deg is None else np.asarray(tilts_deg, float)
        azimuths = (
            np.zeros(count) if azimuths_deg is None else np.asarray(azimuths_deg, float)
        )
        # tool_orientation is (N, 2) spherical [theta, phi]; theta is the tilt.
        self.tool_orientation = np.column_stack(
            [np.radians(tilts), np.radians(azimuths)]
        ).astype(np.float32)

        self.travel_type = np.zeros(count, dtype=np.int32)
        self.width = np.full(count, 0.9, dtype=np.float32)
        self.height = np.full(count, 0.45, dtype=np.float32)
        self.point_count = count
        self.platform_height = 0.0


@pytest.fixture(scope="module")
def reference_profile():
    return machine_profile.load_profile("reference")


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------


def test_zero_tilt_puts_all_three_screws_at_the_same_height(ti_cpu, reference_profile):
    """With the bed flat, z0 = z1 = z2 = point.z + z_offset."""
    toolpath = FakeToolpath([[150.0, 145.0, 10.0]])
    result = contracts.from_toolpath(toolpath)

    assert result.valid.all()
    screws = result.machine[0, contracts.AXIS_Z0 : contracts.AXIS_Z2 + 1]
    np.testing.assert_allclose(screws, screws[0], atol=1e-4)
    assert screws[0] == pytest.approx(10.0 + reference_profile.z_offset, abs=1e-3)


def test_zero_tilt_leaves_x_and_y_untouched(ti_cpu):
    toolpath = FakeToolpath([[150.0, 145.0, 10.0]])
    result = contracts.from_toolpath(toolpath)

    assert result.machine[0, contracts.AXIS_X] == pytest.approx(150.0, abs=1e-3)
    assert result.machine[0, contracts.AXIS_Y] == pytest.approx(145.0, abs=1e-3)


def test_tilting_separates_the_screws(ti_cpu):
    toolpath = FakeToolpath([[150.0, 145.0, 10.0]], tilts_deg=[10.0])
    result = contracts.from_toolpath(toolpath)

    assert result.valid.all()
    screws = result.machine[0, contracts.AXIS_Z0 : contracts.AXIS_Z2 + 1]
    assert np.ptp(screws) > 1.0, "a 10 degree tilt must move the screws apart"
    assert result.tilt_deg[0] == pytest.approx(10.0, abs=1e-3)


def test_tilt_beyond_the_machine_limit_is_marked_invalid(ti_cpu, reference_profile):
    """`kinematics3z.inverse` signals failure with NaN, and we must not abort."""
    over = reference_profile.max_tilt_angle_deg + 5.0
    toolpath = FakeToolpath([[150.0, 145.0, 10.0]], tilts_deg=[over])

    result = contracts.from_toolpath(toolpath)

    assert not result.valid[0]
    assert result.invalid_count == 1
    assert list(result.invalid_indices) == [0]


def test_one_bad_point_does_not_stop_the_others(ti_cpu, reference_profile):
    """The whole reason this exists: toolpath_to_gcode aborts, we mark."""
    over = reference_profile.max_tilt_angle_deg + 5.0
    toolpath = FakeToolpath(
        [[150.0, 145.0, 10.0]] * 3, tilts_deg=[0.0, over, 5.0]
    )

    result = contracts.from_toolpath(toolpath)

    assert list(result.valid) == [True, False, True]
    assert result.point_count == 3
    assert result.max_tilt_deg == pytest.approx(5.0, abs=1e-3)


def test_a_point_beyond_the_travel_limits_is_invalid(ti_cpu, reference_profile):
    beyond = reference_profile.max_x_axis + 100.0
    result = contracts.from_toolpath(FakeToolpath([[beyond, 145.0, 10.0]]))
    assert not result.valid[0]


def test_a_point_below_the_bed_is_invalid(ti_cpu):
    result = contracts.from_toolpath(FakeToolpath([[150.0, 145.0, -1.0]]))
    assert not result.valid[0]


def test_max_tilt_ignores_unreachable_points(ti_cpu, reference_profile):
    over = reference_profile.max_tilt_angle_deg + 10.0
    result = contracts.from_toolpath(
        FakeToolpath([[150.0, 145.0, 10.0]] * 2, tilts_deg=[8.0, over])
    )
    assert result.max_tilt_deg == pytest.approx(8.0, abs=1e-3)


def test_an_empty_toolpath_solves_to_nothing(ti_cpu):
    result = contracts.from_toolpath(FakeToolpath(np.zeros((0, 3))))
    assert result.point_count == 0
    assert result.invalid_count == 0
    assert result.max_tilt_deg == 0.0


def test_a_mismatched_profile_is_refused(ti_cpu):
    """Machine constants are baked in at import; a different one cannot apply."""
    import warnings

    from atom.machine_profile import PlaceholderProfileWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderProfileWarning)
        ours = machine_profile.load_profile("ours")

    with pytest.raises(ValueError, match="baked into Taichi closures"):
        contracts.from_toolpath(FakeToolpath([[150.0, 145.0, 10.0]]), profile=ours)


# --------------------------------------------------------------------------
# Bed centring
# --------------------------------------------------------------------------


def test_bed_centering_offset_centres_the_part(reference_profile):
    points = np.array([[0.0, 0.0, 0.0], [20.0, 30.0, 10.0]])
    offset = contracts.bed_centering_offset(points, reference_profile)

    centred = points + offset
    assert centred[:, 0].mean() == pytest.approx(reference_profile.max_x_axis / 2.0)
    assert centred[:, 1].mean() == pytest.approx(reference_profile.max_y_axis / 2.0)
    assert offset[2] == 0.0, "centring must not move anything vertically"


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_npz_round_trip(ti_cpu, tmp_path):
    original = contracts.from_toolpath(
        FakeToolpath(
            [[150.0, 145.0, 5.0], [151.0, 145.0, 5.5], [152.0, 146.0, 6.0]],
            tilts_deg=[0.0, 4.0, 9.0],
            azimuths_deg=[0.0, 90.0, 180.0],
        )
    )
    path = tmp_path / "nested" / "machine_toolpath.npz"
    original.save(path)
    loaded = contracts.MachineToolpath.load(path)

    np.testing.assert_array_equal(loaded.point, original.point)
    np.testing.assert_array_equal(loaded.tool_orientation, original.tool_orientation)
    np.testing.assert_allclose(loaded.machine, original.machine)
    np.testing.assert_allclose(loaded.tilt_deg, original.tilt_deg)
    np.testing.assert_array_equal(loaded.valid, original.valid)
    assert loaded.point_count == original.point_count
    assert loaded.profile_name == original.profile_name


def test_a_stale_schema_version_is_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, schema_version=np.array(contracts.SCHEMA_VERSION + 1))

    with pytest.raises(ValueError, match="schema version"):
        contracts.MachineToolpath.load(path)


# --------------------------------------------------------------------------
# Against the real golden toolpath
# --------------------------------------------------------------------------


@pytest.mark.pipeline
def test_golden_toolpath_solves_with_no_invalid_points(ti_cpu, repo_root):
    """Every point of a real, shipped toolpath must be reachable.

    Marked `pipeline` only because it solves ~47 000 points; it needs no GPU
    and no Blender.
    """
    from atom import toolpath3

    golden = repo_root / "tests" / "golden" / "calibration_cube.toolpath.npz"
    if not golden.is_file():
        pytest.skip("golden toolpath not captured; see tests/golden/README.md")

    toolpath = toolpath3.Toolpath()
    toolpath.load(str(golden))
    result = contracts.from_toolpath(toolpath, center_on_bed=True)

    assert result.invalid_count == 0, (
        f"{result.invalid_count} of {result.point_count} points are unreachable, "
        f"first at index {result.invalid_indices[:5]}"
    )
    # The part was sliced with max_slope 7.0; the field must stay inside it.
    assert result.max_tilt_deg <= 7.0 + 1e-3


@pytest.mark.pipeline
def test_golden_machine_axes_match_the_written_gcode(ti_cpu, repo_root):
    """Solving the golden toolpath reproduces the G-code's own axis values.

    This is the check that the contract really models what the printer is
    commanded to do, rather than merely being self-consistent.

    Two details of the comparison, both from `tools/toolpath_to_gcode.py`:

    * It re-centres the toolpath on the bed before solving, so `center_on_bed`
      must be set or nothing lines up. The difference is not a constant: bed
      tilt pivots about the ball joints, so screw travel depends on where a
      point sits relative to them.
    * The G-code header contains purge lines at fixed coordinates
      (``X0.1 Y20``, ``Y200.0``, all three screws at ``0.3 + Z_OFFSET``) which
      are not part of the toolpath. They can only *extend* an axis range, so
      the maxima must match exactly while the G-code minima may be lower.
    """
    from atom import toolpath3

    golden_dir = repo_root / "tests" / "golden"
    toolpath_path = golden_dir / "calibration_cube.toolpath.npz"
    stats_path = golden_dir / "calibration_cube.stats.json"
    if not (toolpath_path.is_file() and stats_path.is_file()):
        pytest.skip("golden baseline not captured; see tests/golden/README.md")

    toolpath = toolpath3.Toolpath()
    toolpath.load(str(toolpath_path))
    result = contracts.from_toolpath(toolpath, center_on_bed=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for column, axis in (
        (contracts.AXIS_Z0, "Z"),
        (contracts.AXIS_Z1, "U"),
        (contracts.AXIS_Z2, "V"),
    ):
        ours = result.machine[result.valid, column]
        written = stats["axis_ranges"][axis]

        assert ours.max() == pytest.approx(written["max"], abs=1e-3), (
            f"axis {axis}: solved max {ours.max():.4f} against G-code "
            f"{written['max']:.4f}"
        )
        assert written["min"] <= ours.min() + 1e-3, (
            f"axis {axis}: G-code min {written['min']:.4f} is above the solved "
            f"min {ours.min():.4f}, which the purge lines cannot explain"
        )

    # X is bounded above by the part, so its maximum must agree exactly too.
    solved_x = result.machine[result.valid, contracts.AXIS_X]
    assert solved_x.max() == pytest.approx(stats["axis_ranges"]["X"]["max"], abs=1e-3)
