"""Tests for the analytic tilt bound (build plan P2.1, `tools/tilt_bound.py`)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import tilt_bound as tb


def test_the_plan_example_45_plus_30_is_75():
    assert tb.steepest_printable_overhang_deg(45, 30) == 75


def test_a_flat_ledge_needs_45_degrees():
    assert tb.required_tilt_deg(90, 45) == 45
    assert tb.steepest_printable_overhang_deg(45, 44.9) < 90
    assert tb.steepest_printable_overhang_deg(45, 45) == 90


def test_no_tilt_is_needed_within_the_threshold():
    assert tb.required_tilt_deg(40, 45) == 0
    assert tb.required_tilt_deg(45, 45) == 0


def test_the_bound_never_passes_90_degrees():
    assert tb.steepest_printable_overhang_deg(45, 50) == 90


def test_the_nozzle_caps_the_usable_tilt_at_50_degrees():
    assert tb.NOZZLE_TILT_CAP_DEG == 50
    assert tb.usable_tilt_deg(30) == 30
    assert tb.usable_tilt_deg(60) == 50
    assert tb.steepest_printable_overhang_deg(30, 60) == 80


def test_the_nozzle_cap_is_atomizers_own():
    """`fff3.MAX_SLOPE_ANGLE` caps the field at (180 - NOZZLE_CONE_ANGLE) / 2."""
    from atom import toolpath3

    assert tb.NOZZLE_TILT_CAP_DEG == pytest.approx(
        (180.0 - math.degrees(toolpath3.NOZZLE_CONE_ANGLE)) / 2.0
    )


def test_the_threshold_is_the_one_the_report_judges_by():
    import overhang_report

    assert tb.DEFAULT_MAX_OVERHANG_DEG == overhang_report.MAX_EFFECTIVE_OVERHANG_DEG


@pytest.mark.parametrize(
    "theta,tilt,expected",
    [
        (45, 30, tb.INSIDE),
        (70, 30, tb.INSIDE),  # 70 <= 45 + 30 - 2
        (74, 30, tb.EDGE),  # inside 75, not inside 73
        (75, 30, tb.EDGE),
        (80, 30, tb.OUT_OF_RANGE),
        (80, 40, tb.INSIDE),
        (90, 45, tb.EDGE),  # needs 47 with the margin
        (90, 50, tb.INSIDE),  # 45 + 50 - 2 = 93: the 90 cap applies after the margin
        (90, 60, tb.INSIDE),  # the nozzle caps 60 at 50
    ],
)
def test_parts_are_classified_as_p2_5_counts_them(theta, tilt, expected):
    assert tb.classify(theta, 45, tilt, margin_deg=2) == expected


def test_the_benchmark_parts_come_from_their_generator():
    parts = dict(tb.benchmark_overhangs())
    assert [name for name in parts if name.startswith("ramp")] == [
        "ramp45", "ramp50", "ramp60", "ramp70", "ramp80", "ramp90",
    ]
    assert parts["ramp60"] == 60
    assert parts["tshape"] == 90  # its default underside, gate D0
    assert parts["twin_domes"] is None
    assert dict(tb.benchmark_overhangs(70))["tshape"] == 70


def test_the_report_at_the_reference_machines_30_degrees(capsys, monkeypatch):
    monkeypatch.delenv("ATOM_MACHINE", raising=False)
    assert tb.main([]) == 0
    out = capsys.readouterr().out
    assert "machine profile 'reference': 30 deg" in out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ramp") or line.startswith("  tshape")}
    assert "75" in out
    assert rows["ramp70"].rstrip().endswith("inside")
    assert rows["ramp80"].rstrip().endswith("out of range")
    assert rows["tshape"].rstrip().endswith("out of range")


def test_several_tilts_give_a_column_each(capsys):
    assert tb.main(["--tilt", "30", "40", "--tshape-underside", "70"]) == 0
    out = capsys.readouterr().out
    ramp80 = next(line for line in out.splitlines() if line.startswith("  ramp80"))
    assert ramp80.split()[-4:] == ["out", "of", "range", "inside"]
    tshape = next(line for line in out.splitlines() if line.startswith("  tshape"))
    assert tshape.split()[1] == "70"


def test_an_impossible_angle_is_refused():
    with pytest.raises(SystemExit):
        tb.main(["--tilt", "120"])


def test_the_nozzle_cap_follows_the_profile(capsys, monkeypatch):
    import warnings

    from atom import machine_profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", machine_profile.PlaceholderProfileWarning)
        dev60 = machine_profile.load_profile("dev60")
    assert tb.nozzle_tilt_cap_deg(dev60) == 60
    assert tb.nozzle_tilt_cap_deg(machine_profile.load_profile("reference")) == 50
    assert tb.usable_tilt_deg(60, nozzle_cap_deg=60) == 60
    assert tb.classify(90, 45, 60, margin_deg=2, nozzle_cap_deg=60) == tb.INSIDE

    monkeypatch.setenv("ATOM_MACHINE", "dev60")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", machine_profile.PlaceholderProfileWarning)
        assert tb.main([]) == 0
    out = capsys.readouterr().out
    assert "60 deg cone, capping the field at 60 deg" in out
    tshape = next(line for line in out.splitlines() if line.startswith("  tshape"))
    assert tshape.rstrip().endswith("inside")
