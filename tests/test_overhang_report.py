"""Tests for the overhang report tool (build plan task P0.8).

The important one is `test_measure_recovers_a_known_overhang_angle`: it builds
a ramp whose overhang angle is known by construction, prints it with a
synthetic toolpath, and checks the tool reports that angle back. If that works,
the numbers in the baseline table mean what they claim to.

No `from __future__ import annotations`: this drives Taichi through the tool.
"""

import json
import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_report
from atom import benchmark_meshes as bm


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------


def test_stage_times_are_parsed_from_a_log():
    log = (
        "# Direction Field Computation\n\n"
        "Grid cell 3D count: [119 119 125]\n"
        "Direction computation took 7.2 seconds\n\n"
        "# Order Atoms\n\n"
        "Atom count: 31630\n"
        "Toolpath planner took 331.6 seconds.\n"
    )
    times = overhang_report.parse_stage_times(log)

    assert times["Direction computation"] == pytest.approx(7.2)
    assert times["Toolpath planner"] == pytest.approx(331.6)


def test_a_log_with_no_timings_gives_an_empty_mapping():
    assert overhang_report.parse_stage_times("nothing to see here") == {}


# --------------------------------------------------------------------------
# Measuring a part whose answer is known
# --------------------------------------------------------------------------


class SyntheticToolpath:
    """A toolpath that deposits along a surface with a chosen build direction."""

    def __init__(self, points, direction):
        self.point = np.asarray(points, dtype=np.float32)
        count = len(self.point)

        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        self.tool_orientation = np.tile(
            [
                math.acos(np.clip(direction[2], -1.0, 1.0)),
                math.atan2(direction[1], direction[0]),
            ],
            (count, 1),
        ).astype(np.float32)

        self.travel_type = np.zeros(count, dtype=np.int32)
        self.width = np.full(count, 0.9, dtype=np.float32)
        self.height = np.full(count, 0.45, dtype=np.float32)
        self.point_count = count
        self.platform_height = 0.0

    def save(self, path):
        np.savez(
            path,
            point=self.point,
            travel_type=self.travel_type,
            tool_orientation=self.tool_orientation,
            width=self.width,
            height=self.height,
            point_count=np.array(self.point_count),
            platform_height=np.array(self.platform_height),
        )


def _points_over_overhang_faces(mesh, spacing=0.4):
    """Deposition points sitting just beneath each overhang face centre."""
    from atom import overhang_metrics as om

    overhangs = om.overhang_face_mask(mesh.face_normals, mesh.triangles_center)
    centres = mesh.triangles_center[overhangs]

    points = []
    for centre in centres:
        for offset in (-spacing, 0.0, spacing):
            points.append([centre[0], centre[1] + offset, centre[2]])
    return np.array(points)


@pytest.mark.parametrize("angle", [45, 60, 90])
def test_measure_recovers_a_known_overhang_angle(ti_cpu, tmp_path, angle):
    """A ramp built at A degrees, printed vertically, must measure A degrees."""
    mesh = bm.make_ramp(angle, **bm.default_dimensions(60.0))
    stl_path = tmp_path / f"ramp{angle}.stl"
    mesh.export(stl_path)

    toolpath_path = tmp_path / f"ramp{angle}_smoothed.npz"
    SyntheticToolpath(
        _points_over_overhang_faces(mesh), direction=[0.0, 0.0, 1.0]
    ).save(toolpath_path)

    report = overhang_report.measure(
        f"ramp{angle}",
        max_slope_deg=7.0,
        deposition_width=0.9,
        toolpath_path=toolpath_path,
        stl_path=stl_path,
    )

    surfaces = [s for s in report["metrics"]["surfaces"] if s["sample_count"] > 0]
    assert surfaces, "no overhang surface was measured"
    assert surfaces[0]["geometric_angle_deg"] == pytest.approx(angle, abs=0.1)
    assert surfaces[0]["max_effective_deg"] == pytest.approx(angle, abs=0.5)
    assert report["metrics"]["max_tool_tilt_deg"] == pytest.approx(0.0, abs=1e-3)


def test_tilting_toward_the_overhang_lowers_the_reported_angle(ti_cpu, tmp_path):
    """The measurement must see the benefit the 5-axis work is meant to deliver."""
    mesh = bm.make_ramp(60, **bm.default_dimensions(60.0))
    stl_path = tmp_path / "ramp60.stl"
    mesh.export(stl_path)

    points = _points_over_overhang_faces(mesh)
    # The ramp's overhang faces outward along +x, so lean the tool that way.
    tilted = np.array([math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0))])

    toolpath_path = tmp_path / "ramp60_smoothed.npz"
    SyntheticToolpath(points, direction=tilted).save(toolpath_path)

    report = overhang_report.measure(
        "ramp60", 30.0, 0.9, toolpath_path=toolpath_path, stl_path=stl_path
    )

    surfaces = [s for s in report["metrics"]["surfaces"] if s["sample_count"] > 0]
    assert surfaces[0]["max_effective_deg"] == pytest.approx(40.0, abs=0.5)
    assert report["metrics"]["max_tool_tilt_deg"] == pytest.approx(20.0, abs=1e-2)


def test_a_box_reports_no_overhang_surfaces(ti_cpu, tmp_path):
    mesh = bm.make_box(**bm.default_dimensions(60.0))
    stl_path = tmp_path / "box.stl"
    mesh.export(stl_path)

    points = np.array([[10.0, 10.0, z] for z in np.arange(0.45, 20.0, 0.45)])
    toolpath_path = tmp_path / "box_smoothed.npz"
    SyntheticToolpath(points, [0.0, 0.0, 1.0]).save(toolpath_path)

    report = overhang_report.measure(
        "box", 7.0, 0.9, toolpath_path=toolpath_path, stl_path=stl_path
    )

    assert report["metrics"]["surfaces"] == []
    assert math.isnan(report["verdict"]["worst_effective_deg"])
    assert report["verdict"]["printable"] is False, (
        "a part with nothing measured must not be reported as a pass"
    )


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------


def _fake_report(part, slope, worst, near_fraction, tilt, printable):
    return {
        "schema_version": overhang_report.SCHEMA_VERSION,
        "part": part,
        "max_slope_deg": slope,
        "metrics": {
            "max_tool_tilt_deg": tilt,
            "unsupported_fraction_near_overhangs": near_fraction,
            "surfaces": [],
        },
        "verdict": {"printable": printable, "worst_effective_deg": worst},
    }


def test_summary_lays_parts_against_slopes():
    reports = [
        _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True),
        _fake_report("ramp45", 30.0, 38.0, 0.001, 22.0, True),
        _fake_report("ramp60", 7.0, 58.0, 0.05, 5.6, False),
        _fake_report("ramp60", 30.0, 46.0, 0.02, 24.0, False),
    ]
    summary = overhang_report.summarize(reports)

    assert "max_slope 7°" in summary and "max_slope 30°" in summary
    assert "`ramp45`" in summary and "`ramp60`" in summary
    assert "✅ 42° / 0.2% / 5.5°" in summary
    assert "❌ 58° / 5.0% / 5.6°" in summary


def test_summary_states_the_printable_overhang_limit():
    reports = [
        _fake_report("ramp45", 7.0, 40.0, 0.001, 5.5, True),
        _fake_report("ramp50", 7.0, 44.0, 0.004, 5.5, True),
        _fake_report("ramp60", 7.0, 58.0, 0.06, 5.6, False),
    ]
    summary = overhang_report.summarize(reports)

    assert "up to **50°**" in summary
    assert "shallowest ramp it failed was 60°" in summary


def test_summary_is_honest_when_nothing_printed():
    reports = [_fake_report("ramp45", 7.0, 70.0, 0.4, 5.5, False)]
    summary = overhang_report.summarize(reports)

    assert "did not print any ramp" in summary


def test_summary_marks_unmeasured_cells_rather_than_scoring_them():
    reports = [_fake_report("box", 7.0, float("nan"), float("nan"), 5.5, False)]
    summary = overhang_report.summarize(reports)

    assert "n/m" in summary
    assert "No ramp parts have been measured" in summary


def test_a_missing_combination_shows_as_a_gap():
    reports = [
        _fake_report("ramp45", 7.0, 40.0, 0.001, 5.5, True),
        _fake_report("ramp60", 30.0, 46.0, 0.02, 24.0, False),
    ]
    summary = overhang_report.summarize(reports)

    assert "—" in summary, "an unrun combination must be visibly absent"


def test_summary_round_trips_through_the_report_schema(ti_cpu, tmp_path):
    """A report written to disk can be read back and summarised."""
    path = tmp_path / "ramp45_ms7.json"
    path.write_text(
        json.dumps(_fake_report("ramp45", 7.0, 40.0, 0.001, 5.5, True)),
        encoding="utf-8",
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert "`ramp45`" in overhang_report.summarize([loaded])
