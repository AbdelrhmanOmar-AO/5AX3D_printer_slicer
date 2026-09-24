"""Tests for the overhang report tool (build plan task P0.8).

The important one is `test_measure_recovers_a_known_overhang_angle`: it builds
a ramp whose overhang angle is known by construction, prints it with a
synthetic toolpath, and checks the tool reports that angle back. If that works,
the numbers in the baseline table mean what they claim to.

No `from __future__ import annotations`: this drives Taichi through the tool.
"""

import csv
import json
import math
import os
import time

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


def _fake_report(part, slope, worst, near_fraction, tilt, printable, machine=None):
    report = {
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
    if machine is not None:
        report["provenance"] = {"pipeline": machine, "scored": machine}
    return report


#: Two machines for the provenance tests, matching the real pair in
#: `docs/handoff.md` section 3.
LAPTOP_MACHINE = {
    "recorded": "captured",
    "host": "laptop",
    "cpu": "AMD Ryzen 5 5600H with Radeon Graphics",
    "cpu_logical": 12,
    "taichi": "1.7.4",
    "blender": "Blender 5.2.1 LTS",
    "ti_arch": None,
    "machine_profile": "reference",
}
LAB_MACHINE = dict(
    LAPTOP_MACHINE,
    host="labpc",
    cpu="Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz",
    cpu_logical=72,
)


# --------------------------------------------------------------------------
# Naming the stage that actually failed
# --------------------------------------------------------------------------


def _stage_tree(root, part, through=None, mtime=None):
    """Create the artifacts of every stage up to and including `through`."""
    made = []
    for label, template in overhang_report.STAGE_ARTIFACTS:
        path = root / template.format(part=part)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        made.append(label)
        if through is not None and label == through:
            break
    return made


def test_a_complete_run_reports_no_failed_stage(tmp_path):
    started = time.time()
    _stage_tree(tmp_path, "ramp45_xs", mtime=started + 1)
    assert overhang_report.first_failed_stage("ramp45_xs", started, tmp_path) is None


def test_the_first_missing_stage_is_named(tmp_path):
    """The real failure: stage 6 died and twelve more failed behind it.

    `tools/compute_tangents.py` loads `data/image/0.png` as its default tangent
    field, that file was missing from the worker trees, and because `atomize.py`
    ignores stage exit codes the log was a wall of downstream complaints around
    one cause.
    """
    started = time.time()
    _stage_tree(tmp_path, "ramp45_xs", through="5 implicit layers",
                mtime=started + 1)

    failure = overhang_report.first_failed_stage("ramp45_xs", started, tmp_path)

    assert failure is not None
    label, path, reason = failure
    assert label == "6 tangents"
    assert "basis" in str(path)
    assert "produced nothing" in reason


def test_the_earliest_failure_is_named_not_the_loudest(tmp_path):
    """A later stage's missing output is a symptom; the first gap is the cause."""
    started = time.time()
    _stage_tree(tmp_path, "ramp45_xs", through="2 point-normal cloud",
                mtime=started + 1)
    # Something much later also exists, as it would from a previous run.
    late = tmp_path / "data" / "gcode" / "ramp45_xs.gcode"
    late.parent.mkdir(parents=True, exist_ok=True)
    late.write_bytes(b"x")

    label, _, _ = overhang_report.first_failed_stage("ramp45_xs", started, tmp_path)
    assert label == "3 SDF"


def test_a_stale_artifact_counts_as_a_failure(tmp_path):
    """The dangerous case: the file exists, so nothing looks wrong, but it is
    the *previous* run's. Without this the report would be quietly wrong."""
    started = time.time()
    _stage_tree(tmp_path, "ramp45_xs", mtime=started + 1)

    stale = tmp_path / "data" / "basis" / "ramp45_xs.npz"
    old = started - 3600
    os.utime(stale, (old, old))

    failure = overhang_report.first_failed_stage("ramp45_xs", started, tmp_path)
    assert failure is not None
    label, _, reason = failure
    assert label == "6 tangents"
    assert "earlier run" in reason


def test_freshness_allows_for_timestamp_granularity(tmp_path):
    """A file written a moment before the recorded start is not stale."""
    started = time.time()
    _stage_tree(tmp_path, "ramp45_xs",
                mtime=started - overhang_report.FRESHNESS_TOLERANCE_S / 2)
    assert overhang_report.first_failed_stage("ramp45_xs", started, tmp_path) is None


def test_the_stage_list_covers_the_whole_pipeline():
    """13 stages, ending at the G-code, so no stage can fail unnoticed."""
    labels = [label for label, _ in overhang_report.STAGE_ARTIFACTS]
    assert len(labels) == 13
    assert labels[0].startswith("1 ")
    assert "G-code" in labels[-1]
    assert any("tangents" in label for label in labels)


# --------------------------------------------------------------------------
# Provenance: which machine produced these numbers
# --------------------------------------------------------------------------


def test_a_fresh_run_records_this_machine(ti_cpu, tmp_path):
    """The pipeline record must describe the machine that made the toolpath."""
    from atom import provenance as prov

    mesh = bm.make_ramp(45.0, **bm.default_dimensions(60.0))
    stl = tmp_path / "ramp45.stl"
    mesh.export(stl)
    npz = tmp_path / "ramp45_smoothed.npz"
    SyntheticToolpath(
        _points_over_overhang_faces(mesh), direction=[0.0, 0.0, 1.0]
    ).save(npz)

    mine = prov.fingerprint()
    report = overhang_report.measure(
        "ramp45", 7.0, 0.9,
        toolpath_path=npz, stl_path=stl, pipeline_provenance=mine,
    )

    assert report["provenance"]["pipeline"]["cpu"] == mine["cpu"]
    assert report["provenance"]["scored"]["cpu"] == mine["cpu"]
    assert report["schema_version"] == 2


def test_rescoring_does_not_move_where_the_toolpath_was_computed(ti_cpu, tmp_path):
    """The guard that matters.

    `--reanalyse` re-scores archived toolpaths. Run on the lab machine, it must
    not relabel 48 laptop-measured runs as lab-measured — that is exactly the
    corruption provenance exists to prevent (corrections 3.9).
    """
    mesh = bm.make_ramp(45.0, **bm.default_dimensions(60.0))
    stl = tmp_path / "ramp45.stl"
    mesh.export(stl)
    npz = tmp_path / "ramp45_smoothed.npz"
    SyntheticToolpath(
        _points_over_overhang_faces(mesh), direction=[0.0, 0.0, 1.0]
    ).save(npz)

    report = overhang_report.measure(
        "ramp45", 7.0, 0.9,
        toolpath_path=npz, stl_path=stl,
        pipeline_provenance=LAPTOP_MACHINE,
    )

    assert report["provenance"]["pipeline"]["cpu"] == LAPTOP_MACHINE["cpu"]
    assert report["provenance"]["pipeline"] is not report["provenance"]["scored"]
    # The scoring record is this machine, whatever it is — but not the laptop's,
    # unless the test happens to run on one.
    assert report["provenance"]["scored"]["recorded"] == "captured"


def test_the_summary_names_the_machine():
    reports = [
        _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True, machine=LAPTOP_MACHINE),
        _fake_report("ramp60", 7.0, 58.0, 0.05, 5.6, False, machine=LAPTOP_MACHINE),
    ]
    summary = overhang_report.summarize(reports)
    assert "Measured on" in summary
    assert "5600H" in summary
    assert "more than one machine" not in summary


def test_the_summary_shouts_when_two_machines_are_mixed():
    """A table mixing machines credits a hardware difference to the result."""
    reports = [
        _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True, machine=LAPTOP_MACHINE),
        _fake_report("ramp60", 7.0, 58.0, 0.05, 5.6, False, machine=LAB_MACHINE),
    ]
    summary = overhang_report.summarize(reports)

    assert "more than one machine" in summary
    assert "5600H" in summary and "6254" in summary
    assert "3.9" in summary, "the warning should cite the correction that explains why"


def test_a_forced_backend_counts_as_a_different_machine():
    """3.9 is about backends, not only hardware: ATOM_TI_ARCH changes the toolpath."""
    reports = [
        _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True, machine=LAPTOP_MACHINE),
        _fake_report("ramp60", 7.0, 58.0, 0.05, 5.6, False,
                     machine=dict(LAPTOP_MACHINE, ti_arch="cpu")),
    ]
    assert "more than one machine" in overhang_report.summarize(reports)


def test_the_summary_says_so_when_no_machine_was_recorded():
    """Reports written before provenance existed must not look verified."""
    reports = [
        _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True),
        _fake_report("ramp60", 7.0, 58.0, 0.05, 5.6, False),
    ]
    summary = overhang_report.summarize(reports)
    assert "Machine not recorded" in summary
    assert "more than one machine" not in summary, (
        "several unknowns are one unknown, not several machines"
    )


def test_version_1_reports_are_still_readable(tmp_path, monkeypatch):
    """The 48 committed baseline reports are schema 1. Refusing them would have
    thrown the baseline away."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)
    old = _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True)
    old["schema_version"] = 1
    old["metrics_version"] = overhang_report.METRICS_VERSION
    (tmp_path / "ramp45_ms7.json").write_text(json.dumps(old), encoding="utf-8")

    assert len(overhang_report.load_reports(quiet=True)) == 1


def test_an_unknown_schema_is_skipped_rather_than_guessed_at(tmp_path, monkeypatch):
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)
    future = _fake_report("ramp45", 7.0, 42.0, 0.002, 5.5, True)
    future["schema_version"] = 99
    (tmp_path / "ramp45_ms7.json").write_text(json.dumps(future), encoding="utf-8")

    assert overhang_report.load_reports(quiet=True) == []


# --------------------------------------------------------------------------
# Progress log
# --------------------------------------------------------------------------


def test_the_progress_log_gains_a_machine_column(tmp_path):
    """An older log must be migrated, not left ragged.

    The log is the crash trail from correction 4.7, so it has to stay readable
    by anything that opens it — appending a tenth field to nine-field rows would
    break that quietly.
    """
    log = tmp_path / "matrix_progress.csv"
    old_header = [
        "finished_utc", "part", "max_slope_deg", "metrics_version",
        "worst_effective_deg", "unsupported_near_overhangs",
        "max_tilt_used_deg", "printable", "runtime_s",
    ]
    with log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(old_header)
        writer.writerow(["2026-09-23T12:00:00+00:00", "ramp45_xs", "7", "2",
                         "45.00", "0.0024", "0.61", "True", "341"])

    assert overhang_report.migrate_progress_header(log) is True

    with log.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert tuple(rows[0]) == overhang_report.PROGRESS_COLUMNS
    assert len(rows[1]) == len(overhang_report.PROGRESS_COLUMNS)
    assert rows[1][-1] == "", "an old row has no machine, and must not gain a made-up one"
    assert rows[1][1] == "ramp45_xs", "the existing data must survive the migration"


def test_migrating_an_already_current_log_does_nothing(tmp_path):
    log = tmp_path / "matrix_progress.csv"
    with log.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(list(overhang_report.PROGRESS_COLUMNS))
    before = log.read_bytes()

    assert overhang_report.migrate_progress_header(log) is False
    assert log.read_bytes() == before


def test_migrating_a_missing_log_is_not_an_error(tmp_path):
    assert overhang_report.migrate_progress_header(tmp_path / "nope.csv") is False


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


# --------------------------------------------------------------------------
# Runtime / scaling section
# --------------------------------------------------------------------------


def _timed_report(part, slope, total_s, ordering_s, volume, points):
    report = _fake_report(part, slope, 40.0, 0.001, 5.5, True)
    report["runtime"] = {
        "total_s": total_s,
        "stages": {"Toolpath planner": ordering_s, "Direction computation": 7.2},
    }
    report["mesh"] = {"volume_mm3": volume, "face_count": 12, "extents_mm": [1, 2, 3]}
    report["toolpath"] = {"point_count": points, "deposition_count": points}
    return report


def test_runtime_table_reports_each_run():
    summary = overhang_report.summarize(
        [_timed_report("ramp60_s", 7.0, 1500.0, 1200.0, 29757.0, 112051)]
    )

    assert "## Runtime" in summary
    assert "`ramp60_s`" in summary
    assert "29,757" in summary          # volume
    assert "112,051" in summary         # toolpath points
    assert "20.0" in summary            # ordering minutes


def test_runtime_table_says_when_it_is_a_single_point():
    summary = overhang_report.summarize(
        [_timed_report("ramp60_s", 7.0, 1500.0, 1200.0, 29757.0, 112051)]
    )
    assert "single point rather than a scaling curve" in summary


def test_runtime_table_becomes_a_curve_with_more_than_one_size():
    summary = overhang_report.summarize(
        [
            _timed_report("ramp60_xs", 7.0, 300.0, 240.0, 6428.0, 24203),
            _timed_report("ramp60_s", 7.0, 1500.0, 1200.0, 29757.0, 112051),
        ]
    )
    assert "single point rather than a scaling curve" not in summary

    # Check the order inside the Runtime section only: the parts table above it
    # is sorted alphabetically, which puts `ramp60_s` before `ramp60_xs`.
    runtime = summary[summary.index("## Runtime") :]
    assert runtime.index("`ramp60_xs`") < runtime.index("`ramp60_s`"), (
        "the runtime table should read smallest size first, so the trend is "
        "visible down the column"
    )


def test_runtime_section_is_honest_when_nothing_was_timed():
    summary = overhang_report.summarize(
        [_fake_report("ramp45", 7.0, 40.0, 0.001, 5.5, True)]
    )
    assert "No run recorded a duration" in summary


@pytest.mark.parametrize(
    "part,expected",
    [("ramp60_xs", "xs"), ("twin_domes_l", "l"), ("tshape_m", "m"),
     ("calibration_cube", None)],
)
def test_size_suffix_is_recognised(part, expected):
    assert overhang_report._size_of(part) == expected


def test_archived_toolpaths_are_not_committed(repo_root):
    """A full matrix archives ~220 MB of toolpaths; they must stay local.

    They exist so `--reanalyse` can re-score without re-slicing, which is a
    local concern. The JSON reports are the committed artifact.
    """
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "reports/toolpaths/example_ms7.npz"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "reports/toolpaths/ is not gitignored; a matrix run would try to commit "
        "hundreds of megabytes of regenerable data"
    )


def test_archive_and_report_paths_agree_on_naming():
    """`--reanalyse` finds an archive by rebuilding its name from the report."""
    assert (
        overhang_report.archive_path("ramp60_s", 7.0).stem
        == overhang_report.report_path("ramp60_s", 7.0).stem
    )


# --------------------------------------------------------------------------
# Face subdivision before sampling
#
# Both metrics search near each face's centroid, so one large flat face is
# represented by a single point. A ramp's overhang is two triangles covering
# 172 mm^2; the first baseline matrix measured it from 108 deposition points.
# --------------------------------------------------------------------------


def test_subdivision_preserves_normals_and_area():
    """Splitting a triangle in its own plane must change neither."""
    mesh = bm.make_ramp(45, **bm.default_dimensions(30.0))
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/ramp45.stl"
        mesh.export(path)

        original, _, _, coarse_areas = overhang_report.load_mesh_arrays(path)
        _, fine_normals, fine_centres, fine_areas = overhang_report.load_mesh_arrays(
            path, max_edge=0.9
        )

    assert len(fine_areas) > 10 * len(coarse_areas), "the mesh was not subdivided"
    assert fine_areas.sum() == pytest.approx(coarse_areas.sum(), rel=1e-9)

    from atom import overhang_metrics as om

    fine_overhangs = om.overhang_face_mask(fine_normals, fine_centres)
    # The overhang still measures 45 degrees, and still covers the same area.
    angles = om.geometric_overhang_angle_deg(fine_normals[fine_overhangs])
    assert np.allclose(angles, 45.0, atol=0.1)
    assert fine_areas[fine_overhangs].sum() == pytest.approx(171.8, rel=1e-2)


def test_subdivision_widens_the_sampled_region(tmp_path):
    """The point of the fix: far more of the surface is actually measured."""
    from atom import overhang_metrics as om

    mesh = bm.make_ramp(45, **bm.default_dimensions(30.0))
    path = tmp_path / "ramp45.stl"
    mesh.export(path)

    low, high = mesh.bounds
    cloud = np.mgrid[
        low[0] : high[0] : 0.45, low[1] : high[1] : 0.45, 0.45 : high[2] : 0.45
    ].reshape(3, -1).T

    _, coarse_n, coarse_c, _ = overhang_report.load_mesh_arrays(path)
    _, fine_n, fine_c, _ = overhang_report.load_mesh_arrays(path, max_edge=0.9)

    coarse = om.points_near_overhangs(cloud, coarse_n, coarse_c, radius=1.8).sum()
    fine = om.points_near_overhangs(cloud, fine_n, fine_c, radius=1.8).sum()

    assert fine > 5 * coarse, (
        f"subdivision sampled {fine} points against {coarse}; the surface is "
        "still being represented by too few centroids"
    )


def test_measure_records_how_many_faces_it_sampled(ti_cpu, tmp_path):
    """The report must say how densely the surface was measured."""
    mesh = bm.make_ramp(60, **bm.default_dimensions(60.0))
    stl_path = tmp_path / "ramp60.stl"
    mesh.export(stl_path)

    toolpath_path = tmp_path / "ramp60_smoothed.npz"
    SyntheticToolpath(
        _points_over_overhang_faces(mesh), direction=[0.0, 0.0, 1.0]
    ).save(toolpath_path)

    report = overhang_report.measure(
        "ramp60", 7.0, 0.9, toolpath_path=toolpath_path, stl_path=stl_path
    )

    assert report["mesh"]["sampled_face_count"] > report["mesh"]["face_count"]


# --------------------------------------------------------------------------
# Crash recovery
#
# A matrix run overwrites the previous run's report files, so after an
# interrupted re-run the combinations not yet reached still hold results from
# the old metric definition. Without a metrics version they look complete.
# --------------------------------------------------------------------------


def test_reports_record_which_metric_definition_produced_them(ti_cpu, tmp_path):
    mesh = bm.make_ramp(60, **bm.default_dimensions(60.0))
    stl_path = tmp_path / "ramp60.stl"
    mesh.export(stl_path)
    toolpath_path = tmp_path / "ramp60_smoothed.npz"
    SyntheticToolpath(
        _points_over_overhang_faces(mesh), direction=[0.0, 0.0, 1.0]
    ).save(toolpath_path)

    report = overhang_report.measure(
        "ramp60", 7.0, 0.9, toolpath_path=toolpath_path, stl_path=stl_path
    )
    assert report["metrics_version"] == overhang_report.METRICS_VERSION


def test_status_counts_a_stale_report_as_missing(tmp_path, monkeypatch):
    """The trap: an old report for a not-yet-redone combination looks done."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)

    current = _fake_report("ramp45_xs", 7.0, 40.0, 0.002, 5.5, True)
    current["metrics_version"] = overhang_report.METRICS_VERSION
    (tmp_path / "ramp45_xs_ms7.json").write_text(json.dumps(current))

    old = _fake_report("ramp50_xs", 7.0, 50.0, 0.30, 5.5, False)
    old["metrics_version"] = overhang_report.METRICS_VERSION - 1
    (tmp_path / "ramp50_xs_ms7.json").write_text(json.dumps(old))

    present, missing, stale = overhang_report.matrix_status(
        ["xs"], parts=("ramp45", "ramp50"), slopes=(7.0,)
    )

    assert ("ramp45_xs", 7.0) in present
    assert ("ramp50_xs", 7.0) in missing, "a stale report must not count as done"
    assert ("ramp50_xs", 7.0) in stale


def test_a_report_with_no_version_counts_as_stale(tmp_path, monkeypatch):
    """Reports from before versioning existed must be re-run, not trusted."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)
    (tmp_path / "ramp45_xs_ms7.json").write_text(
        json.dumps(_fake_report("ramp45_xs", 7.0, 40.0, 0.002, 5.5, True))
    )

    _, missing, stale = overhang_report.matrix_status(
        ["xs"], parts=("ramp45",), slopes=(7.0,)
    )
    assert ("ramp45_xs", 7.0) in missing
    assert ("ramp45_xs", 7.0) in stale


def test_a_truncated_report_is_reported_not_fatal(tmp_path, monkeypatch, capsys):
    """A power cut mid-write must not take the whole summary down."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)

    good = _fake_report("ramp45_xs", 7.0, 40.0, 0.002, 5.5, True)
    good["metrics_version"] = overhang_report.METRICS_VERSION
    (tmp_path / "ramp45_xs_ms7.json").write_text(json.dumps(good))
    (tmp_path / "ramp50_xs_ms7.json").write_text('{"part": "ramp50_xs", "met')

    reports = overhang_report.load_reports()

    assert len(reports) == 1
    assert "CORRUPT" in capsys.readouterr().out


def test_progress_log_appends_a_row_per_run(tmp_path, monkeypatch):
    """An append-only trail that survives a crash."""
    log = tmp_path / "matrix_progress.csv"
    monkeypatch.setattr(overhang_report, "PROGRESS_LOG", log)

    for part in ("ramp45_xs", "ramp50_xs"):
        report = _fake_report(part, 7.0, 40.0, 0.002, 5.5, True)
        report["metrics_version"] = overhang_report.METRICS_VERSION
        report["runtime"] = {"total_s": 420.0, "stages": {}}
        overhang_report.append_progress(report)

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "a header and one row per run"
    assert lines[0].startswith("finished_utc,part,max_slope_deg,metrics_version")
    assert "ramp45_xs" in lines[1]
    assert "ramp50_xs" in lines[2]


@pytest.mark.parametrize(
    "version,expected_exit",
    [(overhang_report.METRICS_VERSION, 0), (overhang_report.METRICS_VERSION - 1, 1)],
)
def test_check_done_exit_code(tmp_path, monkeypatch, version, expected_exit):
    """-Resume relies on the exit code, so it must be right."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)
    report = _fake_report("ramp45_s", 30.0, 40.0, 0.002, 5.5, True)
    report["metrics_version"] = version
    (tmp_path / "ramp45_s_ms30.json").write_text(json.dumps(report))

    assert overhang_report.main(["--check-done", "ramp45_s", "30"]) == expected_exit


def test_check_done_reports_missing_and_corrupt_as_not_done(tmp_path, monkeypatch):
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)

    assert overhang_report.main(["--check-done", "nothing_here", "7"]) == 1

    (tmp_path / "broken_s_ms7.json").write_text('{"metrics_ver')
    assert overhang_report.main(["--check-done", "broken_s", "7"]) == 1


def test_check_done_prints_nothing(tmp_path, monkeypatch, capsys):
    """Output would be noise in the matrix log; the exit code is the answer."""
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path)
    overhang_report.main(["--check-done", "absent", "7"])
    assert capsys.readouterr().out == ""


def test_a_part_with_no_overhang_is_not_reported_as_a_failure(ti_cpu, tmp_path):
    """twin_domes checks that smooth surfaces are not made worse.

    It has no overhang, so "not printable" misreads what it is for.
    """
    mesh = bm.make_box(**bm.default_dimensions(60.0))
    stl_path = tmp_path / "box.stl"
    mesh.export(stl_path)
    toolpath_path = tmp_path / "box_smoothed.npz"
    SyntheticToolpath(
        [[10.0, 10.0, z] for z in np.arange(0.45, 20.0, 0.45)], [0.0, 0.0, 1.0]
    ).save(toolpath_path)

    report = overhang_report.measure(
        "box", 7.0, 0.9, toolpath_path=toolpath_path, stl_path=stl_path
    )

    assert report["verdict"]["assessable"] is False
    assert report["verdict"]["printable"] is False

    cell = overhang_report._format_cell(report)
    assert "no overhang" in cell
    assert "❌" not in cell, "nothing to assess is not a failure"
    assert "✅" not in cell, "nor is it a pass"
