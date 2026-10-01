"""Tests for the field-only evaluation mode (build plan P2.0).

A field-only run stops after atom extraction and measures the atoms in
``data/frame/<part>.npz`` instead of a toolpath, skipping `order_atoms`, 86 %
of a run. The unit tests pin the reader to the real writer, show that the same
points give the same numbers either way, and check that field-only results can
never be mistaken for full ones. The `pipeline` test at the end is the plan's
proof on real output: on `ramp60_xs` at a 30-degree budget, the field-only
effective overhang angles match the full run's within 1 degree.

No `from __future__ import annotations`: this drives Taichi through the tool.
"""

import json
import math
import time
import types

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_report as orep
from atom import benchmark_meshes as bm
from atom import frame_atoms
from atom import overhang_metrics as om


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _spherical(direction, count):
    """``(count, 2)`` float32 ``[theta, phi]`` for one Cartesian direction."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    theta = math.acos(np.clip(direction[2], -1.0, 1.0))
    phi = math.atan2(direction[1], direction[0])
    return np.tile([theta, phi], (count, 1)).astype(np.float32)


def _write_frame(path, points, direction):
    """A frame file in the layout `extract_explicit_atoms` writes."""
    points = np.asarray(points, dtype=np.float32)
    np.savez(
        path,
        point=points,
        normal=_spherical(direction, len(points)),
        phi_t=np.zeros(len(points), dtype=np.float32),
    )


def _write_toolpath(path, points, direction):
    """A `_smoothed` toolpath depositing at ``points``, for `measure`."""
    points = np.asarray(points, dtype=np.float32)
    count = len(points)
    np.savez(
        path,
        point=points,
        travel_type=np.zeros(count, dtype=np.int32),
        tool_orientation=_spherical(direction, count),
        width=np.full(count, 0.9, dtype=np.float32),
        height=np.full(count, 0.45, dtype=np.float32),
        point_count=np.array(count),
        platform_height=np.array(0.0),
    )


def _points_under_overhangs(mesh, spacing=0.4):
    """Points just beneath each overhang face centre."""
    overhangs = om.overhang_face_mask(mesh.face_normals, mesh.triangles_center)
    points = []
    for centre in mesh.triangles_center[overhangs]:
        for offset in (-spacing, 0.0, spacing):
            points.append([centre[0], centre[1] + offset, centre[2]])
    return np.array(points)


def _ramp(tmp_path, angle):
    mesh = bm.make_ramp(angle, **bm.default_dimensions(60.0))
    stl_path = tmp_path / f"ramp{angle}.stl"
    mesh.export(stl_path)
    return mesh, stl_path


def _tilted(degrees_toward_x):
    t = math.radians(degrees_toward_x)
    return [math.sin(t), 0.0, math.cos(t)]


# --------------------------------------------------------------------------
# The frame reader
# --------------------------------------------------------------------------


def test_the_reader_reads_what_the_extraction_stage_writes(ti_cpu, tmp_path):
    """Written by `frame3.Field.save_active_frame_set` itself, read back exactly."""
    ti = ti_cpu
    from atom import frame3

    shape = (2, 2, 2)
    points = np.arange(24, dtype=np.float32).reshape(*shape, 3)
    points[0, 0, 1] = np.nan  # atoms the extraction filtered out
    points[1, 1, 0] = np.nan
    normals = np.stack(
        [np.linspace(0.0, 0.5, 8), np.linspace(-3.0, 3.0, 8)], axis=1
    ).astype(np.float32).reshape(*shape, 2)
    phi_t = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(shape)

    field = frame3.Field()
    field.point = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    field.normal = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    field.phi_t = ti.field(dtype=ti.f32, shape=shape)
    field.point.from_numpy(points)
    field.normal.from_numpy(normals)
    field.phi_t.from_numpy(phi_t)
    path = tmp_path / "part.npz"
    field.save_active_frame_set(str(path))

    atoms = frame_atoms.load_frame_atoms(path)

    active = ~np.isnan(points.reshape(-1, 3)).any(axis=1)
    assert atoms.point_count == 6
    np.testing.assert_array_equal(atoms.point, points.reshape(-1, 3)[active])
    np.testing.assert_array_equal(atoms.tool_orientation, normals.reshape(-1, 2)[active])
    np.testing.assert_array_equal(atoms.phi_t, phi_t.ravel()[active])
    assert atoms.dropped_non_finite == 0


def test_every_atom_counts_as_deposition(tmp_path):
    path = tmp_path / "part.npz"
    _write_frame(path, [[0, 0, 1], [0, 0, 2], [0, 0, 3]], [0, 0, 1])
    atoms = frame_atoms.load_frame_atoms(path)
    assert atoms.travel_type.tolist() == [0, 0, 0]
    assert om.deposition_mask(atoms).all()


@pytest.mark.parametrize(
    "arrays,message",
    [
        ({"point": np.zeros((2, 3)), "normal": np.zeros((2, 2))}, "no phi_t"),
        ({"point": np.zeros((2, 3)), "normal": np.zeros((2, 3)), "phi_t": np.zeros(2)}, "normal has shape"),
        ({"point": np.zeros((2, 2)), "normal": np.zeros((2, 2)), "phi_t": np.zeros(2)}, "point has shape"),
        ({"point": np.zeros((2, 3)), "normal": np.zeros((2, 2)), "phi_t": np.zeros(3)}, "phi_t has shape"),
        ({"point": np.full((2, 3), np.nan), "normal": np.zeros((2, 2)), "phi_t": np.zeros(2)}, "no atom"),
    ],
)
def test_the_reader_refuses_what_is_not_a_frame_file(tmp_path, arrays, message):
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        frame_atoms.load_frame_atoms(path)


def test_an_atom_without_an_orientation_is_left_out_and_counted(tmp_path):
    path = tmp_path / "part.npz"
    normal = np.array([[0.1, 0.0], [np.nan, 0.0], [0.2, 1.0]], dtype=np.float32)
    np.savez(path, point=np.zeros((3, 3), np.float32), normal=normal, phi_t=np.zeros(3, np.float32))
    atoms = frame_atoms.load_frame_atoms(path)
    assert atoms.point_count == 2
    assert atoms.dropped_non_finite == 1


# --------------------------------------------------------------------------
# Measuring atoms
# --------------------------------------------------------------------------


def test_the_same_points_measure_the_same_either_way(tmp_path):
    """Atoms and a toolpath carrying the same points give identical metrics."""
    mesh, stl_path = _ramp(tmp_path, 60)
    points = _points_under_overhangs(mesh)
    direction = _tilted(20.0)
    _write_frame(tmp_path / "atoms.npz", points, direction)
    _write_toolpath(tmp_path / "toolpath.npz", points, direction)

    field = orep.measure_field_only(
        "ramp60", 30.0, 0.9, frame_path=tmp_path / "atoms.npz", stl_path=stl_path
    )
    full = orep.measure(
        "ramp60", 30.0, 0.9, toolpath_path=tmp_path / "toolpath.npz", stl_path=stl_path
    )

    assert field["metrics"]["surfaces"] == full["metrics"]["surfaces"]
    assert field["metrics"]["max_tool_tilt_deg"] == full["metrics"]["max_tool_tilt_deg"]
    assert field["verdict"]["worst_effective_deg"] == full["verdict"]["worst_effective_deg"]
    assert field["metrics"]["atoms_near_overhangs"] == full["metrics"]["deposition_points_near_overhangs"]
    # A 20-degree lean toward the overhang takes 60 degrees to 40.
    assert field["verdict"]["worst_effective_deg"] == pytest.approx(40.0, abs=0.5)


def test_a_field_only_report_says_what_it_cannot_measure(tmp_path):
    mesh, stl_path = _ramp(tmp_path, 60)
    _write_frame(tmp_path / "atoms.npz", _points_under_overhangs(mesh), [0, 0, 1])

    report = orep.measure_field_only(
        "ramp60", 7.0, 0.9, runtime_s=91.5, frame_path=tmp_path / "atoms.npz", stl_path=stl_path
    )

    assert report["mode"] == orep.MODE_FIELD_ONLY
    assert report["metrics"]["unsupported_fraction_overall"] is None
    assert report["metrics"]["unsupported_fraction_near_overhangs"] is None
    assert report["metrics"]["unsupported_note"].startswith("n/a (field-only)")
    assert "deposition_points_near_overhangs" not in report["metrics"]
    assert report["atoms"] == {"count": len(_points_under_overhangs(mesh)), "dropped_non_finite": 0}
    assert report["runtime"]["total_s"] == 91.5
    assert report["metrics_version"] == orep.METRICS_VERSION
    json.dumps(report, allow_nan=True)  # it must serialise as written


@pytest.mark.parametrize(
    "angle,tilt,printable,within",
    [
        (60, 0.0, False, False),  # 60 degrees: over the threshold, ruled out
        (60, 20.0, False, False),  # leaning 20 away makes it 80: ruled out
        (45, 0.0, None, True),  # 45: the angle passes; unsupported is unknown
    ],
)
def test_field_only_can_rule_a_part_out_but_never_in(tmp_path, angle, tilt, printable, within):
    mesh, stl_path = _ramp(tmp_path, angle)
    # The ramps overhang toward +x, so a negative x lean tilts away from it.
    _write_frame(tmp_path / "atoms.npz", _points_under_overhangs(mesh), _tilted(-tilt))

    verdict = orep.measure_field_only(
        f"ramp{angle}", 30.0, 0.9, frame_path=tmp_path / "atoms.npz", stl_path=stl_path
    )["verdict"]

    assert verdict["printable"] is printable
    assert verdict["effective_within_threshold"] is within


def test_a_part_with_no_overhang_is_not_assessed(tmp_path):
    mesh = bm.make_box(**bm.default_dimensions(60.0))
    stl_path = tmp_path / "box.stl"
    mesh.export(stl_path)
    _write_frame(tmp_path / "atoms.npz", [[10.0, 10.0, z] for z in np.arange(0.45, 20.0, 0.45)], [0, 0, 1])

    report = orep.measure_field_only("box", 7.0, 0.9, frame_path=tmp_path / "atoms.npz", stl_path=stl_path)

    assert report["verdict"]["assessable"] is False
    assert report["verdict"]["printable"] is None
    assert report["verdict"]["effective_within_threshold"] is None
    assert orep._format_field_only_cell(report).startswith("– no overhang")


def test_missing_atoms_stop_the_tool_with_a_message(tmp_path):
    _, stl_path = _ramp(tmp_path, 60)
    with pytest.raises(SystemExit, match="No atoms at"):
        orep.measure_field_only("ramp60", 7.0, 0.9, frame_path=tmp_path / "none.npz", stl_path=stl_path)


# --------------------------------------------------------------------------
# Never confused with full runs
# --------------------------------------------------------------------------


def _report(part, slope, mode, worst=60.0, printable=False, assessable=True):
    report = {
        "schema_version": orep.SCHEMA_VERSION,
        "metrics_version": orep.METRICS_VERSION,
        "part": part,
        "max_slope_deg": slope,
        "runtime": {"total_s": 0.0, "stages": {}},
        "metrics": {
            "max_tool_tilt_deg": 5.0,
            "unsupported_fraction_near_overhangs": None if mode == orep.MODE_FIELD_ONLY else 0.05,
            "surfaces": [],
        },
        "verdict": {"printable": printable, "assessable": assessable, "worst_effective_deg": worst},
        "provenance": None,
        "mesh": {"volume_mm3": 1.0},
        "toolpath": {"point_count": 10},
    }
    if mode == orep.MODE_FIELD_ONLY:
        report["mode"] = mode
    return report


def test_field_only_reports_get_their_own_table(tmp_path):
    full = [_report("ramp60_xs", 30.0, orep.MODE_FULL)]
    field = [
        _report("ramp60_xs", 30.0, orep.MODE_FIELD_ONLY),
        _report("ramp45_xs", 30.0, orep.MODE_FIELD_ONLY, worst=44.0, printable=None),
    ]

    alone = orep.summarize(full)
    both = orep.summarize(full, field)

    before, section = both.split("## Field-only runs (build plan P2.0)")
    assert "## Runtime" in section  # the section sits before the runtime table
    assert before.rstrip() == alone.split("\n## Runtime")[0].rstrip()
    assert "| `ramp60_xs` | ❌ 60° / n/a / 5.0° |" in section
    assert "| `ramp45_xs` | ❔ 44° / n/a / 5.0° |" in section
    assert "Not comparable with the table above" in section
    assert "ramp45_xs" not in before


def test_no_field_only_reports_leaves_the_summary_as_it_was():
    full = [_report("ramp60_xs", 30.0, orep.MODE_FULL)]
    assert orep.summarize(full, []) == orep.summarize(full)


def test_the_committed_summary_is_unchanged_by_p2_0(repo_root):
    """The committed reports still produce the committed summary exactly.

    Built as `--summarize` builds it: the stock runs, then (since P2.5) the
    overhang-aware ones beside them, and any field-only runs.
    """
    assert orep.summarize(
        orep.load_reports(quiet=True),
        orep.load_reports(quiet=True, field_only=True, overhang_aware=None),
        orep.load_reports(quiet=True, overhang_aware=True),
    ) == (repo_root / "reports" / "baseline_overhang.md").read_text(encoding="utf-8")


def test_a_report_in_the_wrong_folder_is_refused(tmp_path, monkeypatch, capsys):
    full_dir, field_dir = tmp_path / "full", tmp_path / "field"
    full_dir.mkdir()
    field_dir.mkdir()
    monkeypatch.setattr(orep, "REPORT_DIR", full_dir)
    monkeypatch.setattr(orep, "FIELD_ONLY_DIR", field_dir)
    for directory, report in (
        (full_dir, _report("a", 7.0, orep.MODE_FULL)),
        (full_dir, _report("b", 7.0, orep.MODE_FIELD_ONLY)),
        (field_dir, _report("c", 7.0, orep.MODE_FIELD_ONLY)),
        (field_dir, _report("d", 7.0, orep.MODE_FULL)),
    ):
        (directory / f"{report['part']}.json").write_text(json.dumps(report), encoding="utf-8")

    assert [r["part"] for r in orep.load_reports()] == ["a"]
    assert [r["part"] for r in orep.load_reports(field_only=True)] == ["c"]
    printed = capsys.readouterr().out
    assert "Skipping b.json: a field_only report" in printed
    assert "Skipping d.json: a full report" in printed


def test_field_only_paths_are_apart_from_the_full_runs():
    assert orep.FIELD_ONLY_DIR != orep.REPORT_DIR
    assert orep.FRAME_ARCHIVE != orep.TOOLPATH_ARCHIVE
    assert orep.field_only_report_path("ramp60_xs", 30.0).name == "ramp60_xs_ms30.json"
    assert orep.frame_archive_path("ramp60_xs", 30.0).name == "ramp60_xs_ms30.npz"


def test_archived_atoms_are_not_committed(repo_root):
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "reports/frames/ramp60_xs_ms30.npz"],
        cwd=repo_root,
    )
    assert result.returncode == 0, "reports/frames/ must be gitignored"


# --------------------------------------------------------------------------
# Running only the first stages
# --------------------------------------------------------------------------


def test_a_field_only_run_stops_after_atom_extraction(tmp_path, monkeypatch):
    param = tmp_path / "part.json"
    param.write_text(json.dumps({"solid_name": "part", "max_slope": 7.0}), encoding="utf-8")
    seen = []

    def fake_run(command, cwd, env):
        seen.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(orep.subprocess, "run", fake_run)
    orep.run_pipeline(param, 30.0, verify_stages=False, field_only=True)
    orep.run_pipeline(param, 30.0, verify_stages=False)

    assert seen[0][-2:] == ["--stop-after", "extract_explicit_atoms"]
    assert "--stop-after" not in seen[1]


def test_a_field_only_run_checks_only_the_stages_it_ran(tmp_path):
    started = time.time()
    for _, template in orep.FIELD_ONLY_ARTIFACTS:
        path = tmp_path / template.format(part="part")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert orep.FIELD_ONLY_ARTIFACTS[-1][0] == "8 explicit atoms"
    assert orep.first_failed_stage(
        "part", started, root=tmp_path, artifacts=orep.FIELD_ONLY_ARTIFACTS
    ) is None
    label, _, reason = orep.first_failed_stage("part", started, root=tmp_path)
    assert (label, reason) == ("9 order atoms", "produced nothing")


def test_the_command_line_writes_a_field_only_report_apart(tmp_path, monkeypatch, capsys):
    """`--field-only --skip-pipeline` measures data/frame and writes nothing else."""
    root = tmp_path / "repo"
    mesh = bm.make_ramp(60, **bm.default_dimensions(60.0))
    (root / "data" / "mesh").mkdir(parents=True)
    (root / "data" / "frame").mkdir(parents=True)
    mesh.export(root / "data" / "mesh" / "ramp60_t.stl")
    _write_frame(root / "data" / "frame" / "ramp60_t.npz", _points_under_overhangs(mesh), [0, 0, 1])
    param = tmp_path / "ramp60_t.json"
    param.write_text(json.dumps({"solid_name": "ramp60_t", "deposition_width": 0.9, "max_slope": 30.0}), encoding="utf-8")

    monkeypatch.setattr(orep, "REPO_ROOT", root)
    monkeypatch.setattr(orep, "REPORT_DIR", root / "reports" / "baseline_overhang")
    monkeypatch.setattr(orep, "FIELD_ONLY_DIR", root / "reports" / "field_only")
    monkeypatch.setattr(orep, "PROGRESS_LOG", root / "reports" / "matrix_progress.csv")
    monkeypatch.setattr(orep, "init_taichi", lambda arch: None)

    assert orep.main([str(param), "--field-only", "--skip-pipeline"]) == 0

    written = json.loads((root / "reports" / "field_only" / "ramp60_t_ms30.json").read_text(encoding="utf-8"))
    assert written["mode"] == orep.MODE_FIELD_ONLY
    assert written["verdict"]["worst_effective_deg"] == pytest.approx(60.0, abs=0.5)
    assert not (root / "reports" / "baseline_overhang").exists()
    assert not (root / "reports" / "matrix_progress.csv").exists()
    out = capsys.readouterr().out
    assert "unsupported near overhang: n/a (field-only)" in out
    assert "False (effective angle over the threshold)" in out


def test_reanalyse_rescores_archived_atoms(tmp_path, monkeypatch):
    mesh, stl_path = _ramp(tmp_path, 60)
    monkeypatch.setattr(orep, "FIELD_ONLY_DIR", tmp_path / "field_only")
    monkeypatch.setattr(orep, "FRAME_ARCHIVE", tmp_path / "frames")
    monkeypatch.setattr(orep, "REPO_ROOT", tmp_path)
    (tmp_path / "field_only").mkdir()
    (tmp_path / "frames").mkdir()
    (tmp_path / "data" / "mesh").mkdir(parents=True)
    mesh.export(tmp_path / "data" / "mesh" / "ramp60_t.stl")
    _write_frame(orep.frame_archive_path("ramp60_t", 30.0), _points_under_overhangs(mesh), _tilted(10.0))

    stale = _report("ramp60_t", 30.0, orep.MODE_FIELD_ONLY, worst=99.0)
    stale["deposition_width_mm"] = 0.9
    missing = _report("ramp70_t", 30.0, orep.MODE_FIELD_ONLY)

    rewritten, skipped = orep.reanalyse_field_only([stale, missing])

    assert [r["part"] for r in rewritten] == ["ramp60_t"]
    assert rewritten[0]["verdict"]["worst_effective_deg"] == pytest.approx(50.0, abs=0.5)
    assert skipped == ["ramp70_t @ 30 field-only (no archived atoms)"]
    on_disk = json.loads(orep.field_only_report_path("ramp60_t", 30.0).read_text(encoding="utf-8"))
    assert on_disk["verdict"]["worst_effective_deg"] == rewritten[0]["verdict"]["worst_effective_deg"]


# --------------------------------------------------------------------------
# The plan's proof on real output (laptop)
# --------------------------------------------------------------------------

#: Build plan P2.0: "on ramp60_xs, field-only theta_eff matches the full-run
#: theta_eff within 1 degree". At a 30-degree budget, where stock Atomizer tilts
#: `ramp60` by ~29 degrees, so the check is not trivially met by an untilted field.
PIPELINE_PART = "ramp60_xs"
PIPELINE_SLOPE = 30.0
AGREEMENT_DEG = 1.0


def _atoms(path):
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


@pytest.mark.pipeline
def test_field_only_measures_what_the_full_run_measures(repo_root, capsys):
    """Field-only, full, then field-only again, on one part and budget.

    The first field-only run can pay Taichi's kernel compilation: on its first
    laptop run (2026-09-30) the direction and layer stages took 46 and 45 s
    against 10 and 12 s one run later, and the timing check failed on that
    alone. So only the **third** run is timed against the full one; by then
    every kernel it uses has been compiled twice. About 20 minutes on the
    laptop. The report is printed whatever happens.
    """
    param = repo_root / "data" / "param" / f"{PIPELINE_PART}.json"
    frame_path = repo_root / "data" / "frame" / f"{PIPELINE_PART}.npz"
    toolpath_path = repo_root / "data" / "toolpath" / f"{PIPELINE_PART}.npz"
    width = json.loads(param.read_text(encoding="utf-8"))["deposition_width"]

    first_s, _, _ = orep.run_pipeline(param, PIPELINE_SLOPE, field_only=True)
    first_atoms = _atoms(frame_path)

    full_s, _, _ = orep.run_pipeline(param, PIPELINE_SLOPE)
    full = orep.measure(PIPELINE_PART, PIPELINE_SLOPE, width, full_s)
    full_atoms = _atoms(frame_path)
    with np.load(repo_root / "data" / "toolpath" / f"{PIPELINE_PART}_smoothed.npz") as data:
        count = int(data["point_count"])
        deposits = data["travel_type"][:count] == om.TRAVEL_TYPE_DEPOSITION
        deposited_orientations = np.array(data["tool_orientation"][:count][deposits])
    toolpath_written = toolpath_path.stat().st_mtime

    field_s, _, _ = orep.run_pipeline(param, PIPELINE_SLOPE, field_only=True)
    field = orep.measure_field_only(PIPELINE_PART, PIPELINE_SLOPE, width, field_s)
    field_atoms = _atoms(frame_path)

    field_surfaces = [s for s in field["metrics"]["surfaces"] if s["sample_count"]]
    full_surfaces = [s for s in full["metrics"]["surfaces"] if s["sample_count"]]
    field_worst = field["verdict"]["worst_effective_deg"]
    full_worst = full["verdict"]["worst_effective_deg"]
    with capsys.disabled():
        print(
            f"\n\nP2.0 check, {PIPELINE_PART} at max_slope {PIPELINE_SLOPE:g}:\n"
            f"  field-only run, first : {first_s:7.1f} s (may include kernel compilation)\n"
            f"  full run              : {full_s:7.1f} s, {int(np.count_nonzero(deposits))} deposition points\n"
            f"  field-only run, warm  : {field_s:7.1f} s, {field['atoms']['count']} atoms\n"
            f"  time ratio (warm/full): {field_s / full_s:.3f}\n"
            f"  worst effective overhang: field-only {field_worst:.3f}, full {full_worst:.3f}\n"
            f"  max tilt used           : field-only {field['metrics']['max_tool_tilt_deg']:.3f}, "
            f"full {full['metrics']['max_tool_tilt_deg']:.3f}\n"
        )

    # A field-only run stops before ordering: it leaves the toolpath alone.
    assert toolpath_path.stat().st_mtime == toolpath_written

    # The field-only stages are the same computation in all three runs.
    for atoms in (full_atoms, field_atoms):
        assert set(atoms) == set(first_atoms)
        for key in first_atoms:
            np.testing.assert_array_equal(atoms[key], first_atoms[key], err_msg=key)

    # Every orientation the full run deposits with is one of the atoms' own:
    # ordering copies it and smoothing moves positions only (atom.frame_atoms).
    atom_orientations = {tuple(row) for row in field_atoms["normal"].tolist()}
    strays = [row for row in deposited_orientations.tolist() if tuple(row) not in atom_orientations]
    assert not strays, f"{len(strays)} deposited orientations are not an atom's"
    assert field["metrics"]["max_tool_tilt_deg"] >= full["metrics"]["max_tool_tilt_deg"] - 1e-9

    # The plan's criterion.
    assert abs(field_worst - full_worst) <= AGREEMENT_DEG
    assert [s["geometric_angle_deg"] for s in field_surfaces] == [
        s["geometric_angle_deg"] for s in full_surfaces
    ]
    for f, g in zip(field_surfaces, full_surfaces):
        assert abs(f["max_effective_deg"] - g["max_effective_deg"]) <= AGREEMENT_DEG
        assert abs(f["mean_effective_deg"] - g["mean_effective_deg"]) <= AGREEMENT_DEG

    # "Done when: ... finishes in a small fraction of the full-run time."
    assert field_s < 0.5 * full_s
