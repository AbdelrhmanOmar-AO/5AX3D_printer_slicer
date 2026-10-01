"""`tools/overhang_report.py --overhang-aware` (build plan P2.2).

The report tool runs the overhang-aware field through the parameter file, and
keeps its results apart from stock ones: their own file names, their own
sections of the summary, and out of the stock progress log. Without the
option nothing changes. The field itself is tested in
`tests/test_orientation_field.py`; the atomize commands in
`tests/test_atomize_stages.py`.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_report as orep
from atom import benchmark_meshes as bm
from atom import overhang_metrics as om


def _report(part, slope, aware, mode=orep.MODE_FULL, worst=60.0):
    report = {
        "schema_version": orep.SCHEMA_VERSION,
        "metrics_version": orep.METRICS_VERSION,
        "part": part,
        "max_slope_deg": slope,
        "overhang_aware": aware,
        "runtime": {"total_s": 0.0, "stages": {}},
        "metrics": {
            "max_tool_tilt_deg": 5.0,
            "unsupported_fraction_near_overhangs": None if mode == orep.MODE_FIELD_ONLY else 0.05,
            "surfaces": [],
        },
        "verdict": {"printable": False, "assessable": True, "worst_effective_deg": worst},
        "provenance": None,
        "mesh": {"volume_mm3": 1.0},
        "toolpath": {"point_count": 10},
    }
    if mode == orep.MODE_FIELD_ONLY:
        report["mode"] = mode
    return report


# --------------------------------------------------------------------------
# Running the pipeline
# --------------------------------------------------------------------------


@pytest.fixture
def seen_params(tmp_path, monkeypatch):
    """Run `run_pipeline` against a stub, and return the parameter files it wrote."""
    param = tmp_path / "part.json"
    param.write_text(json.dumps({"solid_name": "part", "max_slope": 7}), encoding="utf-8")
    seen = []

    def fake_run(command, cwd, env):
        with open(command[2], encoding="utf-8") as handle:
            seen.append(json.load(handle))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(orep.subprocess, "run", fake_run)
    return param, seen


def test_an_overhang_aware_run_sets_the_key_and_the_profiles_budget(seen_params, monkeypatch):
    """Operator, 2026-09-30: the budget is the machine profile's, not the part file's 7."""
    param, seen = seen_params
    monkeypatch.setenv("ATOM_MACHINE", "dev60")
    orep.run_pipeline(param, None, verify_stages=False, overhang_aware=True)
    orep.run_pipeline(param, 25.0, verify_stages=False, overhang_aware=True)
    assert seen[0] == {"solid_name": "part", "max_slope": 60.0, "overhang_aware": True}
    assert seen[1]["max_slope"] == 25.0


def test_a_stock_run_leaves_the_part_file_as_it_is(seen_params):
    param, seen = seen_params
    orep.run_pipeline(param, None, verify_stages=False)
    assert seen == [{"solid_name": "part", "max_slope": 7}]


@pytest.mark.parametrize("machine,slope", [("reference", 30.0), ("dev60", 60.0)])
def test_the_budget_follows_atom_machine(monkeypatch, machine, slope):
    monkeypatch.setenv("ATOM_MACHINE", machine)
    assert orep.overhang_aware_slope() == slope


# --------------------------------------------------------------------------
# Kept apart from stock runs
# --------------------------------------------------------------------------


def test_overhang_aware_files_are_named_apart():
    for path_of in (orep.report_path, orep.field_only_report_path):
        assert path_of("ramp60_xs", 30.0).name == "ramp60_xs_ms30.json"
        assert path_of("ramp60_xs", 30.0, True).name == "ramp60_xs_ms30_aware.json"
    for path_of in (orep.archive_path, orep.frame_archive_path):
        assert path_of("ramp60_xs", 30.0).name == "ramp60_xs_ms30.npz"
        assert path_of("ramp60_xs", 30.0, True).name == "ramp60_xs_ms30_aware.npz"


def test_loading_picks_stock_aware_or_both(tmp_path, monkeypatch):
    monkeypatch.setattr(orep, "REPORT_DIR", tmp_path)
    for report in (_report("a", 7.0, False), _report("b", 7.0, True), _report("c", 7.0, None)):
        (tmp_path / f"{report['part']}.json").write_text(json.dumps(report), encoding="utf-8")
    old = _report("d", 7.0, False)
    del old["overhang_aware"]  # a report from before the key meant anything
    (tmp_path / "d.json").write_text(json.dumps(old), encoding="utf-8")

    assert sorted(r["part"] for r in orep.load_reports()) == ["a", "c", "d"]
    assert [r["part"] for r in orep.load_reports(overhang_aware=True)] == ["b"]
    assert len(orep.load_reports(overhang_aware=None)) == 4


def _provenance(machine="labpc", metrics_version=None):
    """A complete provenance block, as `atom.provenance.is_known` wants it."""
    return {
        "source": "pipeline", "machine": machine, "os": "Windows-10", "python": "3.10.21",
        "taichi": "1.7.4", "ti_arch_setting": "stock mix", "stage_arches": {"order_atoms": "x64"},
        "machine_profile": "reference", "git_commit": "abc123", "code_modified": False,
        "recorded_utc": "2026-10-01T00:00:00Z",
        "metrics_version": orep.METRICS_VERSION if metrics_version is None else metrics_version,
        "scored": {},
    }


def _paired(worst_stock=89.0, worst_aware=43.0, aware_machine="labpc", aware_version=None):
    stock = _report("ramp60_s", 30.0, False, worst=worst_stock)
    stock["provenance"] = _provenance()
    aware = _report("ramp60_s", 30.0, True, worst=worst_aware)
    aware["provenance"] = _provenance(aware_machine)
    aware["verdict"]["printable"] = worst_aware <= 45
    if aware_version is not None:
        aware["metrics_version"] = aware_version
    return stock, aware


def test_overhang_aware_runs_sit_beside_the_stock_run():
    """Build plan P2.5: stock -> overhang-aware in one cell, after the conclusion."""
    stock, aware = _paired()
    alone = orep.summarize([stock])
    both = orep.summarize([stock], (), [aware])

    assert orep.summarize([stock], (), ()) == alone
    before, section = both.split("## Stock vs overhang-aware (build plan P2.5)")
    assert before.rstrip() == alone.split("\n## Runtime")[0].rstrip()
    assert "| `ramp60_s` | ❌ 89° / 5.0% → ✅ 43° / 5.0% |" in section
    assert "share one provenance group" in section and "Comparable." in section
    assert "Printable, over the 1 part and slope pair(s) with both runs: stock 0, overhang-aware 1." in section
    assert "## Runtime" in section


def test_a_pair_from_two_machines_is_flagged():
    stock, aware = _paired(aware_machine="laptop")
    section = orep.summarize([stock], (), [aware]).split("## Stock vs overhang-aware")[1]
    assert "Not comparable as they stand: 2 provenance group(s)" in section
    assert "stock: machine labpc" in section and "overhang-aware: machine laptop" in section


def test_a_pair_scored_with_two_metrics_versions_is_flagged():
    """Tomorrow's case until the lab re-scores: stock at version 2, overhang-aware at 3."""
    stock, aware = _paired()
    stock["metrics_version"] = orep.METRICS_VERSION - 1
    section = orep.summarize([stock], (), [aware]).split("## Stock vs overhang-aware")[1]
    assert "Scored with different metrics versions" in section


def test_an_overhang_aware_run_with_no_stock_partner_shows_a_dash():
    stock, aware = _paired()
    lonely = dict(aware, part="tshape_s")
    section = orep.summarize([stock], (), [aware, lonely]).split("## Stock vs overhang-aware")[1]
    assert "| `tshape_s` | — → ✅ 43° / 5.0% |" in section
    assert "over the 1 part and slope pair(s)" in section


def _with_top(report, fraction):
    report["metrics"]["top_surface"] = {"fraction_on_target": fraction}
    return report


def test_top_surface_quality_sits_beside_the_stock_run():
    """Build plan P2.5: overhang-aware >= stock - 5 points, flagged when missed."""
    stock, aware = _paired()
    section = orep.summarize([_with_top(stock, 0.95)], (), [_with_top(aware, 0.91)]).split("### Top-surface quality")[1]
    assert "| `ramp60_s` | 95% → 91% |" in section
    assert "Pairs within 5 points of stock or better: 1 of 1." in section

    stock, aware = _paired()
    section = orep.summarize([_with_top(stock, 0.95)], (), [_with_top(aware, 0.80)]).split("### Top-surface quality")[1]
    assert "| `ramp60_s` | 95% → 80% ⚠️ |" in section
    assert "Pairs within 5 points of stock or better: 0 of 1." in section


def test_reports_without_the_top_surface_measure_say_not_scored():
    stock, aware = _paired()
    section = orep.summarize([stock], (), [_with_top(aware, float("nan"))]).split("### Top-surface quality")[1]
    assert "| `ramp60_s` | not scored → n/m |" in section
    assert "1 pair(s) not scored yet." in section


def test_tilt_rate_sits_beside_the_stock_run():
    stock, aware = _paired()
    stock["metrics"]["tilt_rate"] = {"max_deg_per_mm": 4.2, "fraction_over_limit": 0.01}
    aware["metrics"]["tilt_rate"] = {"max_deg_per_mm": 50.6, "fraction_over_limit": 0.264}
    section = orep.summarize([stock], (), [aware]).split("### Tilt rate")[1]
    assert "| `ramp60_s` | 4°/mm (1%) → 51°/mm (26%) |" in section


def test_field_only_runs_split_stock_from_aware():
    field = [
        _report("ramp60_xs", 30.0, False, orep.MODE_FIELD_ONLY),
        _report("ramp60_xs", 30.0, True, orep.MODE_FIELD_ONLY, worst=44.0),
    ]
    section = orep.summarize([_report("ramp60_xs", 30.0, False)], field).split(
        "## Field-only runs (build plan P2.0)"
    )[1]
    stock_part, aware_part = section.split("### Overhang-aware (build plan P2.2)")
    assert "### Stock" in stock_part
    assert "60°" in stock_part and "44°" not in stock_part
    assert "44°" in aware_part


def test_only_aware_field_only_runs_say_so():
    field = [_report("ramp60_xs", 30.0, True, orep.MODE_FIELD_ONLY)]
    section = orep.summarize([_report("ramp60_xs", 30.0, False)], field).split(
        "## Field-only runs (build plan P2.0)"
    )[1]
    assert "All overhang-aware (build plan P2.2)." in section
    assert "### Stock" not in section


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A repository with one ramp's mesh, atoms and toolpath already in data/.

    ``ATOM_MACHINE`` is dev60 here, for the budget. `atom.kinematics3z` and
    `atom.toolpath3` read the profile once, at import, so they are imported
    first: under dev60 they would keep a 60-degree nozzle for every later test.
    """
    import atom.kinematics3z  # noqa: F401
    import atom.toolpath3  # noqa: F401

    root = tmp_path / "repo"
    mesh = bm.make_ramp(60, **bm.default_dimensions(60.0))
    for sub in ("mesh", "frame", "toolpath"):
        (root / "data" / sub).mkdir(parents=True)
    mesh.export(root / "data" / "mesh" / "ramp60_t.stl")

    overhangs = om.overhang_face_mask(mesh.face_normals, mesh.triangles_center)
    points = mesh.triangles_center[overhangs].astype(np.float32)
    up = np.tile([0.0, 0.0], (len(points), 1)).astype(np.float32)
    np.savez(root / "data" / "frame" / "ramp60_t.npz", point=points, normal=up,
             phi_t=np.zeros(len(points), dtype=np.float32))
    np.savez(
        root / "data" / "toolpath" / "ramp60_t_smoothed.npz",
        point=points,
        travel_type=np.zeros(len(points), dtype=np.int32),
        tool_orientation=up,
        width=np.full(len(points), 0.9, dtype=np.float32),
        height=np.full(len(points), 0.45, dtype=np.float32),
        point_count=np.array(len(points)),
        platform_height=np.array(0.0),
    )
    param = tmp_path / "ramp60_t.json"
    param.write_text(json.dumps({"solid_name": "ramp60_t", "deposition_width": 0.9, "max_slope": 7}), encoding="utf-8")

    monkeypatch.setattr(orep, "REPO_ROOT", root)
    monkeypatch.setattr(orep, "REPORT_DIR", root / "reports" / "baseline_overhang")
    monkeypatch.setattr(orep, "FIELD_ONLY_DIR", root / "reports" / "field_only")
    monkeypatch.setattr(orep, "PROGRESS_LOG", root / "reports" / "matrix_progress.csv")
    monkeypatch.setattr(orep, "init_taichi", lambda arch: None)
    monkeypatch.setenv("ATOM_MACHINE", "dev60")
    return root, param


@pytest.mark.parametrize(
    "extra,folder",
    [([], "baseline_overhang"), (["--field-only"], "field_only")],
)
def test_the_command_line_writes_an_aware_report_apart(tiny_repo, capsys, extra, folder):
    """Named ``_aware``, at the profile's budget; a full run is logged as overhang-aware (P2.5)."""
    import csv

    root, param = tiny_repo
    assert orep.main([str(param), "--skip-pipeline", "--overhang-aware", *extra]) == 0

    written = json.loads((root / "reports" / folder / "ramp60_t_ms60_aware.json").read_text(encoding="utf-8"))
    assert written["overhang_aware"] is True
    assert written["max_slope_deg"] == 60.0
    assert not (root / "reports" / folder / "ramp60_t_ms60.json").exists()
    log = root / "reports" / "matrix_progress.csv"
    if "--field-only" in extra:
        assert not log.exists()  # the log has no column for field-only runs
    else:
        with log.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [(r["part"], r["overhang_aware"]) for r in rows] == [("ramp60_t", "True")]
    assert "ramp60_t at max_slope 60°, overhang-aware" in capsys.readouterr().out


def test_the_toolpath_is_archived_before_it_is_scored(tiny_repo, monkeypatch):
    """A scoring failure must not lose the run: the archive is written first."""
    root, param = tiny_repo
    monkeypatch.setattr(orep, "TOOLPATH_ARCHIVE", root / "reports" / "toolpaths")
    monkeypatch.setattr(orep, "run_pipeline", lambda *a, **k: (1.0, json.loads(param.read_text()), {}))

    def broken(*args, **kwargs):
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(orep, "measure", broken)
    with pytest.raises(RuntimeError, match="scoring failed"):
        orep.main([str(param), "--overhang-aware"])
    assert (root / "reports" / "toolpaths" / "ramp60_t_ms60_aware.npz").is_file()
    assert not (root / "reports" / "baseline_overhang" / "ramp60_t_ms60_aware.json").exists()


def test_without_the_option_the_command_line_is_stock(tiny_repo):
    root, param = tiny_repo
    assert orep.main([str(param), "--skip-pipeline"]) == 0
    written = json.loads((root / "reports" / "baseline_overhang" / "ramp60_t_ms7.json").read_text(encoding="utf-8"))
    assert written["overhang_aware"] is False
    assert (root / "reports" / "matrix_progress.csv").is_file()


# --------------------------------------------------------------------------
# Re-scoring stays on the machine that measured (plan_corrections P2-14)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field_only", [False, True])
def test_reanalyse_skips_reports_from_another_machine(tmp_path, monkeypatch, field_only):
    """Archives are found by part and slope only; another machine's report must not meet this one's files."""
    import platform

    monkeypatch.setattr(orep, "TOOLPATH_ARCHIVE", tmp_path / "toolpaths")
    monkeypatch.setattr(orep, "FRAME_ARCHIVE", tmp_path / "frames")
    monkeypatch.setattr(orep, "init_taichi", lambda arch: None)
    theirs = _report("ramp60_xs", 30.0, False, orep.MODE_FIELD_ONLY if field_only else orep.MODE_FULL)
    theirs["provenance"] = {"machine": "some-other-machine"}
    ours = dict(theirs, provenance={"machine": platform.node()})

    rescore = orep.reanalyse_field_only if field_only else orep.reanalyse
    rewritten, skipped = rescore([theirs, ours])

    assert rewritten == []
    assert "measured on some-other-machine; re-score it there" in skipped[0]
    assert "no archived" in skipped[1]  # this machine's report is looked up as before
    assert orep.measured_elsewhere({"provenance": None}) is None
