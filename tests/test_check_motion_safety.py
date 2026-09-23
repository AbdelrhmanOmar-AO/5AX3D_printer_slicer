"""Tests for tools/check_motion_safety.py (build plan P4).

The checks themselves are tested in their own modules; these test the tool
around them: inputs, exit status and the JSON report.
"""

# No `from __future__ import annotations`: the swept tests drive Taichi kernels.

import json

import numpy as np
import pytest

import check_motion_safety as cms
from test_nozzle_material_check import stem_and_arm


def write_toolpath(path, points, directions, deposit):
    count = len(points)
    spherical = np.column_stack([
        np.arccos(np.clip(directions[:, 2], -1, 1)),
        np.arctan2(directions[:, 1], directions[:, 0]),
    ])
    np.savez(
        path,
        point=points.astype(np.float32),
        travel_type=np.where(deposit, 0, 1).astype(np.int32),
        tool_orientation=spherical.astype(np.float32),
        width=np.full(count, 0.9, np.float32),
        height=np.full(count, 0.45, np.float32),
        point_count=np.array(count),
    )
    return path


@pytest.fixture
def toolpaths(tmp_path):
    folder = tmp_path / "toolpaths"
    folder.mkdir()
    *tall, _ = stem_and_arm(stem_top=10.0)
    *short, _ = stem_and_arm(stem_top=4.5)
    return {
        "collides": write_toolpath(folder / "a_collides.npz", *tall),
        "clear": write_toolpath(folder / "b_clear.npz", *short),
        "folder": folder,
    }


def test_a_clear_toolpath_exits_zero(toolpaths, capsys):
    assert cms.main([str(toolpaths["clear"]), "--checks", "nozzle"]) == 0
    assert "clear" in capsys.readouterr().out


def test_a_collision_exits_one_and_is_reported(toolpaths, tmp_path, capsys):
    report_path = tmp_path / "out" / "report.json"
    status = cms.main([str(toolpaths["collides"]), "--checks", "nozzle",
                       "--json", str(report_path), "--limit", "2"])
    assert status == 1
    assert "COLLISION" in capsys.readouterr().out

    report = json.loads(report_path.read_text())
    assert report["ok"] is False
    assert report["machine"] == "reference"
    assert report["clearance_model"]["model"] == "reference"
    entry = report["files"][0]
    assert entry["points"] == 35
    assert len(entry["sha256"]) == 64
    nozzle = entry["checks"]["nozzle"]
    assert nozzle["collisions"] > 0 and len(nozzle["deepest"]) == 2


def test_a_directory_is_expanded_in_order(toolpaths, tmp_path):
    files = cms.expand_inputs([toolpaths["folder"]])
    assert [path.name for path in files] == ["a_collides.npz", "b_clear.npz"]

    report_path = tmp_path / "report.json"
    assert cms.main([str(toolpaths["folder"]), "--checks", "nozzle",
                     "--json", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    assert [entry["ok"] for entry in report["files"]] == [False, True]


def test_settings_reach_the_check(toolpaths, tmp_path):
    report_path = tmp_path / "report.json"
    cms.main([str(toolpaths["collides"]), "--checks", "nozzle", "--tolerance", "0.5",
              "--subsample", "0.25", "--json", str(report_path)])
    settings = json.loads(report_path.read_text())["files"][0]["checks"]["nozzle"]["settings"]
    assert settings["tolerance_mm"] == 0.5
    assert settings["subsample_mm"] == 0.25
    assert settings["height_mm"] == 70.0


def test_wildcards_are_expanded_by_the_tool(toolpaths):
    """PowerShell hands `*` to Python as it is."""
    files = cms.expand_inputs([str(toolpaths["folder"] / "*_clear.npz")])
    assert [path.name for path in files] == ["b_clear.npz"]
    assert cms.expand_inputs([str(toolpaths["folder"] / "*.nothing")]) == []


def test_missing_input_exits_two(tmp_path, capsys):
    assert cms.main([str(tmp_path / "nothing.npz")]) == 2
    assert "No such toolpath" in capsys.readouterr().err


def test_unknown_checks_exit_two(toolpaths, capsys):
    assert cms.main([str(toolpaths["clear"]), "--checks", "nozzle,bogus"]) == 2
    assert "bogus" in capsys.readouterr().err


def test_both_checks_run_by_default_and_record_the_backend(ti_cpu, tmp_path):
    """The travel through a block: clear at the points, caught between them."""
    from test_tilt_motion_check import travel_case

    tp, crossing = travel_case(1.0)
    path = write_toolpath(tmp_path / "travel.npz", tp.point.astype(float),
                          _directions(tp.tool_orientation), tp.travel_type == 0)
    report_path = tmp_path / "report.json"

    assert cms.main([str(path), "--json", str(report_path)]) == 1

    report = json.loads(report_path.read_text())
    assert report["checks"] == ["nozzle", "swept"]
    assert report["taichi_backend"] == "x64"
    checks = report["files"][0]["checks"]
    assert checks["nozzle"]["ok"] is True
    assert checks["swept"]["ok"] is False
    assert checks["swept"]["worst"][0]["move"] == crossing
    assert checks["swept"]["worst"][0]["kind"] == "nozzle_vs_material"


def _directions(spherical):
    theta, phi = spherical[:, 0].astype(float), spherical[:, 1].astype(float)
    return np.column_stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                            np.cos(theta)])


def test_bed_hits_before_the_platform_come_with_a_hint():
    assert cms.before_platform("reports/toolpaths/ramp60_s_ms30.npz")
    assert cms.before_platform("data/toolpath/ramp60_s_smoothed.npz")
    assert not cms.before_platform("data/toolpath/ramp60_s_platform.npz")
    assert not cms.before_platform("tests/golden/calibration_cube.toolpath.npz")

    entry = {"file": "ramp60_s_ms30.npz", "points": 10, "max_tilt_deg": 29.4, "ok": False,
             "checks": {"swept": {"ok": False, "seconds": 1.0, "violations": 2,
                                  "violations_by_kind": {"bed": 2}}}}
    assert cms.BED_HINT in cms.summary_line(entry)
    entry["file"] = "ramp60_s_platform.npz"
    assert cms.BED_HINT not in cms.summary_line(entry)
