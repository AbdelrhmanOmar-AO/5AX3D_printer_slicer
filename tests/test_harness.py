"""Tests for the test harness itself (build plan task P0.3).

The build plan splits tests into three tiers, and the slow two must not run
unless explicitly asked for; otherwise CI (no GPU, no Blender) would fail on
tests it cannot possibly run. These tests pin that behaviour down.
"""

from __future__ import annotations

import pytest


def test_unmarked_test_is_treated_as_unit(request):
    """A test with no marker is given the `unit` marker during collection."""
    assert "unit" in request.node.keywords


@pytest.mark.unit
def test_explicit_unit_marker_runs():
    assert True


@pytest.mark.pipeline
def test_dummy_pipeline_is_skipped_without_the_flag():
    """Sentinel: must be skipped in CI, and must run with --run-pipeline.

    The operator should see this as PASSED when running
    `scripts/run_pipeline_tests.ps1`, which confirms the flag plumbing works.
    """
    assert True


@pytest.mark.benchmark
def test_dummy_benchmark_is_skipped_without_the_flag():
    """Sentinel: must be skipped unless --run-benchmark is passed."""
    assert True


def test_tmp_data_dir_is_an_isolated_copy(tmp_data_dir, repo_root):
    """Pipeline tests must never write into the committed `data/` tree."""
    assert tmp_data_dir.is_dir()
    assert tmp_data_dir != repo_root / "data"

    # Inputs are copied in...
    assert (tmp_data_dir / "mesh" / "calibration_cube.stl").is_file()
    assert (tmp_data_dir / "param" / "calibration_cube.json").is_file()

    # ...and the stage output directories exist but start empty.
    for name in ("gcode", "sdf", "toolpath", "direction", "log"):
        output_dir = tmp_data_dir / name
        assert output_dir.is_dir()
        assert not any(output_dir.iterdir())


def test_tmp_data_dir_writes_do_not_touch_the_repo(tmp_data_dir, repo_root):
    written = tmp_data_dir / "gcode" / "scratch.gcode"
    written.write_text("G21\n", encoding="utf-8")

    assert written.is_file()
    assert not (repo_root / "data" / "gcode" / "scratch.gcode").exists()
