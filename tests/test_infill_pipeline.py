"""Does ``infill_period`` reach the infill stage? (build plan P1.3, pipeline tier)

Runs the calibration cube once with ``infill_period`` 12 instead of upstream's
8, and compares its G-code with the golden record. A longer period spaces the
gyroid walls further apart, so the part holds less material: total extrusion
must change, and should fall. If it does not change at all, the parameter
never reached the stage.

The run writes the calibration cube's intermediate files under ``data/``,
the same paths the golden test uses (all gitignored). Run it on its own, or
after the golden test, not in the middle of it:

    pytest --run-pipeline tests/test_infill_pipeline.py

About 8 to 13 minutes on the operator's laptop.

No `from __future__ import annotations` needed; kept consistent with
test_golden.py, which drives the pipeline the same way.
"""

import json
import os
import subprocess
import sys

import pytest

import gcode_stats

PART = "calibration_cube"


@pytest.mark.pipeline
def test_a_longer_infill_period_extrudes_less(repo_root, tmp_path):
    from atom import gcode_check, machine_profile

    golden = json.loads(
        (repo_root / "tests" / "golden" / f"{PART}.stats.json").read_text(encoding="utf-8")
    )

    params = json.loads((repo_root / "data" / "param" / f"{PART}.json").read_text(encoding="utf-8"))
    assert params.get("infill") is True, "the calibration cube must have infill for this check"
    params["infill_period"] = 12
    param_path = tmp_path / f"{PART}.json"
    param_path.write_text(json.dumps(params, indent=4), encoding="utf-8")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "tools/atomize.py", str(param_path)],
        cwd=repo_root, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert result.returncode == 0, "\n".join((result.stdout or "").splitlines()[-40:])

    log = (repo_root / "data" / "log" / f"{PART}.log").read_text(encoding="utf-8")
    assert "--infill-period 12" in log, "the option never reached the stage's command"

    gcode_path = repo_root / "data" / "gcode" / f"{PART}.gcode"
    stats = gcode_stats.stats_for_file(gcode_path)
    change = stats["total_extrusion_mm"] / golden["total_extrusion_mm"] - 1.0
    print(
        f"total extrusion: golden {golden['total_extrusion_mm']:.3f} mm, "
        f"infill_period 12: {stats['total_extrusion_mm']:.3f} mm ({change:+.2%})"
    )
    assert stats["total_extrusion_mm"] != golden["total_extrusion_mm"]
    assert change < 0.0, "a sparser gyroid should hold less material"

    report = gcode_check.check_file(gcode_path, machine_profile.load_profile())
    assert report.ok, report.violations[:10]
