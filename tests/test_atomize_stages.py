"""Tests for `tools/atomize.py`'s stages as functions (build plan P5.1a).

P5.1a moved the stage commands out of `__main__` into `build_stage_commands`
and `run_stages`, and added `--stop-after` for P2.0's field-only evaluation.
The command line must behave exactly as before. That is checked here without
a GPU or Blender: `tests/_atomize_record.py` runs the script with every stage
stubbed out and records each command, printed line and log line, and
`tests/fixtures/atomize/recorded.json` holds the same record taken from the
script *before* the change. The golden test (`tests/test_golden.py`, laptop)
checks the same thing end to end, through the G-code.

Everything runs in one child process: importing `atomize.py` starts Taichi,
which would reset the runtime under every other test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "atomize" / "recorded.json"
RECORDER = REPO_ROOT / "tests" / "_atomize_record.py"

RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASE_PARAMS = {case["name"]: case["params"] for case in RECORDED["cases"]}

#: Every stage for a part with infill, in order, as `STAGE_NAMES` must list it.
ALL_STAGES = [
    "process_for_atomizer",
    "obj_to_bpn",
    "bpn_to_sdf",
    "sdf_to_isdf",
    "compute_tool_orientations",
    "sdf_df_to_layers",
    "compute_tangents",
    "align_atoms",
    "extract_explicit_atoms",
    "order_atoms",
    "smooth_toolpath_point",
    "tesselate_toolpath_orientations",
    "add_platform",
    "toolpath_to_gcode",
    "ratrig_to_craftware",
]


def _new_cases():
    """Cases for the behaviour P5.1a added, recorded fresh on every run."""
    cube = CASE_PARAMS["calibration_cube"]
    plain = CASE_PARAMS["no_infill"]
    extract = ["--stop-after", "extract_explicit_atoms"]
    return [
        {"name": "stop_cube", "params": cube, "argv": extract},
        {"name": "stop_no_infill", "params": plain, "argv": extract},
        {"name": "stop_warmup", "params": cube, "argv": [*extract, "--warmup"]},
        {"name": "stop_unknown", "params": cube, "argv": ["--stop-after", "nonsense"]},
        {"name": "api_all", "params": cube, "mode": "api"},
        {"name": "api_only_order", "params": cube, "mode": "api", "only": ["order_atoms"]},
        {
            "name": "api_reversed",
            "params": cube,
            "mode": "api",
            "only": ["add_platform", "smooth_toolpath_point"],
        },
        {
            "name": "api_mid_step",
            "params": cube,
            "mode": "api",
            "only": ["tesselate_toolpath_orientations"],
        },
        {
            "name": "api_skip",
            "params": cube,
            "mode": "api",
            "skip": ["process_for_atomizer", "order_atoms"],
        },
        {"name": "api_absent_stage", "params": plain, "mode": "api", "only": ["sdf_to_isdf"]},
        {"name": "api_unknown", "params": cube, "mode": "api", "only": ["remesh"]},
    ]


@pytest.fixture(scope="module")
def recorded_now(tmp_path_factory):
    """Run every case through the current `atomize.py`, in one child process."""
    work = tmp_path_factory.mktemp("atomize")
    cases_path = work / "cases.json"
    out_path = work / "out.json"
    cases = RECORDED["cases"] + _new_cases()
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    env = dict(os.environ, ATOM_TI_ARCH="cpu", PYTHONIOENCODING="utf-8")
    env.pop("ATOM_TI_ARCH_LOG", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, str(RECORDER), str(cases_path), str(out_path), str(work / "runs")],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The command line is unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", [c["name"] for c in RECORDED["cases"]])
@pytest.mark.parametrize("field", ["commands", "stdout", "log", "error"])
def test_the_command_line_behaves_exactly_as_before(recorded_now, case, field):
    """Every command, printed line and log line matches the pre-P5.1a record."""
    assert recorded_now[case][field] == RECORDED["results"][case][field]


def test_the_record_covers_what_varies_between_runs():
    """The fixture exercises infill on and off, every optional key, and --warmup."""
    names = {case["name"] for case in RECORDED["cases"]}
    assert {"calibration_cube", "no_infill", "every_option", "warmup"} <= names
    every = CASE_PARAMS["every_option"]
    for key in (
        "infill", "infill_period", "shell_thickness", "bed_temp", "nozzle_temp",
        "top_lines", "bottom_lines", "ortho_to_wall", "all_up",
    ):
        assert key in every, key
    assert not CASE_PARAMS["no_infill"].get("infill", False)


# --------------------------------------------------------------------------
# --stop-after
# --------------------------------------------------------------------------


def _through_extraction(commands):
    """The recorded commands up to and including atom extraction."""
    last = max(
        index for index, command in enumerate(commands)
        if command.startswith("python tools/extract_explicit_atoms.py")
    )
    return commands[: last + 1]


@pytest.mark.parametrize(
    "case,recorded",
    [("stop_cube", "calibration_cube"), ("stop_no_infill", "no_infill"), ("stop_warmup", "warmup")],
)
def test_stop_after_runs_exactly_the_stages_up_to_atom_extraction(recorded_now, case, recorded):
    """The same commands as a full run, cut after `extract_explicit_atoms`."""
    full = RECORDED["results"][recorded]["commands"]
    stopped = recorded_now[case]["commands"]
    assert stopped == _through_extraction(full)
    assert not any("order_atoms.py" in command for command in stopped)


def test_stop_after_says_so_on_screen_and_in_the_log(recorded_now):
    run = recorded_now["stop_cube"]
    message = "Stopped after extract_explicit_atoms (--stop-after); the later stages did not run."
    assert message in run["stdout"]
    assert message in run["log"]
    assert "Step 9 Starting" not in run["stdout"]
    assert "# Order Atoms" not in run["log"]


def test_stop_after_keeps_the_log_header_of_a_full_run(recorded_now):
    """Parameters and the command lists are written as for a full run."""
    full_log = RECORDED["results"]["calibration_cube"]["log"]
    header = full_log[: full_log.index("\n# Direction Field Computation")]
    assert recorded_now["stop_cube"]["log"].startswith(header)


def test_stop_after_refuses_an_unknown_stage(recorded_now):
    run = recorded_now["stop_unknown"]
    assert run["error"] == "SystemExit(2)"
    assert run["commands"] == []


# --------------------------------------------------------------------------
# run_stages
# --------------------------------------------------------------------------


def _returned(run):
    """The stage names `run_stages` returned, from the recorder's printed line."""
    line = next(
        line for line in run["stdout"].splitlines() if line.startswith("run_stages returned ")
    )
    return json.loads(line[len("run_stages returned "):])


def test_every_stage_in_order(recorded_now):
    run = recorded_now["api_all"]
    assert run["error"] is None
    assert _returned(run) == ALL_STAGES
    assert run["commands"] == RECORDED["results"]["calibration_cube"]["commands"]


def test_every_stage_is_a_tool_in_this_repository():
    for name in ALL_STAGES:
        assert (REPO_ROOT / "tools" / f"{name}.py").is_file(), name


def test_only_runs_the_named_stage(recorded_now):
    run = recorded_now["api_only_order"]
    assert _returned(run) == ["order_atoms"]
    assert len(run["commands"]) == 1
    assert run["commands"][0].startswith("python tools/order_atoms.py ")
    assert "Step 9 Starting: Order Atoms" in run["stdout"]


def test_names_run_in_pipeline_order_and_a_shared_step_prints_once(recorded_now):
    run = recorded_now["api_reversed"]
    assert _returned(run) == ["smooth_toolpath_point", "add_platform"]
    assert run["stdout"].count("Step 10 Starting") == 1


def test_a_step_entered_part_way_is_still_announced(recorded_now):
    run = recorded_now["api_mid_step"]
    assert _returned(run) == ["tesselate_toolpath_orientations"]
    assert "Step 10 Starting: Smooth, Tesselate and Add a Platform" in run["stdout"]


def test_skip_leaves_stages_out(recorded_now):
    returned = _returned(recorded_now["api_skip"])
    assert returned == [n for n in ALL_STAGES if n not in ("process_for_atomizer", "order_atoms")]


def test_a_stage_the_part_does_not_use_is_simply_absent(recorded_now):
    """`sdf_to_isdf` is a valid name, but a part without infill has no such stage."""
    run = recorded_now["api_absent_stage"]
    assert run["error"] is None
    assert _returned(run) == []
    assert run["commands"] == []


def test_an_unknown_stage_name_is_refused(recorded_now):
    run = recorded_now["api_unknown"]
    assert run["error"].startswith("ValueError: Unknown stage name(s): remesh.")
    assert run["commands"] == []
