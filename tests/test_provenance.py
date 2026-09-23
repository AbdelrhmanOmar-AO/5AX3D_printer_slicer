"""Tests for report provenance (build plan task P1.7).

Plan §0 rule 10: every reported number carries the machine and backend it came
from, and a table that pools runs which are not comparable says so. These tests
cover building the block, recording each stage's real backend through
`atom.ti_env`, the summary's grouping and warning, and the backfilled reports
committed in ``reports/baseline_overhang/``.

No `from __future__ import annotations`: this drives Taichi through
`atom.ti_env` and the report tool.
"""

import copy
import json
import subprocess
import types

import pytest

pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import backfill_provenance
import overhang_report
from atom import provenance as prov
from atom import ti_env


@pytest.fixture
def no_overrides(monkeypatch):
    monkeypatch.delenv("ATOM_TI_ARCH", raising=False)
    monkeypatch.delenv("ATOM_MACHINE", raising=False)


def pipeline_block(repo_root, stage_arches=None):
    return prov.collect(
        prov.SOURCE_PIPELINE,
        stage_arches if stage_arches is not None else {"order_atoms": "x64", "bpn_to_sdf": "cuda"},
        repo_root,
        overhang_report.METRICS_VERSION,
    )


def report_with(block, part="ramp45_xs", slope=7.0, printable=True):
    return {
        "schema_version": overhang_report.SCHEMA_VERSION,
        "part": part,
        "max_slope_deg": slope,
        "metrics": {
            "max_tool_tilt_deg": 5.5,
            "unsupported_fraction_near_overhangs": 0.002,
            "surfaces": [],
        },
        "verdict": {"printable": printable, "worst_effective_deg": 45.0},
        "provenance": block,
    }


# --------------------------------------------------------------------------
# Building the block
# --------------------------------------------------------------------------


def test_a_pipeline_block_is_complete(repo_root, no_overrides):
    block = pipeline_block(repo_root)
    assert prov.problems(block) == []
    assert set(prov.REQUIRED_FIELDS) <= set(block)
    assert block["ti_arch_setting"] == prov.STOCK_MIX
    assert block["machine_profile"] == "reference"
    assert block["metrics_version"] == overhang_report.METRICS_VERSION
    assert block["stage_arches"] == {"bpn_to_sdf": "cuda", "order_atoms": "x64"}
    assert block["taichi"].count(".") == 2  # "1.7.4"


def test_the_backend_and_profile_overrides_are_recorded(repo_root, monkeypatch):
    monkeypatch.setenv("ATOM_TI_ARCH", " CPU ")
    monkeypatch.setenv("ATOM_MACHINE", "ours")
    block = pipeline_block(repo_root)
    assert block["ti_arch_setting"] == "cpu"
    assert block["machine_profile"] == "ours"


def test_the_commit_is_recorded(repo_root, no_overrides):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    ).stdout.strip()
    block = pipeline_block(repo_root)
    assert block["git_commit"] == head
    assert block["scored"]["git_commit"] == head
    assert isinstance(block["code_modified"], bool)


def test_a_skip_pipeline_block_admits_its_origin_is_unknown(repo_root):
    block = prov.collect(prov.SOURCE_SKIP_PIPELINE, {}, repo_root, 2)
    assert prov.problems(block) == []
    assert block["machine"] is None and block["stage_arches"] == {}
    assert block["scored"]["machine"]  # but the scoring machine is known
    assert not prov.is_known(block)
    assert "unknown origin" in prov.describe(block)


def test_an_unknown_source_is_refused(repo_root):
    with pytest.raises(ValueError, match="source"):
        prov.collect("guess", {}, repo_root, 2)


def test_problems_names_what_is_missing(repo_root, no_overrides):
    assert prov.problems(None) == ["no provenance block"]
    block = pipeline_block(repo_root)
    del block["taichi"]
    assert prov.problems(block) == ["missing field 'taichi'"]

    block = pipeline_block(repo_root, stage_arches={})
    assert "stage_arches is empty" in prov.problems(block)


def test_rescoring_keeps_the_run_and_updates_the_scoring(repo_root, no_overrides):
    block = pipeline_block(repo_root)
    block["machine"] = "the-laptop"
    block["scored"]["utc"] = "2000-01-01T00:00:00+00:00"

    updated = prov.rescored(block, repo_root, 99)
    assert updated["machine"] == "the-laptop"
    assert updated["stage_arches"] == block["stage_arches"]
    assert updated["metrics_version"] == 99
    assert updated["scored"]["metrics_version"] == 99
    assert updated["scored"]["utc"] != "2000-01-01T00:00:00+00:00"
    assert block["metrics_version"] != 99, "the original must not be modified"


# --------------------------------------------------------------------------
# Each stage's real backend, through atom.ti_env
# --------------------------------------------------------------------------


def test_ti_env_records_the_backend_it_actually_got(ti_cpu, tmp_path, monkeypatch):
    log = tmp_path / "arches.jsonl"
    monkeypatch.setenv(ti_env.ARCH_LOG_ENV_VAR, str(log))
    monkeypatch.setattr("sys.argv", ["tools/order_atoms.py"])

    ti_env._record_arch("cpu")

    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry == {"stage": "order_atoms", "requested": "cpu", "actual": ti_env.current_arch_name()}
    assert prov.read_arch_log(log) == {"order_atoms": ti_env.current_arch_name()}


def test_ti_env_records_nothing_unless_asked(ti_cpu, tmp_path, monkeypatch):
    monkeypatch.delenv(ti_env.ARCH_LOG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    ti_env._record_arch("cpu")
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_log_warns_rather_than_aborting_the_stage(ti_cpu, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ti_env.ARCH_LOG_ENV_VAR, str(tmp_path / "missing_dir" / "log.jsonl"))
    ti_env._record_arch("cpu")
    assert "could not record the Taichi backend" in capsys.readouterr().err


def test_the_arch_log_keeps_disagreements_visible(tmp_path):
    log = tmp_path / "arches.jsonl"
    log.write_text(
        "\n".join([
            '{"stage": "align_atoms", "requested": "gpu", "actual": "cuda"}',
            '{"stage": "align_atoms", "requested": "gpu", "actual": "cuda"}',  # warm-up
            '{"stage": "bpn_to_sdf", "requested": "gpu", "actual": "cuda"}',
            '{"stage": "bpn_to_sdf", "requested": "gpu", "actual": "x64"}',
            "not json",
            "",
        ]),
        encoding="utf-8",
    )
    assert prov.read_arch_log(log) == {"align_atoms": "cuda", "bpn_to_sdf": "cuda+x64"}
    assert prov.read_arch_log(tmp_path / "absent.jsonl") == {}


def test_run_pipeline_asks_every_stage_to_record_its_backend(tmp_path, monkeypatch, repo_root):
    """The env var reaches atomize.py, and what the stages wrote comes back."""
    param = tmp_path / "part.json"
    param.write_text(json.dumps({"solid_name": "part", "max_slope": 7.0}), encoding="utf-8")
    seen = {}

    def fake_run(command, cwd, env):
        seen["env"] = env
        with open(env[ti_env.ARCH_LOG_ENV_VAR], "a", encoding="utf-8") as handle:
            handle.write('{"stage": "order_atoms", "requested": "cpu", "actual": "x64"}\n')
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(overhang_report.subprocess, "run", fake_run)
    _, params, stage_arches = overhang_report.run_pipeline(param, 30.0)

    assert ti_env.ARCH_LOG_ENV_VAR in seen["env"]
    assert stage_arches == {"order_atoms": "x64"}
    assert params["max_slope"] == 30.0


# --------------------------------------------------------------------------
# Comparability and the summary
# --------------------------------------------------------------------------


def test_the_commit_does_not_make_runs_incomparable(repo_root, no_overrides):
    a = pipeline_block(repo_root)
    b = copy.deepcopy(a)
    b["git_commit"], b["code_modified"], b["os"] = "0" * 40, True, "another OS build"
    b["machine"] = a["machine"].upper()  # host names are case-insensitive
    assert prov.comparability_key(a) == prov.comparability_key(b)


@pytest.mark.parametrize(
    "field, value",
    [
        ("machine", "university-cpu-node"),
        ("ti_arch_setting", "cpu"),
        ("stage_arches", {"order_atoms": "x64", "bpn_to_sdf": "x64"}),
        ("taichi", "1.8.0"),
        ("machine_profile", "ours"),
    ],
)
def test_each_comparability_field_separates_runs(repo_root, no_overrides, field, value):
    a = pipeline_block(repo_root)
    b = copy.deepcopy(a)
    b[field] = value
    assert prov.comparability_key(a) != prov.comparability_key(b)


def test_one_group_is_stated_above_the_table(repo_root, no_overrides):
    block = pipeline_block(repo_root)
    reports = [report_with(block, slope=7.0), report_with(block, slope=30.0)]
    summary = overhang_report.summarize(reports)

    assert f"Measured on: {prov.describe(block)}. All 2 runs are comparable." in summary
    assert "Mixed provenance" not in summary and "[A]" not in summary
    assert overhang_report.provenance_warning(reports) is None


def test_a_mixed_table_warns_and_labels_every_cell(repo_root, no_overrides):
    laptop = pipeline_block(repo_root)
    university = copy.deepcopy(laptop)
    university["machine"] = "university-cpu-node"
    university["ti_arch_setting"] = "cpu"
    reports = [
        report_with(laptop, part="ramp45_s", slope=7.0),
        report_with(laptop, part="ramp45_s", slope=15.0),
        report_with(university, part="ramp45_s", slope=30.0),
    ]
    summary = overhang_report.summarize(reports)

    assert "Mixed provenance: 2 groups" in summary
    assert "**[A]** 2 run(s)" in summary and "**[B]** 1 run(s)" in summary
    row = next(line for line in summary.splitlines() if line.startswith("| `ramp45_s`"))
    assert row.count("[A]") == 2 and row.count("[B]") == 1

    warning = overhang_report.provenance_warning(reports)
    assert warning.startswith("WARNING: this table mixes 2 provenance groups")
    assert "university-cpu-node" in warning


def test_reports_without_provenance_are_flagged(repo_root, no_overrides):
    reports = [report_with(None), report_with(None, slope=30.0)]
    summary = overhang_report.summarize(reports)
    assert "Origin unknown" in summary
    assert "unknown" in overhang_report.provenance_warning(reports)

    mixed = [report_with(pipeline_block(repo_root)), report_with(None, slope=30.0)]
    assert "no provenance recorded" in overhang_report.provenance_warning(mixed)


def test_summarize_prints_the_warning(repo_root, no_overrides, tmp_path, monkeypatch, capsys):
    laptop = pipeline_block(repo_root)
    other = copy.deepcopy(laptop)
    other["taichi"] = "1.8.0"
    monkeypatch.setattr(overhang_report, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(overhang_report, "SUMMARY_PATH", tmp_path / "summary.md")
    (tmp_path / "reports").mkdir()
    for index, block in enumerate((laptop, other)):
        (tmp_path / "reports" / f"r{index}.json").write_text(
            json.dumps(report_with(block, slope=7.0 + index)), encoding="utf-8"
        )

    assert overhang_report.main(["--summarize"]) == 0
    assert "WARNING: this table mixes 2 provenance groups" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The backfilled baseline
# --------------------------------------------------------------------------


def committed_reports(repo_root):
    paths = sorted((repo_root / "reports" / "baseline_overhang").glob("*.json"))
    assert len(paths) == 48
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_every_committed_report_carries_complete_provenance(repo_root):
    for report in committed_reports(repo_root):
        assert prov.problems(report.get("provenance")) == [], report["part"]


def test_the_committed_baseline_is_one_comparable_group(repo_root):
    reports = committed_reports(repo_root)
    assert len(overhang_report.provenance_groups(reports)) == 1
    assert overhang_report.provenance_warning(reports) is None
    block = reports[0]["provenance"]
    assert block["machine"] == "AbdoYasser"
    assert block["ti_arch_setting"] == prov.STOCK_MIX
    assert block["stage_arches"]["order_atoms"] == "x64"
    assert block["stage_arches"]["compute_tool_orientations"] == "cuda"


def test_the_committed_summary_states_where_the_numbers_came_from(repo_root):
    summary = (repo_root / "reports" / "baseline_overhang.md").read_text(encoding="utf-8")
    assert "Measured on: machine AbdoYasser, backend stock mix" in summary


def test_backfill_covers_every_stage_that_starts_taichi_in_a_run(repo_root):
    """Every pipeline stage in atomize.py that calls init_taichi, and nothing else."""
    atomize = (repo_root / "tools" / "atomize.py").read_text(encoding="utf-8")
    for stage, default in backfill_provenance.STAGE_DEFAULTS.items():
        source = (repo_root / "tools" / f"{stage}.py").read_text(encoding="utf-8")
        assert f'init_taichi("{default}"' in source, stage
        if stage != "atomize":
            assert f"tools/{stage}.py" in atomize, stage


def test_backfill_is_idempotent_and_never_overwrites(tmp_path, repo_root, no_overrides):
    fresh = report_with(None)
    fresh["metrics_version"] = 2
    recorded = report_with(pipeline_block(repo_root), slope=30.0)
    recorded["metrics_version"] = 2
    (tmp_path / "a.json").write_text(json.dumps(fresh), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(recorded), encoding="utf-8")

    assert backfill_provenance.main(["--report-dir", str(tmp_path), "--check"]) == 1
    assert backfill_provenance.main(["--report-dir", str(tmp_path)]) == 0
    first = (tmp_path / "a.json").read_text(encoding="utf-8")
    assert backfill_provenance.main(["--report-dir", str(tmp_path)]) == 0
    assert (tmp_path / "a.json").read_text(encoding="utf-8") == first

    assert json.loads(first)["provenance"]["source"] == prov.SOURCE_BACKFILLED
    kept = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["provenance"]
    assert kept["source"] == prov.SOURCE_PIPELINE


def test_backfill_refreshes_its_own_blocks_but_not_recorded_ones(repo_root, no_overrides):
    stale = report_with(copy.deepcopy(backfill_provenance.BACKFILL))
    stale["metrics_version"] = 2
    stale["provenance"]["machine"] = "an-old-guess"
    assert backfill_provenance.backfill_report(stale)
    assert stale["provenance"]["machine"] == backfill_provenance.LAPTOP

    recorded = report_with(pipeline_block(repo_root))
    recorded["metrics_version"] = 2
    before = copy.deepcopy(recorded)
    assert not backfill_provenance.backfill_report(recorded)
    assert recorded == before


def test_backfill_matches_what_the_laptop_recorded():
    """The first run after P1.7 on the laptop (2026-09-23), as pasted by the operator.

    Everything that decides comparability must agree, or the baseline would
    split from every new laptop run.
    """
    recorded = {
        "source": "pipeline",
        "machine": "AbdoYasser",
        "os": "Windows-10-10.0.26200-SP0",
        "python": "3.10.21",
        "taichi": "1.7.4",
        "ti_arch_setting": "stock mix",
        "stage_arches": {
            "add_platform": "cuda", "align_atoms": "cuda", "atomize": "x64",
            "bpn_to_sdf": "cuda", "compute_tangents": "cuda",
            "compute_tool_orientations": "cuda", "extract_explicit_atoms": "cuda",
            "obj_to_bpn": "x64", "order_atoms": "x64", "sdf_df_to_layers": "cuda",
            "sdf_to_isdf": "x64", "smooth_toolpath_point": "x64",
            "tesselate_toolpath_orientations": "x64", "toolpath_to_gcode": "cuda",
        },
        "machine_profile": "reference",
        "git_commit": "8d435f7eb6b266a6070f551723eaff8a1d649334",
        "code_modified": False,
        "recorded_utc": "2026-09-23T20:47:37+00:00",
        "metrics_version": 2,
        "scored": {
            "machine": "AbdoYasser",
            "git_commit": "8d435f7eb6b266a6070f551723eaff8a1d649334",
            "code_modified": False,
            "utc": "2026-09-23T20:47:37+00:00",
            "metrics_version": 2,
        },
    }
    assert prov.comparability_key(recorded) == prov.comparability_key(backfill_provenance.BACKFILL)
    assert recorded["os"] == backfill_provenance.BACKFILL["os"]


def test_backfill_refuses_a_different_metrics_version():
    report = report_with(None)
    report["metrics_version"] = 1
    with pytest.raises(ValueError, match="metrics version 1"):
        backfill_provenance.backfill_report(report)
