"""Golden regression test for the stock pipeline (build plan task P0.2).

Re-runs the calibration cube end to end and compares the resulting G-code
against the committed baseline. This is the safety net for every edit to a
vendored Atomizer file: if behaviour changed, this fails.

The pipeline is deterministic **on one backend mix** (see
``tests/golden/baseline.md``), so on that mix the comparison is exact: the
SHA-256 of the G-code must match.

It is not deterministic *across* backends. Forcing every stage onto the CPU
with ``ATOM_TI_ARCH=cpu`` changes the toolpath itself, not just its last
digits: 2.4 % more points, three fewer deposition runs, twenty fewer fan
toggles. The measurement and the reason are recorded as
``docs/plan_corrections.md`` 3.9. So the hash assertion runs only on the
captured mix, and a forced backend is checked against invariants and
tolerances instead.

Marked `pipeline`: it needs a GPU and Blender, and takes several minutes
(`order_atoms` alone is ~5.5 minutes on the reference laptop; the whole run is
~20 minutes with every stage forced onto the CPU). Run with::

    pytest --run-pipeline tests/test_golden.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gcode_stats

PART = "calibration_cube"
GOLDEN_STATS = Path(__file__).parent / "golden" / f"{PART}.stats.json"

#: The environment variable that overrides every stage's Taichi backend.
#: Kept as a literal rather than imported from ``atom.ti_env`` so this test
#: does not need Taichi just to decide whether to skip.
ARCH_ENV_VAR = "ATOM_TI_ARCH"

#: Fields compared one by one when the hashes differ, to make the failure
#: readable. `head_lines`/`tail_lines` are excluded: they are long, and a real
#: difference in them shows up in the hash anyway.
COMPARED_FIELDS = (
    "line_count",
    "g0_count",
    "g1_count",
    "command_counts",
    "axis_ranges",
    "total_extrusion_mm",
    "total_retraction_mm",
    "retract_count",
    "prime_count",
)

#: How far a forced-backend run may drift from the golden before it counts as a
#: real regression rather than the cross-backend difference of 3.9. The CPU run
#: measured on 2026-09-23 sits at +2.44 % on points and -0.25 % on extrusion;
#: these leave headroom over that without admitting a genuine change of
#: behaviour, which moves these figures by far more.
MAX_POINT_COUNT_DRIFT = 0.06
MAX_EXTRUSION_DRIFT = 0.02


def _forced_arch() -> str | None:
    """The backend every stage was forced onto, or None for each stage's own."""
    value = os.environ.get(ARCH_ENV_VAR)
    return value.strip().lower() if value and value.strip() else None


def _describe_differences(expected: dict, actual: dict) -> str:
    """Human-readable summary of which statistics moved, for the failure text."""
    lines = []
    for field in COMPARED_FIELDS:
        if expected.get(field) != actual.get(field):
            lines.append(f"  {field}:\n    golden: {expected.get(field)!r}\n    now:    {actual.get(field)!r}")
    if not lines:
        return (
            "  No tracked statistic differs, but the SHA-256 does. The change is "
            "in G-code text that the statistics do not cover (comments, "
            "whitespace, feed rates, or line ordering)."
        )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def golden_stats() -> dict:
    if not GOLDEN_STATS.is_file():
        pytest.skip(
            f"No golden baseline at {GOLDEN_STATS}. Capture it first — see "
            "tests/golden/README.md."
        )
    return json.loads(GOLDEN_STATS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline_stats(repo_root) -> dict:
    """Run the pipeline once and return the statistics of its G-code.

    Module-scoped: the run costs 6 to 20 minutes depending on the backend, and
    both pipeline tests below compare against the same output.
    """
    param_path = repo_root / "data" / "param" / f"{PART}.json"
    gcode_path = repo_root / "data" / "gcode" / f"{PART}.gcode"

    # The stage commands in tools/atomize.py use paths relative to the
    # repository root, so it has to run from there. Only files under the
    # data/ output directories are written, and those are gitignored.
    #
    # `encoding`/`errors` are explicit, and PYTHONIOENCODING is set for the
    # child: without them the reader thread decodes the stage output with the
    # locale codec, which on Windows is cp1252, and a single non-ASCII byte
    # raises UnicodeDecodeError inside subprocess — losing the whole output,
    # including the message of a stage that genuinely failed.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "tools/atomize.py", str(param_path.relative_to(repo_root))],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert result.returncode == 0, (
        f"atomize.py exited {result.returncode}\n"
        f"--- stdout (last 40 lines) ---\n" + "\n".join((result.stdout or "").splitlines()[-40:]) +
        f"\n--- stderr (last 40 lines) ---\n" + "\n".join((result.stderr or "").splitlines()[-40:])
    )
    assert gcode_path.is_file(), f"the pipeline wrote no G-code at {gcode_path}"

    return gcode_stats.stats_for_file(gcode_path)


@pytest.mark.pipeline
def test_pipeline_reproduces_the_golden_gcode(golden_stats, pipeline_stats):
    """The stock pipeline still produces byte-identical G-code.

    Only on the backend mix the baseline was captured on. A forced backend is
    a different computation, not a regression — see 3.9 and the test below.
    """
    forced = _forced_arch()
    if forced is not None:
        pytest.skip(
            f"{ARCH_ENV_VAR}={forced!r} forces every stage onto one backend. The "
            "golden SHA-256 was captured on each stage's own default (the five "
            "field stages on CUDA, ordering and smoothing on the CPU) and is not "
            "reproducible on another mix — see docs/plan_corrections.md 3.9. "
            "test_forced_backend_stays_within_tolerance covers this run instead."
        )

    assert pipeline_stats["sha256"] == golden_stats["sha256"], (
        "The G-code changed against the golden baseline "
        f"({GOLDEN_STATS.name}, commit recorded in tests/golden/baseline.md).\n"
        f"  golden sha256: {golden_stats['sha256']}\n"
        f"  actual sha256: {pipeline_stats['sha256']}\n"
        "Differences in the tracked statistics:\n"
        + _describe_differences(golden_stats, pipeline_stats)
    )


@pytest.mark.pipeline
def test_forced_backend_stays_within_tolerance(golden_stats, pipeline_stats):
    """With every stage forced onto one backend, check invariants and drift.

    The hash cannot hold here (3.9), but the output must still be valid 5-axis
    G-code of roughly the golden's size. This is what catches a real regression
    on a machine that has no CUDA — the university CPU machines, or CI if the
    pipeline tier is ever enabled there.
    """
    forced = _forced_arch()
    if forced is None:
        pytest.skip(
            f"{ARCH_ENV_VAR} is unset, so this run is on the captured mix and "
            "test_pipeline_reproduces_the_golden_gcode checks it exactly."
        )

    # Still 5-axis G-code: all three bed screws move, and the extruder runs.
    for axis in ("Z", "U", "V"):
        assert pipeline_stats["axis_ranges"][axis]["count"] > 0, (
            f"the G-code from the {forced!r} backend never moves {axis}"
        )
    assert pipeline_stats["g1_count"] > 0

    point_drift = (
        pipeline_stats["g1_count"] / golden_stats["g1_count"] - 1.0
    )
    extrusion_drift = (
        pipeline_stats["total_extrusion_mm"] / golden_stats["total_extrusion_mm"] - 1.0
    )

    assert abs(point_drift) <= MAX_POINT_COUNT_DRIFT, (
        f"the {forced!r} backend changed the G1 count by {point_drift:+.2%}, more "
        f"than the {MAX_POINT_COUNT_DRIFT:.0%} allowed for a cross-backend "
        "difference. That is a change of behaviour, not backend float noise.\n"
        + _describe_differences(golden_stats, pipeline_stats)
    )
    assert abs(extrusion_drift) <= MAX_EXTRUSION_DRIFT, (
        f"the {forced!r} backend changed total extrusion by {extrusion_drift:+.2%}, "
        f"more than the {MAX_EXTRUSION_DRIFT:.0%} allowed for a cross-backend "
        "difference.\n" + _describe_differences(golden_stats, pipeline_stats)
    )


def test_golden_statistics_are_self_consistent(golden_stats):
    """Guards against a corrupted or hand-edited baseline file.

    A `unit` test: it only reads the committed JSON, so CI can run it. It was
    marked `pipeline` by mistake, which meant the check that the baseline file
    is intact never ran anywhere the pipeline could not.
    """
    assert golden_stats["schema_version"] == 1
    assert golden_stats["line_count"] > 0
    assert golden_stats["g1_count"] > 0
    assert golden_stats["total_extrusion_mm"] > 0

    # The triple-Z machine drives three independent bed screws; G-code that
    # moved none of them would not be 5-axis output at all.
    for axis in ("Z", "U", "V"):
        assert golden_stats["axis_ranges"][axis]["count"] > 0, (
            f"the baseline G-code never moves {axis}"
        )


def test_golden_baseline_record_exists(repo_root):
    """`baseline.md` documents the commit, environment and determinism result.

    A `unit` test, so CI notices if the record goes missing even though the
    pipeline itself cannot run there.
    """
    record = repo_root / "tests" / "golden" / "baseline.md"
    assert record.is_file(), "tests/golden/baseline.md is missing"

    text = record.read_text(encoding="utf-8")
    for expected in ("Baseline commit", "Determinism", "Environment"):
        assert expected in text, f"baseline.md has no '{expected}' section"


def test_golden_data_files_are_committed(repo_root):
    """The captured baseline must be in the repository, not just on one laptop.

    `golden_stats` skips when the stats file is absent, which is right for a
    fresh clone before the baseline has ever been captured. Once it has been,
    a missing file means it was captured locally and never committed, and the
    regression test then skips on every other machine while looking healthy.
    That is exactly the failure this catches.

    A `unit` test, so CI notices without needing to run the pipeline.
    """
    import subprocess

    # `git ls-tree HEAD`, not `git ls-files`: the latter lists the index, so a
    # file that was staged with `git add` but never committed would pass. That
    # is precisely the state this test exists to catch, and it did pass on it
    # once.
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "tests/golden/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout, or git is unavailable")

    tracked = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    record = repo_root / "tests" / "golden" / "baseline.md"

    # The record documents a captured baseline, so the data must be there too.
    if record.is_file() and "not yet captured" not in record.read_text(
        encoding="utf-8"
    ):
        expected = f"tests/golden/{PART}.stats.json"
        assert expected in tracked, (
            f"{record.name} documents a captured baseline, but {expected} is not "
            "committed. It was produced on the operator's laptop and never "
            "pushed, so tests/test_golden.py silently skips everywhere else. "
            "Commit it."
        )
