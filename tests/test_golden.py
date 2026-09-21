"""Golden regression test for the stock pipeline (build plan task P0.2).

Re-runs the calibration cube end to end and compares the resulting G-code
against the committed baseline. This is the safety net for every edit to a
vendored Atomizer file: if behaviour changed, this fails.

The pipeline is deterministic on the reference environment (see
``tests/golden/baseline.md``), so the comparison is exact — the SHA-256 of the
G-code must match. If a future environment turns out not to be deterministic,
record that in `baseline.md` and relax the comparison there rather than
loosening it silently here.

Marked `pipeline`: it needs a GPU and Blender, and takes several minutes
(`order_atoms` alone is ~5.5 minutes on the reference laptop). Run with::

    pytest --run-pipeline tests/test_golden.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import gcode_stats

PART = "calibration_cube"
GOLDEN_STATS = Path(__file__).parent / "golden" / f"{PART}.stats.json"

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


@pytest.mark.pipeline
def test_pipeline_reproduces_the_golden_gcode(golden_stats, repo_root):
    """The stock pipeline still produces byte-identical G-code."""
    param_path = repo_root / "data" / "param" / f"{PART}.json"
    gcode_path = repo_root / "data" / "gcode" / f"{PART}.gcode"

    # The stage commands in tools/atomize.py use paths relative to the
    # repository root, so it has to run from there. Only files under the
    # data/ output directories are written, and those are gitignored.
    result = subprocess.run(
        [sys.executable, "tools/atomize.py", str(param_path.relative_to(repo_root))],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"atomize.py exited {result.returncode}\n"
        f"--- stdout (last 40 lines) ---\n" + "\n".join(result.stdout.splitlines()[-40:]) +
        f"\n--- stderr (last 40 lines) ---\n" + "\n".join(result.stderr.splitlines()[-40:])
    )
    assert gcode_path.is_file(), f"the pipeline wrote no G-code at {gcode_path}"

    actual = gcode_stats.stats_for_file(gcode_path)

    assert actual["sha256"] == golden_stats["sha256"], (
        "The G-code changed against the golden baseline "
        f"({GOLDEN_STATS.name}, commit recorded in tests/golden/baseline.md).\n"
        f"  golden sha256: {golden_stats['sha256']}\n"
        f"  actual sha256: {actual['sha256']}\n"
        "Differences in the tracked statistics:\n"
        + _describe_differences(golden_stats, actual)
    )


@pytest.mark.pipeline
def test_golden_statistics_are_self_consistent(golden_stats):
    """Guards against a corrupted or hand-edited baseline file."""
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
