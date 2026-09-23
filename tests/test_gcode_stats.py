"""Unit tests for tools/gcode_stats.py (build plan task P0.2).

Every expected value below is hand-computed from
``tests/fixtures/gcode/tiny.gcode``, so a change in the parser that silently
alters the golden record will fail here rather than in a laptop pipeline run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import gcode_stats

FIXTURE = Path(__file__).parent / "fixtures" / "gcode" / "tiny.gcode"


@pytest.fixture(scope="module")
def stats() -> dict:
    return gcode_stats.stats_for_file(FIXTURE)


def test_strip_comment_removes_trailing_comment():
    assert gcode_stats.strip_comment("G1 X1 ; move") == "G1 X1"
    assert gcode_stats.strip_comment("; comment only") == ""
    assert gcode_stats.strip_comment("   ") == ""


def test_parse_words_reads_signs_and_decimals():
    words = gcode_stats.parse_words("G1 X0.1 Y-20 Z75.3 E-2.0 F2700")
    assert words == {
        "G": 1.0,
        "X": 0.1,
        "Y": -20.0,
        "Z": 75.3,
        "E": -2.0,
        "F": 2700.0,
    }


def test_parse_words_ignores_non_numeric_words():
    # RepRapFirmware macro calls carry a quoted path, not a number.
    assert gcode_stats.parse_words('M98 P"/macros/enable3Z.g"') == {"M": 98.0}


def test_line_count(stats):
    assert stats["line_count"] == 20


def test_move_counts(stats):
    assert stats["g1_count"] == 9
    assert stats["g0_count"] == 1


def test_command_counts(stats):
    assert stats["command_counts"] == {
        "G0": 1,
        "G1": 9,
        "G21": 1,
        "G90": 1,
        "G92": 2,
        "M82": 2,
        "M83": 1,
        "M104": 1,
        "M400": 1,
    }


@pytest.mark.parametrize(
    "axis,expected_min,expected_max,expected_count",
    [
        ("X", 0.1, 21.0, 6),
        ("Y", 10.0, 200.0, 6),
        ("Z", 75.3, 78.0, 6),
        ("U", 75.3, 78.1, 6),
        ("V", 75.3, 78.0, 6),
        ("E", -2.0, 15.0, 8),
    ],
)
def test_axis_ranges(stats, axis, expected_min, expected_max, expected_count):
    entry = stats["axis_ranges"][axis]
    assert entry["min"] == pytest.approx(expected_min)
    assert entry["max"] == pytest.approx(expected_max)
    assert entry["count"] == expected_count


def test_total_extrusion_spans_both_extrusion_modes(stats):
    # Absolute (M82): E15 from a G92 E0 reference -> 15.0 mm.
    # Relative (M83): 0.5 + 0.25 + 2.0 (prime) + 0.3 -> 3.05 mm.
    assert stats["total_extrusion_mm"] == pytest.approx(18.05)


def test_retraction_is_counted_separately_from_extrusion(stats):
    # Absolute E13 after E15 (-2.0), relative E-2.0, absolute E-2.0 after G92 E0.
    assert stats["total_retraction_mm"] == pytest.approx(6.0)


def test_retract_and_prime_lines(stats):
    # Extruder-only moves: three negative (retract), one positive (prime).
    assert stats["retract_count"] == 3
    assert stats["prime_count"] == 1


def test_sha256_matches_file_contents(stats):
    expected = hashlib.sha256(
        FIXTURE.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert stats["sha256"] == expected


def test_head_and_tail_cover_a_short_file(stats):
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    assert stats["head_lines"] == lines
    assert stats["tail_lines"] == lines


def test_head_and_tail_are_clipped_on_a_long_file():
    text = "\n".join(f"G1 X{i}" for i in range(200))
    result = gcode_stats.compute_stats(text)
    assert len(result["head_lines"]) == gcode_stats.CONTEXT_LINES
    assert len(result["tail_lines"]) == gcode_stats.CONTEXT_LINES
    assert result["head_lines"][0] == "G1 X0"
    assert result["tail_lines"][-1] == "G1 X199"


def test_g92_resets_the_absolute_extrusion_reference():
    # Without the G92 reset the second E5 would count as zero extrusion.
    text = "M82\nG92 E0\nG1 X1 E5\nG92 E0\nG1 X2 E5\n"
    result = gcode_stats.compute_stats(text)
    assert result["total_extrusion_mm"] == pytest.approx(10.0)


def test_cli_writes_json_to_output_path(tmp_path):
    destination = tmp_path / "nested" / "stats.json"
    exit_code = gcode_stats.main([str(FIXTURE), "-o", str(destination)])

    assert exit_code == 0
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["line_count"] == 20
    assert written["file"] == "tiny.gcode"


def test_cli_rejects_a_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        gcode_stats.main([str(tmp_path / "does_not_exist.gcode")])
