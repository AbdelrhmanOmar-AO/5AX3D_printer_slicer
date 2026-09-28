"""Tests for the G-code validator (build plan task P1.4).

Each fixture under ``tests/fixtures/gcode/validate_*.gcode`` is the same
minimal file with one line broken, so it must trigger exactly one check ID, on
a known line. The clean fixture and a real excerpt of the calibration cube must
pass with no violations.

No `from __future__ import annotations` here; two tests drive the Taichi
kernels in `atom.kinematics3z`.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import gcode_stats
import validate_gcode
from atom import gcode_check, machine_profile, tilt

FIXTURES = Path(__file__).parent / "fixtures" / "gcode"
CLEAN = FIXTURES / "validate_clean.gcode"


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


def check(text, profile, **options):
    return gcode_check.check_text(text, profile, **options)


def with_body(body: str) -> str:
    """The clean fixture's header, ``body`` in relative extrusion, its footer."""
    lines = CLEAN.read_text().splitlines()
    header = lines[: lines.index("M83 ; relative extrusion") + 1]
    footer = lines[lines.index("M82 ; absolute extrusion"):]
    return "\n".join(header + body.strip("\n").splitlines() + footer) + "\n"


# --------------------------------------------------------------------------
# Clean files pass
# --------------------------------------------------------------------------


def test_clean_fixture_passes(reference):
    report = gcode_check.check_file(CLEAN, reference)
    assert report.ok, report.violations
    assert report.stats["retract_count"] == 3  # header, one in the body, footer
    assert report.stats["prime_count"] == 2
    assert report.stats["max_tilt_deg"] == pytest.approx(0.1854, abs=1e-3)


def test_real_calibration_cube_excerpt_passes(reference):
    """Header, first 400 moves and footer of a real pipeline G-code file."""
    report = gcode_check.check_file(FIXTURES / "calibration_cube_head.gcode", reference)
    assert report.ok, report.violations[:5]


def test_the_writers_own_header_and_footer_pass(reference):
    """kinematics3z.HEADER and FOOTER, with a short body, pass as written.

    Also pins STRUCTURE_MARKERS to the macro calls the writer really emits.
    """
    pytest.importorskip("taichi")
    import atom.kinematics3z as k

    markers = gcode_check.STRUCTURE_MARKERS["rrf"]
    assert markers["enable_3z"] in k.HEADER
    assert markers["disable_3z"] in k.FOOTER

    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0.000000 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X151.0 Y145.0 Z85.0 U85.0 V85.0 E0.080000 F600\n"
    )
    report = check(k.HEADER + body + k.FOOTER, reference)
    assert report.ok, report.violations


# --------------------------------------------------------------------------
# One fixture per check
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, check_id, line",
    [
        ("axis_range", gcode_check.AXIS_RANGE, 17),
        ("tilt_limit", gcode_check.TILT_LIMIT, 15),
        ("nan", gcode_check.NAN, 14),
        ("feed", gcode_check.FEED, 19),
        ("extrusion", gcode_check.EXTRUSION, 14),
        ("structure", gcode_check.STRUCTURE, 7),
    ],
)
def test_each_fixture_triggers_exactly_its_check_on_the_right_line(reference, name, check_id, line):
    report = gcode_check.check_file(FIXTURES / f"validate_{name}.gcode", reference)
    assert [(v.id, v.line) for v in report.violations] == [(check_id, line)]
    assert report.stats["violation_counts"][check_id] == 1


def test_every_check_has_a_fixture():
    covered = {"axis_range", "tilt_limit", "nan", "feed", "extrusion", "structure"}
    assert {check_id.lower() for check_id in gcode_check.CHECK_IDS} == covered


# --------------------------------------------------------------------------
# AXIS_RANGE
# --------------------------------------------------------------------------


def test_header_purge_lines_are_checked_like_any_move(reference):
    text = CLEAN.read_text().replace("G1 X0.1 Y20 Z75.3", "G1 X-0.1 Y20 Z75.3")
    report = check(text, reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.AXIS_RANGE, 8)]


def test_a_screw_above_its_travel_is_out_of_range(reference):
    report = check(with_body("G1 X150.0 Y145.0 Z281.0 U281.0 V281.0 E0 F3000"), reference)
    assert report.ids() == {gcode_check.AXIS_RANGE}
    assert len(report.violations) == 3  # Z, U and V


# --------------------------------------------------------------------------
# TILT_LIMIT
# --------------------------------------------------------------------------


def test_tilt_just_inside_the_limit_passes(reference):
    # Z - U = sin(29.99 deg) * 309 along +X: 29.99 degrees.
    rise = np.sin(np.radians(29.99)) * 309.0
    body = f"G1 X150.0 Y145.0 Z200.0 U{200.0 - rise:.6f} V{200.0 - rise / 2:.6f} E0 F3000"
    report = check(with_body(body), reference)
    assert report.ok, report.violations
    assert report.stats["max_tilt_deg"] == pytest.approx(29.99, abs=1e-3)


def test_box_limit_permits_more_on_the_diagonal(reference, ti_cpu):
    """(25, 25) degrees is 34.78 in total: outside a 30-degree cone, inside a box."""
    import atom.kinematics3z as k

    direction = tilt.direction_from_tilts(25.0, 25.0)
    theta = np.arccos(direction[2])
    phi = np.arctan2(direction[1], direction[0])
    machine = np.zeros((1, 5), np.float32)
    k.toolpath_from_cartesian_toolpath(
        np.array([[150.0, 145.0, 60.0]], np.float32),
        np.array([[theta, phi]], np.float32),
        machine,
    )
    x, y, z0, z1, z2 = (float(value) for value in machine[0])
    # At this tilt the U screw lands below zero. Tilt depends only on the screw
    # differences, so lift all three into their travel.
    z0, z1, z2 = z0 + 60.0, z1 + 60.0, z2 + 60.0
    text = with_body(f"G1 X{x:.6f} Y{y:.6f} Z{z0:.6f} U{z1:.6f} V{z2:.6f} E0 F3000")

    cone = check(text, reference)
    assert cone.ids() == {gcode_check.TILT_LIMIT}
    assert cone.stats["max_tilt_deg"] == pytest.approx(tilt.total_tilt_from_tilts_deg(25.0, 25.0), abs=1e-3)

    box = check(text, dataclasses.replace(reference, tilt_limit_shape="box"))
    assert box.ok, box.violations


def test_box_limit_still_catches_a_single_axis_overrun(reference):
    report = check(
        (FIXTURES / "validate_tilt_limit.gcode").read_text(),
        dataclasses.replace(reference, tilt_limit_shape="box"),
    )
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.TILT_LIMIT, 15)]


# --------------------------------------------------------------------------
# NAN
# --------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["Xnan", "ZNaN", "E-inf", "Finf", "Uinfinity"])
def test_non_finite_words_are_caught(reference, word):
    report = check(with_body(f"G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000 {word}"), reference)
    assert gcode_check.NAN in report.ids()


def test_ordinary_words_are_not_mistaken_for_nan(reference):
    text = with_body(
        "M117 Loading information about infill ; banana\n"
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000"
    )
    assert check(text, reference).ok


# --------------------------------------------------------------------------
# FEED
# --------------------------------------------------------------------------


def test_feed_has_no_upper_limit_unless_given(reference):
    text = with_body("G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F10725")
    assert check(text, reference).ok
    limited = check(text, reference, max_feed=10000)
    assert [(v.id, v.line) for v in limited.violations] == [(gcode_check.FEED, 12)]


def test_negative_feed_is_rejected(reference):
    report = check(with_body("G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F-600"), reference)
    assert report.ids() == {gcode_check.FEED}


# --------------------------------------------------------------------------
# EXTRUSION
# --------------------------------------------------------------------------


def test_max_e_per_mm_is_adjustable(reference):
    """The fixture's line 14 pushes 7.5 mm over 1 mm of travel."""
    text = (FIXTURES / "validate_extrusion.gcode").read_text()
    report = check(text, reference)
    assert "over 1 mm of travel" in report.violations[0].detail
    assert check(text, reference, max_e_per_mm=8.0).ok
    assert gcode_check.DEFAULT_MAX_E_PER_MM == 0.5
    assert report.stats["limits"]["max_e_per_mm"] == 0.5


def test_a_long_extruding_move_is_judged_per_mm(reference):
    """A 30 mm platform edge at the nominal bead pushes out 5.05 mm of
    filament: over the old 5 mm single-move limit, but 0.168 mm per mm
    (plan_corrections 7b, P4-7)."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X180.0 Y145.0 Z85.0 U85.0 V85.0 E5.051400 F600\n"
    )
    report = check(with_body(body), reference)
    assert report.ok, report.violations
    assert report.stats["largest_e_per_mm"] == pytest.approx(5.0514 / 30.0)


def test_a_short_move_with_a_blob_is_rejected(reference):
    """0.4 mm of filament over 0.5 mm of travel is under any absolute limit
    but 0.8 mm per mm."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X150.5 Y145.0 Z85.0 U85.0 V85.0 E0.4 F600\n"
    )
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 14)]
    assert "0.5 mm of travel" in report.violations[0].detail


def test_travel_is_the_five_axis_machine_distance(reference):
    """Screw moves count: 0.25 mm of filament over dX = 1, dZ = dU = 0.5 is
    0.25 / sqrt(1.5) = 0.204 per mm, within 0.21 but over 0.2."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X151.0 Y145.0 Z85.5 U85.5 V85.0 E0.25 F600\n"
    )
    assert check(with_body(body), reference, max_e_per_mm=0.21).ok
    report = check(with_body(body), reference, max_e_per_mm=0.2)
    assert report.ids() == {gcode_check.EXTRUSION}


def test_extruding_without_moving_is_rejected(reference):
    """Axis words that go nowhere are zero travel: any E there is a blob."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X150.0 Y145.0 E0.05 F600\n"
    )
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 14)]


def test_max_e_caps_a_prime(reference):
    """An E with no axis word has no travel: the absolute ``max_e_mm`` holds."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E-6.00 F2700 ; retract\n"
        "G1 E6.00 F2700 ; prime\n"
    )
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 14)]
    assert "without moving" in report.violations[0].detail
    assert check(with_body(body), reference, max_e_mm=8.0).ok
    assert gcode_check.DEFAULT_MAX_E_MM == 5.0


def test_absolute_purge_line_is_not_a_single_move_overrun(reference):
    """The header's E15/E30 are absolute positions, not 15 mm in one move.

    The limit sits above the file's 2 mm primes and far below the purge's 15.
    """
    assert gcode_check.check_file(CLEAN, reference, max_e_mm=2.5).ok


def test_prime_without_a_retract(reference):
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 E2.00 F2700 ; prime again\n"
    )
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 14)]
    assert "no retract pending" in report.violations[0].detail


def test_prime_larger_than_the_retract(reference):
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
        "G1 E3.00 F2700 ; prime more than the header retracted\n"
    )
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 13)]
    assert "more than the 2 mm retracted" in report.violations[0].detail


def test_extruding_while_retracted(reference):
    body = "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0.08 F600 ; the header left 2 mm retracted\n"
    report = check(with_body(body), reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.EXTRUSION, 12)]


def test_quoted_macro_path_is_not_read_as_extrusion(reference):
    """`M98 P"/macros/enable3Z.g"` in relative mode must not count as E3 (hazard 2)."""
    body = (
        'M98 P"/macros/enable3Z.g"\n'
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0 F3000\n"
    )
    report = check(with_body(body), reference, max_e_mm=1.0)
    assert report.ok, report.violations
    assert report.stats["largest_relative_e_mm"] == 0.0


# --------------------------------------------------------------------------
# STRUCTURE
# --------------------------------------------------------------------------


def test_missing_footer_macro(reference):
    text = CLEAN.read_text().replace('M98 P"/macros/disable3Z.g"', "; no disable macro")
    report = check(text, reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.STRUCTURE, 19)]


def test_missing_units_line(reference):
    text = CLEAN.read_text().replace("G21 ; set units to millimeters", "; no G21")
    report = check(text, reference)
    assert [(v.id, v.line) for v in report.violations] == [(gcode_check.STRUCTURE, 7)]


def test_empty_file_is_not_ok(reference):
    assert check("; nothing\n", reference).ids() == {gcode_check.STRUCTURE}


def test_klipper_is_refused(reference):
    with pytest.raises(NotImplementedError, match="Gate E1 chose RepRapFirmware"):
        check(CLEAN.read_text(), dataclasses.replace(reference, firmware_dialect="klipper"))


# --------------------------------------------------------------------------
# Line numbers, report, shared parser, CLI
# --------------------------------------------------------------------------


def test_line_numbers_count_blank_and_comment_lines(reference):
    text = "\n; comment\n" + (FIXTURES / "validate_feed.gcode").read_text()
    report = check(text, reference)
    assert [v.line for v in report.violations] == [21]


def test_report_serialises_to_the_documented_shape(reference):
    report = gcode_check.check_file(FIXTURES / "validate_feed.gcode", reference)
    data = json.loads(json.dumps(report.to_dict()))
    assert set(data) == {"ok", "violations", "stats"}
    assert data["ok"] is False
    assert set(data["violations"][0]) == {"id", "line", "detail"}


def test_gcode_stats_uses_the_same_parser():
    """One G-code word parser in the repository (plan P1.4 step 1)."""
    assert gcode_stats.parse_words is gcode_check.parse_words
    assert gcode_stats.strip_comment is gcode_check.strip_comment


def test_cli_exit_codes(capsys):
    assert validate_gcode.main([str(CLEAN), "--machine", "reference"]) == 0
    assert "OK: no violations." in capsys.readouterr().out

    assert validate_gcode.main([str(FIXTURES / "validate_nan.gcode"), "--machine", "reference"]) == 1
    assert "line      14  NAN" in capsys.readouterr().out


def test_cli_json_output(capsys):
    code = validate_gcode.main(
        [str(FIXTURES / "validate_feed.gcode"), "--machine", "reference", "--json"]
    )
    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert data["violations"] == [{"id": "FEED", "line": 19, "detail": "F0 is not above zero"}]


def test_cli_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        validate_gcode.main([str(tmp_path / "absent.gcode")])
    assert error.value.code == 2


# --------------------------------------------------------------------------
# The rules as shared functions (the swept check, P4.2, applies them too)
# --------------------------------------------------------------------------


def _screw_states(path):
    """(line, Z, U, V) of every move that sets all three screws."""
    states = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        words = gcode_check.parse_words(gcode_check.strip_comment(raw))
        if {"Z", "U", "V"} <= words.keys():
            states.append((number, words["Z"], words["U"], words["V"]))
    return states


def test_axis_limits_are_the_ones_the_validator_reports(reference):
    stats = check(CLEAN.read_text(), reference).stats
    assert {axis: list(bounds) for axis, bounds in gcode_check.axis_limits(reference).items()} \
        == stats["limits"]["axes"]


def test_axis_range_problems_on_arrays(reference):
    states = np.array([
        [150.0, 145.0, 85.0, 85.0, 85.0],     # fine
        [-0.5, 145.0, 85.0, 85.0, 85.0],      # X below 0
        [150.0, 300.0, 85.0, -1.0, 281.0],    # Y over 293, U below 0, V over 280
        [0.0, 293.0, 0.0, 280.0, 0.0],        # exactly on the limits: fine
    ])
    problems = gcode_check.axis_range_problems(states, reference)
    assert [(row, axis) for row, axis, _ in problems] == [(1, "X"), (2, "Y"), (2, "U"), (2, "V")]
    assert problems[0][2] == "X-0.5 is outside [0, 300]"
    assert problems[0][2] == gcode_check.axis_range_detail("X", -0.5, 0.0, 300.0)


def test_tilt_limit_problems_say_exactly_what_the_validator_says(reference):
    """One rule: the screw states of the tilt fixture, fed to the shared
    function, give the validator's violation word for word."""
    path = FIXTURES / "validate_tilt_limit.gcode"
    report = gcode_check.check_file(path, reference)
    (violation,) = [v for v in report.violations if v.id == gcode_check.TILT_LIMIT]

    states = _screw_states(path)
    problems, total = gcode_check.tilt_limit_problems(
        np.array([state[1:] for state in states]), reference)

    (row, detail), = problems
    assert states[row][0] == violation.line
    assert detail == violation.detail
    assert report.stats["max_tilt_deg"] == pytest.approx(np.nanmax(total))


def test_tilt_limit_problems_report_heights_no_bed_can_reach(reference):
    problems, total = gcode_check.tilt_limit_problems(
        np.array([[85.0, 85.0, 85.0], [0.0, 900.0, 0.0]]), reference)
    assert np.isnan(total[1]) and total[0] == pytest.approx(0.0)
    assert problems == [(1, "screw heights no rigid bed can reach "
                            "(their differences are too large)")]


def test_tilt_limit_problems_follow_the_limit_shape(reference):
    box = dataclasses.replace(reference, tilt_limit_shape="box")
    odd = dataclasses.replace(reference, tilt_limit_shape="sphere")
    level = np.array([[85.0, 85.0, 85.0]])
    assert gcode_check.tilt_limit_problems(level, box)[0] == []
    with pytest.raises(ValueError, match="tilt_limit_shape"):
        gcode_check.tilt_limit_problems(level, odd)
