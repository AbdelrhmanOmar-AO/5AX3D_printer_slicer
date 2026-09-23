"""Tests for the templated G-code header and footer (build plan task P1.2).

The fixtures ``header_rrf_reference.gcode`` and ``footer_rrf_reference.gcode``
are upstream's text exactly, as evaluated from the f-strings in
``src/atom/kinematics3z.py`` at commit e7b71ea (reference profile). They carry
upstream's trailing space after ``M400 ; wait``; keep it if you edit them.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atom import gcode_check, gcode_templates, machine_profile

FIXTURES = Path(__file__).parent / "fixtures" / "gcode"
HEADER_FIXTURE = FIXTURES / "header_rrf_reference.gcode"
FOOTER_FIXTURE = FIXTURES / "footer_rrf_reference.gcode"


@pytest.fixture(scope="module")
def reference():
    return machine_profile.load_profile("reference")


def read_exact(path: Path) -> str:
    """The file's text, every character kept, line endings read as ``\\n``.

    Universal newlines on purpose: Git for Windows checks text files out with
    CRLF endings, and `toolpath_to_gcode.py` writes CRLF there too, while the
    templates use ``\\n``. Trailing spaces are kept either way.
    """
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Byte-identical to upstream with the defaults
# --------------------------------------------------------------------------


def test_rrf_header_with_defaults_is_upstreams(reference):
    assert gcode_templates.make_header(reference) == read_exact(HEADER_FIXTURE)


def test_rrf_footer_is_upstreams(reference):
    assert gcode_templates.make_footer(reference) == read_exact(FOOTER_FIXTURE)


def test_kinematics3z_header_and_footer_keep_their_names_and_text():
    """`tools/toolpath_to_gcode.py` still writes `kinematics3z.HEADER`/`FOOTER`."""
    pytest.importorskip("taichi")
    import atom.kinematics3z as k

    assert k.HEADER == read_exact(HEADER_FIXTURE)
    assert k.FOOTER == read_exact(FOOTER_FIXTURE)


def test_the_fixtures_keep_upstreams_trailing_space():
    """Editors strip trailing spaces; this one is part of the golden G-code."""
    for path in (HEADER_FIXTURE, FOOTER_FIXTURE):
        assert "M400 ; wait \n" in read_exact(path), path.name
    assert gcode_templates.WAIT_LINE == "M400 ; wait "


# --------------------------------------------------------------------------
# Temperatures land where expected, and nowhere else
# --------------------------------------------------------------------------


def test_temperatures_change_exactly_three_lines(reference):
    default = gcode_templates.make_header(reference).splitlines()
    warmer = gcode_templates.make_header(reference, bed_temp=60, nozzle_temp=215).splitlines()

    changed = [(old, new) for old, new in zip(default, warmer) if old != new]
    assert len(default) == len(warmer)
    assert changed == [
        ("M190 S55 ; wait for bed temperature to be reached", "M190 S60 ; wait for bed temperature to be reached"),
        ("M104 S210 ; set temperature", "M104 S215 ; set temperature"),
        ("M109 S210 ; wait for temperature to be reached", "M109 S215 ; wait for temperature to be reached"),
    ]


def test_each_temperature_is_independent(reference):
    header = gcode_templates.make_header(reference, bed_temp=70)
    assert "M190 S70 ;" in header and "M104 S210 ;" in header
    header = gcode_templates.make_header(reference, nozzle_temp=230)
    assert "M190 S55 ;" in header and "M109 S230 ;" in header


@pytest.mark.parametrize("value, text", [(55, "55"), (55.0, "55"), (60.5, "60.5"), (0, "0")])
def test_temperature_formatting(value, text):
    assert gcode_templates.format_temperature(value) == text


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "hot", None, True])
def test_impossible_temperatures_are_refused(reference, value):
    with pytest.raises(ValueError):
        gcode_templates.format_temperature(value)
    with pytest.raises(ValueError):
        gcode_templates.make_header(reference, bed_temp=value)


def test_purge_heights_follow_the_profile(reference):
    header = gcode_templates.make_header(dataclasses.replace(reference, z_offset=80.0))
    assert "G1 Z81.3 U81.3 V81.3 F500" in header
    assert "G1 X0.1 Y20 Z80.3 U80.3 V80.3 F1000.0" in header


def test_the_header_still_passes_the_validator(reference):
    """A body between a warmer header and the footer is still valid G-code."""
    body = (
        "G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0.000000 F3000\n"
        "G1 E2.00 F2700 ; prime\n"
        "G1 X151.0 Y145.0 Z85.0 U85.0 V85.0 E0.080000 F600\n"
    )
    text = (
        gcode_templates.make_header(reference, bed_temp=60, nozzle_temp=215)
        + body
        + gcode_templates.make_footer(reference)
    )
    report = gcode_check.check_text(text, reference)
    assert report.ok, report.violations


def test_the_validator_checks_the_same_macro_strings_the_header_writes():
    assert gcode_check.STRUCTURE_MARKERS is gcode_templates.MACRO_CALLS


# --------------------------------------------------------------------------
# Gate E1
# --------------------------------------------------------------------------


def test_klipper_is_gated(reference):
    klipper = dataclasses.replace(reference, firmware_dialect="klipper")
    with pytest.raises(NotImplementedError, match="GATE E1"):
        gcode_templates.make_header(klipper)
    with pytest.raises(NotImplementedError, match="GATE E1"):
        gcode_templates.make_footer(klipper)


# --------------------------------------------------------------------------
# The command line: atomize.py -> toolpath_to_gcode.py
# --------------------------------------------------------------------------


def test_no_temperatures_means_no_extra_arguments():
    """A parameter file without the keys gives upstream's command exactly."""
    assert gcode_templates.temperature_arguments() == ""
    assert gcode_templates.temperature_arguments(None, None) == ""


def test_temperature_arguments():
    assert gcode_templates.temperature_arguments(60, None) == " --bed-temp 60"
    assert gcode_templates.temperature_arguments(None, 215.5) == " --nozzle-temp 215.5"
    assert gcode_templates.temperature_arguments(60, 215) == " --bed-temp 60 --nozzle-temp 215"
    with pytest.raises(ValueError):
        gcode_templates.temperature_arguments(-1, None)


def test_atomize_reads_the_optional_keys(repo_root):
    """The keys are read in `Parameters.load` and reach the G-code command.

    Checked on the source text: importing `tools/atomize.py` starts Taichi,
    which would reset the runtime under every other test.
    """
    source = (repo_root / "tools" / "atomize.py").read_text(encoding="utf-8")
    assert 'param_dict.get("bed_temp")' in source
    assert 'param_dict.get("nozzle_temp")' in source
    assert "temperature_arguments(self.bed_temp, self.nozzle_temp)" in source
    assert "{params.gcode_path}{params.temperature_arguments}" in source


def _run_toolpath_to_gcode(repo_root, tmp_path, *options):
    """Convert the first 300 points of the golden toolpath, in its own process."""
    golden = dict(np.load(repo_root / "tests" / "golden" / "calibration_cube.toolpath.npz"))
    count = 300
    small = {
        key: (value[:count] if getattr(value, "ndim", 0) >= 1 else value)
        for key, value in golden.items()
    }
    small["point_count"] = np.array(count)
    toolpath = tmp_path / "small.npz"
    np.savez(toolpath, **small)
    gcode = tmp_path / "small.gcode"

    env = dict(os.environ, ATOM_TI_ARCH="cpu", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root / "src"), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "tools/toolpath_to_gcode.py", str(toolpath), str(gcode), *options],
        cwd=repo_root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=120,
    )
    return result, gcode


def test_toolpath_to_gcode_writes_the_given_temperatures(repo_root, tmp_path, reference):
    result, gcode = _run_toolpath_to_gcode(repo_root, tmp_path, "--bed-temp", "60", "--nozzle-temp", "215")
    assert result.returncode == 0, result.stderr[-2000:]
    text = read_exact(gcode)
    assert text.startswith(gcode_templates.make_header(reference, bed_temp=60, nozzle_temp=215))
    assert text.endswith(gcode_templates.make_footer(reference))


def test_toolpath_to_gcode_refuses_an_impossible_temperature(repo_root, tmp_path):
    result, gcode = _run_toolpath_to_gcode(repo_root, tmp_path, "--nozzle-temp", "-5")
    assert result.returncode != 0
    assert "temperature must be" in result.stderr
    assert not gcode.exists(), "no G-code may be written with a refused temperature"
