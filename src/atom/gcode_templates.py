"""G-code header and footer, built from templates (build plan task P1.2).

Upstream Atomizer writes the header and footer as f-strings in
`atom.kinematics3z`, evaluated at import, with the temperatures hard-coded
(``M190 S55``, ``M104/M109 S210``). This module builds the same text from a
template chosen by the machine profile's firmware dialect, with the
temperatures as arguments.

``kinematics3z.HEADER`` and ``FOOTER`` keep their names and are now built here
with the defaults. The ``rrf`` template with the default temperatures is
**byte-identical** to the upstream text, as the golden test requires.
``tests/fixtures/gcode/header_rrf_reference.gcode`` and
``footer_rrf_reference.gcode`` hold the exact upstream strings.

Dialects
--------
``rrf``
    RepRapFirmware, the upstream dialect: ``G32`` homes and calibrates the bed,
    and ``M98 P"..."`` calls the macros that switch the three bed screws into
    3Z mode and back.
``klipper``
    Not written yet. Gate E1 (firmware, and how the three screws are exposed)
    is open, so building it raises ``NotImplementedError`` rather than
    emitting commands Klipper does not have.

Units: degrees Celsius for temperatures, millimetres for positions.
"""

from __future__ import annotations

import math

#: Upstream's temperatures, from the hard-coded header at commit e7b71ea.
DEFAULT_BED_TEMP_C = 55
DEFAULT_NOZZLE_TEMP_C = 210

#: The macro calls that switch the three bed screws on and off, per dialect.
#: `atom.gcode_check` checks a file's structure against these same strings.
MACRO_CALLS = {
    "rrf": {
        "enable_3z": 'M98 P"/macros/enable3Z.g"',
        "disable_3z": 'M98 P"/macros/disable3Z.g"',
    },
}

#: Upstream writes this line with a trailing space. It is part of the golden
#: G-code byte for byte, and editors strip trailing spaces from source files,
#: so the templates insert it from here instead of spelling it out.
WAIT_LINE = "M400 ; wait" + " "

_RRF_HEADER = """G21 ; set units to millimeters
G90 ; use absolute coordinates
M190 S{bed_temp} ; wait for bed temperature to be reached
G32 ; homing and bed calibration
M104 S{nozzle_temp} ; set temperature
M109 S{nozzle_temp} ; wait for temperature to be reached
T0
M82 ; use absolute distances for extrusion
; switch to enable 3Z mode
{enable_3z}
{wait}
; purging line
G92 E0
G1 Z{lift_z} U{lift_z} V{lift_z} F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z{line_z} U{line_z} V{line_z} F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z{line_z} U{line_z} V{line_z} F1000.0 E15 ; draw 1st line
G1 X0.4 Y200.0 Z{line_z} U{line_z} V{line_z} F1000.0 ; move to side a little
G1 X0.4 Y20 Z{line_z} U{line_z} V{line_z} F1000.0 E30 ; draw 2nd line
G1 E28.0 F2700 ; retract
; done purging extruder
M83 ; relative extrusion
"""

_RRF_FOOTER = """M82 ; absolute extrusion
G92 E0
G1 E-2.0 F2700 ; retract
G92 E0
M104 S0 ; turn off temperature
M140 S0
M106 S0    ; fan off
; switch to disable 3Z mode
{disable_3z}
{wait}
"""


def _gate_e1(dialect: str) -> NotImplementedError:
    return NotImplementedError(
        f"GATE E1: the G-code header and footer for firmware dialect {dialect!r} "
        "are not written yet. Only 'rrf' is implemented; the firmware and how "
        "the three bed screws are exposed are still open (build plan P1.5)."
    )


def format_temperature(value) -> str:
    """A temperature as a G-code ``S`` value: ``55`` for 55 or 55.0, ``60.5`` for 60.5.

    Must be a finite number, zero or above. Anything else is refused: a
    ``Snan`` or a negative target would reach the heaters.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"temperature must be a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"temperature must be a finite number of 0 or more, got {value!r}")
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def temperature_arguments(bed_temp=None, nozzle_temp=None) -> str:
    """The ``--bed-temp`` / ``--nozzle-temp`` options for ``toolpath_to_gcode.py``.

    Empty when neither is given, so a parameter file without temperatures
    produces exactly the upstream command, and therefore the upstream G-code.
    Values are checked here, before an hour of pipeline runs, rather than at
    the last stage.
    """
    arguments = ""
    if bed_temp is not None:
        arguments += f" --bed-temp {format_temperature(bed_temp)}"
    if nozzle_temp is not None:
        arguments += f" --nozzle-temp {format_temperature(nozzle_temp)}"
    return arguments


def make_header(profile, bed_temp=DEFAULT_BED_TEMP_C, nozzle_temp=DEFAULT_NOZZLE_TEMP_C) -> str:
    """The G-code header for a machine profile, with the given temperatures (deg C).

    The purge lines sit at the profile's ``z_offset``: all three screws at
    ``1.3 + z_offset`` mm to lift off the bed, then ``0.3 + z_offset`` for the
    lines, exactly as upstream wrote them.
    """
    dialect = profile.firmware_dialect
    if dialect != "rrf":
        raise _gate_e1(dialect)
    return _RRF_HEADER.format(
        bed_temp=format_temperature(bed_temp),
        nozzle_temp=format_temperature(nozzle_temp),
        enable_3z=MACRO_CALLS["rrf"]["enable_3z"],
        wait=WAIT_LINE,
        # Python's float formatting, as upstream's f-string used.
        lift_z=f"{1.3 + profile.z_offset}",
        line_z=f"{0.3 + profile.z_offset}",
    )


def make_footer(profile) -> str:
    """The G-code footer for a machine profile: heaters and fan off, 3Z mode off."""
    dialect = profile.firmware_dialect
    if dialect != "rrf":
        raise _gate_e1(dialect)
    return _RRF_FOOTER.format(disable_3z=MACRO_CALLS["rrf"]["disable_3z"], wait=WAIT_LINE)
