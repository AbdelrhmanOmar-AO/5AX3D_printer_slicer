"""The infill period and shell thickness as parameters (build plan task P1.3).

Upstream hard-codes both inside ``tools/sdf_to_isdf.py``: ``infill_period = 8``
and ``shell_thickness = 2``. They now come from the part's parameter file
(``infill_period``, ``shell_thickness``), through `tools/atomize.py`, to the
stage's ``--infill-period`` / ``--shell-thickness`` options. Without them the
stage uses exactly upstream's values, and the golden G-code is unchanged.

Units
-----
Both are in **deposition widths**, as `atom.fff3.sdf_generate_infill` uses
them: the gyroid repeats every ``infill_period`` widths, and the solid shell
under the surface is ``shell_thickness`` widths thick. With the calibration
cube's 0.9 mm width, the defaults are a 7.2 mm period and a 1.8 mm shell.

Values
------
The kernel takes floats, so fractional values are accepted. The period must be
above zero, since the kernel takes the position modulo it, and the shell must
be zero or more. No upper limit is set: these are print settings, not machine
numbers.
"""

from __future__ import annotations

import math

#: Upstream's values, from tools/sdf_to_isdf.py at commit e7b71ea. Kept as ints,
#: exactly as upstream passed them to the kernel.
DEFAULT_INFILL_PERIOD = 8
DEFAULT_SHELL_THICKNESS = 2


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def check_infill_period(value):
    """The infill period in deposition widths, if valid: a number above zero."""
    if _number("infill_period", value) <= 0:
        raise ValueError(f"infill_period must be above zero, got {value!r}")
    return value


def check_shell_thickness(value):
    """The shell thickness in deposition widths, if valid: a number of zero or more."""
    if _number("shell_thickness", value) < 0:
        raise ValueError(f"shell_thickness must be zero or more, got {value!r}")
    return value


def _format(value) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def infill_arguments(infill_period=None, shell_thickness=None) -> str:
    """The ``--infill-period`` / ``--shell-thickness`` options for ``sdf_to_isdf.py``.

    Empty when neither is given, so a parameter file without them runs exactly
    the upstream command. Values are checked here, when the parameter file is
    read, rather than at the stage.
    """
    arguments = ""
    if infill_period is not None:
        arguments += f" --infill-period {_format(check_infill_period(infill_period))}"
    if shell_thickness is not None:
        arguments += f" --shell-thickness {_format(check_shell_thickness(shell_thickness))}"
    return arguments
