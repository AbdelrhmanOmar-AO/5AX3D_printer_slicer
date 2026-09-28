"""G-code validator: the shared safety net for P3, P4, P5 and P7 (build plan P1.4).

Reads a G-code file line by line and checks it against a machine profile. It
never changes the file. Every problem is reported as a violation with a stable
ID, the 1-based line number, and a readable detail, so a report can be diffed
between runs and a test can assert on exactly one ID.

Checks
------
``AXIS_RANGE``
    X within ``[0, max_x_axis]``, Y within ``[0, max_y_axis]``, and each screw
    (Z, U, V) within ``[0, max_z_axis]``: the same limits
    ``kinematics3z.inverse`` enforces. The header's purge lines are real moves
    and are checked like any other.
``TILT_LIMIT``
    The bed tilt implied by (Z, U, V), from `atom.screw_tilt`, within
    ``max_tilt_angle_deg`` under the profile's ``tilt_limit_shape``. Uses the
    same 1e-4 degree tolerance as ``inverse``.
``NAN``
    No NaN or infinite value in any word. Checked on the raw text, because
    ``Xnan`` is not a number the word parser would read: without this check it
    would be silently dropped and the axis would keep its previous value.
``FEED``
    Every F word a finite number above zero, and at most ``max_feed`` when one
    is given. **No upper limit by default**: the machine profile has no maximum
    feed rate, and the golden G-code legitimately reaches F10725, because
    ``toolpath_to_gcode`` scales the feed by machine distance over tool-tip
    distance (``docs/plan_corrections.md`` 7a, P1-2).
``EXTRUSION``
    In relative mode (``M83``), no single E above ``max_e_mm`` (default 5 mm,
    agreed with the operator 2026-09-23; the golden file's largest is a 2 mm
    prime). Retracts and primes must balance: no prime without a pending
    retract, no prime larger than what was retracted, and no extruding move
    while the filament is still retracted.
``STRUCTURE``
    The header and footer are present: ``G21`` and ``G90`` before the first
    move, the 3Z enable macro before the first U or V word, and the disable
    macro after the last one. Only the ``rrf`` dialect is defined; ``klipper``
    raises until gate E1 is answered.

Parsing
-------
The tokeniser (`strip_comment`, `parse_words`) lives here and
``tools/gcode_stats.py`` imports it, so there is one G-code word parser in the
repository. It removes double-quoted strings before reading words: the
``P"/macros/enable3Z.g"`` of an RRF macro call otherwise yields an ``E3`` word
(``docs/plan_corrections.md`` 4.1). Motion words are modal: a move that omits
an axis keeps its last value.

Units: millimetres, mm/min for F, degrees for tilt. Axis letters: X, Y the
CoreXY head, Z, U, V the three bed screws (``z0``, ``z1``, ``z2`` of
``kinematics3z.inverse``, ``docs/conventions.md`` section 4), E the extruder.

Taichi is not needed: this module is numpy only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import gcode_templates, screw_tilt, tilt

# --------------------------------------------------------------------------
# Tokeniser (shared with tools/gcode_stats.py)
# --------------------------------------------------------------------------

#: A G-code word: a letter followed by a number (``X12.5``, ``E-2.0``, ``F2700``).
WORD_RE = re.compile(r"(?P<letter>[A-Za-z])(?P<value>-?\d*\.?\d+(?:[eE][-+]?\d+)?)")

#: A double-quoted string parameter, as used by RepRapFirmware macro calls.
#: These are removed before word parsing: the path in
#: ``M98 P"/macros/enable3Z.g"`` otherwise parses as an ``E3`` word and
#: corrupts the extrusion totals.
QUOTED_RE = re.compile(r'"[^"]*"')

#: A word whose value is not a finite number: what Python writes for a NaN or
#: an infinity (``Xnan``, ``E-inf``). `WORD_RE` does not match these at all.
NON_FINITE_WORD_RE = re.compile(
    r"(?<![A-Za-z])(?P<letter>[A-Za-z])(?P<value>[-+]?(?:nan|inf(?:inity)?))(?![A-Za-z0-9])",
    re.IGNORECASE,
)

#: A command at the start of a line: ``G1``, ``M83``, ``T0`` ...
COMMAND_RE = re.compile(r"^([GMT])(\d+)", re.IGNORECASE)


def strip_comment(line: str) -> str:
    """Return ``line`` with any ``;`` comment and surrounding whitespace removed."""
    return line.split(";", 1)[0].strip()


def parse_words(code: str) -> dict[str, float]:
    """Parse the G-code words of one comment-free line into ``{letter: value}``.

    Later occurrences of the same letter win, which matches how firmware reads a
    malformed line. Quoted string parameters (e.g. the ``P"/macros/enable3Z.g"``
    of an RRF macro call) are removed first, so text inside them is never read
    as a word.
    """
    unquoted = QUOTED_RE.sub("", code)
    return {
        match.group("letter").upper(): float(match.group("value"))
        for match in WORD_RE.finditer(unquoted)
    }


def parse_command(code: str) -> str | None:
    """The command of a comment-free line, normalised (``g01`` -> ``G1``)."""
    match = COMMAND_RE.match(code)
    if not match:
        return None
    return f"{match.group(1).upper()}{int(match.group(2))}"


# --------------------------------------------------------------------------
# Check IDs and the report
# --------------------------------------------------------------------------

AXIS_RANGE = "AXIS_RANGE"
TILT_LIMIT = "TILT_LIMIT"
NAN = "NAN"
FEED = "FEED"
EXTRUSION = "EXTRUSION"
STRUCTURE = "STRUCTURE"

#: Every check, in report order.
CHECK_IDS: tuple[str, ...] = (AXIS_RANGE, TILT_LIMIT, NAN, FEED, EXTRUSION, STRUCTURE)

#: Default for the largest single relative extrusion, in mm of filament.
#: Agreed with the operator on 2026-09-23. The golden cube's largest is a
#: 2.0 mm prime; its printing moves are below 0.1 mm.
DEFAULT_MAX_E_MM = 5.0

#: The tilt tolerance ``kinematics3z.inverse`` applies, in degrees.
TILT_TOLERANCE_DEG = 1e-4

#: Retract and prime amounts are written with two decimals; below this they
#: count as equal.
EXTRUSION_TOLERANCE_MM = 1e-6

#: The macro calls that switch the three screws on and off, per dialect: the
#: very strings `atom.gcode_templates` writes into the header and footer.
#: Klipper is gate E1.
STRUCTURE_MARKERS = gcode_templates.MACRO_CALLS

SCREW_WORDS = ("Z", "U", "V")
MOTION_WORDS = frozenset({"X", "Y", "Z", "U", "V"})


@dataclass
class Violation:
    """One problem: which check, where (1-based line number), and what."""

    id: str
    line: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "line": self.line, "detail": self.detail}


@dataclass
class Report:
    """The outcome of one validation. ``ok`` is True when there are no violations."""

    violations: list[Violation] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def ids(self) -> set[str]:
        """The distinct check IDs that fired."""
        return {violation.id for violation in self.violations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [violation.to_dict() for violation in self.violations],
            "stats": self.stats,
        }


# --------------------------------------------------------------------------
# The validator
# --------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def check_lines(
    lines: Iterable[str],
    profile,
    *,
    max_feed: float | None = None,
    max_e_mm: float = DEFAULT_MAX_E_MM,
) -> Report:
    """Validate G-code text, given as lines, against a machine profile.

    Parameters
    ----------
    lines
        The file's lines, with or without trailing newlines.
    profile
        An `atom.machine_profile.MachineProfile`. Its axis ranges, tilt limit,
        tilt-limit shape, ball geometry and firmware dialect are used.
    max_feed
        Upper limit for F, in mm/min. ``None`` (the default) checks only that
        F is finite and positive; see the module docstring for why.
    max_e_mm
        Largest single relative extrusion, in mm of filament.
    """
    dialect = profile.firmware_dialect
    if dialect not in STRUCTURE_MARKERS:
        raise NotImplementedError(
            f"GATE E1: the G-code structure for firmware dialect {dialect!r} is not "
            "defined yet. Only 'rrf' is implemented."
        )
    markers = STRUCTURE_MARKERS[dialect]

    limits = {
        "X": (0.0, float(profile.max_x_axis)),
        "Y": (0.0, float(profile.max_y_axis)),
        "Z": (0.0, float(profile.max_z_axis)),
        "U": (0.0, float(profile.max_z_axis)),
        "V": (0.0, float(profile.max_z_axis)),
    }

    violations: list[Violation] = []
    state: dict[str, float] = {}

    relative_extrusion = False
    last_absolute_e = 0.0
    retracted_mm = 0.0
    retract_count = 0
    prime_count = 0
    largest_relative_e = 0.0
    feeds: list[float] = []

    # Screw states to tilt-check after the scan, vectorised: (line, z0, z1, z2).
    screw_lines: list[int] = []
    screw_states: list[tuple[float, float, float]] = []

    first_move_line: int | None = None
    first_screw_line: int | None = None
    last_screw_line: int | None = None
    enable_lines: list[int] = []
    disable_lines: list[int] = []
    units_line: int | None = None
    absolute_line: int | None = None
    line_count = 0
    move_count = 0

    for number, raw in enumerate(lines, start=1):
        line_count = number
        code = strip_comment(raw.rstrip("\r\n"))
        if not code:
            continue

        unquoted = QUOTED_RE.sub("", code)
        for match in NON_FINITE_WORD_RE.finditer(unquoted):
            violations.append(Violation(
                NAN, number,
                f"{match.group('letter').upper()} is {match.group('value')}, not a finite number",
            ))

        if markers["enable_3z"] in code:
            enable_lines.append(number)
        if markers["disable_3z"] in code:
            disable_lines.append(number)

        command = parse_command(code)
        words = parse_words(code)

        if command == "G21" and units_line is None:
            units_line = number
        elif command == "G90" and absolute_line is None:
            absolute_line = number
        elif command == "M82":
            relative_extrusion = False
            continue
        elif command == "M83":
            relative_extrusion = True
            continue
        elif command == "G92":
            if "E" in words:
                last_absolute_e = words["E"]
            continue

        if command not in ("G0", "G1"):
            continue

        move_count += 1
        if first_move_line is None:
            first_move_line = number

        # --- FEED ---
        if "F" in words:
            feed = words["F"]
            feeds.append(feed)
            if not feed > 0.0:
                violations.append(Violation(FEED, number, f"F{_fmt(feed)} is not above zero"))
            elif max_feed is not None and feed > max_feed:
                violations.append(Violation(
                    FEED, number, f"F{_fmt(feed)} exceeds the limit of {_fmt(max_feed)} mm/min",
                ))

        # --- AXIS_RANGE, and the modal state ---
        for axis in MOTION_WORDS & words.keys():
            value = words[axis]
            low, high = limits[axis]
            if not low <= value <= high:
                violations.append(Violation(
                    AXIS_RANGE, number,
                    f"{axis}{_fmt(value)} is outside [{_fmt(low)}, {_fmt(high)}]",
                ))
            state[axis] = value

        if words.keys() & {"U", "V"}:
            if first_screw_line is None:
                first_screw_line = number
            last_screw_line = number

        if words.keys() & set(SCREW_WORDS) and all(axis in state for axis in SCREW_WORDS):
            screw_lines.append(number)
            screw_states.append(tuple(state[axis] for axis in SCREW_WORDS))

        # --- EXTRUSION ---
        if "E" not in words:
            continue
        if relative_extrusion:
            delta = words["E"]
            if delta > largest_relative_e:
                largest_relative_e = delta
            if delta > max_e_mm:
                violations.append(Violation(
                    EXTRUSION, number,
                    f"E{_fmt(delta)} extrudes more than {_fmt(max_e_mm)} mm in one move",
                ))
        else:
            delta = words["E"] - last_absolute_e
            last_absolute_e = words["E"]

        moves_axes = bool(MOTION_WORDS & words.keys())
        if not moves_axes and delta < 0.0:
            retract_count += 1
            retracted_mm += -delta
        elif not moves_axes and delta > 0.0:
            prime_count += 1
            if retracted_mm <= EXTRUSION_TOLERANCE_MM:
                violations.append(Violation(
                    EXTRUSION, number, f"prime of {_fmt(delta)} mm with no retract pending",
                ))
            elif delta > retracted_mm + EXTRUSION_TOLERANCE_MM:
                violations.append(Violation(
                    EXTRUSION, number,
                    f"prime of {_fmt(delta)} mm is more than the {_fmt(retracted_mm)} mm retracted",
                ))
            retracted_mm = max(0.0, retracted_mm - delta)
        elif moves_axes and delta > 0.0 and retracted_mm > EXTRUSION_TOLERANCE_MM:
            violations.append(Violation(
                EXTRUSION, number,
                f"extruding move while {_fmt(retracted_mm)} mm of filament is still retracted",
            ))

    # --- TILT_LIMIT, vectorised over every screw state ---
    max_tilt_deg = None
    if screw_states:
        screws = np.asarray(screw_states, dtype=float)
        total = screw_tilt.total_tilt_deg(screws[:, 0], screws[:, 1], screws[:, 2], profile)
        limit = float(profile.max_tilt_angle_deg)
        finite = np.isfinite(total)
        for index in np.flatnonzero(~finite):
            violations.append(Violation(
                TILT_LIMIT, screw_lines[index],
                "screw heights no rigid bed can reach (their differences are too large)",
            ))
        if profile.tilt_limit_shape == "cone":
            for index in np.flatnonzero(finite & (total > limit + TILT_TOLERANCE_DEG)):
                violations.append(Violation(
                    TILT_LIMIT, screw_lines[index],
                    f"bed tilt {total[index]:.4f} deg exceeds the {_fmt(limit)} deg cone limit",
                ))
        elif profile.tilt_limit_shape == "box":
            # A box can only be exceeded once the total tilt exceeds the limit,
            # so the direction is needed only for those states.
            suspects = np.flatnonzero(finite & (total > limit + TILT_TOLERANCE_DEG))
            directions = screw_tilt.build_direction(
                screws[suspects, 0], screws[suspects, 1], screws[suspects, 2], profile
            )
            for index, direction in zip(suspects, directions):
                a_deg, b_deg = tilt.tilts_from_direction(direction)
                if max(abs(a_deg), abs(b_deg)) > limit + TILT_TOLERANCE_DEG:
                    violations.append(Violation(
                        TILT_LIMIT, screw_lines[index],
                        f"bed tilt (a {a_deg:.4f}, b {b_deg:.4f}) deg exceeds the "
                        f"{_fmt(limit)} deg box limit",
                    ))
        else:
            raise ValueError(f"unknown tilt_limit_shape {profile.tilt_limit_shape!r}")
        max_tilt_deg = float(total[finite].max()) if finite.any() else None

    # --- STRUCTURE ---
    if first_move_line is not None:
        for name, found in (("G21", units_line), ("G90", absolute_line)):
            if found is None or found > first_move_line:
                violations.append(Violation(
                    STRUCTURE, first_move_line, f"header incomplete: no {name} before the first move",
                ))
    if first_screw_line is not None:
        if not enable_lines or enable_lines[0] > first_screw_line:
            violations.append(Violation(
                STRUCTURE, first_screw_line,
                f"first U/V move comes before the 3Z enable macro ({markers['enable_3z']})",
            ))
        if not disable_lines or disable_lines[-1] < last_screw_line:
            violations.append(Violation(
                STRUCTURE, last_screw_line,
                f"footer incomplete: no 3Z disable macro ({markers['disable_3z']}) after the last U/V move",
            ))
    elif move_count == 0:
        violations.append(Violation(STRUCTURE, max(line_count, 1), "no moves in the file"))

    violations.sort(key=lambda violation: (violation.line, CHECK_IDS.index(violation.id)))

    counts = {check: 0 for check in CHECK_IDS}
    for violation in violations:
        counts[violation.id] += 1

    stats = {
        "profile": profile.name,
        "firmware_dialect": dialect,
        "tilt_limit_shape": profile.tilt_limit_shape,
        "line_count": line_count,
        "move_count": move_count,
        "max_tilt_deg": max_tilt_deg,
        "max_feed": max(feeds) if feeds else None,
        "largest_relative_e_mm": largest_relative_e,
        "retract_count": retract_count,
        "prime_count": prime_count,
        "limits": {
            "tilt_deg": float(profile.max_tilt_angle_deg),
            "max_feed": max_feed,
            "max_e_mm": max_e_mm,
            "axes": {axis: list(bounds) for axis, bounds in limits.items()},
        },
        "violation_counts": counts,
    }
    return Report(violations=violations, stats=stats)


def check_text(text: str, profile, **options) -> Report:
    """Validate G-code given as one string. Options as in `check_lines`."""
    return check_lines(text.splitlines(), profile, **options)


def check_file(path, profile, **options) -> Report:
    """Validate a G-code file, streaming it line by line. Options as in `check_lines`."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        report = check_lines(handle, profile, **options)
    report.stats["file"] = str(path)
    return report
