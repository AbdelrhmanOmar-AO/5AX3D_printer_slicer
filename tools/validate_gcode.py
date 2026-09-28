"""Check a G-code file against a machine profile (build plan task P1.4).

The shared safety net for P3, P4, P5 and P7: axis ranges, bed tilt, NaN values,
feed rates, extrusion and retract/prime balance, and the header/footer. The
checks themselves live in `atom.gcode_check`; this is the command line.

Usage
-----
    python tools/validate_gcode.py data/gcode/calibration_cube.gcode
    python tools/validate_gcode.py <gcode> --machine ours --json
    python tools/validate_gcode.py <gcode> --max-feed 12000 --max-e 5

Exit code
---------
0 when the file passes, 1 when it has violations, 2 on a usage error (a
missing file, an unknown option). Scripts can rely on the exit code alone.

The machine is the ``--machine`` argument, else the ``ATOM_MACHINE``
environment variable, else ``reference``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atom import gcode_check, machine_profile

#: How many violations the human-readable output lists before summarising.
#: ``--json`` always carries all of them.
SHOWN_VIOLATIONS = 20


def format_report(report: gcode_check.Report) -> str:
    """A short human-readable summary of a report."""
    stats = report.stats
    lines = [
        f"File:    {stats.get('file', '(text)')}",
        f"Machine: {stats['profile']} ({stats['firmware_dialect']}, "
        f"{stats['tilt_limit_shape']} tilt limit of {stats['limits']['tilt_deg']:g} deg)",
        f"Lines:   {stats['line_count']}, moves: {stats['move_count']}",
    ]
    if stats["max_tilt_deg"] is not None:
        lines.append(f"Largest bed tilt: {stats['max_tilt_deg']:.3f} deg")
    if stats["max_feed"] is not None:
        lines.append(f"Largest feed rate: F{stats['max_feed']:g}")
    lines.append(
        f"Retracts / primes: {stats['retract_count']} / {stats['prime_count']}; "
        f"largest single extrusion {stats['largest_relative_e_mm']:g} mm"
    )
    lines.append("")

    if report.ok:
        lines.append("OK: no violations.")
        return "\n".join(lines)

    counts = ", ".join(f"{check} {count}" for check, count in stats["violation_counts"].items() if count)
    lines.append(f"FAILED: {len(report.violations)} violation(s) ({counts})")
    for violation in report.violations[:SHOWN_VIOLATIONS]:
        lines.append(f"  line {violation.line:>7}  {violation.id:<10}  {violation.detail}")
    hidden = len(report.violations) - SHOWN_VIOLATIONS
    if hidden > 0:
        lines.append(f"  ... and {hidden} more (use --json for all of them)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check a G-code file against a machine profile: axis ranges, bed "
            "tilt, NaN values, feed rates, extrusion, header and footer "
            "(build plan P1.4). Exits 0 when the file passes, 1 when it does not."
        )
    )
    parser.add_argument("gcode_path", type=Path, help="The .gcode file to check.")
    parser.add_argument(
        "--machine",
        default=None,
        help="Machine profile name or JSON path (default: $ATOM_MACHINE, else 'reference').",
    )
    parser.add_argument(
        "--max-feed",
        type=float,
        default=None,
        help="Upper limit for F in mm/min. Default: none, only F > 0 is checked.",
    )
    parser.add_argument(
        "--max-e",
        type=float,
        default=gcode_check.DEFAULT_MAX_E_MM,
        help=(
            "Largest single relative extrusion in mm of filament "
            f"(default {gcode_check.DEFAULT_MAX_E_MM:g})."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON instead of the summary.",
    )
    args = parser.parse_args(argv)

    if not args.gcode_path.is_file():
        parser.error(f"no such file: {args.gcode_path}")

    profile = machine_profile.load_profile(args.machine)
    report = gcode_check.check_file(
        args.gcode_path, profile, max_feed=args.max_feed, max_e_mm=args.max_e
    )

    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        print(format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
