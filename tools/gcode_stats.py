"""Summarise a G-code file into a stable JSON record.

This is the measuring stick for the golden baseline (build plan task P0.2): run
it once on the output of the unmodified pipeline, commit the result, and every
later change can be shown to leave the G-code unchanged (or to change it only
where intended).

Usage
-----
    python tools/gcode_stats.py data/gcode/calibration_cube.gcode
    python tools/gcode_stats.py <gcode> -o tests/golden/calibration_cube.stats.json

Conventions used by this module
-------------------------------
Axis ranges
    Reported over the values **explicitly written in the file**, not over the
    modal state. A line that omits ``Z`` does not contribute to the ``Z`` range
    even though the machine's Z has not changed. This keeps the statistic a pure
    function of the file's text, which is what a baseline comparison needs.

Total extrusion
    Tracks the extrusion mode. ``M82`` selects absolute extrusion, ``M83``
    relative; ``G92 E<v>`` resets the absolute reference without extruding.
    In relative mode each positive ``E`` word adds to the total; in absolute
    mode each positive change in ``E`` does. Negative amounts (retractions) are
    counted separately rather than subtracted, so the total is filament pushed
    out of the nozzle.

Retract / prime lines
    A move that carries an ``E`` word and no axis word (X, Y, Z, U, V) moves
    only the extruder: negative is a retract, positive a prime. This matches how
    ``atom.kinematics3z`` emits ``G1 E-2.0 F2700 ; retract``.

Axis letters
    X, Y are the CoreXY head; Z, U, V are the three independent bed lead screws
    on the triple-Z machine; E is the extruder. See docs added in later tasks.

All distances are millimetres, matching the ``G21`` in the file header.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# The tokeniser is shared with the G-code validator (build plan P1.4), so the
# repository has one G-code word parser. `strip_comment` and `parse_words` stay
# importable from this module for existing callers (tools/visualize_5ax.py).
from atom.gcode_check import parse_words, strip_comment  # noqa: F401

#: Axis words tracked in the per-axis min/max table, in report order.
AXIS_WORDS: tuple[str, ...] = ("X", "Y", "Z", "U", "V", "E")

#: Axis words that move the machine rather than the extruder.
MOTION_WORDS: frozenset[str] = frozenset({"X", "Y", "Z", "U", "V"})

#: Number of leading/trailing lines kept verbatim in the record.
CONTEXT_LINES: int = 40


def compute_stats(text: str) -> dict[str, Any]:
    """Compute the statistics record for the contents of a G-code file."""
    lines = text.splitlines()

    axis_min: dict[str, float] = {}
    axis_max: dict[str, float] = {}
    axis_count: dict[str, int] = {axis: 0 for axis in AXIS_WORDS}

    command_counts: dict[str, int] = {}
    total_extrusion_mm = 0.0
    total_retraction_mm = 0.0
    retract_count = 0
    prime_count = 0

    # Upstream headers start in absolute extrusion (M82) and switch to relative
    # (M83) before printing; assume absolute until told otherwise.
    relative_extrusion = False
    last_e = 0.0

    for raw_line in lines:
        code = strip_comment(raw_line)
        if not code:
            continue

        words = parse_words(code)
        command_match = re.match(r"^([GM])(\d+)", code, re.IGNORECASE)
        if command_match:
            command = f"{command_match.group(1).upper()}{int(command_match.group(2))}"
            command_counts[command] = command_counts.get(command, 0) + 1
        else:
            command = None

        if command == "M82":
            relative_extrusion = False
            continue
        if command == "M83":
            relative_extrusion = True
            continue
        if command == "G92":
            # Resets the extruder reference without moving it.
            if "E" in words:
                last_e = words["E"]
            continue

        if command not in ("G0", "G1"):
            continue

        for axis in AXIS_WORDS:
            if axis not in words:
                continue
            value = words[axis]
            axis_count[axis] += 1
            axis_min[axis] = value if axis not in axis_min else min(axis_min[axis], value)
            axis_max[axis] = value if axis not in axis_max else max(axis_max[axis], value)

        if "E" in words:
            delta = words["E"] if relative_extrusion else words["E"] - last_e
            last_e = words["E"] if not relative_extrusion else last_e + words["E"]

            if delta > 0.0:
                total_extrusion_mm += delta
            elif delta < 0.0:
                total_retraction_mm += -delta

            if not (MOTION_WORDS & words.keys()):
                if delta < 0.0:
                    retract_count += 1
                elif delta > 0.0:
                    prime_count += 1

    axis_ranges = {
        axis: {
            "min": axis_min.get(axis),
            "max": axis_max.get(axis),
            "count": axis_count[axis],
        }
        for axis in AXIS_WORDS
    }

    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line_count": len(lines),
        "g0_count": command_counts.get("G0", 0),
        "g1_count": command_counts.get("G1", 0),
        "command_counts": dict(sorted(command_counts.items())),
        "axis_ranges": axis_ranges,
        "total_extrusion_mm": round(total_extrusion_mm, 6),
        "total_retraction_mm": round(total_retraction_mm, 6),
        "retract_count": retract_count,
        "prime_count": prime_count,
        "head_lines": lines[:CONTEXT_LINES],
        "tail_lines": lines[-CONTEXT_LINES:],
    }


def stats_for_file(path: Path) -> dict[str, Any]:
    """Compute the statistics record for the G-code file at ``path``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    stats = compute_stats(text)
    stats["file"] = path.name
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise a G-code file into a stable JSON record, used as the "
            "golden baseline for regression comparisons (build plan P0.2)."
        )
    )
    parser.add_argument("gcode_path", type=Path, help="Path to the .gcode file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the JSON here instead of to stdout.",
    )
    args = parser.parse_args(argv)

    if not args.gcode_path.is_file():
        parser.error(f"no such file: {args.gcode_path}")

    stats = stats_for_file(args.gcode_path)
    payload = json.dumps(stats, indent=2, sort_keys=True) + "\n"

    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
