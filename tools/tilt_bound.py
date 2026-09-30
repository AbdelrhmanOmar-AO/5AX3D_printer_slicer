"""The steepest overhang a tilting bed can print, by analysis (build plan P2.1).

Tilting the tool by ``t`` toward an overhang lowers its effective angle by
``t``: ``theta_eff = theta_geo - t``, the relation the P0.8 baseline measured
with the sign the wrong way round. A surface prints without support while
``theta_eff <= max_overhang``, so

    steepest printable overhang = max_overhang + usable tilt   (at most 90)

and a surface at ``theta_geo`` needs ``theta_geo - max_overhang`` of tilt. At
the placeholder 45-degree threshold a flat ledge (90 degrees) needs 45.

**Usable tilt** is the smaller of the machine's limit and the nozzle's.
Atomizer never tilts the field further than ``90 - NOZZLE_HALF_ANGLE_DEG`` =
50 degrees (`fff3.MAX_SLOPE_ANGLE`): beyond that a flat layer puts its own
earlier beads inside the nozzle cone (plan_corrections P4-2).

**Margin.** P2.2 asks for ``theta_geo - max_overhang + margin`` of tilt, and
P2.5 counts a part toward its success criteria only when
``theta_geo <= max_overhang + usable tilt - margin``. Parts between that and
the bound are shown as "edge": physically inside, but not counted.

This is an upper limit, not a prediction. It assumes the full tilt is
available where the overhang is (gate M2 (c): the reference machine loses
tilt near the bed, plan_corrections P4-3), that the field reaches it before
the overhang starts (P2.4), and the placeholder threshold (gates D0, D1).

Usage::

    python tools/tilt_bound.py                          # the machine profile's tilt
    python tools/tilt_bound.py --max-overhang 45 --tilt 30
    python tools/tilt_bound.py --tilt 30 40 45 --tshape-underside 70
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom import clearance  # noqa: E402
from atom import machine_profile  # noqa: E402

#: GATE D0/D1: placeholder, team decision pending. The effective-angle
#: threshold `tools/overhang_report.py` judges by (a test keeps them equal).
DEFAULT_MAX_OVERHANG_DEG = 45.0

#: GATE D1: placeholder. P2.2's ``margin_deg`` and P2.5's "- 2 degrees".
DEFAULT_MARGIN_DEG = 2.0

#: The tilt Atomizer's field can never exceed, whatever the machine allows:
#: ``(180 - NOZZLE_CONE_ANGLE) / 2`` in `fff3`, from the same cone as
#: `atom.clearance`.
NOZZLE_TILT_CAP_DEG = 90.0 - clearance.NOZZLE_HALF_ANGLE_DEG

INSIDE = "inside"
EDGE = "edge"
OUT_OF_RANGE = "out of range"
NO_OVERHANG = "no overhang"


def usable_tilt_deg(machine_tilt_deg: float) -> float:
    """The tilt the field can actually use, in degrees: the machine's, capped by the nozzle."""
    return min(float(machine_tilt_deg), NOZZLE_TILT_CAP_DEG)


def steepest_printable_overhang_deg(max_overhang_deg: float, tilt_deg: float) -> float:
    """``max_overhang + usable tilt``, at most 90 degrees (a flat ledge)."""
    return min(90.0, float(max_overhang_deg) + usable_tilt_deg(tilt_deg))


def required_tilt_deg(theta_geo_deg: float, max_overhang_deg: float) -> float:
    """Tilt toward the overhang that brings ``theta_geo`` down to the threshold."""
    return max(0.0, float(theta_geo_deg) - float(max_overhang_deg))


def classify(
    theta_geo_deg: float,
    max_overhang_deg: float,
    tilt_deg: float,
    margin_deg: float = DEFAULT_MARGIN_DEG,
) -> str:
    """`INSIDE`, `EDGE` or `OUT_OF_RANGE` for an overhang at ``theta_geo_deg``.

    `INSIDE` is P2.5's criterion, ``theta_geo <= max_overhang + usable tilt -
    margin``: the tilt P2.2 asks for, margin included, is available. `EDGE`
    is inside the bound only without the margin.
    """
    # Uncapped on purpose: the 90-degree cap is a fact about surfaces, not
    # about tilt, so 45 + 50 - 2 = 93 still counts a flat ledge as inside.
    reach = float(max_overhang_deg) + usable_tilt_deg(tilt_deg)
    if theta_geo_deg <= max_overhang_deg:
        return INSIDE
    if theta_geo_deg <= reach - margin_deg:
        return INSIDE
    if theta_geo_deg <= reach:
        return EDGE
    return OUT_OF_RANGE


def counted_up_to_deg(max_overhang_deg: float, tilt_deg: float, margin_deg: float) -> float:
    """The steepest overhang P2.5 counts: ``max_overhang + usable tilt - margin``, at most 90."""
    return min(90.0, float(max_overhang_deg) + usable_tilt_deg(tilt_deg) - float(margin_deg))


def benchmark_overhangs(tshape_underside_deg: float | None = None):
    """``[(part, theta_geo or None), ...]`` for the P0.7 benchmark parts.

    Read from `atom.benchmark_meshes`, so a change to the parts shows up
    here: the ramps' angles, and the T-shape's underside at its default (90,
    gate D0) unless ``tshape_underside_deg`` is given.
    """
    from atom import benchmark_meshes as bm

    if tshape_underside_deg is None:
        tshape_underside_deg = (
            inspect.signature(bm.make_tshape).parameters["underside_angle_deg"].default
        )
    parts = [(f"ramp{angle}", float(angle)) for angle in bm.RAMP_ANGLES]
    parts.append(("tshape", float(tshape_underside_deg)))
    parts.append(("twin_domes", None))
    return parts


def table(max_overhang_deg, tilts_deg, margin_deg, parts):
    """The report as lines of text."""
    lines = [
        "Steepest printable overhang = max overhang + usable tilt (build plan P2.1)",
        f"  max overhang {max_overhang_deg:g} deg (placeholder, gates D0/D1), "
        f"margin {margin_deg:g} deg (placeholder, gate D1)",
        f"  nozzle cap {NOZZLE_TILT_CAP_DEG:g} deg: Atomizer never tilts the field "
        "further, whatever the machine allows",
        "",
        f"  {'machine tilt':<14}{'usable':<9}{'steepest printable':<21}counted by P2.5 up to",
    ]
    for tilt in tilts_deg:
        bound = steepest_printable_overhang_deg(max_overhang_deg, tilt)
        counted = counted_up_to_deg(max_overhang_deg, tilt, margin_deg)
        lines.append(f"  {tilt:<14g}{usable_tilt_deg(tilt):<9g}{bound:<21g}{counted:g}")

    lines += ["", "  A flat ledge (90 deg) needs "
              f"{required_tilt_deg(90.0, max_overhang_deg):g} deg of usable tilt "
              f"({required_tilt_deg(90.0, max_overhang_deg) + margin_deg:g} with the margin).", ""]

    header = f"  {'part':<12}{'theta_geo':<11}{'tilt needed (+margin)':<23}" + "".join(
        f"{f'at {t:g} deg':<16}" for t in tilts_deg
    )
    lines.append(header)
    for name, theta in parts:
        if theta is None:
            lines.append(
                f"  {name:<12}{'-':<11}{'-':<23}" + "".join(f"{NO_OVERHANG:<16}" for _ in tilts_deg)
            )
            continue
        needed = required_tilt_deg(theta, max_overhang_deg)
        asked = needed + margin_deg if needed > 0 else 0.0
        needed_text = f"{needed:g} ({asked:g})"
        cells = "".join(
            f"{classify(theta, max_overhang_deg, t, margin_deg):<16}" for t in tilts_deg
        )
        lines.append(f"  {name:<12}{theta:<11g}{needed_text:<23}{cells}")

    lines += [
        "",
        "  inside       : counted in P2.5's success criteria",
        "  edge         : inside the bound only without the margin; not counted",
        "  out of range : no field can print it support-free at this tilt; P2.5 reports it,",
        "                 never passes it",
        "",
        "  An upper limit: it assumes the full tilt is usable where the overhang is (gate M2)",
        "  and reached before the overhang starts (P2.4).",
    ]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-overhang", type=float, default=DEFAULT_MAX_OVERHANG_DEG,
        help="Largest printable effective overhang, degrees from vertical "
        f"(default {DEFAULT_MAX_OVERHANG_DEG:g}, a gate D0/D1 placeholder).",
    )
    parser.add_argument(
        "--tilt", type=float, nargs="+", default=None,
        help="The machine's tilt limit(s) in degrees, one column each. Default: "
        "the machine profile's max_tilt_angle_deg (ATOM_MACHINE, else reference).",
    )
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN_DEG,
        help=f"Margin in degrees (default {DEFAULT_MARGIN_DEG:g}, a gate D1 placeholder).",
    )
    parser.add_argument(
        "--tshape-underside", type=float, default=None,
        help="The T-shape's underside angle to assess (default: the benchmark's, 90).",
    )
    args = parser.parse_args(argv)

    tilts = args.tilt
    if tilts is None:
        profile = machine_profile.load_profile()
        tilts = [profile.max_tilt_angle_deg]
        print(f"Tilt limit from machine profile '{profile.name}': {profile.max_tilt_angle_deg:g} deg\n")
    for value in [args.max_overhang, args.margin, *tilts]:
        if not 0.0 <= value <= 90.0:
            parser.error(f"angles must be between 0 and 90 degrees, got {value:g}")

    lines = table(args.max_overhang, tilts, args.margin, benchmark_overhangs(args.tshape_underside))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
