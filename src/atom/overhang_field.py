"""The overhang rule of the overhang-aware field, in numpy (build plan P2.2).

Stock Atomizer constrains the tool-orientation field only on the first layer
and on upward-facing "ceilings" (`docs/orientation_field.md` section 2.2); no
downward-facing surface is ever constrained. P2.2 adds that constraint: on a
boundary cell whose surface faces down at a geometric overhang ``theta_geo``
steeper than ``max_overhang``, the tool is tilted **toward** the overhang by

    t = min(theta_geo - max_overhang + margin, max_tilt)

so its effective overhang ``theta_eff = theta_geo - t`` drops to
``max_overhang - margin`` where the tilt allows. The direction is spherical
``[theta, phi] = [t, atan2(n_y, n_x)]``: tilt ``t`` from +Z toward the
horizontal part of the surface's outward normal ``n``.

This module is the rule on its own, numpy only, so it can be tested and
reasoned about without Taichi. `atom.orientation_field` applies the same rule
inside the field's initialisation kernel, from the SDF's own normals;
`tests/test_orientation_field.py` checks that the kernel asks for this
module's tilt and azimuth on a ramp.

Units: degrees at the public interface, radians inside. The defaults are the
plan's placeholders for gate D1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ramp_in import DEFAULT_MAX_TILT_RATE_DEG_PER_MM, RampSettings

#: GATE D1: placeholder, team decision pending. The effective overhang the
#: rule aims for, and the threshold `tools/overhang_report.py` judges by.
DEFAULT_MAX_OVERHANG_DEG = 45.0
#: GATE D1: placeholder. Extra tilt beyond the bare minimum.
DEFAULT_MARGIN_DEG = 2.0
#: The operator's decision, 2026-09-30: hold the overhang constraints through
#: the field's 32 final smoothing passes unless a part says otherwise
#: (plan_corrections 7c P2-12).
DEFAULT_HOLD_OVERHANG = True


@dataclass(frozen=True)
class OverhangSettings:
    """The rule's parameters, degrees."""

    #: Surfaces steeper than this (from vertical) are constrained.
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG
    #: Added to the needed tilt.
    margin_deg: float = DEFAULT_MARGIN_DEG

    def __post_init__(self):
        if not 0.0 <= self.max_overhang_deg <= 90.0:
            raise ValueError(f"max_overhang_deg must be in [0, 90], got {self.max_overhang_deg}")
        if not 0.0 <= self.margin_deg <= 90.0:
            raise ValueError(f"margin_deg must be in [0, 90], got {self.margin_deg}")


def geometric_overhang_deg(normal) -> float:
    """``theta_geo`` of a surface with outward unit normal ``normal``, degrees.

    0 for a vertical wall, 90 for a flat ceiling facing straight down; 0 or
    less for a surface that does not face down. The same quantity as
    `overhang_metrics.geometric_overhang_angle_deg`.
    """
    n = np.asarray(normal, dtype=np.float64)
    return 90.0 - math.degrees(math.acos(float(np.clip(-n[2], -1.0, 1.0))))


@dataclass(frozen=True)
class OverhangTarget:
    """What the rule asks of one boundary cell."""

    #: Tilt from +Z, degrees: the ``theta`` of the constrained direction.
    tilt_deg: float
    #: Azimuth from +X, degrees: the ``phi``, toward the overhang.
    azimuth_deg: float
    #: The tilt the rule would ask for with no cap.
    wanted_deg: float

    @property
    def capped(self) -> bool:
        """True when ``max_tilt`` stopped the tilt short of what was wanted."""
        return self.tilt_deg < self.wanted_deg

    @property
    def spherical_rad(self) -> tuple[float, float]:
        """``[theta, phi]`` in radians, as the field stores it."""
        return math.radians(self.tilt_deg), math.radians(self.azimuth_deg)


def overhang_target(
    normal, max_tilt_deg: float, settings: OverhangSettings = OverhangSettings()
) -> OverhangTarget | None:
    """The constrained direction for a surface, or None if the rule does not act.

    ``normal`` is the surface's outward unit normal. The rule acts only on a
    downward-facing surface steeper than ``settings.max_overhang_deg``; a wall,
    an upward-facing surface or a gentle overhang gets None, as in stock
    Atomizer. ``max_tilt_deg`` caps the tilt: the machine's reach at that
    point, which P2.3's map will supply, and until then the field's budget.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n[2] >= 0.0:
        return None
    theta_geo = geometric_overhang_deg(n)
    if theta_geo <= settings.max_overhang_deg:
        return None
    wanted = theta_geo - settings.max_overhang_deg + settings.margin_deg
    return OverhangTarget(
        tilt_deg=min(wanted, float(max_tilt_deg)),
        azimuth_deg=math.degrees(math.atan2(n[1], n[0])),
        wanted_deg=wanted,
    )


def effective_overhang_deg(normal, tilt_deg: float, azimuth_deg: float) -> float:
    """``theta_eff`` of a surface printed with the tool at ``[tilt, azimuth]``.

    The P0.8 definition, ``90 - degrees(arccos(n . -d))``, for one surface.
    """
    t, p = math.radians(tilt_deg), math.radians(azimuth_deg)
    d = np.array([math.cos(p) * math.sin(t), math.sin(p) * math.sin(t), math.cos(t)])
    n = np.asarray(normal, dtype=np.float64)
    return 90.0 - math.degrees(math.acos(float(np.clip(n @ -d, -1.0, 1.0))))


# --------------------------------------------------------------------------
# The parameter-file keys (build plan P2.2/P2.6), as `tools/atomize.py` reads them
# --------------------------------------------------------------------------

#: Optional keys of a part's parameter file. Absent, the stage runs upstream's
#: field. The others need ``overhang_aware: true``.
PARAMETER_KEYS = (
    "overhang_aware", "max_overhang_deg", "overhang_margin_deg", "hold_overhang",
    "flat_overhangs_outward", "overhang_edges", "ramp_in", "max_tilt_rate_deg_per_mm",
)


def _flag(name: str, value) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


def _angle(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of degrees, got {value!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 90.0:
        raise ValueError(f"{name} must be between 0 and 90 degrees, got {value!r}")
    return float(value)


def _rate(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of degrees per mm, got {value!r}")
    try:
        return RampSettings(float(value)).max_tilt_rate_deg_per_mm
    except ValueError:
        raise ValueError(f"{name} must be a positive number of degrees per mm, got {value!r}") from None


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def overhang_arguments(
    overhang_aware=None,
    max_overhang_deg=None,
    overhang_margin_deg=None,
    hold_overhang=None,
    ramp_in=None,
    max_tilt_rate_deg_per_mm=None,
    overhang_edges=None,
    flat_overhangs_outward=None,
) -> str:
    """The stage options for ``compute_tool_orientations.py`` from the parameter keys.

    Empty when ``overhang_aware`` is absent or false, so a parameter file
    without the keys runs exactly upstream's command. Otherwise the rule's
    values are written out in full (defaults included, hold as
    ``--hold_overhang`` or ``--no_hold_overhang``, flat overhangs as
    ``--flat_overhangs_outward`` or ``--no_flat_overhangs_outward``, the edges as
    ``--overhang_edges`` or ``--no_overhang_edges``, the ramp-in of build
    plan P2.4 as ``--ramp_in --max_tilt_rate R`` or ``--no_ramp_in``), so the
    log shows what ran. Values are checked here, when the parameter file is read; a rule key
    given without ``overhang_aware: true`` is refused rather than ignored.
    """
    if not _flag("overhang_aware", overhang_aware):
        given = [
            name
            for name, value in (
                ("max_overhang_deg", max_overhang_deg),
                ("overhang_margin_deg", overhang_margin_deg),
                ("hold_overhang", hold_overhang),
                ("overhang_edges", overhang_edges),
                ("flat_overhangs_outward", flat_overhangs_outward),
                ("ramp_in", ramp_in),
                ("max_tilt_rate_deg_per_mm", max_tilt_rate_deg_per_mm),
            )
            if value is not None
        ]
        if given:
            raise ValueError(f"{', '.join(given)} need(s) \"overhang_aware\": true")
        return ""

    settings = OverhangSettings(
        DEFAULT_MAX_OVERHANG_DEG if max_overhang_deg is None else _angle("max_overhang_deg", max_overhang_deg),
        DEFAULT_MARGIN_DEG if overhang_margin_deg is None else _angle("overhang_margin_deg", overhang_margin_deg),
    )
    arguments = (
        f" --overhang_aware --max_overhang {_format(settings.max_overhang_deg)}"
        f" --overhang_margin {_format(settings.margin_deg)}"
    )
    hold = DEFAULT_HOLD_OVERHANG if hold_overhang is None else _flag("hold_overhang", hold_overhang)
    arguments += " --hold_overhang" if hold else " --no_hold_overhang"
    flat = True if flat_overhangs_outward is None else _flag("flat_overhangs_outward", flat_overhangs_outward)
    arguments += " --flat_overhangs_outward" if flat else " --no_flat_overhangs_outward"
    edges = True if overhang_edges is None else _flag("overhang_edges", overhang_edges)
    arguments += " --overhang_edges" if edges else " --no_overhang_edges"

    if _flag("ramp_in", True if ramp_in is None else ramp_in):
        rate = DEFAULT_MAX_TILT_RATE_DEG_PER_MM
        if max_tilt_rate_deg_per_mm is not None:
            rate = _rate("max_tilt_rate_deg_per_mm", max_tilt_rate_deg_per_mm)
        arguments += f" --ramp_in --max_tilt_rate {_format(rate)}"
    elif max_tilt_rate_deg_per_mm is not None:
        raise ValueError('max_tilt_rate_deg_per_mm needs "ramp_in" left on')
    else:
        arguments += " --no_ramp_in"
    return arguments
