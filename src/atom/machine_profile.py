"""Machine constants as swappable configuration (build plan task P0.5).

Upstream Atomizer hard-codes one machine's geometry as module-level literals at
the top of `atom.kinematics3z`. This module moves those numbers into JSON
profiles under ``config/machines/`` so a different machine is a config change
rather than a code edit.

Which profile is active
-----------------------
The ``ATOM_MACHINE`` environment variable names it; the default is
``reference``.

* ``reference`` - the upstream machine, values copied exactly from the original
  ``kinematics3z.py`` constants block. Used by the golden test, and kept in the
  test matrix permanently so regressions stay detectable.
* ``ours`` - our printer. **Every number in it is a placeholder copied from the
  reference machine** until the mechanical design is finished (gates M1 and M2).
  Loading it warns loudly.

Units
-----
Millimetres for length, degrees for angles, mm/min for feed rates, matching the
upstream constants and the G-code the pipeline emits.

Types
-----
Values are returned with the exact Python types the upstream literals had: the
axis limits and feed rates were integers, the geometry floats. Taichi captures
these as globals when it compiles a kernel, so preserving the types keeps the
generated code, and therefore the golden G-code, identical.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

#: Where the shipped profiles live, relative to the repository root.
#: This resolves correctly for an editable install (`pip install -e .`), which
#: is how README.md and scripts/setup_laptop.ps1 install the package. A plain
#: `pip install .` would not copy config/ into site-packages, and loading any
#: profile then fails with a message listing what it could find.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "machines"

#: Environment variable naming the active profile.
ENV_VAR = "ATOM_MACHINE"

#: Used when the environment variable is unset.
DEFAULT_PROFILE = "reference"

#: `status` values. A PLACEHOLDER profile has numbers nobody has measured.
STATUS_VERIFIED = "verified"
STATUS_PLACEHOLDER = "PLACEHOLDER"
VALID_STATUS = frozenset({STATUS_VERIFIED, STATUS_PLACEHOLDER})

#: G-code dialects. The upstream header is RepRapFirmware; Klipper support is
#: gated on E1 and not implemented.
VALID_FIRMWARE = frozenset({"rrf", "klipper"})

#: Shape of the tilt limit: a cone limits total tilt in any direction, a box
#: limits each axis independently. GATE M2: which one our machine has is a
#: mechanical question, unanswered.
VALID_TILT_LIMIT_SHAPE = frozenset({"cone", "box"})


class PlaceholderProfileWarning(UserWarning):
    """Raised when a profile whose numbers nobody has measured is loaded."""


@dataclass(frozen=True)
class MachineProfile:
    """Every machine-specific constant the pipeline reads.

    Field names mirror the upstream constants in `kinematics3z`, lower-cased.
    Two upstream spellings are corrected here and mapped back at the point of
    use: ``nozzle_to_gantry`` (upstream ``NOZZLE_TO_GAUNTRY``) and
    ``deposition_feedrate`` (upstream ``DEPOSITON_FEEDRATE``).
    """

    name: str
    status: str
    firmware_dialect: str
    tilt_limit_shape: str

    # --- Bed geometry: the three ball joints and their rails ---
    ball_2dpos_0: tuple[float, float]
    ball_2dpos_1: tuple[float, float]
    ball_2dpos_2: tuple[float, float]
    ball_z: float
    rail_angle_0: float
    rail_angle_1: float
    rail_angle_2: float
    z_offset: float

    # --- Travel and clearance limits ---
    max_tilt_angle_deg: float
    max_x_axis: float
    max_y_axis: float
    max_z_axis: float
    ball_to_corner: float
    nozzle_to_gantry: float

    # --- Extrusion and motion ---
    filament_diameter: float
    deposition_feedrate: float
    travel_feedrate: float
    z_fan_on: float
    retract_thresh: float
    retract_length: float
    retract_speed: float

    @property
    def is_placeholder(self) -> bool:
        return self.status == STATUS_PLACEHOLDER


def _validate(data: dict[str, Any], source: str) -> None:
    """Check the parsed JSON before it becomes a profile.

    Keys beginning with an underscore are notes (`_comment`, `_gate`) and are
    ignored. Both missing and unexpected keys are errors: a typo in a machine
    constant should fail loudly rather than silently leave the default in place.
    """
    expected = {field.name for field in fields(MachineProfile)}
    present = {key for key in data if not key.startswith("_")}

    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"{source}: missing required key(s): {', '.join(missing)}")

    unexpected = sorted(present - expected)
    if unexpected:
        raise ValueError(f"{source}: unknown key(s): {', '.join(unexpected)}")

    if data["status"] not in VALID_STATUS:
        raise ValueError(
            f"{source}: status must be one of {sorted(VALID_STATUS)}, "
            f"got {data['status']!r}"
        )
    if data["firmware_dialect"] not in VALID_FIRMWARE:
        raise ValueError(
            f"{source}: firmware_dialect must be one of {sorted(VALID_FIRMWARE)}, "
            f"got {data['firmware_dialect']!r}"
        )
    if data["tilt_limit_shape"] not in VALID_TILT_LIMIT_SHAPE:
        raise ValueError(
            f"{source}: tilt_limit_shape must be one of "
            f"{sorted(VALID_TILT_LIMIT_SHAPE)}, got {data['tilt_limit_shape']!r}"
        )

    for key in ("ball_2dpos_0", "ball_2dpos_1", "ball_2dpos_2"):
        value = data[key]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{source}: {key} must be a list of two numbers")


def _warn_if_placeholder(profile: MachineProfile, source: str) -> None:
    if not profile.is_placeholder:
        return

    message = (
        f"Machine profile {profile.name!r} ({source}) is marked "
        f"{STATUS_PLACEHOLDER}: its numbers are copied from the reference "
        "machine and have not been measured. Any G-code produced with it "
        "describes a machine that does not exist. See gates M1 and M2."
    )
    warnings.warn(message, PlaceholderProfileWarning, stacklevel=3)
    # Also print a banner: a warnings filter, or a CLI that never shows
    # warnings, must not be able to hide this one.
    print("=" * 78, file=sys.stderr)
    print(f"WARNING: {message}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)


def resolve_profile_path(name_or_path: str) -> Path:
    """Map a profile name or path to the JSON file that defines it."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" or candidate.parent != Path("."):
        return candidate
    return CONFIG_DIR / f"{name_or_path}.json"


def load_profile(name_or_path: str | None = None) -> MachineProfile:
    """Load a machine profile by name, by path, or from ``ATOM_MACHINE``.

    Parameters
    ----------
    name_or_path
        A profile name (``"reference"``), or a path to a JSON file. When
        omitted, the ``ATOM_MACHINE`` environment variable is used, defaulting
        to ``"reference"``.
    """
    if name_or_path is None:
        name_or_path = os.environ.get(ENV_VAR, DEFAULT_PROFILE)

    path = resolve_profile_path(name_or_path)
    if not path.is_file():
        available = (
            ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.json")))
            if CONFIG_DIR.is_dir()
            else "none"
        )
        raise FileNotFoundError(
            f"No machine profile at {path}. Available profiles: {available}."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    _validate(data, str(path))

    values = {key: value for key, value in data.items() if not key.startswith("_")}
    for key in ("ball_2dpos_0", "ball_2dpos_1", "ball_2dpos_2"):
        values[key] = tuple(values[key])

    profile = MachineProfile(**values)
    _warn_if_placeholder(profile, str(path))
    return profile
