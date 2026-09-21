"""Tests for machine profiles (build plan task P0.5).

The point of this task is that moving the machine constants out of the code
changes nothing about what the pipeline produces. The first test is therefore
the important one: the reference profile must reproduce the upstream literals
exactly, values and Python types alike, because Taichi captures them as globals
when it compiles a kernel.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from atom import machine_profile
from atom.machine_profile import (
    MachineProfile,
    PlaceholderProfileWarning,
    load_profile,
)

#: The constants block from src/atom/kinematics3z.py at commit e7b71ea, before
#: task P0.5 moved it into configuration. Hard-coded on purpose: if the profile
#: and the code were read from the same place the test would prove nothing.
UPSTREAM_CONSTANTS: dict[str, object] = {
    "BALL_2DPOS_0": np.array((-4.07, -12.16)),
    "BALL_2DPOS_1": np.array((304.93, -12.16)),
    "BALL_2DPOS_2": np.array((150.43, 296.84)),
    "BALL_Z": -45.7,
    "RAIL_ANGLE_0": 29.89,
    "RAIL_ANGLE_1": -29.89,
    "RAIL_ANGLE_2": 90.0,
    "Z_OFFSET": 75.0,
    "MAX_TILT_ANGLE_DEG": 30.0,
    "MAX_X_AXIS": 300,
    "MAX_Y_AXIS": 293,
    "MAX_Z_AXIS": 280,
    "BALL_TO_CORNER": 10,
    "NOZZLE_TO_GAUNTRY": 70,
    "FILAMENT_DIAMETER": 1.75,
    "DEPOSITON_FEEDRATE": 600,
    "TRAVEL_FEEDRATE": 3000,
    "Z_FAN_ON": 2.0,
    "RETRACT_THRESH": 1.8,
    "RETRACT_LENGTH": 2.0,
    "RETRACT_SPEED": 2700,
}


@pytest.mark.parametrize("name", sorted(UPSTREAM_CONSTANTS))
def test_kinematics_constants_match_the_upstream_literals(name):
    """Every module-level constant still has its original value and type."""
    import atom.kinematics3z as kinematics3z

    expected = UPSTREAM_CONSTANTS[name]
    actual = getattr(kinematics3z, name)

    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    else:
        assert actual == expected
        # Taichi compiles a kernel against the Python type of the global it
        # captures, so int must stay int and float must stay float.
        assert type(actual) is type(expected), (
            f"{name} changed type: {type(actual).__name__} "
            f"(was {type(expected).__name__})"
        )


def test_reference_profile_is_verified_and_rrf():
    profile = load_profile("reference")

    assert profile.name == "reference"
    assert profile.status == machine_profile.STATUS_VERIFIED
    assert not profile.is_placeholder
    assert profile.firmware_dialect == "rrf"
    assert profile.tilt_limit_shape == "cone"  # GATE M2 for our own machine


def test_default_profile_is_reference(monkeypatch):
    monkeypatch.delenv(machine_profile.ENV_VAR, raising=False)
    assert load_profile().name == "reference"


def test_env_var_selects_the_profile(monkeypatch):
    monkeypatch.setenv(machine_profile.ENV_VAR, "ours")
    with pytest.warns(PlaceholderProfileWarning):
        assert load_profile().name == "ours"


def test_loading_our_placeholder_profile_warns_loudly(capsys):
    """Nobody should produce G-code from unmeasured numbers without noticing."""
    with pytest.warns(PlaceholderProfileWarning, match="PLACEHOLDER"):
        profile = load_profile("ours")

    assert profile.is_placeholder
    # The banner goes to stderr too, so a warnings filter cannot hide it.
    assert "PLACEHOLDER" in capsys.readouterr().err


def test_our_profile_is_a_copy_of_reference_until_the_gates_are_answered():
    """`ours` must not drift into looking measured while it is still guesswork."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderProfileWarning)
        ours = load_profile("ours")
    reference = load_profile("reference")

    differing = [
        field
        for field in ("ball_z", "z_offset", "max_tilt_angle_deg", "max_x_axis")
        if getattr(ours, field) != getattr(reference, field)
    ]
    assert not differing, (
        f"ours.json now differs from reference.json in {differing}. If these are "
        "real measurements, set its status to 'verified' and delete this test."
    )


def test_missing_key_is_rejected(tmp_path):
    data = json.loads(
        (machine_profile.CONFIG_DIR / "reference.json").read_text(encoding="utf-8")
    )
    del data["ball_z"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key"):
        load_profile(str(path))


def test_unknown_key_is_rejected(tmp_path):
    """A typo'd constant must fail rather than silently leave the default."""
    data = json.loads(
        (machine_profile.CONFIG_DIR / "reference.json").read_text(encoding="utf-8")
    )
    data["bal_z"] = -45.7
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown key"):
        load_profile(str(path))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("status", "probably_fine", "status must be one of"),
        ("firmware_dialect", "marlin", "firmware_dialect must be one of"),
        ("tilt_limit_shape", "sphere", "tilt_limit_shape must be one of"),
    ],
)
def test_enumerated_fields_are_validated(tmp_path, field, value, message):
    data = json.loads(
        (machine_profile.CONFIG_DIR / "reference.json").read_text(encoding="utf-8")
    )
    data[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_profile(str(path))


def test_ball_position_must_be_a_pair(tmp_path):
    data = json.loads(
        (machine_profile.CONFIG_DIR / "reference.json").read_text(encoding="utf-8")
    )
    data["ball_2dpos_0"] = [1.0, 2.0, 3.0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="two numbers"):
        load_profile(str(path))


def test_missing_profile_names_the_available_ones():
    with pytest.raises(FileNotFoundError, match="Available profiles:.*reference"):
        load_profile("no_such_machine")


def test_profile_is_immutable():
    """A stray write must not silently change the machine mid-run."""
    profile = load_profile("reference")
    with pytest.raises(Exception):
        profile.z_offset = 1.0  # type: ignore[misc]


def test_every_shipped_profile_loads():
    """Each file in config/machines/ parses and declares its own filename."""
    paths = sorted(machine_profile.CONFIG_DIR.glob("*.json"))
    assert paths, "no machine profiles found in config/machines/"

    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderProfileWarning)
            profile = load_profile(path.stem)

        assert isinstance(profile, MachineProfile)
        assert profile.name == path.stem, (
            f"{path.name} declares name {profile.name!r}, which will not match "
            "the value of ATOM_MACHINE that selects it"
        )
