"""Tests for `atom.provenance` (which machine produced a result).

The point of these records is to stop a lab-machine run being mistaken for a
laptop one, so the tests that matter are the ones about *not* claiming two
results are comparable when they are not.
"""

import csv
import json
import sys

import pytest

from atom import provenance as prov

LAPTOP = {
    "recorded": prov.CAPTURED,
    "host": "laptop",
    "cpu": "AMD Ryzen 5 5600H with Radeon Graphics",
    "cpu_logical": 12,
    "taichi": "1.7.4",
    "blender": "Blender 5.2.1 LTS",
    "ti_arch": None,
    "machine_profile": "reference",
}
LAB = dict(LAPTOP, host="labpc", cpu="Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz",
           cpu_logical=72)


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_records_what_can_change_a_result():
    record = prov.fingerprint(include_blender=False)
    for field in ("cpu", "cpu_logical", "python", "ti_arch", "machine_profile"):
        assert field in record, field
    assert record["recorded"] == prov.CAPTURED


def test_a_scored_record_skips_the_blender_subprocess():
    """The metrics never touch Blender, so a `scored` record must not pay for it."""
    assert "blender" not in prov.fingerprint(include_blender=False)
    assert "blender" in prov.fingerprint(include_blender=True)


def test_no_ti_arch_override_reads_as_none_not_empty(monkeypatch):
    """None means the stock backend mix the golden was captured on (3.9)."""
    monkeypatch.delenv("ATOM_TI_ARCH", raising=False)
    assert prov.fingerprint(include_blender=False)["ti_arch"] is None

    monkeypatch.setenv("ATOM_TI_ARCH", "")
    assert prov.fingerprint(include_blender=False)["ti_arch"] is None

    monkeypatch.setenv("ATOM_TI_ARCH", "cpu")
    assert prov.fingerprint(include_blender=False)["ti_arch"] == "cpu"


def test_the_machine_profile_defaults_to_reference(monkeypatch):
    monkeypatch.delenv("ATOM_MACHINE", raising=False)
    assert prov.fingerprint(include_blender=False)["machine_profile"] == "reference"


def test_cpu_name_is_more_than_the_architecture():
    """`platform.processor()` returns bare "x86_64" on Linux, which would make a
    Xeon Gold 6254 and a Xeon Silver 4112 indistinguishable — the exact
    distinction these records exist to draw."""
    name = prov.cpu_name()
    assert name and name != "unknown"
    if sys.platform.startswith("linux"):
        assert name not in {"x86_64", "aarch64"}


# --------------------------------------------------------------------------
# comparable
# --------------------------------------------------------------------------


def test_the_same_machine_is_comparable_with_itself():
    assert prov.comparable(LAPTOP, dict(LAPTOP))


def test_a_different_cpu_is_not_comparable():
    assert not prov.comparable(LAPTOP, LAB)


def test_a_different_backend_on_one_machine_is_not_comparable():
    """3.9: forcing every stage onto the CPU changes the toolpath."""
    assert not prov.comparable(LAPTOP, dict(LAPTOP, ti_arch="cpu"))


def test_a_different_blender_is_not_comparable():
    """Blender does the remesh, so its version is part of the computation."""
    assert not prov.comparable(LAPTOP, dict(LAPTOP, blender="Blender 4.2.0 LTS"))


def test_a_different_machine_profile_is_not_comparable():
    assert not prov.comparable(LAPTOP, dict(LAPTOP, machine_profile="ours"))


def test_renaming_a_machine_does_not_make_it_a_different_one():
    assert prov.comparable(LAPTOP, dict(LAPTOP, host="renamed"))


@pytest.mark.parametrize("other", [None, {}])
def test_unknown_is_never_comparable(other):
    """An unrecorded machine must not be assumed to match a recorded one.

    This is the case that matters: the 48 baseline reports predate provenance,
    and quietly treating "unknown" as "same as mine" would defeat the guard.
    """
    assert not prov.comparable(LAPTOP, other)
    assert not prov.comparable(other, LAPTOP)


def test_two_unknowns_are_not_declared_comparable_either():
    assert not prov.comparable(None, None)


# --------------------------------------------------------------------------
# grouping and description
# --------------------------------------------------------------------------


def test_grouping_separates_two_machines():
    buckets = prov.group_by_machine([LAPTOP, LAB, dict(LAPTOP), LAB])
    assert len(buckets) == 2
    sizes = sorted(len(members) for _, members in buckets)
    assert sizes == [2, 2]


def test_grouping_one_machine_gives_one_bucket():
    buckets = prov.group_by_machine([LAPTOP, dict(LAPTOP), dict(LAPTOP)])
    assert len(buckets) == 1
    assert buckets[0][1] == [0, 1, 2]


def test_unknown_records_bucket_together():
    """Otherwise 48 provenance-less reports would read as 48 different machines."""
    buckets = prov.group_by_machine([None, None, None])
    assert len(buckets) == 1


def test_grouping_keeps_unknowns_apart_from_a_known_machine():
    buckets = prov.group_by_machine([LAPTOP, None])
    assert len(buckets) == 2


def test_describe_names_the_cpu_and_the_backend():
    text = prov.describe(LAPTOP)
    assert "5600H" in text
    assert "stock backend mix" in text
    assert "laptop" in text


def test_describe_says_when_a_backend_was_forced():
    assert "cpu" in prov.describe(dict(LAPTOP, ti_arch="cpu"))


def test_describe_flags_a_reconstructed_record():
    text = prov.describe(dict(LAPTOP, recorded=prov.RECONSTRUCTED))
    assert "not measured" in text


def test_describe_handles_nothing_at_all():
    assert prov.describe(None) == "unknown machine"
