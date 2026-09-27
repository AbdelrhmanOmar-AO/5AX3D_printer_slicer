"""Tests for the order_atoms kernel-profiler switch (build plan task P1.6).

Whether switching the profiler off changes nothing but the time is proved by
the golden test on the laptop (the G-code SHA-256 must not move). These tests
cover the switch itself.
"""

from __future__ import annotations

import pytest

from atom import ti_profiler


@pytest.fixture
def unset(monkeypatch):
    monkeypatch.delenv(ti_profiler.ENV_VAR, raising=False)


def test_off_by_default(unset):
    assert ti_profiler.profiler_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_values_that_switch_it_on(monkeypatch, value):
    monkeypatch.setenv(ti_profiler.ENV_VAR, value)
    assert ti_profiler.profiler_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "No", "off", "  "])
def test_values_that_leave_it_off(monkeypatch, value):
    monkeypatch.setenv(ti_profiler.ENV_VAR, value)
    assert ti_profiler.profiler_enabled() is False


@pytest.mark.parametrize("value", ["2", "enabled", "tru"])
def test_anything_else_is_refused_not_guessed(monkeypatch, value):
    monkeypatch.setenv(ti_profiler.ENV_VAR, value)
    with pytest.raises(ValueError, match=ti_profiler.ENV_VAR):
        ti_profiler.profiler_enabled()


def test_order_atoms_uses_the_switch_for_both_the_profiler_and_its_table(repo_root):
    """Checked on the source: importing tools/order_atoms.py starts Taichi."""
    source = (repo_root / "tools" / "order_atoms.py").read_text(encoding="utf-8")
    assert "kernel_profiler=True" not in source
    assert "KERNEL_PROFILER = profiler_enabled()" in source
    assert "kernel_profiler=KERNEL_PROFILER" in source
    assert "if KERNEL_PROFILER:\n        ti.profiler.print_kernel_profiler_info()" in source
    # The stage keeps its CPU default (checked more fully in test_ti_env.py).
    assert 'init_taichi("cpu"' in source
