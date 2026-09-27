"""The viewer tests' display gate (`tests/_display.py`).

Worth its own test because the gate is what stops one machine's broken graphics
context from taking down the whole suite: on the 36-core lab machine a VTK
teardown raised a Windows access violation, which kills the pytest process
rather than failing a test, so the other 450 tests died with it.

A `unit` test: it reloads the module under a patched environment rather than
relying on the machine it runs on.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import _display


def _reload(monkeypatch, *, env=None, platform=None, display=None):
    """Re-import `_display` under a chosen environment, and return it."""
    monkeypatch.delenv(_display.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    if env is not None:
        monkeypatch.setenv(_display.SKIP_ENV_VAR, env)
    if display is not None:
        monkeypatch.setenv("DISPLAY", display)
    if platform is not None:
        monkeypatch.setattr(sys, "platform", platform)
    return importlib.reload(_display)


@pytest.fixture(autouse=True)
def _restore_module():
    """Leave `_display` as the rest of the session found it."""
    yield
    importlib.reload(_display)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_the_opt_out_is_honoured(monkeypatch, value):
    """Any reasonable spelling of "yes" skips, on any platform."""
    mod = _reload(monkeypatch, env=value, platform="win32")
    assert mod.NO_DISPLAY
    assert mod.SKIP_ENV_VAR in mod.NO_DISPLAY_REASON


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_anything_else_does_not_skip_on_windows(monkeypatch, value):
    """The gate must not skip by accident: the laptop found 4.13 by running these."""
    mod = _reload(monkeypatch, env=value, platform="win32")
    assert not mod.NO_DISPLAY


def test_headless_linux_still_skips_without_the_variable(monkeypatch):
    """The original CI condition is preserved."""
    mod = _reload(monkeypatch, platform="linux")
    assert mod.NO_DISPLAY
    assert "xvfb" in mod.NO_DISPLAY_REASON


def test_linux_with_a_display_does_not_skip(monkeypatch):
    mod = _reload(monkeypatch, platform="linux", display=":99")
    assert not mod.NO_DISPLAY


def test_the_reason_names_the_variable_when_both_apply(monkeypatch):
    """With both reasons live, the explicit one is the more useful message."""
    mod = _reload(monkeypatch, env="1", platform="linux")
    assert mod.NO_DISPLAY
    assert mod.SKIP_ENV_VAR in mod.NO_DISPLAY_REASON
