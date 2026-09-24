"""Whether the display-dependent viewer tests can run on this machine.

The viewer tests (``test_visualize_5ax.py``, ``test_viewer_qt.py``) build real
VTK or Qt windows. They are worth running by default — the Windows-only Play
bug in ``docs/plan_corrections.md`` 4.13 was invisible to every headless test —
but they need a graphics context that actually works, and on a machine without
one VTK does not fail politely.

On the 36-core lab machine (2026-09-24, over a remote session) the very last
test aborted the whole run with::

    Windows fatal exception: access violation
      pyvista/plotting/renderer.py, line 3939 in close

An access violation kills the pytest process, so a single unusable graphics
context takes down the other 450 tests with it and ``pytest`` stops being a
usable check on that machine at all. Hence an explicit opt-out:

.. code-block:: powershell

    $env:ATOM_SKIP_DISPLAY_TESTS = "1"

Opt-out rather than opt-in, so the operator's laptop — which can run them, and
found 4.13 by doing so — keeps the coverage without having to ask for it.

Set it on any machine where a real window cannot open: a remote desktop
session, a headless server, or a GPU whose driver VTK does not get on with.
Nothing in the slicing pipeline depends on these tests.
"""

from __future__ import annotations

import os
import sys

#: Environment variable that forces the viewer tests to skip.
SKIP_ENV_VAR = "ATOM_SKIP_DISPLAY_TESTS"

_TRUTHY = {"1", "true", "yes", "on"}


def _skip_requested() -> bool:
    return os.environ.get(SKIP_ENV_VAR, "").strip().lower() in _TRUTHY


def _headless_linux() -> bool:
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY")


#: True when the display-dependent tests must not run.
NO_DISPLAY: bool = _skip_requested() or _headless_linux()

#: Reason shown in the skip report, naming which of the two applied.
NO_DISPLAY_REASON: str = (
    f"{SKIP_ENV_VAR} is set (no usable graphics context on this machine)"
    if _skip_requested()
    else "needs a display (run under xvfb-run on Linux)"
)
