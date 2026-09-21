"""Smoke test that the environment is installed correctly (build plan P0.1).

If this fails, nothing else in the suite is meaningful: either the `atom`
package was not installed (`pip install -e ".[dev]"`) or Taichi is missing.

.. warning::
   This module must **not** use ``from __future__ import annotations``.
   That flag turns every annotation into a string, and Taichi reads kernel
   argument annotations as live objects, so every ``@ti.kernel`` in the module
   fails with ``TaichiSyntaxError: Invalid type annotation``. Any module that
   defines Taichi kernels is subject to the same restriction.
"""

import pytest


def test_import_atom_package():
    import atom

    # `src/atom/` has no `__init__.py` upstream, so `atom` is an implicit
    # namespace package and `__file__` is None. `__path__` is the thing that
    # tells us the package was actually located.
    assert list(atom.__path__), "the `atom` package was found but has no path"


def test_import_kinematics_without_taichi_init():
    """`atom.kinematics3z` must import before any `ti.init()` call.

    Taichi kernels are compiled lazily, so importing the module is safe. Tests
    and tools rely on this to read the machine constants without starting a
    Taichi runtime.
    """
    import atom.kinematics3z as kinematics3z

    # Spot-check constants from the upstream machine definition
    # (kinematics3z.py, constants block). These move into a machine profile in
    # build plan task P0.5; until then they are the reference machine's values.
    assert kinematics3z.MAX_TILT_ANGLE_DEG == 30.0
    assert kinematics3z.Z_OFFSET == 75.0


def test_import_taichi():
    import taichi as ti

    assert ti.__version__ is not None


def test_taichi_runs_a_kernel_on_cpu(ti_cpu):
    """The `ti_cpu` fixture gives unit tests a deterministic CPU backend."""
    ti = ti_cpu
    field = ti.field(dtype=ti.f32, shape=4)

    @ti.kernel
    def fill(value: ti.f32):
        for i in field:
            field[i] = value * i

    fill(2.0)
    assert list(field.to_numpy()) == pytest.approx([0.0, 2.0, 4.0, 6.0])
