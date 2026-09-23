"""Tests for the infill period and shell thickness parameters (build plan P1.3).

Unit tests only. The pipeline check (a calibration-cube run with
``infill_period`` 12 must extrude a different amount) is in
``tests/test_infill_pipeline.py``.
"""

from __future__ import annotations

import pytest

from atom import infill_options


def test_defaults_are_upstreams_exactly():
    """Ints, as upstream passed them to the kernel: the golden depends on it."""
    assert infill_options.DEFAULT_INFILL_PERIOD == 8
    assert infill_options.DEFAULT_SHELL_THICKNESS == 2
    assert type(infill_options.DEFAULT_INFILL_PERIOD) is int
    assert type(infill_options.DEFAULT_SHELL_THICKNESS) is int


def test_no_settings_means_no_extra_arguments():
    assert infill_options.infill_arguments() == ""
    assert infill_options.infill_arguments(None, None) == ""


def test_infill_arguments():
    assert infill_options.infill_arguments(12, None) == " --infill-period 12"
    assert infill_options.infill_arguments(None, 3) == " --shell-thickness 3"
    assert infill_options.infill_arguments(12.0, 1.5) == " --infill-period 12 --shell-thickness 1.5"
    assert infill_options.infill_arguments(None, 0) == " --shell-thickness 0"


@pytest.mark.parametrize("value", [0, -4, float("nan"), float("inf"), "8", True])
def test_impossible_periods_are_refused(value):
    with pytest.raises(ValueError, match="infill_period"):
        infill_options.infill_arguments(value, None)


@pytest.mark.parametrize("value", [-1, float("nan"), "2", False])
def test_impossible_shells_are_refused(value):
    with pytest.raises(ValueError, match="shell_thickness"):
        infill_options.infill_arguments(None, value)


# --------------------------------------------------------------------------
# The stage's command line
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage():
    """tools/sdf_to_isdf.py. Importing it does not start Taichi, but it needs
    the package's runtime dependencies (pyvista, Pillow)."""
    pytest.importorskip("pyvista")
    pytest.importorskip("PIL")
    import sdf_to_isdf

    return sdf_to_isdf


def test_stage_defaults_are_upstreams_values(stage):
    args = stage.parse_args(["in.npz", "sdf.npz", "out.npz", "no_gui=True"])
    assert args.no_gui is True
    assert args.infill_period == 8 and type(args.infill_period) is int
    assert args.shell_thickness == 2 and type(args.shell_thickness) is int


def test_stage_reads_the_options(stage):
    args = stage.parse_args(
        ["in.npz", "sdf.npz", "out.npz", "no_gui=True", "--infill-period", "12", "--shell-thickness", "1.5"]
    )
    assert args.infill_period == 12.0
    assert args.shell_thickness == 1.5


def test_stage_keeps_upstreams_gui_switch(stage):
    """The positional no_gui still reads as upstream did: "False" or "0" means the GUI."""
    assert stage.parse_args(["a", "b", "c"]).no_gui is False
    assert stage.parse_args(["a", "b", "c", "0"]).no_gui is False
    assert stage.parse_args(["a", "b", "c", "no_gui=True"]).no_gui is True


def test_stage_refuses_an_impossible_period(stage):
    with pytest.raises(ValueError, match="infill_period"):
        stage.parse_args(["a", "b", "c", "no_gui=True", "--infill-period", "0"])


def test_atomize_reads_the_optional_keys(repo_root):
    """Checked on the source text: importing tools/atomize.py starts Taichi."""
    source = (repo_root / "tools" / "atomize.py").read_text(encoding="utf-8")
    assert 'param_dict.get("infill_period")' in source
    assert 'param_dict.get("shell_thickness")' in source
    assert "infill_arguments(self.infill_period, self.shell_thickness)" in source
    assert "no_gui=True{params.infill_arguments}" in source
