"""Tests for the Taichi backend switch (build plan task P0.4).

Note: no `from __future__ import annotations` here or anywhere that defines
Taichi kernels; that flag stringifies annotations and Taichi reads kernel
argument annotations as live objects.
"""

import re
from pathlib import Path

import pytest

from atom import ti_env


# --------------------------------------------------------------------------
# Resolving which backend to use
# --------------------------------------------------------------------------


def test_default_arch_is_used_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(ti_env.ENV_VAR, raising=False)
    assert ti_env.resolve_arch_name("gpu") == "gpu"
    assert ti_env.resolve_arch_name("cpu") == "cpu"


def test_environment_variable_overrides_the_default(monkeypatch):
    monkeypatch.setenv(ti_env.ENV_VAR, "cuda")
    assert ti_env.resolve_arch_name("cpu") == "cuda"


@pytest.mark.parametrize("value", ["CUDA", " cuda ", "Cuda\n"])
def test_the_variable_is_case_and_whitespace_insensitive(monkeypatch, value):
    monkeypatch.setenv(ti_env.ENV_VAR, value)
    assert ti_env.resolve_arch_name("cpu") == "cuda"


def test_an_unknown_backend_name_is_an_error(monkeypatch):
    """A typo must not quietly run hours of compute on the wrong backend."""
    monkeypatch.setenv(ti_env.ENV_VAR, "guda")
    with pytest.raises(ValueError, match="Unknown Taichi backend"):
        ti_env.resolve_arch_name("cpu")


def test_the_error_names_the_environment_variable_when_that_is_the_culprit(monkeypatch):
    monkeypatch.setenv(ti_env.ENV_VAR, "nonsense")
    with pytest.raises(ValueError, match="ATOM_TI_ARCH='nonsense'"):
        ti_env.resolve_arch_name("cpu")


def test_an_unknown_default_is_also_an_error(monkeypatch):
    monkeypatch.delenv(ti_env.ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="default_arch"):
        ti_env.resolve_arch_name("quantum")


# --------------------------------------------------------------------------
# Passing through to ti.init
# --------------------------------------------------------------------------


def test_keyword_arguments_reach_ti_init_untouched(monkeypatch):
    """Each stage keeps its own debug/cache/profiler settings."""
    import taichi as ti

    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ti, "init", fake_init)
    monkeypatch.delenv(ti_env.ENV_VAR, raising=False)

    ti_env.init_taichi(
        "cpu", offline_cache_cleaning_policy="never", kernel_profiler=True, debug=False
    )

    assert captured["arch"] is ti.cpu
    assert captured["offline_cache_cleaning_policy"] == "never"
    assert captured["kernel_profiler"] is True
    assert captured["debug"] is False


def test_the_environment_variable_reaches_ti_init(monkeypatch):
    import taichi as ti

    captured = {}
    monkeypatch.setattr(ti, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv(ti_env.ENV_VAR, "cuda")

    ti_env.init_taichi("cpu")
    assert captured["arch"] is ti.cuda


def test_current_arch_name_reports_the_live_backend(ti_cpu):
    """Taichi silently falls back to CPU, so the arch must be read, not assumed."""
    assert ti_env.current_arch_name() in ("x64", "arm64", "cpu")


# --------------------------------------------------------------------------
# Every stage must go through the helper
# --------------------------------------------------------------------------


TOOLS = sorted(
    path
    for path in (Path(__file__).resolve().parent.parent / "tools").glob("*.py")
    if "init_taichi(" in path.read_text(encoding="utf-8")
    or "ti.init(" in path.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("path", TOOLS, ids=[p.name for p in TOOLS])
def test_no_stage_calls_ti_init_directly(path):
    """A stage calling ti.init directly would ignore ATOM_TI_ARCH."""
    text = path.read_text(encoding="utf-8")
    assert "ti.init(" not in text, (
        f"{path.name} calls ti.init directly; use init_taichi so the backend "
        "stays switchable"
    )
    assert "from atom.ti_env import init_taichi" in text


#: The backend each stage asked for before task P0.4, read off the original
#: source. Hard-coded so a careless edit to a default is caught.
ORIGINAL_DEFAULTS = {
    "add_platform.py": "gpu",
    "align_atoms.py": "gpu",
    "atomize.py": "cpu",
    "bpn_to_sdf.py": "gpu",
    "cl_fr_dw.py": "cpu",
    "compute_tangents.py": "gpu",
    "compute_tool_orientations.py": "gpu",
    "extract_explicit_atoms.py": "gpu",
    "obj_to_bpn.py": "cpu",
    "order_atoms.py": "cpu",
    "sdf_df_to_layers.py": "gpu",
    "sdf_to_isdf.py": "cpu",
    "smooth_toolpath_point.py": "cpu",
    "tesselate_toolpath_orientations.py": "cpu",
    "toolpath_to_gcode.py": "gpu",
    "visualize_bases.py": "gpu",
    "visualize_bpn.py": "cpu",
    "visualize_bpn_df.py": "gpu",
    "visualize_bpn_layers.py": "gpu",
    "visualize_bpn_sdf.py": "gpu",
    "visualize_explicit_atoms.py": "gpu",
    "visualize_implicit_atoms.py": "gpu",
    "visualize_toolpath.py": "cpu",
}


@pytest.mark.parametrize("name,expected", sorted(ORIGINAL_DEFAULTS.items()))
def test_each_stage_keeps_its_original_default_backend(name, expected):
    """P0.4 must not change any stage's behaviour when ATOM_TI_ARCH is unset."""
    path = Path(__file__).resolve().parent.parent / "tools" / name
    text = path.read_text(encoding="utf-8")

    found = re.findall(r'init_taichi\(\s*"(\w+)"', text)
    assert found, f"{name} has no init_taichi call"
    assert found[0] == expected, (
        f"{name} now defaults to {found[0]!r}, was {expected!r} upstream"
    )
