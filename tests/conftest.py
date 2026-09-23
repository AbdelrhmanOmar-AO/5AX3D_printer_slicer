"""Shared pytest configuration for the 5AX3D slicer fork.

Three test tiers are used across the build plan:

``unit``
    CPU-only, fast, runs anywhere (developer machine, CI). This is the default:
    a test with no marker is treated as ``unit``.
``pipeline``
    Drives real Atomizer stages on small meshes. Needs the operator's laptop
    (CUDA GPU + Blender on PATH). Skipped unless ``--run-pipeline`` is given.
``benchmark``
    Full runs on the benchmark parts; can take a long time. Skipped unless
    ``--run-benchmark`` is given.

Skipping by default is what lets CI run the whole suite on a GPU-less Ubuntu
runner without the slow tiers silently failing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

# `tools/` holds CLI scripts rather than an importable package, so tests that
# exercise their logic (e.g. tools/gcode_stats.py) import them by module name.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pipeline",
        action="store_true",
        default=False,
        help="Run tests marked `pipeline` (needs a GPU laptop with Blender).",
    )
    parser.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="Run tests marked `benchmark` (long full-part runs).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Default unmarked tests to `unit`, and skip the slow tiers by default."""
    run_pipeline = config.getoption("--run-pipeline")
    run_benchmark = config.getoption("--run-benchmark")

    skip_pipeline = pytest.mark.skip(reason="needs --run-pipeline (operator's laptop)")
    skip_benchmark = pytest.mark.skip(
        reason="needs --run-benchmark (operator's laptop, slow)"
    )

    for item in items:
        keywords = item.keywords
        if not any(tier in keywords for tier in ("unit", "pipeline", "benchmark")):
            item.add_marker(pytest.mark.unit)

        if "pipeline" in keywords and not run_pipeline:
            item.add_marker(skip_pipeline)
        if "benchmark" in keywords and not run_benchmark:
            item.add_marker(skip_benchmark)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def ti_cpu():
    """Initialise Taichi on the CPU backend for the whole test session.

    Unit tests that touch Taichi kernels request this fixture so results are
    deterministic and do not depend on a GPU being present.
    """
    ti = pytest.importorskip("taichi", reason="taichi is not installed")
    ti.init(arch=ti.cpu, offline_cache_cleaning_policy="never")
    yield ti
    ti.reset()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A throwaway copy of the `data/` directory layout.

    Pipeline tests write their intermediate stage outputs here instead of into
    the real ``data/``, so a test run can never clobber committed inputs or the
    golden baseline. Mesh and param inputs are copied; output directories are
    created empty.
    """
    data_root = tmp_path / "data"
    for name in ("mesh", "param"):
        source = REPO_ROOT / "data" / name
        if source.is_dir():
            shutil.copytree(source, data_root / name)
        else:
            (data_root / name).mkdir(parents=True, exist_ok=True)

    for name in (
        "basis",
        "direction",
        "frame",
        "gcode",
        "log",
        "phasor",
        "point_normal",
        "sdf",
        "toolpath",
        "toolpath_planner",
        "triphasor",
    ):
        (data_root / name).mkdir(parents=True, exist_ok=True)

    return data_root
