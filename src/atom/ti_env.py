"""One switch for Taichi's backend across every stage (build plan task P0.4).

Each stage under ``tools/`` hard-codes its Taichi backend at import time. This
module routes those calls through one place so the backend can be overridden by
the ``ATOM_TI_ARCH`` environment variable, without editing any stage.

Two reasons this matters:

1. **Deterministic tests.** Unit tests force the CPU backend regardless of what
   the machine has.
2. **Measuring where the time goes.** ``order_atoms`` is 86 % of pipeline
   runtime and initialises Taichi on the CPU (see ``tests/golden/baseline.md``).
   Whether the GPU would be faster is an open question, and this makes it a
   one-command experiment rather than a code change.

Observed behaviour of ``ti.init`` with no GPU, Taichi 1.7.4
-----------------------------------------------------------
``ti.init(arch=ti.gpu)`` does **not** raise on a machine without a GPU. It logs::

    [W] [cuda_driver.cpp] libcuda.so lib not found.
    [W] [misc.py] Arch=[<Arch.cuda: 3>] is not supported, falling back to CPU

and then reports ``Arch.x64``. A stage asking for the GPU therefore degrades
silently and runs many times slower rather than failing. Anything checking
which backend is live must compare the resulting arch, not catch an exception.
``current_arch_name`` does exactly that.

Usage
-----
Replace ``ti.init(arch=ti.gpu, debug=False)`` with
``init_taichi("gpu", debug=False)``. The default argument preserves the stage's
original backend, so behaviour is unchanged unless ``ATOM_TI_ARCH`` is set.

    $env:ATOM_TI_ARCH = "cuda"     # PowerShell: force every stage onto CUDA
    $env:ATOM_TI_ARCH = "cpu"      # force every stage onto the CPU
    Remove-Item Env:\\ATOM_TI_ARCH  # back to each stage's own default

Recording the backend each stage really used (build plan P1.7)
--------------------------------------------------------------
Because a GPU request can silently become the CPU, a report must record the
backend each stage **actually** started on, not the one it asked for. When
``ATOM_TI_ARCH_LOG`` names a file, `init_taichi` appends one JSON line to it
after ``ti.init``::

    {"stage": "order_atoms", "requested": "cpu", "actual": "x64"}

``tools/overhang_report.py`` sets it for the pipeline it launches. Unset, as
it is everywhere else, nothing is written and nothing changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import taichi as ti

#: Environment variable overriding every stage's backend.
ENV_VAR = "ATOM_TI_ARCH"

#: Environment variable naming a file that records each stage's actual backend.
ARCH_LOG_ENV_VAR = "ATOM_TI_ARCH_LOG"

#: Backend names accepted in the environment variable and as `default_arch`.
#: "gpu" asks Taichi to pick any available GPU backend; the others are specific.
ARCH_NAMES: dict[str, object] = {
    "cpu": ti.cpu,
    "x64": ti.x64,
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
    "opengl": ti.opengl,
    "metal": ti.metal,
}


def resolve_arch_name(default_arch: str) -> str:
    """Return the backend name to use: the environment's, or ``default_arch``.

    The environment value is case-insensitive and whitespace is ignored, so a
    stray ``ATOM_TI_ARCH=" CUDA "`` behaves as expected. An unrecognised value
    is an error rather than a silent fallback: a typo that quietly ran the whole
    matrix on the wrong backend would waste hours and invalidate the timings.
    """
    requested = os.environ.get(ENV_VAR)
    name = (requested if requested is not None else default_arch).strip().lower()

    if name not in ARCH_NAMES:
        source = f"{ENV_VAR}={requested!r}" if requested is not None else (
            f"default_arch={default_arch!r}"
        )
        raise ValueError(
            f"Unknown Taichi backend from {source}. "
            f"Valid names: {', '.join(sorted(ARCH_NAMES))}."
        )
    return name


def init_taichi(default_arch: str, **kwargs):
    """Initialise Taichi, honouring ``ATOM_TI_ARCH`` over ``default_arch``.

    Every keyword argument is passed through to ``ti.init`` untouched, so a
    stage keeps its own ``debug``, ``offline_cache_cleaning_policy`` and
    ``kernel_profiler`` settings.
    """
    name = resolve_arch_name(default_arch)
    result = ti.init(arch=ARCH_NAMES[name], **kwargs)
    _record_arch(name)
    return result


def stage_name() -> str:
    """The running stage's name: its script's file name without ``.py``."""
    return Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "unknown"


def _record_arch(requested: str) -> None:
    """Append this stage's actual backend to ``ATOM_TI_ARCH_LOG``, if set.

    A failure to write is reported on stderr rather than raised: losing one
    provenance line must not abort an hour-long pipeline run. The report then
    shows that stage's backend as unrecorded.
    """
    path = os.environ.get(ARCH_LOG_ENV_VAR)
    if not path:
        return
    entry = {"stage": stage_name(), "requested": requested, "actual": current_arch_name()}
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as exc:
        print(f"WARNING: could not record the Taichi backend in {path}: {exc}", file=sys.stderr)


def current_arch_name() -> str:
    """Name of the backend Taichi actually started on.

    Call after ``init_taichi`` to find out whether a requested GPU backend was
    honoured or silently replaced by the CPU.
    """
    return str(ti.lang.impl.current_cfg().arch).rsplit(".", 1)[-1].lower()
