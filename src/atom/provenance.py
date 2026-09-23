"""Where a reported number came from (build plan task P1.7).

Plan §0 rule 10: record the machine and backend beside every reported number,
and never pool runs from different machines or backends. Forcing the stages
onto one backend changes the toolpath structurally, not just its last digits
(``docs/plan_corrections.md`` 3.9), so a stock run from one machine compared
with an overhang-aware run from another would credit the difference to the
contribution.

Every overhang report carries a ``provenance`` block built here:

``source``
    ``"pipeline"`` when the report tool ran the pipeline itself,
    ``"skip-pipeline"`` when it measured a toolpath already on disk (its origin
    is then unknown), or ``"backfilled"`` for reports written before P1.7 and
    filled in afterwards from known values.
``machine``, ``os``, ``python``
    The computer that ran the pipeline: host name, OS and Python version.
``taichi``
    Taichi version, as ``"1.7.4"``.
``ti_arch_setting``
    ``ATOM_TI_ARCH`` as set for the run, or ``"stock mix"`` when unset (each
    stage on its own default backend).
``stage_arches``
    ``{stage: backend}``: the backend each stage **actually** started on, as
    recorded by `atom.ti_env` (``"cuda"``, ``"x64"``, ...). A GPU request that
    silently fell back to the CPU shows up here.
``machine_profile``
    ``ATOM_MACHINE``, or ``"reference"`` when unset.
``git_commit``, ``code_modified``
    The commit the pipeline ran from, and whether ``src/``, ``tools/`` or
    ``config/`` had uncommitted changes at the time. ``None`` when unknown.
``recorded_utc``
    When the pipeline run finished.
``metrics_version``
    The metric definition the numbers were computed with.
``scored``
    Where and when the metrics were last computed. ``--reanalyse`` re-scores an
    archived toolpath, possibly on another machine and commit, while everything
    above still describes the run that produced the toolpath.

Comparability
-------------
Two runs are comparable when they share `COMPARABILITY_FIELDS`: machine,
backend (the setting and every stage's actual backend), Taichi version and
machine profile. Agreed with the operator on 2026-09-23. The git commit is
deliberately excluded, since code changes between runs are the point of a
comparison. Host names compare case-insensitively.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: ``ti_arch_setting`` when ``ATOM_TI_ARCH`` is unset.
STOCK_MIX = "stock mix"

SOURCE_PIPELINE = "pipeline"
SOURCE_SKIP_PIPELINE = "skip-pipeline"
SOURCE_BACKFILLED = "backfilled"
VALID_SOURCES = frozenset({SOURCE_PIPELINE, SOURCE_SKIP_PIPELINE, SOURCE_BACKFILLED})

#: Every key a provenance block must have (values may be None where unknown).
REQUIRED_FIELDS: tuple[str, ...] = (
    "source",
    "machine",
    "os",
    "python",
    "taichi",
    "ti_arch_setting",
    "stage_arches",
    "machine_profile",
    "git_commit",
    "code_modified",
    "recorded_utc",
    "metrics_version",
    "scored",
)

#: The fields two runs must share to be pooled in one table.
COMPARABILITY_FIELDS: tuple[str, ...] = (
    "machine",
    "ti_arch_setting",
    "stage_arches",
    "taichi",
    "machine_profile",
)

#: Directories whose uncommitted changes would make ``git_commit`` misleading.
CODE_DIRECTORIES: tuple[str, ...] = ("src", "tools", "config")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def taichi_version() -> str | None:
    """Taichi's version as ``"1.7.4"``, or None if Taichi is not installed."""
    try:
        import taichi
    except ImportError:
        return None
    version = taichi.__version__
    return ".".join(str(part) for part in version) if isinstance(version, tuple) else str(version)


def git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    """``(commit, code_modified)`` for the repository, or ``(None, None)``."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_DIRECTORIES], cwd=repo_root,
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, bool(changed)


def read_arch_log(path: Path) -> dict[str, str]:
    """``{stage: backend}`` from an ``ATOM_TI_ARCH_LOG`` file.

    A stage run twice (``atomize.py``'s warm-up) normally reports the same
    backend both times. If it ever reports two, both are kept, joined with
    ``+``, so the disagreement stays visible instead of one silently winning.
    """
    seen: dict[str, list[str]] = {}
    if not Path(path).is_file():
        return {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        stage, actual = entry.get("stage"), entry.get("actual")
        if stage and actual and actual not in seen.setdefault(stage, []):
            seen[stage].append(actual)
    return {stage: "+".join(values) for stage, values in sorted(seen.items())}


def scoring_block(repo_root: Path, metrics_version: int) -> dict[str, Any]:
    """Where and when the metrics are being computed, now."""
    commit, modified = git_state(repo_root)
    return {
        "machine": platform.node() or None,
        "git_commit": commit,
        "code_modified": modified,
        "utc": utc_now(),
        "metrics_version": metrics_version,
    }


def collect(
    source: str,
    stage_arches: dict[str, str] | None,
    repo_root: Path,
    metrics_version: int,
) -> dict[str, Any]:
    """A provenance block for a run on this machine, now.

    For ``skip-pipeline`` the toolpath's origin is unknown, so the fields that
    describe the run that produced it are None; only ``scored`` describes this
    machine.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    scored = scoring_block(repo_root, metrics_version)
    if source == SOURCE_SKIP_PIPELINE:
        return {
            "source": source,
            "machine": None,
            "os": None,
            "python": None,
            "taichi": None,
            "ti_arch_setting": None,
            "stage_arches": {},
            "machine_profile": None,
            "git_commit": None,
            "code_modified": None,
            "recorded_utc": None,
            "metrics_version": metrics_version,
            "scored": scored,
        }

    setting = os.environ.get("ATOM_TI_ARCH", "").strip().lower()
    return {
        "source": source,
        "machine": platform.node() or None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "taichi": taichi_version(),
        "ti_arch_setting": setting or STOCK_MIX,
        "stage_arches": dict(sorted((stage_arches or {}).items())),
        "machine_profile": os.environ.get("ATOM_MACHINE", "").strip() or "reference",
        "git_commit": scored["git_commit"],
        "code_modified": scored["code_modified"],
        "recorded_utc": scored["utc"],
        "metrics_version": metrics_version,
        "scored": scored,
    }


def rescored(block: dict[str, Any], repo_root: Path, metrics_version: int) -> dict[str, Any]:
    """The same run's provenance, with the scoring updated to now.

    Everything describing the run that produced the toolpath is kept.
    """
    updated = dict(block)
    updated["metrics_version"] = metrics_version
    updated["scored"] = scoring_block(repo_root, metrics_version)
    return updated


def problems(block: Any) -> list[str]:
    """What is wrong with a provenance block; an empty list means it is complete."""
    if not isinstance(block, dict):
        return ["no provenance block"]
    found = [f"missing field {name!r}" for name in REQUIRED_FIELDS if name not in block]
    if found:
        return found
    if block["source"] not in VALID_SOURCES:
        found.append(f"unknown source {block['source']!r}")
    if not isinstance(block["stage_arches"], dict):
        found.append("stage_arches is not a mapping")
    if block["source"] != SOURCE_SKIP_PIPELINE:
        for name in COMPARABILITY_FIELDS:
            if not block[name]:
                found.append(f"{name} is empty")
    if not isinstance(block["scored"], dict):
        found.append("scored is not a mapping")
    return found


def comparability_key(block: Any) -> tuple:
    """The values two runs must share to be comparable (see the module docstring).

    Reports with no provenance share one key, as do skip-pipeline reports, so
    each forms its own group rather than merging with a known machine.
    """
    if not isinstance(block, dict) or problems(block):
        return ("no provenance recorded",)
    machine = block["machine"].lower() if isinstance(block["machine"], str) else None
    return (
        machine,
        block["ti_arch_setting"],
        tuple(sorted(block["stage_arches"].items())),
        block["taichi"],
        block["machine_profile"],
    )


def is_known(block: Any) -> bool:
    """Whether a block says where its run came from: complete, and not skip-pipeline."""
    return (
        isinstance(block, dict)
        and not problems(block)
        and block["source"] != SOURCE_SKIP_PIPELINE
    )


def describe(block: Any) -> str:
    """One line naming the machine, backend, Taichi version and profile."""
    if not isinstance(block, dict) or problems(block):
        return "no provenance recorded"
    if block["source"] == SOURCE_SKIP_PIPELINE:
        return "unknown origin (measured with --skip-pipeline)"
    arches = sorted(set(block["stage_arches"].values()))
    backend = block["ti_arch_setting"]
    if arches:
        backend += f" ({' + '.join(arches)})"
    return (
        f"machine {block['machine']}, backend {backend}, Taichi {block['taichi']}, "
        f"profile {block['machine_profile']}"
    )
