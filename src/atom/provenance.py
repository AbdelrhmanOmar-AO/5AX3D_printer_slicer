"""Which machine and toolchain produced a result.

Why this exists
---------------
`docs/plan_corrections.md` 3.9 established that the same part sliced on two
different backends gives a different toolpath — 2.4 % more points, three fewer
deposition runs. Different machines are the same problem: the committed baseline
was measured on the operator's laptop, and a run on the 36-core lab machine
**overwrites the report for its part and slope** with nothing in the file to show
a different machine made it.

That is how a stock-vs-overhang-aware comparison silently credits a hardware
difference to the contribution. This module makes each report say where it came
from.

Two distinct facts, deliberately kept apart
-------------------------------------------
``pipeline``
    The machine and toolchain that produced the **toolpath**. This is the one
    that matters for comparability, and ``--reanalyse`` must carry it forward
    unchanged: re-scoring an archived toolpath on another machine does not move
    where that toolpath was computed.
``scored``
    The machine that computed the **metrics** from that toolpath. Refreshed by
    ``--reanalyse``. Useful for tracing a metrics change, not for comparability.

Conflating the two would have made ``--reanalyse`` on the lab machine relabel
all 48 laptop-measured runs as lab-measured, which is exactly the corruption
this is meant to prevent.

No ``from __future__ import annotations`` here: this is imported alongside
Taichi kernels and the repository has walked into that twice already
(corrections 3.2 and 4.10).
"""

import os
import platform
import subprocess
import sys

#: Fields that decide whether two results are comparable. Host name is
#: deliberately excluded: the same machine renamed is still the same machine,
#: and two identically-specified machines are still two machines — the CPU
#: string plus core count is the better discriminator, and `host` is kept for
#: humans rather than for the comparison.
COMPARABLE_FIELDS = (
    "cpu",
    "cpu_logical",
    "ti_arch",
    "taichi",
    "blender",
    "machine_profile",
)

#: Marks how a record was obtained. Reports written before provenance existed
#: can be filled in from `docs/handoff.md` and `tests/golden/baseline.md`, which
#: is worth doing — they are the baseline — but must not be passed off as
#: measured.
CAPTURED = "captured"
RECONSTRUCTED = "reconstructed-from-docs"

_blender_version = None  # cached; the query costs a subprocess


def cpu_name():
    """The processor's marketing name, or the best available substitute.

    ``platform.processor()`` gives "Intel64 Family 6 Model 85 Stepping 7" on
    Windows and bare "x86_64" on Linux — neither distinguishes a Xeon Gold 6254
    from a Xeon Silver 4112, which is the distinction that matters here.
    """
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return platform.processor() or platform.machine() or "unknown"


def blender_version(refresh=False):
    """Blender's version string, or None if it cannot be asked.

    Blender runs the pipeline's first stage (the remesh), so its version is part
    of what determines the output — `tests/golden/baseline.md` records it for
    that reason. Cached: one subprocess per process is plenty.
    """
    global _blender_version
    if _blender_version is not None and not refresh:
        return _blender_version or None

    version = ""
    try:
        out = subprocess.run(
            ["blender", "--version"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
        if out.returncode == 0:
            # First line is e.g. "Blender 5.2.1 LTS"; the rest is build detail.
            first = (out.stdout or "").strip().splitlines()
            if first:
                version = first[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    _blender_version = version
    return version or None


def git_commit():
    """The checked-out commit, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def taichi_version():
    """Taichi's version, or None if it is not importable."""
    try:
        import taichi as ti

        return ".".join(str(part) for part in ti.__version__)
    except Exception:
        return None


def fingerprint(include_blender=True):
    """Everything about this machine that could change a result.

    ``include_blender`` exists because the metrics do not touch Blender: a
    ``scored`` record has no reason to pay for the subprocess.
    """
    record = {
        "recorded": CAPTURED,
        "host": platform.node() or None,
        "cpu": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "taichi": taichi_version(),
        # None means no override, i.e. each stage's own default — the "stock
        # mix" the golden baseline was captured on. See corrections 3.9.
        "ti_arch": os.environ.get("ATOM_TI_ARCH") or None,
        "machine_profile": os.environ.get("ATOM_MACHINE") or "reference",
        "git_commit": git_commit(),
    }
    if include_blender:
        record["blender"] = blender_version()
    return record


def comparable(a, b):
    """Whether two records describe the same computation.

    ``None`` on either side means unknown, and unknown is not the same as
    matching: a report with no provenance cannot be declared comparable to one
    that has it.
    """
    if not a or not b:
        return False
    return all(a.get(f) == b.get(f) for f in COMPARABLE_FIELDS)


def describe(record):
    """One short line for a summary table or a warning."""
    if not record:
        return "unknown machine"
    cpu = record.get("cpu") or "unknown CPU"
    host = record.get("host")
    arch = record.get("ti_arch") or "stock backend mix"
    label = f"{cpu} ({arch})"
    if host:
        label += f" on {host}"
    if record.get("recorded") == RECONSTRUCTED:
        label += " [from docs, not measured]"
    return label


def group_by_machine(records):
    """Group records into buckets of mutually comparable ones.

    Used to warn when one summary table mixes machines. Returns a list of
    ``(representative_record, [indices])``, unknown records bucketed together.
    """
    buckets = []
    for index, record in enumerate(records):
        for representative, members in buckets:
            if comparable(representative, record) or (
                not representative and not record
            ):
                members.append(index)
                break
        else:
            buckets.append((record, [index]))
    return buckets
