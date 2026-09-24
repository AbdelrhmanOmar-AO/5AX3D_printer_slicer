"""Fill in provenance on overhang reports written before P1.7 (build plan P1.7).

The 48 baseline reports in ``reports/baseline_overhang/`` were produced before
reports recorded where they came from. Re-running them would cost about 20
hours of laptop time, and their origin is known, so it is filled in instead.
The plan asks for exactly this.

What is known, and how
----------------------
Every value below comes from ``docs/handoff.md`` (sections 3 and 7b) or from
the operator on 2026-09-23:

* all 48 ran on the operator's laptop. Its host name is ``AbdoYasser``, which
  is what ``platform.node()`` records there. The operator first gave the name
  ``Abdelrahman-personal-laptop``; the first run after P1.7 recorded
  ``AbdoYasser``, and the operator confirmed it is the same laptop;
* ``scripts/run_baseline_matrix.ps1`` sets neither ``ATOM_TI_ARCH`` nor
  ``ATOM_MACHINE``, so every run used the stock backend mix and the
  ``reference`` profile;
* Windows 11 (10.0.26200, which ``platform.platform()`` reports as
  ``Windows-10-10.0.26200-SP0``), Python 3.10.21, Taichi 1.7.4, and a working
  CUDA driver (Taichi started on CUDA reports ``Arch.cuda``).

What is **inferred** rather than recorded: ``stage_arches``. It is each
stage's default backend (``docs/plan_corrections.md`` 1.5), with the GPU
stages on ``cuda`` because the laptop's CUDA works. The first run on the
laptop after P1.7 (``ramp45_xs`` at 7 degrees, 2026-09-23) recorded exactly
these 14 values.

What is **not known**: the commit each run used, since the runs spanned several
commits on 2026-09-22 and 2026-09-23, and each run's finishing time. Those are
left as None rather than guessed.

Usage
-----
    python tools/backfill_provenance.py            # fill in, then report
    python tools/backfill_provenance.py --check    # exit 1 if any report lacks it

It fills reports that have no provenance, and refreshes blocks it wrote itself
(``source: backfilled``) when the values above are corrected. It never touches
a block that a real run recorded, and running it twice changes nothing.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom import provenance as prov  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "baseline_overhang"

#: The stages `tools/atomize.py` runs for a part with infill, and the backend
#: each asks for by default. `ratrig_to_craftware` and the Blender remesh do
#: not use Taichi.
STAGE_DEFAULTS = {
    "atomize": "cpu",
    "obj_to_bpn": "cpu",
    "bpn_to_sdf": "gpu",
    "sdf_to_isdf": "cpu",
    "compute_tool_orientations": "gpu",
    "sdf_df_to_layers": "gpu",
    "compute_tangents": "gpu",
    "align_atoms": "gpu",
    "extract_explicit_atoms": "gpu",
    "order_atoms": "cpu",
    "smooth_toolpath_point": "cpu",
    "tesselate_toolpath_orientations": "cpu",
    "add_platform": "gpu",
    "toolpath_to_gcode": "gpu",
}

#: What `atom.ti_env.current_arch_name` reports for each request on the
#: operator's laptop: an x86-64 CPU, and CUDA for "gpu".
LAPTOP_ARCH = {"cpu": "x64", "gpu": "cuda"}

LAPTOP = "AbdoYasser"

BACKFILL = {
    "source": prov.SOURCE_BACKFILLED,
    "machine": LAPTOP,
    "os": "Windows-10-10.0.26200-SP0",
    "python": "3.10.21",
    "taichi": "1.7.4",
    "ti_arch_setting": prov.STOCK_MIX,
    "stage_arches": {stage: LAPTOP_ARCH[arch] for stage, arch in sorted(STAGE_DEFAULTS.items())},
    "machine_profile": "reference",
    "git_commit": None,
    "code_modified": None,
    "recorded_utc": None,
    "metrics_version": 2,
    "scored": {
        "machine": LAPTOP,
        "git_commit": None,
        "code_modified": None,
        "utc": None,
        "metrics_version": 2,
    },
    "_note": (
        "Backfilled 2026-09-23 (build plan P1.7) from docs/handoff.md and the "
        "operator. stage_arches is inferred from each stage's default backend, "
        "not recorded at run time; the first run after P1.7 on the same laptop "
        "recorded identical values. The commit and finishing time were not "
        "recorded and are left as null."
    ),
}


def backfill_report(report: dict) -> bool:
    """Give one report the backfill block. True if it changed.

    Fills a report with no provenance, and refreshes a block this script wrote
    earlier if the values have since been corrected. A block a real run
    recorded is never touched. Refuses a report whose metrics version is not
    the one the backfill describes, since those numbers were computed
    differently.
    """
    existing = report.get("provenance")
    if isinstance(existing, dict):
        if existing.get("source") != prov.SOURCE_BACKFILLED or existing == BACKFILL:
            return False
    if report.get("metrics_version") != BACKFILL["metrics_version"]:
        raise ValueError(
            f"{report.get('part')} at {report.get('max_slope_deg')}: metrics "
            f"version {report.get('metrics_version')}, but the backfill describes "
            f"version {BACKFILL['metrics_version']}"
        )
    report["provenance"] = copy.deepcopy(BACKFILL)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="Change nothing; exit 1 if any report lacks a complete provenance block.",
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    paths = sorted(args.report_dir.glob("*.json"))
    changed, incomplete = 0, []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not args.check and backfill_report(report):
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            changed += 1
        found = prov.problems(report.get("provenance"))
        if found:
            incomplete.append(f"{path.name}: {'; '.join(found)}")

    if not args.check:
        print(f"Backfilled {changed} of {len(paths)} report(s) in {args.report_dir}.")
    for line in incomplete:
        print(f"  incomplete: {line}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
