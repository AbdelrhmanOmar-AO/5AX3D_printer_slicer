"""Check a toolpath for collisions the printer would make (build plan P4).

Checks available:

``nozzle`` (P4.3)
    At every position the nozzle visits, is material printed earlier inside
    it? The nozzle is the 40-degree cone of `atom.clearance`, placed along
    each point's own tool orientation, up to the gantry level. numpy and
    scipy only, so it runs anywhere in seconds.
``swept`` (P4.2)
    Every machine state *between* the points too: moves that turn the tool
    and all travel are split into small steps, the bed is posed by the
    forward kinematics, and the part so far, the bed corners and the nozzle
    are checked against the clearance model. Axis ranges and the tilt limit
    between points are not checked yet (they come from the P1.4 validator).
    Needs Taichi; runs on the CPU unless ``ATOM_TI_ARCH`` says otherwise.

Every file is solved re-centred on the bed, as `toolpath_to_gcode` does.
Several files, or a directory of them, can be given at once; the P0.8
archive ``reports/toolpaths/`` holds the stock toolpaths of the whole
baseline matrix, up to 30 degrees of tilt.

Usage
-----
    python tools/check_motion_safety.py data/toolpath/ramp60_s_platform.npz
    python tools/check_motion_safety.py reports/toolpaths --json reports/motion_safety/stock.json
    python tools/check_motion_safety.py "reports/toolpaths/*_xs_*.npz"
    python tools/check_motion_safety.py part.npz --checks nozzle

Exit status: 0 when every file is clear, 1 when any collision is found, 2 on
bad input.
"""

from __future__ import annotations  # no Taichi kernels in this module

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom import clearance, machine_profile  # noqa: E402
from atom import nozzle_material_check as nm  # noqa: E402
from atom import overhang_metrics as om  # noqa: E402
from atom import toolpath_view as tv  # noqa: E402

#: Version of the report layout.
SCHEMA_VERSION = 1

CHECKS = ("nozzle", "swept")


def expand_inputs(paths) -> list[Path]:
    """Files as given, every ``.npz`` inside a directory, and wildcard
    patterns such as ``reports/toolpaths/*_xs_*.npz`` (PowerShell passes
    those to Python unexpanded). Sorted within each argument."""
    import glob

    files = []
    for text in map(str, paths):
        path = Path(text)
        if path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
        elif any(char in text for char in "*?["):
            files.extend(sorted(Path(match) for match in glob.glob(text)))
        else:
            files.append(path)
    return files


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, profile, settings: dict, limit: int,
               checks=CHECKS, model=None) -> dict:
    """Run the requested checks on one toolpath file; a JSON-ready dict."""
    arrays = tv.load_toolpath_arrays(path)
    toolpath = SimpleNamespace(**arrays)
    tilt = om.tilt_from_vertical_deg(om.spherical_to_cartesian(arrays["tool_orientation"]))
    entry = {
        "file": str(path),
        "sha256": file_digest(path),
        "points": int(arrays["point_count"]),
        "max_tilt_deg": round(float(tilt.max()), 3) if len(tilt) else 0.0,
        "checks": {},
    }

    if "nozzle" in checks:
        started = time.perf_counter()
        result = nm.check_toolpath(toolpath, profile, **settings)
        report = result.to_dict(limit=limit)
        report["seconds"] = round(time.perf_counter() - started, 2)
        entry["checks"]["nozzle"] = report

    if "swept" in checks:
        from atom import tilt_motion_check as tmc

        started = time.perf_counter()
        result = tmc.check_toolpath(
            toolpath, profile, model,
            material_subsample_mm=settings["subsample_mm"],
            tolerance_mm=max(settings["tolerance_mm"], tmc.SweptCheckSettings.tolerance_mm),
        )
        report = result.to_dict(limit=limit)
        report["seconds"] = round(time.perf_counter() - started, 2)
        entry["checks"]["swept"] = report

    entry["ok"] = all(check["ok"] for check in entry["checks"].values())
    return entry


#: Printed after bed-corner hits on a toolpath from before `add_platform`.
BED_HINT = ("bed hits before add_platform are mostly the lift the platform adds; "
            "check the _platform toolpath")


def before_platform(path) -> bool:
    """True for a toolpath from before `add_platform`: the P0.8 archive
    (``<part>_ms<deg>.npz``) and ``_smoothed`` / ``_smoothed_tesselated``."""
    name = Path(path).name
    return "_platform" not in name and name != "calibration_cube.toolpath.npz"


def summary_line(entry: dict) -> str:
    parts = []
    seconds = 0.0
    for name, check in entry["checks"].items():
        seconds += check["seconds"]
        if check["ok"]:
            continue
        if name == "nozzle":
            parts.append(f"nozzle: {check['collisions']} positions "
                         f"({check['collisions_while_printing']} printing, "
                         f"{check['collisions_while_travelling']} travelling), "
                         f"deepest {check['max_depth_mm']:.2f} mm")
        else:
            kinds = ", ".join(f"{kind} {n}" for kind, n in check["violations_by_kind"].items())
            parts.append(f"swept: {check['violations']} moves ({kinds})")
            if "bed" in check["violations_by_kind"] and before_platform(entry["file"]):
                parts.append(BED_HINT)
    verdict = "clear" if entry["ok"] else "COLLISION  " + "; ".join(parts)
    return (f"{Path(entry['file']).name:<34} {entry['points']:>8} pts  "
            f"tilt <= {entry['max_tilt_deg']:5.1f} deg  "
            f"{seconds:6.1f} s  {verdict}")


def ensure_taichi() -> str:
    """Start Taichi (CPU unless ``ATOM_TI_ARCH`` says otherwise) if nothing has
    yet, and name the backend in use. Re-initialising would reset a runtime
    another caller (a test session, say) is already using."""
    import taichi as ti

    from atom.ti_env import init_taichi
    from atom.ti_env import current_arch_name

    if ti.lang.impl.get_runtime().prog is None:
        init_taichi("cpu", log_level="error")
    return current_arch_name()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="toolpath .npz files, or directories of them")
    parser.add_argument("--machine", default=None,
                        help="machine profile (default: ATOM_MACHINE, else reference)")
    parser.add_argument("--checks", default=",".join(CHECKS),
                        help=f"comma-separated, from {', '.join(CHECKS)} (default: all)")
    parser.add_argument("--subsample", type=float, default=None, metavar="MM",
                        help="thin earlier material to one point per voxel of this size")
    parser.add_argument("--tolerance", type=float, default=0.0, metavar="MM",
                        help="ignore material less than this far inside the nozzle")
    parser.add_argument("--limit", type=int, default=100,
                        help="collisions listed per file in the JSON (deepest first)")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH",
                        help="write the full report here")
    args = parser.parse_args(argv)

    files = expand_inputs(args.paths)
    missing = [str(path) for path in files if not path.is_file()]
    if missing or not files:
        print(f"No such toolpath file(s): {', '.join(missing) or 'none given'}",
              file=sys.stderr)
        return 2

    checks = tuple(name.strip() for name in args.checks.split(",") if name.strip())
    unknown = sorted(set(checks) - set(CHECKS))
    if unknown or not checks:
        print(f"Unknown check(s): {', '.join(unknown) or 'none given'}; "
              f"choose from {', '.join(CHECKS)}", file=sys.stderr)
        return 2

    if args.machine is not None:
        # The kinematics bake the machine in when first imported (hazard 12),
        # which happens lazily below, so this reaches them.
        os.environ[machine_profile.ENV_VAR] = args.machine
    profile = machine_profile.load_profile(args.machine)
    model = clearance.load_clearance(profile)
    settings = {"subsample_mm": args.subsample, "tolerance_mm": args.tolerance}

    backend = None
    if "swept" in checks:
        backend = ensure_taichi()

    entries = []
    for path in files:
        entry = check_file(path, profile, settings, args.limit, checks, model)
        print(summary_line(entry), flush=True)
        entries.append(entry)

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/check_motion_safety.py",
        "checks": list(checks),
        "machine": profile.name,
        "machine_status": profile.status,
        "clearance_model": model.describe(),
        # Plan rule 10: the backend beside every number. The nozzle check is
        # numpy only; the swept check's kinematics run on this backend.
        "taichi_backend": backend,
        "files": entries,
        "ok": all(entry["ok"] for entry in entries),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.json}")

    bad = sum(not entry["ok"] for entry in entries)
    print(f"{len(entries) - bad} of {len(entries)} clear.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
