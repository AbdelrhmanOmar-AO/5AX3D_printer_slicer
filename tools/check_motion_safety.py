"""Check a toolpath for collisions the printer would make (build plan P4).

Checks available:

``nozzle`` (P4.3)
    At every position the nozzle visits, is material printed earlier inside
    it? The nozzle is the 40-degree cone of `atom.clearance`, placed along
    each point's own tool orientation, up to the gantry level. numpy and
    scipy only, so it runs anywhere in seconds.

Every file is checked in its own part frame. Several files, or a directory
of them, can be given at once; the P0.8 archive ``reports/toolpaths/`` holds
the stock toolpaths of the whole baseline matrix, up to 30 degrees of tilt.

Usage
-----
    python tools/check_motion_safety.py data/toolpath/ramp60_s_smoothed.npz
    python tools/check_motion_safety.py reports/toolpaths --json reports/motion_safety/stock_nozzle.json

Exit status: 0 when every file is clear, 1 when any collision is found, 2 on
bad input.
"""

from __future__ import annotations  # no Taichi kernels in this module

import argparse
import hashlib
import json
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

CHECKS = ("nozzle",)


def expand_inputs(paths) -> list[Path]:
    """Files as given, and every ``.npz`` inside a directory, sorted."""
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
        else:
            files.append(path)
    return files


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, profile, settings: dict, limit: int) -> dict:
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

    started = time.perf_counter()
    result = nm.check_toolpath(toolpath, profile, **settings)
    report = result.to_dict(limit=limit)
    report["seconds"] = round(time.perf_counter() - started, 2)
    entry["checks"]["nozzle"] = report
    entry["ok"] = all(check["ok"] for check in entry["checks"].values())
    return entry


def summary_line(entry: dict) -> str:
    nozzle = entry["checks"]["nozzle"]
    verdict = "clear" if entry["ok"] else "COLLISION"
    detail = ""
    if not nozzle["ok"]:
        detail = (f"  {nozzle['collisions']} positions "
                  f"({nozzle['collisions_while_printing']} printing, "
                  f"{nozzle['collisions_while_travelling']} travelling), "
                  f"deepest {nozzle['max_depth_mm']:.2f} mm")
    return (f"{Path(entry['file']).name:<34} {entry['points']:>8} pts  "
            f"tilt <= {entry['max_tilt_deg']:5.1f} deg  "
            f"{nozzle['seconds']:6.1f} s  {verdict}{detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="toolpath .npz files, or directories of them")
    parser.add_argument("--machine", default=None,
                        help="machine profile (default: ATOM_MACHINE, else reference)")
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

    profile = machine_profile.load_profile(args.machine)
    model = clearance.load_clearance(profile)
    settings = {"subsample_mm": args.subsample, "tolerance_mm": args.tolerance}

    entries = []
    for path in files:
        entry = check_file(path, profile, settings, args.limit)
        print(summary_line(entry), flush=True)
        entries.append(entry)

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/check_motion_safety.py",
        "checks": list(CHECKS),
        "machine": profile.name,
        "machine_status": profile.status,
        "clearance_model": model.describe(),
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
