"""Build plan P2.4's pipeline test: has the tilt built up where the overhang starts?

The plan: "on `ramp60_xs`, the tilt measured from the toolpath at the first
overhang layer is >= t - 2". ``t`` is the overhang rule's tilt for the ramp's
underside: 60 - 45 + 2 = 17 degrees at a 30-degree budget, so the bar is 15.

Read here as: the atoms by the underside (the effective overhang angle's own
points) in the **first 0.5 mm past the column's edge**, out over the air
(`overhang_metrics.overhang_start_tilts`; 0.5 mm as plan_corrections P2-13
measured it). The atoms rather than the toolpath: a field-only run takes a
fraction of a full one, and every orientation the toolpath deposits with is
an atom's own (`test_field_only.py` checks that on this part).

**It reports and does not yet judge the tilt** (the operator's decision,
2026-10-01): on the CPU (the pipeline-like SDF, plan_corrections P2-20) the
strip's 13 atoms average 15.0 degrees and the lowest is 13.5, right at the
bar, so the pass rule is set once the operator has seen the laptop's numbers
on the remeshed part. About 3-5 minutes on the laptop.

No `from __future__ import annotations`: this drives Taichi through the tool.
"""

import json

import numpy as np
import pytest

pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

import overhang_report as orep
from atom import frame_atoms
from atom import overhang_field as of
from atom import overhang_metrics as om

PART = "ramp60_xs"
SLOPE = 30.0
#: How far past the column's edge the strip reaches, mm (P2-13), and one
#: bead width for comparison.
STRIPS_MM = (0.5, 0.9)
#: The plan's allowance below the rule's tilt.
ALLOWANCE_DEG = 2.0


def _summary(start, bar):
    tilts = start.tilts_deg
    return (
        f"{start.count} atoms, tilt mean {tilts.mean():.1f}, lowest {tilts.min():.1f}, "
        f"p10 {np.percentile(tilts, 10):.1f}, at or over {bar:g}: {np.mean(tilts >= bar) * 100:.0f} %"
    )


@pytest.mark.pipeline
def test_the_tilt_where_the_overhang_starts(repo_root, capsys):
    param = repo_root / "data" / "param" / f"{PART}.json"
    width = json.loads(param.read_text(encoding="utf-8"))["deposition_width"]
    elapsed, _, _ = orep.run_pipeline(param, SLOPE, field_only=True, overhang_aware=True)

    atoms = frame_atoms.load_frame_atoms(repo_root / "data" / "frame" / f"{PART}.npz")
    count = int(np.asarray(atoms.point_count).item())
    keep = om.deposition_mask(atoms)
    points = np.asarray(atoms.point[:count], dtype=np.float64)[keep]
    directions = om.tool_directions(atoms)[:count][keep]

    mesh, _, _, _ = orep.load_mesh_arrays(repo_root / "data" / "mesh" / f"{PART}.stl")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = om.unit_face_normals(triangles)
    underside = normals[om.overhang_face_mask(normals, triangles.mean(axis=1))][0]
    rule = of.overhang_target(underside, SLOPE)
    bar = rule.tilt_deg - ALLOWANCE_DEG

    starts = {
        strip: om.overhang_start_tilts(points, directions, triangles, underside, strip, width)
        for strip in STRIPS_MM
    }
    first = starts[STRIPS_MM[0]]
    lowest = np.argsort(first.tilts_deg)[:5]
    with capsys.disabled():
        print(f"\n\nP2.4 check, {PART} at max_slope {SLOPE:g}, overhang-aware, field-only ({elapsed:.0f} s):")
        print(f"  the rule's tilt t = {rule.tilt_deg:g}, the plan's bar t - {ALLOWANCE_DEG:g} = {bar:g}")
        for strip, start in starts.items():
            print(f"  first {strip:g} mm past the column's edge: " + (_summary(start, bar) if start.count else "no atoms"))
        print("  lowest in the first strip:  x, y, z (mm) | tilt")
        for i in lowest:
            x, y, z = first.points[i]
            print(f"    {x:6.2f} {y:6.2f} {z:6.2f} | {first.tilts_deg[i]:5.1f}")
        print("  No pass rule on the tilt yet: the operator sets it after these numbers (2026-10-01).\n")

    assert first.count > 0, "no atoms by the underside where it starts: nothing to measure"
