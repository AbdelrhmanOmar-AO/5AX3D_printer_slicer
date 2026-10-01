"""Measure how a part's overhangs were actually printed (build plan P0.8).

Runs a part through the stock pipeline at a chosen ``max_slope``, applies the
overhang metrics to the resulting toolpath, and writes a JSON report. With
``--summarize`` it collects every report written so far into one comparison
table.

These are the "before" numbers. P2.5 re-runs the same matrix with the
overhang-aware field and adds its columns beside them, so the two are measured
on identical parts with identical thresholds.

Usage
-----
    python tools/overhang_report.py data/param/ramp60_s.json --max-slope 7
    python tools/overhang_report.py data/param/ramp60_s.json --skip-pipeline
    python tools/overhang_report.py data/param/ramp60_xs.json --max-slope 30 --field-only
    python tools/overhang_report.py --summarize

Which toolpath is measured
--------------------------
``data/toolpath/<part>_smoothed.npz``: the toolpath after ordering and
smoothing, before `tesselate_toolpath_orientations` (which only subdivides
segments so the orientation steps stay small) and before `add_platform` (which
adds sacrificial material beneath the part that is not part of its geometry).

Archived toolpaths
------------------
Every run of a given part writes to the same ``data/`` paths regardless of
``max_slope``, so a later run overwrites the previous one's intermediates. Each
run therefore copies its toolpath to
``reports/toolpaths/<part>_ms<deg>.npz``.

That copy is what makes ``--reanalyse`` possible: a change to the metrics
re-scores every run already performed, in seconds, instead of re-slicing. The
first baseline matrix took 20 hours, and a flaw found in the bed-contact rule
afterwards would otherwise have cost all 20 again.

Provenance
----------
Every report carries a ``provenance`` block (`atom.provenance`, build plan
P1.7): the machine, the backend each stage actually ran on, the Taichi
version, the machine profile and the commit. ``--summarize`` states it above
the table, and when a table pools runs that are not comparable it warns and
labels each cell with its group (plan §0 rule 10).

Field-only mode (build plan P2.0)
---------------------------------
``--field-only`` runs the pipeline only up to atom extraction
(`tools/atomize.py --stop-after extract_explicit_atoms`) and measures the
atoms in ``data/frame/<part>.npz`` instead of a toolpath (`atom.frame_atoms`).
It skips `order_atoms`, 86 % of a run, so a change to the orientation field
(P2.2 to P2.4) can be tried in minutes rather than hours.

It measures the effective overhang angles and the tilt used. It **cannot**
measure unsupported deposition, which needs a print order, and reports it as
``null`` with "n/a (field-only)". So a field-only run can show a part is *not*
printable (its effective angle is over the threshold) but never that it *is*.

Field-only reports go to ``reports/field_only/`` and their atoms to
``reports/frames/`` (gitignored), never beside the full runs, and
``--summarize`` puts them in a table of their own. Their provenance lists
fewer stages than a full run's, so they never share a comparability group
with one either.
"""

# No `from __future__ import annotations`: this module drives Taichi kernels
# through atom.contracts. See docs/plan_corrections.md 3.2.

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom import frame_atoms  # noqa: E402
from atom import overhang_metrics as om  # noqa: E402
from atom import layer_thickness as lt  # noqa: E402
from atom import provenance as prov  # noqa: E402
from atom.ti_env import ARCH_LOG_ENV_VAR  # noqa: E402
from atom.ramp_in import DEFAULT_MAX_TILT_RATE_DEG_PER_MM  # noqa: E402
from atom.ti_env import init_taichi  # noqa: E402

#: Where individual reports and the summary live.
REPORT_DIR = REPO_ROOT / "reports" / "baseline_overhang"
SUMMARY_PATH = REPO_ROOT / "reports" / "baseline_overhang.md"
#: Each run's toolpath, kept so the metrics can be recomputed without re-slicing.
TOOLPATH_ARCHIVE = REPO_ROOT / "reports" / "toolpaths"
#: Field-only reports (build plan P2.0). Kept apart from REPORT_DIR so a
#: field-only result can never be read as a full run's, by a person or by
#: `--status`, `--check-done` and `run_matrix_parallel.py --resume`.
FIELD_ONLY_DIR = REPO_ROOT / "reports" / "field_only"
#: Each field-only run's atoms (``data/frame/<part>.npz``), for `--reanalyse`.
FRAME_ARCHIVE = REPO_ROOT / "reports" / "frames"
#: The ``mode`` a field-only report records. A report without the key is a
#: full run's, as every report written before P2.0 is.
MODE_FIELD_ONLY = "field_only"
MODE_FULL = "full"
#: Where a field-only run stops (`tools/atomize.py --stop-after`).
FIELD_ONLY_LAST_STAGE = "extract_explicit_atoms"
#: What a field-only report says in place of the unsupported fraction.
FIELD_ONLY_UNSUPPORTED_NOTE = (
    "n/a (field-only): unsupported deposition needs a print order, which "
    "only order_atoms produces"
)

SCHEMA_VERSION = 1

#: Version of the *metric semantics*, distinct from the file layout. Bump it
#: whenever a change alters what the numbers mean, so a report produced by an
#: older definition is recognised as stale rather than silently mixed with new
#: ones.
#:
#: A matrix run overwrites the previous run's report files, so after an
#: interrupted re-run the combinations not yet reached still hold results from
#: the old definition. Without this they look complete.
#:
#: 1 - the original definitions.
#: 2 - bed contact anchored to the part's lowest point (plan_corrections 4.5);
#:     surfaces sampled densely rather than by face centroid (4.6);
#:     support radius 2.5x height so the 65-degree cone governs (1.8).
#: 3 - the effective overhang angle counts only points out over the air (one
#:     layer back along the build direction is outside the part and above the
#:     bed); points printed onto the wall below a corner no longer count
#:     (the operator's decision, 2026-09-30, plan_corrections P2-14).
#: 4 - instead, it counts the points whose nearest surface is an overhang
#:     face. Version 3 also dropped the half-supported beads of gentle
#:     overhangs and could find nothing to measure (`ramp45_xs`); the
#:     operator's decision, 2026-10-01, plan_corrections P2-17.
METRICS_VERSION = 4

#: An append-only record of every run, written as each finishes. Survives a
#: crash and gives a human-readable trail beside the per-run JSON.
PROGRESS_LOG = REPO_ROOT / "reports" / "matrix_progress.csv"

#: GATE D0: placeholder thresholds, team decision pending. The same values are
#: used in P2.5, so stock and overhang-aware are judged on one scale.
MAX_EFFECTIVE_OVERHANG_DEG = 45.0
MAX_UNSUPPORTED_FRACTION = 0.01
#: Build plan P2.5: top-surface quality may fall at most this many percentage
#: points below stock's (a placeholder, like the two above).
TOP_SURFACE_ALLOWANCE_POINTS = 5.0
#: GATE D3: the tilt rate's placeholder limit, degrees per mm of printing;
#: the report counts the printing that turns faster. The ramp-in's default.
TILT_RATE_LIMIT_DEG_PER_MM = DEFAULT_MAX_TILT_RATE_DEG_PER_MM

#: Search radius for deposition points near a face, in deposition widths.
SURFACE_SEARCH_WIDTHS = 1.0
#: Radius defining "near an overhang" for the unsupported measurement.
NEAR_OVERHANG_WIDTHS = 2.0
#: Atomizer's layer height as a share of the deposition width
#: (`fff3.LAYER_HEIGHT_WRT_NOZZLE`). A top surface's top layer lies within one
#: layer height of it (`overhang_metrics.top_surface_quality`).
LAYER_HEIGHT_WRT_DEPOSITION_WIDTH = 0.5

_STAGE_TIME = re.compile(
    r"^(?P<label>[A-Z][^\n]*?)\s+took\s+(?P<seconds>[\d.]+)\s+seconds", re.MULTILINE
)


def parse_stage_times(log_text):
    """Per-stage durations from an atomize.py log, as {label: seconds}."""
    return {
        m.group("label").strip(): float(m.group("seconds"))
        for m in _STAGE_TIME.finditer(log_text)
    }


#: What each pipeline stage must leave behind, in order, as
#: (stage label, path template). Used to name the *first* stage that produced
#: nothing, because `atomize.py` cannot.
#:
#: `tools/atomize.py` runs all 13 stages with `os.system` and **ignores every
#: return code**. One early failure therefore produces twelve more, and the only
#: thing that finally errors is this tool finding no toolpath — behind a log full
#: of downstream noise. Diagnosing the first parallel run's failure took a
#: directory listing per stage to find the one real cause.
#:
#: Worse, in a working tree that has run that part before, the *previous* run's
#: outputs are still sitting there, so a failed stage can be read as a
#: successful one and the report is quietly wrong. The freshness half of this
#: check is what catches that.
STAGE_ARTIFACTS = (
    ("1 remesh (Blender)", "data/mesh/{part}.obj"),
    ("2 point-normal cloud", "data/point_normal/{part}.npz"),
    ("3 SDF", "data/sdf/{part}.npz"),
    ("4 direction field", "data/direction/{part}.npz"),
    ("5 implicit layers", "data/phasor/{part}.npz"),
    ("6 tangents", "data/basis/{part}.npz"),
    ("7 atom alignment", "data/triphasor/{part}.npz"),
    ("8 explicit atoms", "data/frame/{part}.npz"),
    ("9 order atoms", "data/toolpath/{part}.npz"),
    ("10a smooth", "data/toolpath/{part}_smoothed.npz"),
    ("10b tesselate", "data/toolpath/{part}_smoothed_tesselated.npz"),
    ("10c add platform", "data/toolpath/{part}_platform.npz"),
    ("11 G-code", "data/gcode/{part}.gcode"),
)

#: The stages a field-only run makes, up to and including the atoms.
FIELD_ONLY_ARTIFACTS = STAGE_ARTIFACTS[
    : [label for label, _ in STAGE_ARTIFACTS].index("8 explicit atoms") + 1
]

#: Slack on the freshness comparison, for filesystem timestamp granularity and
#: any clock adjustment during a long run.
FRESHNESS_TOLERANCE_S = 5.0


def first_failed_stage(part, started_wall_s, root=None, artifacts=STAGE_ARTIFACTS):
    """The first stage whose artifact is missing or older than this run.

    Returns (label, path, reason) or None if every stage delivered. "Older than
    this run" is the case that matters most: the file exists, so nothing looks
    wrong, but it belongs to a previous run of the same part. ``artifacts`` is
    the stages to check, `FIELD_ONLY_ARTIFACTS` for a field-only run.
    """
    root = REPO_ROOT if root is None else Path(root)
    for label, template in artifacts:
        path = root / template.format(part=part)
        if not path.is_file():
            return label, path, "produced nothing"
        if path.stat().st_mtime < started_wall_s - FRESHNESS_TOLERANCE_S:
            return label, path, (
                "left a file from an earlier run; this run did not rewrite it"
            )
    return None


def run_pipeline(param_path, max_slope_deg, verify_stages=True, field_only=False,
                 overhang_aware=False):
    """Run `tools/atomize.py`, optionally overriding ``max_slope``.

    The override is applied through a temporary copy of the parameter file, so
    the committed one is untouched and no vendored stage is modified.

    Returns ``(elapsed_s, params, stage_arches)``. ``stage_arches`` is the
    backend each stage actually started on, which every stage records through
    `atom.ti_env` when ``ATOM_TI_ARCH_LOG`` is set for it.

    ``verify_stages`` checks afterwards that every stage left a fresh artifact
    (see `first_failed_stage`). Only a unit test that stubs `subprocess.run`
    should turn it off: such a test never runs the pipeline, so of course no
    artifact appears, and the check would fire on a test about something else.

    ``field_only`` stops the pipeline after atom extraction (build plan P2.0)
    and checks only the stages that ran.

    ``overhang_aware`` (build plan P2.2) runs the overhang-aware field. Its
    tilt budget, unless ``max_slope_deg`` is given, is the active machine
    profile's limit rather than the part file's ``max_slope``, which is
    written for stock Atomizer (`overhang_aware_slope`).
    """
    params = json.loads(Path(param_path).read_text(encoding="utf-8"))
    if overhang_aware:
        params["overhang_aware"] = True
        if max_slope_deg is None:
            max_slope_deg = overhang_aware_slope()
    if max_slope_deg is not None:
        params["max_slope"] = float(max_slope_deg)

    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp) / Path(param_path).name
        temporary.write_text(json.dumps(params, indent=4), encoding="utf-8")
        arch_log = Path(tmp) / "stage_arches.jsonl"
        env = dict(os.environ, **{ARCH_LOG_ENV_VAR: str(arch_log)})

        command = [sys.executable, "tools/atomize.py", str(temporary)]
        if field_only:
            command += ["--stop-after", FIELD_ONLY_LAST_STAGE]

        started = time.perf_counter()
        started_wall = time.time()
        result = subprocess.run(command, cwd=REPO_ROOT, env=env)
        elapsed = time.perf_counter() - started
        stage_arches = prov.read_arch_log(arch_log)

    if result.returncode != 0:
        raise SystemExit(
            f"atomize.py exited {result.returncode} for {param_path}. "
            "The report cannot be written."
        )

    # atomize.py exits 0 even when a stage failed, so its exit code proves
    # nothing. Find the first stage that did not deliver and say so.
    failure = (
        first_failed_stage(
            params["solid_name"],
            started_wall,
            artifacts=FIELD_ONLY_ARTIFACTS if field_only else STAGE_ARTIFACTS,
        )
        if verify_stages else None
    )
    if failure is not None:
        label, path, reason = failure
        raise SystemExit(
            f"Stage {label} {reason}: {path}\n"
            "atomize.py runs every stage with os.system and ignores the exit "
            "codes, so the stages after this one will have failed too and the "
            "log will be mostly their complaints. This is the one to fix."
        )

    return elapsed, params, stage_arches


def load_mesh_arrays(stl_path, max_edge=None):
    """Face normals, centres and areas from an STL, via trimesh.

    Both metrics sample a surface by searching near each face's **centroid**,
    so a large flat face is represented by a single point. A ramp's overhang is
    two triangles covering 172 mm^2; searching 1.8 mm around their two
    centroids samples 20 mm^2 of it, at two arbitrary spots, and the first
    baseline matrix measured "unsupported near overhangs" from 108 deposition
    points rather than thousands.

    Subdividing every face to at most ``max_edge`` (one deposition width) puts
    centroids roughly a bead apart, so the search covers the surface. Splitting
    a triangle in its own plane changes neither the normals nor the total area,
    which is asserted in the tests.

    Returns the original mesh (for volume and extents) alongside the subdivided
    arrays used for sampling.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dev dependency
        raise SystemExit(
            "trimesh is required to read the mesh. Run: pip install -e \".[dev]\""
        ) from exc

    mesh = trimesh.load_mesh(str(stl_path))
    if max_edge is None or max_edge <= 0:
        return mesh, mesh.face_normals, mesh.triangles_center, mesh.area_faces

    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge
    )
    sampled = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, sampled.face_normals, sampled.triangles_center, sampled.area_faces


def measure(
    part,
    max_slope_deg,
    deposition_width,
    runtime_s=0.0,
    stage_times=None,
    toolpath_path=None,
    stl_path=None,
    provenance=None,
    overhang_aware=False,
):
    """Apply the three metrics and assemble the report.

    ``toolpath_path`` and ``stl_path`` default to the pipeline's own output
    locations; they are arguments so this can be exercised on synthetic data.
    ``provenance`` is the block from `atom.provenance`; None records that it
    is unknown, and the summary then warns.
    """
    from atom import toolpath3

    stage_times = {} if stage_times is None else stage_times
    toolpath_path = Path(
        toolpath_path or REPO_ROOT / "data" / "toolpath" / f"{part}_smoothed.npz"
    )
    if not toolpath_path.is_file():
        raise SystemExit(
            f"No toolpath at {toolpath_path}. Run without --skip-pipeline first."
        )
    stl_path = Path(stl_path or REPO_ROOT / "data" / "mesh" / f"{part}.stl")
    if not stl_path.is_file():
        raise SystemExit(f"No mesh at {stl_path}.")

    toolpath = toolpath3.Toolpath()
    toolpath.load(str(toolpath_path))
    mesh, normals, centres, areas = load_mesh_arrays(
        stl_path, max_edge=deposition_width
    )

    surfaces = om.effective_overhang_angles(
        normals,
        centres,
        areas,
        toolpath,
        search_radius=SURFACE_SEARCH_WIDTHS * deposition_width,
        part_triangles=mesh.triangles,
    )
    overall, near_fraction, near_count = om.unsupported_near_overhangs(
        toolpath,
        normals,
        centres,
        radius=NEAR_OVERHANG_WIDTHS * deposition_width,
    )
    max_tilt = om.max_tool_tilt_deg(toolpath)
    top = om.top_surface_quality(
        toolpath,
        mesh.triangles,
        top_surface_angle_deg(max_slope_deg, provenance),
        LAYER_HEIGHT_WRT_DEPOSITION_WIDTH * deposition_width,
    )
    rate = om.tilt_rate(toolpath, TILT_RATE_LIMIT_DEG_PER_MM)
    machine = _machine_block(toolpath, deposition_width, provenance)
    layers = lt.layer_thickness(toolpath, LAYER_HEIGHT_WRT_DEPOSITION_WIDTH * deposition_width)
    near_points = om.points_near_overhangs(
        np.asarray(toolpath.point[: int(np.asarray(toolpath.point_count).item())], dtype=np.float64)[
            om.deposition_mask(toolpath)
        ],
        normals, centres, NEAR_OVERHANG_WIDTHS * deposition_width,
    )

    measured = [s for s in surfaces if s.measured]
    worst_effective = max((s.max_effective_deg for s in measured), default=float("nan"))

    # A part with no overhang surface at all is not a failure; there is simply
    # nothing to assess. twin_domes exists to check that smooth surfaces are not
    # made worse, and reporting it as "not printable" misreads that.
    assessable = bool(measured)
    printable = bool(
        assessable
        and worst_effective <= MAX_EFFECTIVE_OVERHANG_DEG
        and near_count > 0
        and near_fraction < MAX_UNSUPPORTED_FRACTION
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "metrics_version": METRICS_VERSION,
        "part": part,
        "max_slope_deg": max_slope_deg,
        "overhang_aware": bool(overhang_aware),
        "deposition_width_mm": deposition_width,
        "mesh": _mesh_block(mesh, areas),
        "toolpath": {
            "point_count": int(np.asarray(toolpath.point_count).item()),
            "deposition_count": int(np.count_nonzero(om.deposition_mask(toolpath))),
        },
        "runtime": {"total_s": runtime_s, "stages": stage_times},
        "metrics": {
            "max_tool_tilt_deg": max_tilt,
            "unsupported_fraction_overall": overall.fraction,
            "unsupported_fraction_near_overhangs": near_fraction,
            "deposition_points_near_overhangs": near_count,
            "surfaces": _surfaces_block(surfaces),
            "top_surface": _top_surface_block(top),
            "tilt_rate": _tilt_rate_block(rate),
            "machine": machine,
            "layers": _layers_block(layers, near_points),
        },
        "verdict": {
            "printable": printable,
            "assessable": assessable,
            "worst_effective_deg": worst_effective,
            "thresholds": _thresholds_block(),
        },
        "provenance": provenance,
    }


def _mesh_block(mesh, areas):
    """A report's ``mesh`` entry: the part as modelled, and how densely sampled."""
    return {
        "volume_mm3": float(mesh.volume),
        "face_count": int(len(mesh.faces)),
        "sampled_face_count": int(len(areas)),
        "extents_mm": [float(v) for v in mesh.extents],
    }


def _surfaces_block(surfaces):
    """A report's ``surfaces`` list, one entry per group of equally sloped faces."""
    return [
        {
            "geometric_angle_deg": s.geometric_angle_deg,
            "area_mm2": s.area_mm2,
            "face_count": s.face_count,
            "sample_count": s.sample_count,
            "max_effective_deg": s.max_effective_deg,
            "mean_effective_deg": s.mean_effective_deg,
            "max_tilt_used_deg": s.max_tilt_used_deg,
            "skipped_samples": s.skipped_samples,
        }
        for s in surfaces
    ]


def _top_surface_block(top):
    """A report's ``top_surface`` entry (build plan P2.5, `top_surface_quality`)."""
    return {
        "fraction_on_target": top.fraction_on_target,
        "sample_count": top.sample_count,
        "on_target_count": top.on_target_count,
        "mean_deviation_deg": top.mean_deviation_deg,
        "max_deviation_deg": top.max_deviation_deg,
        "skipped_samples": top.skipped_samples,
        "top_max_angle_deg": top.top_max_angle_deg,
        "tolerance_deg": top.tolerance_deg,
    }


def _tilt_rate_block(rate):
    """A report's ``tilt_rate`` entry (build plan P2.5, `overhang_metrics.tilt_rate`)."""
    return {
        "max_deg_per_mm": rate.max_deg_per_mm,
        "max_at_mm": list(rate.max_at_mm),
        "fraction_over_limit": rate.fraction_over_limit,
        "limit_deg_per_mm": rate.limit_deg_per_mm,
        "stretch_mm": rate.stretch_mm,
        "printing_moves": rate.printing_moves,
    }


def _layers_block(layers, near_points):
    """A report's ``layers`` entry (build plan P3.2, `atom.layer_thickness`)."""

    def counts(where):
        judged = int(np.count_nonzero(layers.judged & where))
        thin = int(np.count_nonzero(layers.thin & where))
        thick = int(np.count_nonzero(layers.thick & where))
        return {
            "judged": judged,
            "thin": thin,
            "thick": thick,
            "fraction_thin": thin / judged if judged else float("nan"),
            "fraction_thick": thick / judged if judged else float("nan"),
            "nothing_below": int(np.count_nonzero(layers.nothing_below & where)),
        }

    thickness = layers.thickness_mm[layers.judged]
    block = counts(np.ones(len(layers.on_bed), dtype=bool))
    block.update({
        "nominal_mm": layers.nominal_mm,
        "min_fraction": layers.min_fraction,
        "max_fraction": layers.max_fraction,
        "median_mm": float(np.median(thickness)) if len(thickness) else float("nan"),
        "p1_mm": float(np.percentile(thickness, 1)) if len(thickness) else float("nan"),
        "p99_mm": float(np.percentile(thickness, 99)) if len(thickness) else float("nan"),
        "near_overhangs": counts(np.asarray(near_points, dtype=bool)),
    })
    return block


def _machine_block(toolpath, deposition_width, provenance):
    """A report's ``machine`` entry: points out of reach and the platform (`atom.machine_reach`).

    The kinematics carry the profile they were imported with; a run made on
    another profile is not scored against the wrong machine but says so.
    """
    from atom import kinematics3z, machine_reach

    wanted = provenance.get("machine_profile") if isinstance(provenance, dict) else None
    active = kinematics3z._PROFILE.name
    if wanted and wanted != active:
        return {"error": f"made on profile {wanted}; scoring loaded {active}. "
                         f"Re-score with ATOM_MACHINE={wanted}."}
    # As `tools/atomize.py` passes it to add_platform: a float32 (0.44999998...).
    layer_height = float(np.float32(LAYER_HEIGHT_WRT_DEPOSITION_WIDTH * deposition_width))
    reach = machine_reach.platform_and_reach(toolpath, deposition_width, layer_height)
    return {
        "profile": reach.profile,
        "points_checked": reach.points_checked,
        "unreachable_points": reach.unreachable_points,
        "platform_mm": reach.platform_mm,
        "lift_mm": reach.lift_mm,
        "unsettled": reach.unsettled,
        "tesselated": reach.tesselated,
    }


def top_surface_angle_deg(max_slope_deg, provenance=None):
    """Which faces are top surfaces: upstream's ceiling angle, `fff3.CEIL_MAX_ANGLE`.

    ``max_slope`` capped at ``(180 - nozzle cone) / 2`` by the machine profile
    the run used (its provenance; the current one when that is not recorded),
    as `compute_tool_orientations.py` sets it. Every matrix slope is under
    the cap on both profiles (50 on `reference`, 60 on `dev60`).
    """
    import warnings

    from atom import machine_profile

    name = provenance.get("machine_profile") if isinstance(provenance, dict) else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cone = machine_profile.load_profile(name).nozzle_cone_angle_deg
    except (OSError, ValueError, KeyError, TypeError):
        return float(max_slope_deg)
    return min(float(max_slope_deg), (180.0 - cone) / 2.0)


def _thresholds_block():
    return {
        "max_effective_overhang_deg": MAX_EFFECTIVE_OVERHANG_DEG,
        "max_unsupported_fraction": MAX_UNSUPPORTED_FRACTION,
        "_gate": "D0: placeholders, team decision pending",
    }


def report_mode(report):
    """``"field_only"`` or ``"full"``; a report without ``mode`` is a full run's."""
    return report.get("mode", MODE_FULL)


def measure_field_only(
    part,
    max_slope_deg,
    deposition_width,
    runtime_s=0.0,
    stage_times=None,
    frame_path=None,
    stl_path=None,
    provenance=None,
    overhang_aware=False,
):
    """The field-only report: the metrics measured on the atoms (build plan P2.0).

    Measures what `measure` does, on ``data/frame/<part>.npz`` instead of a
    toolpath (`atom.frame_atoms` says why the two agree), except unsupported
    deposition, which needs a print order and is reported as ``None``.

    The verdict follows from that: ``printable`` is False when the worst
    effective angle is over the threshold, since that alone rules a part out,
    and None otherwise, because the unsupported half of the test is unknown.
    A field-only run never reports a part as printable.
    """
    stage_times = {} if stage_times is None else stage_times
    frame_path = Path(frame_path or REPO_ROOT / "data" / "frame" / f"{part}.npz")
    if not frame_path.is_file():
        raise SystemExit(
            f"No atoms at {frame_path}. Run without --skip-pipeline first."
        )
    stl_path = Path(stl_path or REPO_ROOT / "data" / "mesh" / f"{part}.stl")
    if not stl_path.is_file():
        raise SystemExit(f"No mesh at {stl_path}.")

    try:
        atoms = frame_atoms.load_frame_atoms(frame_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    mesh, normals, centres, areas = load_mesh_arrays(
        stl_path, max_edge=deposition_width
    )
    surfaces = om.effective_overhang_angles(
        normals,
        centres,
        areas,
        atoms,
        search_radius=SURFACE_SEARCH_WIDTHS * deposition_width,
        part_triangles=mesh.triangles,
    )
    near = om.points_near_overhangs(
        np.asarray(atoms.point, dtype=np.float64),
        normals,
        centres,
        radius=NEAR_OVERHANG_WIDTHS * deposition_width,
    )
    max_tilt = om.max_tool_tilt_deg(atoms)

    measured = [s for s in surfaces if s.measured]
    worst_effective = max((s.max_effective_deg for s in measured), default=float("nan"))
    assessable = bool(measured)
    within = (
        bool(worst_effective <= MAX_EFFECTIVE_OVERHANG_DEG) if assessable else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "metrics_version": METRICS_VERSION,
        "mode": MODE_FIELD_ONLY,
        "part": part,
        "max_slope_deg": max_slope_deg,
        "overhang_aware": bool(overhang_aware),
        "deposition_width_mm": deposition_width,
        "mesh": _mesh_block(mesh, areas),
        "atoms": {
            "count": atoms.point_count,
            "dropped_non_finite": atoms.dropped_non_finite,
        },
        "runtime": {"total_s": runtime_s, "stages": stage_times},
        "metrics": {
            "max_tool_tilt_deg": max_tilt,
            "unsupported_fraction_overall": None,
            "unsupported_fraction_near_overhangs": None,
            "unsupported_note": FIELD_ONLY_UNSUPPORTED_NOTE,
            "atoms_near_overhangs": int(np.count_nonzero(near)),
            "surfaces": _surfaces_block(surfaces),
        },
        "verdict": {
            "printable": False if within is False else None,
            "assessable": assessable,
            "worst_effective_deg": worst_effective,
            "effective_within_threshold": within,
            "thresholds": _thresholds_block(),
        },
        "provenance": provenance,
    }


#: Columns of the progress log. `machine` was appended once a second machine
#: started running the pipeline; `migrate_progress_header` brings an older file
#: up to it rather than writing ragged rows into it.
#: `overhang_aware` was appended for build plan P2.5; older rows are padded
#: with an empty value, and every one of them is a stock run (nothing else was
#: logged before the column existed).
PROGRESS_COLUMNS = (
    "finished_utc", "part", "max_slope_deg", "metrics_version",
    "worst_effective_deg", "unsupported_near_overhangs",
    "max_tilt_used_deg", "printable", "runtime_s", "machine",
    "overhang_aware",
)


def migrate_progress_header(path=None):
    """Bring a progress log written before `machine` existed up to date.

    Returns True if the file was rewritten. Old rows are padded with an empty
    machine rather than guessed at: the log predates the record, and inventing
    a machine for it would be worse than admitting none.

    A ragged CSV is the failure being avoided. This file is the crash trail from
    correction 4.7, so it must stay readable by anything that opens it.
    """
    path = PROGRESS_LOG if path is None else path
    if not path.is_file():
        return False

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or tuple(rows[0]) == PROGRESS_COLUMNS:
        return False

    width = len(PROGRESS_COLUMNS)
    padded = [list(PROGRESS_COLUMNS)]
    for row in rows[1:]:
        padded.append((list(row) + [""] * width)[:width])

    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(padded)
    return True


def append_progress(report):
    """Append one line to the progress log, flushed immediately.

    The per-run JSON is the real artifact; this is the at-a-glance trail that
    survives a crash and shows what had been completed and when — and on which
    machine, since a run on another machine is a different computation
    (corrections 3.9 and 4.14).
    """
    metrics, verdict = report["metrics"], report["verdict"]
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)

    new_file = not PROGRESS_LOG.exists()
    if not new_file:
        migrate_progress_header()

    provenance = report.get("provenance") or {}
    with PROGRESS_LOG.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(list(PROGRESS_COLUMNS))
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            report["part"],
            f"{report['max_slope_deg']:g}",
            report.get("metrics_version", ""),
            f"{verdict['worst_effective_deg']:.2f}",
            f"{metrics['unsupported_fraction_near_overhangs']:.4f}",
            f"{metrics['max_tool_tilt_deg']:.2f}",
            verdict["printable"],
            f"{report['runtime']['total_s']:.0f}",
            provenance.get("machine") or "",
            bool(report.get("overhang_aware", False)),
        ])
        handle.flush()
        os.fsync(handle.fileno())


#: Appended to an overhang-aware run's file names (build plan P2.2), so stock
#: and overhang-aware results of one part and slope never overwrite each other.
AWARE_SUFFIX = "_aware"


def _stem(part, max_slope_deg, overhang_aware=False):
    return f"{part}_ms{max_slope_deg:g}{AWARE_SUFFIX if overhang_aware else ''}"


def report_path(part, max_slope_deg, overhang_aware=False):
    return REPORT_DIR / f"{_stem(part, max_slope_deg, overhang_aware)}.json"


def archive_path(part, max_slope_deg, overhang_aware=False):
    return TOOLPATH_ARCHIVE / f"{_stem(part, max_slope_deg, overhang_aware)}.npz"


def archive_toolpath(part, max_slope_deg, overhang_aware=False):
    """Keep this run's toolpath, so its metrics can be recomputed later."""
    source = REPO_ROOT / "data" / "toolpath" / f"{part}_smoothed.npz"
    if not source.is_file():
        return None
    TOOLPATH_ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = archive_path(part, max_slope_deg, overhang_aware)
    shutil.copy2(source, destination)
    return destination


def field_only_report_path(part, max_slope_deg, overhang_aware=False):
    return FIELD_ONLY_DIR / f"{_stem(part, max_slope_deg, overhang_aware)}.json"


def frame_archive_path(part, max_slope_deg, overhang_aware=False):
    return FRAME_ARCHIVE / f"{_stem(part, max_slope_deg, overhang_aware)}.npz"


def archive_frame(part, max_slope_deg, overhang_aware=False):
    """Keep a field-only run's atoms, so its metrics can be recomputed later."""
    source = REPO_ROOT / "data" / "frame" / f"{part}.npz"
    if not source.is_file():
        return None
    FRAME_ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = frame_archive_path(part, max_slope_deg, overhang_aware)
    shutil.copy2(source, destination)
    return destination


def overhang_aware_slope():
    """The tilt budget of an overhang-aware run: the active machine profile's limit.

    The operator's decision of 2026-09-30 (plan_corrections 7c P2-11): the
    part files' ``max_slope`` values (7 degrees) are stock Atomizer's.
    """
    from atom import machine_profile

    return float(machine_profile.load_profile().max_tilt_angle_deg)


def measured_elsewhere(report):
    """The machine a report was measured on, when it is not this one; else None.

    A report's archived toolpath or atoms are looked up by part and slope
    only, and a machine that ran the same parts itself has its own files under
    those names: re-scoring another machine's report here would pair it with
    this machine's output. None too when the report does not say.
    """
    block = report.get("provenance")
    machine = block.get("machine") if isinstance(block, dict) else None
    here = platform.node() or None
    if isinstance(machine, str) and isinstance(here, str) and machine.lower() != here.lower():
        return machine
    return None


def reanalyse(reports):
    """Re-score every archived run against the current metrics.

    Returns (rewritten, skipped). A run whose toolpath was not archived cannot
    be re-scored and is reported rather than silently left stale.
    """
    init_taichi("cpu")
    rewritten, skipped = [], []

    for old in reports:
        part, slope = old["part"], old["max_slope_deg"]
        aware = bool(old.get("overhang_aware", False))
        elsewhere = measured_elsewhere(old)
        if elsewhere:
            skipped.append(
                f"{part} @ {slope:g}{' aware' if aware else ''} (measured on {elsewhere}; re-score it there)"
            )
            continue
        archived = archive_path(part, slope, aware)
        if not archived.is_file():
            skipped.append(f"{part} @ {slope:g}{' aware' if aware else ''} (no archived toolpath)")
            continue

        # The toolpath still comes from the original run, so its provenance is
        # kept; only the scoring is new. A report without one stays without.
        old_provenance = old.get("provenance")
        fresh = measure(
            part,
            slope,
            old.get("deposition_width_mm", 0.9),
            runtime_s=old.get("runtime", {}).get("total_s", 0.0),
            stage_times=old.get("runtime", {}).get("stages", {}),
            toolpath_path=archived,
            provenance=(
                prov.rescored(old_provenance, REPO_ROOT, METRICS_VERSION)
                if isinstance(old_provenance, dict) else None
            ),
            overhang_aware=aware,
        )
        report_path(part, slope, aware).write_text(
            json.dumps(fresh, indent=2) + "\n", encoding="utf-8"
        )
        rewritten.append(fresh)

    return rewritten, skipped


def reanalyse_field_only(reports):
    """Re-score every field-only run whose atoms were archived.

    Returns (rewritten, skipped), as `reanalyse` does for full runs.
    """
    rewritten, skipped = [], []

    for old in reports:
        part, slope = old["part"], old["max_slope_deg"]
        aware = bool(old.get("overhang_aware", False))
        elsewhere = measured_elsewhere(old)
        if elsewhere:
            skipped.append(
                f"{part} @ {slope:g} field-only{' aware' if aware else ''} "
                f"(measured on {elsewhere}; re-score it there)"
            )
            continue
        archived = frame_archive_path(part, slope, aware)
        if not archived.is_file():
            skipped.append(
                f"{part} @ {slope:g} field-only{' aware' if aware else ''} (no archived atoms)"
            )
            continue

        old_provenance = old.get("provenance")
        fresh = measure_field_only(
            part,
            slope,
            old.get("deposition_width_mm", 0.9),
            runtime_s=old.get("runtime", {}).get("total_s", 0.0),
            stage_times=old.get("runtime", {}).get("stages", {}),
            frame_path=archived,
            provenance=(
                prov.rescored(old_provenance, REPO_ROOT, METRICS_VERSION)
                if isinstance(old_provenance, dict) else None
            ),
            overhang_aware=aware,
        )
        field_only_report_path(part, slope, aware).write_text(
            json.dumps(fresh, indent=2) + "\n", encoding="utf-8"
        )
        rewritten.append(fresh)

    return rewritten, skipped


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def provenance_groups(reports):
    """Group reports by comparability: ``[(label, description, reports), ...]``.

    Largest group first, labelled A, B, C ... A single group means every run in
    the table can be pooled.
    """
    groups = {}
    for report in reports:
        groups.setdefault(prov.comparability_key(report.get("provenance")), []).append(report)
    ordered = sorted(groups.values(), key=lambda members: -len(members))
    return [
        (chr(ord("A") + index), prov.describe(members[0].get("provenance")), members)
        for index, members in enumerate(ordered)
    ]


def provenance_warning(reports):
    """The warning text when ``reports`` are not all comparable, else None.

    Also warns when every run shares one group but its origin is unknown: no
    provenance recorded, or measured with ``--skip-pipeline``.
    """
    groups = provenance_groups(reports)
    if len(groups) == 1:
        if prov.is_known(groups[0][2][0].get("provenance")):
            return None
        return (
            f"WARNING: where these {len(reports)} run(s) came from is unknown "
            f"({groups[0][1]}). Record provenance before comparing them "
            "(plan section 0 rule 10)."
        )
    lines = [
        f"WARNING: this table mixes {len(groups)} provenance groups that are not "
        "comparable (plan section 0 rule 10). Do not compare numbers across groups:"
    ]
    for label, description, members in groups:
        lines.append(f"  [{label}] {len(members)} run(s): {description}")
    return "\n".join(lines)


def _provenance_section(reports):
    """Where the numbers came from, above the table."""
    groups = provenance_groups(reports)
    if len(groups) == 1:
        if prov.is_known(groups[0][2][0].get("provenance")):
            return [f"Measured on: {groups[0][1]}. All {len(reports)} runs are comparable.", ""]
        return [
            f"> ⚠️ **Origin unknown** for all {len(reports)} run(s): {groups[0][1]}. "
            "Record provenance before comparing them (plan §0 rule 10).",
            "",
        ]
    lines = [
        f"> ⚠️ **Mixed provenance: {len(groups)} groups.** Numbers from different "
        "groups are not comparable (plan §0 rule 10). Each cell is labelled with "
        "its group.",
        ">",
    ]
    for label, description, members in groups:
        lines.append(f"> **[{label}]** {len(members)} run(s): {description}")
    return lines + [""]


def _format_cell(report):
    """One table cell: worst effective angle / % unsupported / max tilt used."""
    metrics, verdict = report["metrics"], report["verdict"]
    worst = verdict["worst_effective_deg"]
    near = metrics["unsupported_fraction_near_overhangs"]

    tilt_text = f"{metrics['max_tool_tilt_deg']:.1f}°"

    # No overhang surface to assess is not a failure. Say so rather than
    # marking it failed; twin_domes is in the study to check that smooth
    # surfaces are not made worse, not to pass an overhang threshold.
    if not verdict.get("assessable", True):
        return f"– no overhang / tilt {tilt_text}"

    worst_text = "n/m" if math.isnan(worst) else f"{worst:.0f}°"
    near_text = "n/m" if math.isnan(near) else f"{near * 100:.1f}%"
    mark = "✅" if verdict["printable"] else "❌"
    return f"{mark} {worst_text} / {near_text} / {tilt_text}"


def _format_field_only_cell(report):
    """One field-only cell: worst effective angle / n/a / max tilt used.

    ❌ when the effective angle alone rules the part out; ❔ when it is within
    the threshold, since the unsupported half of the test is unknown.
    """
    metrics, verdict = report["metrics"], report["verdict"]
    tilt_text = f"{metrics['max_tool_tilt_deg']:.1f}°"
    if not verdict.get("assessable", True):
        return f"– no overhang / tilt {tilt_text}"
    mark = "❌" if verdict.get("printable") is False else "❔"
    return f"{mark} {verdict['worst_effective_deg']:.0f}° / n/a / {tilt_text}"


def _table(reports, format_cell):
    """Parts down, slopes across, one cell per report; group labels if mixed."""
    parts = sorted({r["part"] for r in reports})
    slopes = sorted({r["max_slope_deg"] for r in reports})
    by_key = {(r["part"], r["max_slope_deg"]): r for r in reports}

    label_of = {}
    groups = provenance_groups(reports)
    if len(groups) > 1:
        for label, _, members in groups:
            for member in members:
                label_of[id(member)] = f" [{label}]"

    header = "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |"
    lines = [header, "|" + "---|" * (len(slopes) + 1)]
    for part in parts:
        cells = [
            format_cell(by_key[(part, s)]) + label_of.get(id(by_key[(part, s)]), "")
            if (part, s) in by_key else "—"
            for s in slopes
        ]
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")
    return lines


def _field_only_section(reports):
    """The field-only runs, in a table of their own (build plan P2.0 step 3)."""
    lines = [
        "## Field-only runs (build plan P2.0)",
        "",
        "Measured on the extracted atoms, before `order_atoms` "
        "(`overhang_report.py --field-only`). Each cell is **worst effective "
        "overhang angle / unsupported deposition / maximum tilt used**.",
        "",
        "Unsupported deposition needs a print order, so it is `n/a` here. A cell "
        "can therefore show that a part fails (❌: its effective angle is over "
        f"{MAX_EFFECTIVE_OVERHANG_DEG:g}°) but never that it prints (❔: the "
        "angle is within the threshold, the rest is unknown).",
        "",
        "**Not comparable with the table above.** These runs stop after atom "
        "extraction, so they form their own provenance group, and they count "
        "every atom where a full run counts the deposition points of its "
        "toolpath.",
        "",
    ]
    stock = [r for r in reports if not r.get("overhang_aware", False)]
    aware = [r for r in reports if r.get("overhang_aware", False)]
    for title, group in (("Stock", stock), ("Overhang-aware (build plan P2.2)", aware)):
        if not group:
            continue
        if stock and aware:
            lines += [f"### {title}", ""]
        elif aware:
            lines += ["All overhang-aware (build plan P2.2).", ""]
        lines += _provenance_section(group)
        lines += _table(group, _format_field_only_cell) + [""]
    return lines[:-1]


def _short_cell(report):
    """Mark, worst effective angle and unsupported deposition: half a comparison cell."""
    metrics, verdict = report["metrics"], report["verdict"]
    if not verdict.get("assessable", True):
        return "– no overhang"
    worst = verdict["worst_effective_deg"]
    near = metrics["unsupported_fraction_near_overhangs"]
    worst_text = "n/m" if math.isnan(worst) else f"{worst:.0f}°"
    near_text = "n/m" if near is None or math.isnan(near) else f"{near * 100:.1f}%"
    return f"{'✅' if verdict['printable'] else '❌'} {worst_text} / {near_text}"


def _comparison_section(stock, aware):
    """Stock and overhang-aware runs side by side (build plan P2.5).

    One cell per part and slope that has an overhang-aware run: **stock ->
    overhang-aware**. Warns above the table when the pairs are not all from
    one provenance group, or were scored with different metrics versions,
    since then the arrows compare different measurements.
    """
    stock_by_key = {(r["part"], r["max_slope_deg"]): r for r in stock}
    aware_by_key = {(r["part"], r["max_slope_deg"]): r for r in aware}
    pairs = [(stock_by_key.get(key), report) for key, report in sorted(aware_by_key.items())]
    compared = [s for s, _ in pairs if s is not None] + [a for _, a in pairs]

    lines = [
        "## Stock vs overhang-aware (build plan P2.5)",
        "",
        "Each cell is **stock → overhang-aware** for the same part and "
        "max_slope: worst effective overhang angle / unsupported deposition "
        "near overhangs. Overhang-aware runs: `overhang_report.py "
        "--overhang-aware`, or `run_matrix_parallel.py --overhang-aware` for the "
        "matrix. `—` means that side has no run.",
        "",
    ]
    groups = provenance_groups(compared)
    if len(groups) == 1 and prov.is_known(groups[0][2][0].get("provenance")):
        lines += [
            f"Stock and overhang-aware runs share one provenance group: "
            f"{groups[0][1]}. Comparable.",
            "",
        ]
    else:
        lines += [
            f"> ⚠️ **Not comparable as they stand: {len(groups)} provenance "
            "group(s)** (plan §0 rule 10, build plan P2.5: stock and "
            "overhang-aware must come from the same machine and backend).",
            ">",
        ]
        for label, description, members in groups:
            kinds = sorted({"overhang-aware" if m.get("overhang_aware") else "stock" for m in members})
            lines.append(f"> **[{label}]** {len(members)} run(s), {' and '.join(kinds)}: {description}")
        lines.append("")
    versions = sorted({r.get("metrics_version") for r in compared}, key=str)
    if len(versions) > 1:
        lines += [
            f"> ⚠️ **Scored with different metrics versions ({', '.join(map(str, versions))}).** "
            "Re-score the older runs (`overhang_report.py --reanalyse`, on the "
            "machine that measured them) before comparing.",
            "",
        ]

    slopes = sorted({key[1] for key in aware_by_key})
    parts = sorted({key[0] for key in aware_by_key})
    lines += [
        "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |",
        "|" + "---|" * (len(slopes) + 1),
    ]
    for part in parts:
        cells = []
        for slope in slopes:
            a = aware_by_key.get((part, slope))
            st = stock_by_key.get((part, slope))
            if a is None:
                cells.append("—")
            else:
                cells.append(f"{_short_cell(st) if st else '—'} → {_short_cell(a)}")
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")

    both = [(s, a) for s, a in pairs if s is not None]
    stock_ok = sum(bool(s["verdict"]["printable"]) for s, _ in both)
    aware_ok = sum(bool(a["verdict"]["printable"]) for _, a in both)
    lines += [
        "",
        f"Printable, over the {len(both)} part and slope pair(s) with both runs: "
        f"stock {stock_ok}, overhang-aware {aware_ok}.",
    ]
    lines += [""] + _top_surface_section(parts, slopes, stock_by_key, aware_by_key)
    lines += [""] + _tilt_rate_section(parts, slopes, stock_by_key, aware_by_key)
    lines += [""] + _machine_section(parts, slopes, stock_by_key, aware_by_key)
    lines += [""] + _layers_section(parts, slopes, stock_by_key, aware_by_key)
    return lines


def _layers_text(report):
    if report is None:
        return "—"
    layers = report.get("metrics", {}).get("layers")
    if layers is None:
        return "not scored"
    thin, thick = layers.get("fraction_thin"), layers.get("fraction_thick")
    if thin is None or thick is None or math.isnan(thin) or math.isnan(thick):
        return "n/m"
    return f"{thin * 100:.1f}% / {thick * 100:.1f}%"


def _layers_section(parts, slopes, stock_by_key, aware_by_key):
    """Layer thickness, stock -> overhang-aware (build plan P3.2)."""
    lines = [
        "### Layer thickness",
        "",
        "Each cell is **stock → overhang-aware**: the share of deposition points "
        f"whose layer is thinner than {lt.MIN_LAYER_FRACTION:g}x / thicker than "
        f"{lt.MAX_LAYER_FRACTION:g}x the nominal (P3.2's placeholders), the "
        "thickness measured from the toolpath's geometry, not its `height` "
        "(`atom.layer_thickness`; plan_corrections P2-25). A thick one is "
        "mostly a bead with the bead under it missing: a one-layer gap. The "
        "reports also give these near the overhangs.",
        "",
        "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |",
        "|" + "---|" * (len(slopes) + 1),
    ]
    for part in parts:
        cells = []
        for slope in slopes:
            aware = aware_by_key.get((part, slope))
            if aware is None:
                cells.append("—")
            else:
                cells.append(f"{_layers_text(stock_by_key.get((part, slope)))} → {_layers_text(aware)}")
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")
    return lines


def _machine_text(report):
    if report is None:
        return "—"
    machine = report.get("metrics", {}).get("machine")
    if machine is None:
        return "not scored"
    if "error" in machine:
        return "other profile"
    if machine.get("unreachable_points"):
        return f"⚠️ {machine['unreachable_points']} out of reach"
    return f"{machine['platform_mm']:.0f} mm"


def _machine_section(parts, slopes, stock_by_key, aware_by_key):
    """Reach and platform, stock -> overhang-aware (build plan P2.5)."""
    lines = [
        "### Reach and platform",
        "",
        "Each cell is **stock → overhang-aware**: the platform `add_platform` "
        "prints under the part so the tilted bed clears the gantry (`0 mm`: "
        "none), or ⚠️ the number of toolpath points the machine cannot reach "
        "at any lift (`atom.machine_reach`, the toolpath tesselated as the "
        "pipeline does). P2.5 asks for no new points out of reach. A run whose "
        "points are out of reach stops at `add_platform`, so it has no report "
        "and does not appear here; `other profile`: made on another machine "
        "profile than the one scoring it.",
        "",
        "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |",
        "|" + "---|" * (len(slopes) + 1),
    ]
    out_of_reach = 0
    for part in parts:
        cells = []
        for slope in slopes:
            aware = aware_by_key.get((part, slope))
            if aware is None:
                cells.append("—")
                continue
            machine = aware.get("metrics", {}).get("machine") or {}
            out_of_reach += bool(machine.get("unreachable_points"))
            cells.append(f"{_machine_text(stock_by_key.get((part, slope)))} → {_machine_text(aware)}")
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")
    return lines + ["", f"Overhang-aware runs with points out of reach: {out_of_reach}."]


def _tilt_rate_text(report):
    if report is None:
        return "—"
    rate = report.get("metrics", {}).get("tilt_rate")
    if rate is None:
        return "not scored"
    value, share = rate.get("max_deg_per_mm"), rate.get("fraction_over_limit")
    if value is None or math.isnan(value):
        return "n/m"
    return f"{value:.0f}°/mm ({share * 100:.0f}%)"


def _tilt_rate_section(parts, slopes, stock_by_key, aware_by_key):
    """The tilt rate along the printing, stock -> overhang-aware (build plan P2.5)."""
    lines = [
        "### Tilt rate",
        "",
        "Each cell is **stock → overhang-aware**: the largest turn of the tool "
        f"within any {om.TILT_RATE_STRETCH_MM:g} mm of continuous printing, in "
        "degrees per mm, and in brackets the share of printing moves where the "
        f"{om.TILT_RATE_STRETCH_MM:g} mm of printing from there turns faster than "
        f"{TILT_RATE_LIMIT_DEG_PER_MM:g}°/mm (the gate D3 placeholder; "
        "`overhang_metrics.tilt_rate`). Reported, not judged: P2.5 sets no "
        "limit on it.",
        "",
        "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |",
        "|" + "---|" * (len(slopes) + 1),
    ]
    for part in parts:
        cells = []
        for slope in slopes:
            aware = aware_by_key.get((part, slope))
            if aware is None:
                cells.append("—")
            else:
                cells.append(f"{_tilt_rate_text(stock_by_key.get((part, slope)))} → {_tilt_rate_text(aware)}")
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")
    return lines


def _top_surface_value(report):
    """A report's top-surface fraction; None when not scored, NaN when no top."""
    top = (report or {}).get("metrics", {}).get("top_surface")
    if top is None:
        return None
    value = top.get("fraction_on_target")
    return float("nan") if value is None else float(value)


def _top_surface_text(report):
    value = _top_surface_value(report)
    if report is None:
        return "—"
    if value is None:
        return "not scored"
    return "n/m" if math.isnan(value) else f"{value * 100:.0f}%"


def _top_surface_section(parts, slopes, stock_by_key, aware_by_key):
    """Top-surface quality, stock -> overhang-aware (build plan P2.5)."""
    lines = [
        "### Top-surface quality",
        "",
        "Each cell is **stock → overhang-aware**: the share of the top layer's "
        f"deposition points printed within {om.TOP_SURFACE_TOLERANCE_DEG:g}° of "
        "the top surface's normal (`overhang_metrics.top_surface_quality`; top "
        "surfaces are the faces within max_slope of level, upstream's ceilings). "
        f"P2.5 asks for overhang-aware ≥ stock − {TOP_SURFACE_ALLOWANCE_POINTS:g} "
        "points; ⚠️ marks a pair that misses it. `n/m`: no top surface measured; "
        "`not scored`: the report predates the measure (`--reanalyse` adds it).",
        "",
        "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |",
        "|" + "---|" * (len(slopes) + 1),
    ]
    misses = compared = unscored = 0
    for part in parts:
        cells = []
        for slope in slopes:
            aware = aware_by_key.get((part, slope))
            stock = stock_by_key.get((part, slope))
            if aware is None:
                cells.append("—")
                continue
            cell = f"{_top_surface_text(stock)} → {_top_surface_text(aware)}"
            s, a = _top_surface_value(stock), _top_surface_value(aware)
            if stock is None:
                pass
            elif s is None or a is None:
                unscored += 1
            elif not (math.isnan(s) or math.isnan(a)):
                compared += 1
                if a * 100 < s * 100 - TOP_SURFACE_ALLOWANCE_POINTS:
                    misses += 1
                    cell += " ⚠️"
            cells.append(cell)
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")
    summary = (
        f"Pairs within {TOP_SURFACE_ALLOWANCE_POINTS:g} points of stock or better: "
        f"{compared - misses} of {compared}."
    )
    if unscored:
        summary += f" {unscored} pair(s) not scored yet."
    return lines + ["", summary]


def summarize(reports, field_only_reports=(), aware_reports=()):
    """Build the comparison table and the plain-language conclusion.

    ``reports`` are the stock full runs, the baseline. ``aware_reports``
    (full overhang-aware runs) get a section of their own after the
    conclusion, side by side with the stock run of the same part and slope
    (build plan P2.5); ``field_only_reports`` (P2.0, stock and overhang-aware)
    get their own too. Neither enters the main table.
    """
    slopes = sorted({r["max_slope_deg"] for r in reports})

    lines = [
        "# Overhang baseline: stock Atomizer",
        "",
        "Produced by `tools/overhang_report.py --summarize` (build plan P0.8).",
        "",
        "Each cell is **worst effective overhang angle / unsupported deposition "
        "near overhangs / maximum tilt actually used**.",
        "",
        f"A part counts as printable when the worst effective overhang stays at or "
        f"below {MAX_EFFECTIVE_OVERHANG_DEG:g}° *and* unsupported deposition near "
        f"the overhangs is under {MAX_UNSUPPORTED_FRACTION * 100:g}%. "
        "Those thresholds are placeholders pending gate D0.",
        "",
        "`n/m` means not measured: no deposition was found near that surface.",
        "",
    ]
    lines += _provenance_section(reports)
    lines += _table(reports, _format_cell)

    lines += ["", "## Conclusion", ""] + _conclusion(reports, slopes)
    if aware_reports:
        lines += [""] + _comparison_section(reports, list(aware_reports))
    if field_only_reports:
        lines += [""] + _field_only_section(list(field_only_reports))
    lines += [""] + _timing_section(reports)
    return "\n".join(lines) + "\n"


def _size_of(part):
    """The size suffix of a generated benchmark part, or None."""
    match = re.match(r".*_(xs|s|m|l)$", part)
    return match.group(1) if match else None


def run_condition(report):
    """How the run shared its machine: ``"alone"``, ``"8 workers"``, or
    ``"not recorded"`` for a report that predates the field (P1.7 follow-up)."""
    workers = (report.get("provenance") or {}).get("parallel_workers")
    if workers is None:
        return "not recorded"
    return "alone" if workers == 1 else f"{workers} workers"


def _timing_section(reports):
    """How pipeline cost scaled with part volume, from the runs themselves.

    Nobody has published how Atomizer's ordering stage scales, so this is a
    result in its own right rather than only a planning aid. It is built from
    whatever has been run; a single size gives a single row.

    Each row says whether the run had the machine to itself. Runs made in a
    worker pool are slower by contention (1.47x at 8 workers on the lab
    machine), so their times measure the load as much as the part.
    """
    timed = [r for r in reports if r.get("runtime", {}).get("total_s", 0) > 0]
    if not timed:
        return [
            "## Runtime",
            "",
            "No run recorded a duration. Reports produced with `--skip-pipeline` "
            "carry no timing.",
        ]

    order = {"xs": 0, "s": 1, "m": 2, "l": 3}
    rows = []
    for report in sorted(
        timed,
        key=lambda r: (order.get(_size_of(r["part"]), 9), r["part"], r["max_slope_deg"]),
    ):
        stages = report.get("runtime", {}).get("stages", {})
        ordering = next(
            (v for k, v in stages.items() if "planner" in k.lower()), float("nan")
        )
        volume = report["mesh"]["volume_mm3"]
        points = report["toolpath"]["point_count"]
        total = report["runtime"]["total_s"]

        rows.append(
            f"| `{report['part']}` | {report['max_slope_deg']:g}° | "
            f"{volume:,.0f} | {points:,} | "
            + ("n/a" if math.isnan(ordering) else f"{ordering / 60:.1f}")
            + f" | {total / 60:.1f} | {run_condition(report)} |"
        )

    section = [
        "## Runtime",
        "",
        "How the pipeline's cost scaled with part size. `order_atoms` is the "
        "stage that dominates, and it runs on the CPU.",
        "",
        "| Part | max_slope | Volume mm³ | Toolpath points | order_atoms min | Total min | Run |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]

    conditions = {run_condition(r) for r in timed}
    if conditions != {"alone"}:
        section += [
            "",
            "**Not all of these runs had the machine to itself** (the Run "
            "column). A run sharing it with others is slower by contention, "
            "1.47x at 8 workers on the lab machine, so compare times only "
            "between runs made the same way, and do not fit a scaling curve "
            "across them.",
        ]

    sizes = {_size_of(r["part"]) for r in timed} - {None}
    if len(sizes) < 2:
        section += [
            "",
            "Only one part size has been run, so this is a single point rather "
            "than a scaling curve. Run another size to get the trend.",
        ]
    return section


def _conclusion(reports, slopes):
    """Answer P0.8's question: how steep an overhang does stock Atomizer manage?"""
    ramps = [r for r in reports if r["part"].startswith("ramp")]
    if not ramps:
        return [
            "No ramp parts have been measured yet, so the printable overhang "
            "angle cannot be stated. Run the ramp family before drawing "
            "conclusions."
        ]

    def ramp_angle(report):
        return float(re.match(r"ramp(\d+)", report["part"]).group(1))

    ramps = [r for r in ramps if r["verdict"].get("assessable", True)]
    if not ramps:
        return [
            "No ramp presented a measurable overhang surface, so nothing can be "
            "concluded about the printable angle."
        ]
    printable = [r for r in ramps if r["verdict"]["printable"]]
    best = max((ramp_angle(r) for r in printable), default=None)
    worst_failed = min(
        (ramp_angle(r) for r in ramps if not r["verdict"]["printable"]), default=None
    )

    budget_use = [
        (r["max_slope_deg"], r["metrics"]["max_tool_tilt_deg"]) for r in reports
    ]
    spent = [used / limit for limit, used in budget_use if limit]

    sentences = []
    if best is None:
        sentences.append(
            "Stock Atomizer did not print any ramp in the family within the "
            "thresholds, so its unsupported overhang limit is below the "
            f"shallowest angle measured ({min(ramp_angle(r) for r in ramps):.0f}°)."
        )
    else:
        sentences.append(
            f"Stock Atomizer printed overhangs up to **{best:.0f}°** from vertical "
            "within the thresholds."
        )
        if worst_failed is not None:
            sentences.append(
                f"The shallowest ramp it failed was {worst_failed:.0f}°, so the "
                "limit lies between those two angles."
            )

    if spent:
        sentences.append(
            f"Across all runs it used between {min(spent) * 100:.0f}% and "
            f"{max(spent) * 100:.0f}% of the tilt budget it was given, which says "
            "whether raising `max_slope` alone would achieve anything."
        )

    if len(slopes) > 1:
        sentences.append(
            f"Raising `max_slope` from {min(slopes):g}° to {max(slopes):g}° is the "
            "comparison to read across each row: the field constrains only the "
            "first layer and low-curvature top surfaces, so a larger budget need "
            "not produce more tilt where an overhang actually is."
        )

    sentences.append(
        "These are the numbers the overhang-aware field is measured against in "
        "P2.5, on the same parts with the same thresholds."
    )
    return [" ".join(sentences)]


def load_reports(quiet=False, field_only=False, overhang_aware=False):
    """Every valid report on disk: the full runs, or with ``field_only`` the
    field-only ones (build plan P2.0).

    ``overhang_aware`` (build plan P2.2) picks stock runs (False, the
    default), overhang-aware ones (True) or both (None). The default keeps
    every older caller (`--status`, `--check-done`, the parallel runner's
    `--resume`, the baseline table) on stock runs only.

    A report truncated mid-write, which a power loss can do, is reported and
    skipped rather than taking the whole summary down with it. So is a report
    of the other kind found in the wrong folder: pooling a field-only result
    with full runs is the confusion P2.0 exists to prevent.
    """
    directory = FIELD_ONLY_DIR if field_only else REPORT_DIR
    expected_mode = MODE_FIELD_ONLY if field_only else MODE_FULL
    if not directory.is_dir():
        return []

    reports = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  CORRUPT {path.name}: {exc}. Delete it and re-run that part.")
            continue
        if data.get("schema_version") != SCHEMA_VERSION:
            if not quiet:
                print(
                    f"  Skipping {path.name}: schema version "
                    f"{data.get('schema_version')}, expected {SCHEMA_VERSION}"
                )
            continue
        if report_mode(data) != expected_mode:
            print(
                f"  Skipping {path.name}: a {report_mode(data)} report in "
                f"{directory}, which holds {expected_mode} reports only. Move it."
            )
            continue
        if overhang_aware is not None and bool(data.get("overhang_aware", False)) != overhang_aware:
            continue
        reports.append(data)
    return reports


#: The parts and slopes a full baseline matrix covers (build plan P0.8 step 5).
MATRIX_PARTS = (
    "ramp45", "ramp50", "ramp60", "ramp70", "ramp80", "ramp90", "tshape", "twin_domes",
)
MATRIX_SLOPES = (7.0, 15.0, 30.0)


def matrix_status(sizes, parts=MATRIX_PARTS, slopes=MATRIX_SLOPES):
    """Which combinations of the matrix have a report, and which do not.

    Each run writes its report as soon as it finishes, so an interrupted matrix
    keeps everything completed up to that point. This says what is left.
    """
    reports = load_reports(quiet=True)
    done = {
        (r["part"], r["max_slope_deg"])
        for r in reports
        if r.get("metrics_version") == METRICS_VERSION
    }
    stale = {
        (r["part"], r["max_slope_deg"])
        for r in reports
        if r.get("metrics_version") != METRICS_VERSION
    }

    present, missing = [], []
    for size in sizes:
        for part in parts:
            for slope in slopes:
                key = (f"{part}_{size}", slope)
                (present if key in done else missing).append(key)
    return present, missing, stale


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "param_path", nargs="?", type=Path, help="Path to the part's parameter JSON."
    )
    parser.add_argument(
        "--max-slope", type=float, default=None,
        help="Override max_slope in degrees for this run.",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Measure the toolpath already in data/ instead of re-running.",
    )
    parser.add_argument(
        "--overhang-aware", action="store_true",
        help=(
            "Build plan P2.2: run the overhang-aware field. Its tilt budget is "
            "the machine profile's limit (ATOM_MACHINE) unless --max-slope is "
            "given. Reports are named <part>_ms<deg>_aware, beside the stock ones."
        ),
    )
    parser.add_argument(
        "--field-only", action="store_true",
        help=(
            "Build plan P2.0: run the pipeline only up to atom extraction and "
            "measure the atoms, skipping order_atoms (86%% of a run). "
            "Unsupported deposition cannot be measured and is reported as n/a. "
            "Writes to reports/field_only/, never beside the full runs. With "
            "--skip-pipeline, measures the atoms already in data/frame/."
        ),
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Collect every report into reports/baseline_overhang.md.",
    )
    parser.add_argument(
        "--check-done", nargs=2, metavar=("PART", "SLOPE"), default=None,
        help=(
            "Exit 0 if that combination already has a result from the current "
            "metrics, 1 otherwise. Prints nothing. Used by the matrix script's "
            "-Resume; the exit code is the answer, so it cannot be confused by "
            "stray output."
        ),
    )
    parser.add_argument(
        "--status", nargs="*", metavar="SIZE", default=None,
        help=(
            "Report which matrix combinations already have results and which "
            "are missing, then exit. Give the sizes to check, e.g. "
            "--status xs s. Use this after an interrupted run."
        ),
    )
    parser.add_argument(
        "--reanalyse", "--reanalyze", action="store_true", dest="reanalyse",
        help=(
            "Re-score every archived run against the current metrics, without "
            "re-running the pipeline, then rewrite the summary."
        ),
    )
    args = parser.parse_args(argv)

    if args.check_done is not None:
        part, slope_text = args.check_done
        try:
            path = report_path(part, float(slope_text))
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return 1
        return 0 if data.get("metrics_version") == METRICS_VERSION else 1

    if args.status is not None:
        sizes = args.status or ["xs", "s"]
        present, missing, stale = matrix_status(sizes)
        total = len(present) + len(missing)

        print(f"Matrix status for size(s) {', '.join(sizes)}:")
        print(f"  complete (metrics v{METRICS_VERSION}) : {len(present)} of {total}")
        print(f"  missing or stale             : {len(missing)}")
        if stale:
            print(
                f"\n  {len(stale)} report(s) were produced by an older metric "
                "definition and count as missing.\n"
                "  They will be overwritten when those combinations are re-run."
            )
        if missing:
            print("\nStill to run:")
            for part, slope in missing:
                print(f"  {part:<18} max_slope {slope:g}")
            print(
                "\nRe-run the matrix with -Resume to do only these, or run one "
                "directly:\n"
                f"  python tools/overhang_report.py data/param/{missing[0][0]}.json "
                f"--max-slope {missing[0][1]:g}"
            )
        else:
            print("\nNothing missing. Write the summary with --summarize.")
        return 0

    if args.reanalyse:
        existing = load_reports(overhang_aware=None)
        field_existing = load_reports(field_only=True, overhang_aware=None)
        if not existing and not field_existing:
            raise SystemExit(f"No reports in {REPORT_DIR} or {FIELD_ONLY_DIR} to re-score.")
        rewritten, skipped = reanalyse(existing)
        field_rewritten, field_skipped = reanalyse_field_only(field_existing)
        for note in skipped + field_skipped:
            print(f"  skipped: {note}")
        if not rewritten and not field_rewritten:
            raise SystemExit(
                "Nothing could be re-scored: no archived toolpaths or atoms. Runs "
                "made before archiving was added must be repeated."
            )
        if field_rewritten:
            print(f"Re-scored {len(field_rewritten)} field-only run(s) in {FIELD_ONLY_DIR}.")
        stock_rewritten = [r for r in rewritten if not r.get("overhang_aware", False)]
        aware_rewritten = [r for r in rewritten if r.get("overhang_aware", False)]
        if not stock_rewritten:
            print(
                f"The summary was not rewritten: no stock full run could be re-scored, "
                f"and {SUMMARY_PATH.name} is built around them."
            )
            return 0
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            summarize(stock_rewritten, field_rewritten, aware_rewritten), encoding="utf-8"
        )
        print(f"Re-scored {len(rewritten)} run(s); wrote {SUMMARY_PATH}.")
        for label, group in (
            ("", stock_rewritten),
            ("Overhang-aware runs: ", aware_rewritten),
            ("Field-only runs: ", field_rewritten),
        ):
            warning = provenance_warning(group) if group else None
            if warning:
                print("\n" + label + warning)
        return 0

    if args.summarize:
        reports = load_reports()
        aware_reports = load_reports(overhang_aware=True)
        field_reports = load_reports(field_only=True, overhang_aware=None)
        if not reports:
            raise SystemExit(f"No reports in {REPORT_DIR}. Run some parts first.")
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(summarize(reports, field_reports, aware_reports), encoding="utf-8")
        extra = "".join(
            f" + {len(group)} {name}"
            for name, group in (("overhang-aware", aware_reports), ("field-only", field_reports))
            if group
        )
        print(f"Wrote {SUMMARY_PATH} from {len(reports)}{extra} report(s).")
        for label, group in (
            ("", reports),
            ("Overhang-aware runs: ", aware_reports),
            ("Field-only runs: ", field_reports),
        ):
            warning = provenance_warning(group) if group else None
            if warning:
                print("\n" + label + warning)
        return 0

    if args.param_path is None:
        parser.error("a parameter file is required unless --summarize is given")
    if not args.param_path.is_file():
        parser.error(f"no such file: {args.param_path}")

    params = json.loads(args.param_path.read_text(encoding="utf-8"))
    part = params["solid_name"]
    aware = args.overhang_aware
    if args.max_slope is not None:
        max_slope = args.max_slope
    elif aware:
        max_slope = overhang_aware_slope()
    else:
        max_slope = params["max_slope"]

    runtime_s, stage_times, stage_arches = 0.0, {}, {}
    if not args.skip_pipeline:
        kind = "up to atom extraction (field-only)" if args.field_only else "the pipeline"
        field = "overhang-aware" if aware else "stock"
        print(f"Running {kind}: {part}, {field} field, at max_slope {max_slope:g}°")
        # args.max_slope, not max_slope: without --max-slope a stock run uses
        # the part file untouched, and run_pipeline gives an overhang-aware
        # one the profile's limit, as above.
        runtime_s, _, stage_arches = run_pipeline(
            args.param_path, args.max_slope, field_only=args.field_only, overhang_aware=aware
        )

    log_path = REPO_ROOT / "data" / "log" / f"{part}.log"
    if log_path.is_file():
        stage_times = parse_stage_times(log_path.read_text(encoding="utf-8"))

    provenance = prov.collect(
        prov.SOURCE_SKIP_PIPELINE if args.skip_pipeline else prov.SOURCE_PIPELINE,
        stage_arches,
        REPO_ROOT,
        METRICS_VERSION,
    )

    init_taichi("cpu")
    # Archive before scoring: should the scoring fail, the run's toolpath (or
    # atoms) is already kept, and the run can be scored later without
    # re-slicing (the matrix runner brings a failed job's archive back too).
    if args.field_only:
        if not args.skip_pipeline:
            archived = archive_frame(part, max_slope, aware)
            if archived is not None:
                print(f"Archived the atoms to {archived}")
        report = measure_field_only(
            part, max_slope, float(params["deposition_width"]), runtime_s, stage_times,
            provenance=provenance, overhang_aware=aware,
        )
        destination = field_only_report_path(part, max_slope, aware)
    else:
        if not args.skip_pipeline:
            archived = archive_toolpath(part, max_slope, aware)
            if archived is not None:
                print(f"Archived the toolpath to {archived}")
        report = measure(
            part, max_slope, float(params["deposition_width"]), runtime_s, stage_times,
            provenance=provenance, overhang_aware=aware,
        )
        destination = report_path(part, max_slope, aware)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # The progress log records full runs, with a column saying whether the
    # field was overhang-aware (P2.5). It has no column for field-only runs,
    # which would read as full ones, so they stay out.
    if not args.field_only:
        append_progress(report)

    verdict = report["verdict"]
    worst = verdict["worst_effective_deg"]
    near = report["metrics"]["unsupported_fraction_near_overhangs"]
    heading = f"{part} at max_slope {max_slope:g}°{', overhang-aware' if aware else ''}"
    if args.field_only:
        heading += f" (field-only: {report['atoms']['count']} atoms, no print order)"
    print(f"\n{heading}")
    print(f"  worst effective overhang : {worst:.1f}°" if not math.isnan(worst)
          else "  worst effective overhang : not measured")
    print("  unsupported near overhang: "
          + ("n/a (field-only)" if near is None else f"{near * 100:.2f}%"))
    print(f"  max tilt used            : {report['metrics']['max_tool_tilt_deg']:.2f}°")
    if args.field_only:
        printable = {False: "False (effective angle over the threshold)",
                     None: "unknown without a print order"}[verdict["printable"]]
    else:
        printable = verdict["printable"]
    print(f"  printable                : {printable}")
    if args.field_only and report["atoms"]["dropped_non_finite"]:
        print(f"  WARNING: {report['atoms']['dropped_non_finite']} atom(s) without a "
              "finite position or orientation were left out.")
    print(f"  measured on              : {prov.describe(provenance)}")
    if runtime_s:
        print(f"  runtime                  : {runtime_s:.0f} s")
    print(f"\nWrote {destination}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
