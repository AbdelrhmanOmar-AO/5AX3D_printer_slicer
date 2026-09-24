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
"""

# No `from __future__ import annotations`: this module drives Taichi kernels
# through atom.contracts. See docs/plan_corrections.md 3.2.

import argparse
import csv
import json
import math
import os
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

from atom import overhang_metrics as om  # noqa: E402
from atom import provenance as prov  # noqa: E402
from atom.ti_env import init_taichi  # noqa: E402

#: Where individual reports and the summary live.
REPORT_DIR = REPO_ROOT / "reports" / "baseline_overhang"
SUMMARY_PATH = REPO_ROOT / "reports" / "baseline_overhang.md"
#: Each run's toolpath, kept so the metrics can be recomputed without re-slicing.
TOOLPATH_ARCHIVE = REPO_ROOT / "reports" / "toolpaths"

#: File layout version. Bumped to 2 when the `provenance` block was added
#: (which machine produced the toolpath, and which scored it — see
#: `atom.provenance` and corrections 3.9). Version 1 reports remain readable:
#: they are the committed baseline, and refusing them would have thrown it away.
#: A version 1 report simply has no provenance, which reads as "unknown".
SCHEMA_VERSION = 2

#: Layouts this tool can read. Anything else is skipped rather than guessed at.
READABLE_SCHEMA_VERSIONS = (1, 2)

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
METRICS_VERSION = 2

#: An append-only record of every run, written as each finishes. Survives a
#: crash and gives a human-readable trail beside the per-run JSON.
PROGRESS_LOG = REPO_ROOT / "reports" / "matrix_progress.csv"

#: GATE D0: placeholder thresholds, team decision pending. The same values are
#: used in P2.5, so stock and overhang-aware are judged on one scale.
MAX_EFFECTIVE_OVERHANG_DEG = 45.0
MAX_UNSUPPORTED_FRACTION = 0.01

#: Search radius for deposition points near a face, in deposition widths.
SURFACE_SEARCH_WIDTHS = 1.0
#: Radius defining "near an overhang" for the unsupported measurement.
NEAR_OVERHANG_WIDTHS = 2.0

_STAGE_TIME = re.compile(
    r"^(?P<label>[A-Z][^\n]*?)\s+took\s+(?P<seconds>[\d.]+)\s+seconds", re.MULTILINE
)


def parse_stage_times(log_text):
    """Per-stage durations from an atomize.py log, as {label: seconds}."""
    return {
        m.group("label").strip(): float(m.group("seconds"))
        for m in _STAGE_TIME.finditer(log_text)
    }


def run_pipeline(param_path, max_slope_deg):
    """Run `tools/atomize.py`, optionally overriding ``max_slope``.

    The override is applied through a temporary copy of the parameter file, so
    the committed one is untouched and no vendored stage is modified.
    """
    params = json.loads(Path(param_path).read_text(encoding="utf-8"))
    if max_slope_deg is not None:
        params["max_slope"] = float(max_slope_deg)

    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp) / Path(param_path).name
        temporary.write_text(json.dumps(params, indent=4), encoding="utf-8")

        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "tools/atomize.py", str(temporary)], cwd=REPO_ROOT
        )
        elapsed = time.perf_counter() - started

    if result.returncode != 0:
        raise SystemExit(
            f"atomize.py exited {result.returncode} for {param_path}. "
            "The report cannot be written."
        )
    return elapsed, params


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
    pipeline_provenance=None,
):
    """Apply the three metrics and assemble the report.

    ``toolpath_path`` and ``stl_path`` default to the pipeline's own output
    locations; they are arguments so this can be exercised on synthetic data.

    ``pipeline_provenance`` is the machine that produced the **toolpath**, which
    is not necessarily this one: ``--reanalyse`` re-scores an archived toolpath
    and must carry its original machine forward, or every laptop-measured run
    re-scored on the lab machine would be relabelled as lab-measured — the very
    corruption this records guard against (corrections 3.9). It is therefore an
    argument with no default: a caller that has just run the pipeline passes
    this machine's, and one re-scoring passes the old report's, including
    ``None`` when that report predates provenance.
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
    )
    overall, near_fraction, near_count = om.unsupported_near_overhangs(
        toolpath,
        normals,
        centres,
        radius=NEAR_OVERHANG_WIDTHS * deposition_width,
    )
    max_tilt = om.max_tool_tilt_deg(toolpath)

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
        # Which machine produced the toolpath, and which scored it. Kept apart
        # on purpose; see `atom.provenance`.
        "provenance": {
            "pipeline": pipeline_provenance,
            "scored": prov.fingerprint(include_blender=False),
        },
        "part": part,
        "max_slope_deg": max_slope_deg,
        "overhang_aware": False,
        "deposition_width_mm": deposition_width,
        "mesh": {
            "volume_mm3": float(mesh.volume),
            "face_count": int(len(mesh.faces)),
            "sampled_face_count": int(len(areas)),
            "extents_mm": [float(v) for v in mesh.extents],
        },
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
            "surfaces": [
                {
                    "geometric_angle_deg": s.geometric_angle_deg,
                    "area_mm2": s.area_mm2,
                    "face_count": s.face_count,
                    "sample_count": s.sample_count,
                    "max_effective_deg": s.max_effective_deg,
                    "mean_effective_deg": s.mean_effective_deg,
                    "max_tilt_used_deg": s.max_tilt_used_deg,
                }
                for s in surfaces
            ],
        },
        "verdict": {
            "printable": printable,
            "assessable": assessable,
            "worst_effective_deg": worst_effective,
            "thresholds": {
                "max_effective_overhang_deg": MAX_EFFECTIVE_OVERHANG_DEG,
                "max_unsupported_fraction": MAX_UNSUPPORTED_FRACTION,
                "_gate": "D0: placeholders, team decision pending",
            },
        },
    }


#: Columns of the progress log. `machine` was appended when provenance was
#: added; `migrate_progress_header` brings an older file up to it rather than
#: writing ragged rows into it.
PROGRESS_COLUMNS = (
    "finished_utc", "part", "max_slope_deg", "metrics_version",
    "worst_effective_deg", "unsupported_near_overhangs",
    "max_tilt_used_deg", "printable", "runtime_s", "machine",
)


def migrate_progress_header(path=None):
    """Bring a progress log written before `machine` existed up to date.

    Returns True if the file was rewritten. Old rows are padded with an empty
    machine rather than guessed at: the log predates the record, and inventing
    a machine for it would be worse than admitting none.

    A ragged CSV is the failure being avoided. The file is the crash trail from
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
    survives a crash and shows what had been completed and when — and now on
    which machine, since a run on another machine is a different computation
    (corrections 3.9).
    """
    metrics, verdict = report["metrics"], report["verdict"]
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)

    new_file = not PROGRESS_LOG.exists()
    if not new_file:
        migrate_progress_header()

    pipeline = (report.get("provenance") or {}).get("pipeline") or {}
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
            pipeline.get("host") or "",
        ])
        handle.flush()
        os.fsync(handle.fileno())


def report_path(part, max_slope_deg):
    return REPORT_DIR / f"{part}_ms{max_slope_deg:g}.json"


def archive_path(part, max_slope_deg):
    return TOOLPATH_ARCHIVE / f"{part}_ms{max_slope_deg:g}.npz"


def archive_toolpath(part, max_slope_deg):
    """Keep this run's toolpath, so its metrics can be recomputed later."""
    source = REPO_ROOT / "data" / "toolpath" / f"{part}_smoothed.npz"
    if not source.is_file():
        return None
    TOOLPATH_ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = archive_path(part, max_slope_deg)
    shutil.copy2(source, destination)
    return destination


def reanalyse(reports):
    """Re-score every archived run against the current metrics.

    Returns (rewritten, skipped). A run whose toolpath was not archived cannot
    be re-scored and is reported rather than silently left stale.
    """
    init_taichi("cpu")
    rewritten, skipped = [], []

    for old in reports:
        part, slope = old["part"], old["max_slope_deg"]
        archived = archive_path(part, slope)
        if not archived.is_file():
            skipped.append(f"{part} @ {slope:g} (no archived toolpath)")
            continue

        fresh = measure(
            part,
            slope,
            old.get("deposition_width_mm", 0.9),
            runtime_s=old.get("runtime", {}).get("total_s", 0.0),
            stage_times=old.get("runtime", {}).get("stages", {}),
            toolpath_path=archived,
            # The archived toolpath was computed wherever it was computed.
            # Re-scoring it here does not move that.
            pipeline_provenance=old.get("provenance", {}).get("pipeline"),
        )
        report_path(part, slope).write_text(
            json.dumps(fresh, indent=2) + "\n", encoding="utf-8"
        )
        rewritten.append(fresh)

    return rewritten, skipped


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


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


def _provenance_lines(reports):
    """State which machine produced these runs, and shout if it is more than one.

    This is the guard correction 3.9 asks for. A table that silently mixes two
    machines attributes a hardware difference to whatever the table is meant to
    show, and a matrix run on another machine overwrites reports one cell at a
    time, so a mixed table is a realistic accident rather than a hypothetical.
    """
    records = [(r.get("provenance") or {}).get("pipeline") for r in reports]
    buckets = prov.group_by_machine(records)

    if len(buckets) <= 1:
        record = buckets[0][0] if buckets else None
        if record is None:
            return [
                "> **Machine not recorded.** These runs predate provenance "
                "(schema 1), so nothing in the files says where they were "
                "computed. `docs/handoff.md` section 3 has the machine they "
                "were measured on.",
                "",
            ]
        return [f"Measured on **{prov.describe(record)}**.", ""]

    lines = [
        "> ## ⚠️ These results come from more than one machine",
        ">",
        "> A run on another machine is a **different computation** — different "
        "arithmetic gives a different toolpath (`docs/plan_corrections.md` "
        "3.9). Numbers from different machines must not be compared with each "
        "other, and a conclusion drawn across this table is not sound.",
        ">",
    ]
    for record, members in buckets:
        cells = ", ".join(
            f"`{reports[i]['part']}` @ {reports[i]['max_slope_deg']:g}°"
            for i in sorted(members)[:6]
        )
        if len(members) > 6:
            cells += f", and {len(members) - 6} more"
        lines.append(f"> * **{prov.describe(record)}** — {len(members)} run(s): {cells}")
    lines += [
        ">",
        "> Re-run the odd ones out on one machine before drawing anything from "
        "this table.",
        "",
    ]
    return lines


def summarize(reports):
    """Build the comparison table and the plain-language conclusion."""
    parts = sorted({r["part"] for r in reports})
    slopes = sorted({r["max_slope_deg"] for r in reports})
    by_key = {(r["part"], r["max_slope_deg"]): r for r in reports}

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
    ] + _provenance_lines(reports)

    header = "| Part | " + " | ".join(f"max_slope {s:g}°" for s in slopes) + " |"
    lines += [header, "|" + "---|" * (len(slopes) + 1)]
    for part in parts:
        cells = [
            _format_cell(by_key[(part, s)]) if (part, s) in by_key else "—"
            for s in slopes
        ]
        lines.append(f"| `{part}` | " + " | ".join(cells) + " |")

    lines += ["", "## Conclusion", ""] + _conclusion(reports, slopes)
    lines += [""] + _timing_section(reports)
    return "\n".join(lines) + "\n"


def _size_of(part):
    """The size suffix of a generated benchmark part, or None."""
    match = re.match(r".*_(xs|s|m|l)$", part)
    return match.group(1) if match else None


def _timing_section(reports):
    """How pipeline cost scaled with part volume, from the runs themselves.

    Nobody has published how Atomizer's ordering stage scales, so this is a
    result in its own right rather than only a planning aid. It is built from
    whatever has been run; a single size gives a single row.
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
            + f" | {total / 60:.1f} |"
        )

    section = [
        "## Runtime",
        "",
        "How the pipeline's cost scaled with part size. `order_atoms` is the "
        "stage that dominates, and it runs on the CPU.",
        "",
        "| Part | max_slope | Volume mm³ | Toolpath points | order_atoms min | Total min |",
        "|---|---|---|---|---|---|",
        *rows,
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


def load_reports(quiet=False):
    """Every valid report on disk.

    A report truncated mid-write, which a power loss can do, is reported and
    skipped rather than taking the whole summary down with it.
    """
    if not REPORT_DIR.is_dir():
        return []

    reports = []
    for path in sorted(REPORT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  CORRUPT {path.name}: {exc}. Delete it and re-run that part.")
            continue
        if data.get("schema_version") not in READABLE_SCHEMA_VERSIONS:
            if not quiet:
                print(
                    f"  Skipping {path.name}: schema version "
                    f"{data.get('schema_version')}, can read "
                    f"{', '.join(str(v) for v in READABLE_SCHEMA_VERSIONS)}"
                )
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
        existing = load_reports()
        if not existing:
            raise SystemExit(f"No reports in {REPORT_DIR} to re-score.")
        rewritten, skipped = reanalyse(existing)
        for note in skipped:
            print(f"  skipped: {note}")
        if not rewritten:
            raise SystemExit(
                "Nothing could be re-scored: no archived toolpaths. Runs made "
                "before archiving was added must be repeated."
            )
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(summarize(rewritten), encoding="utf-8")
        print(f"Re-scored {len(rewritten)} run(s); wrote {SUMMARY_PATH}.")
        return 0

    if args.summarize:
        reports = load_reports()
        if not reports:
            raise SystemExit(f"No reports in {REPORT_DIR}. Run some parts first.")
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(summarize(reports), encoding="utf-8")
        print(f"Wrote {SUMMARY_PATH} from {len(reports)} report(s).")
        return 0

    if args.param_path is None:
        parser.error("a parameter file is required unless --summarize is given")
    if not args.param_path.is_file():
        parser.error(f"no such file: {args.param_path}")

    params = json.loads(args.param_path.read_text(encoding="utf-8"))
    part = params["solid_name"]
    max_slope = args.max_slope if args.max_slope is not None else params["max_slope"]

    runtime_s, stage_times = 0.0, {}
    if not args.skip_pipeline:
        print(f"Running the pipeline: {part} at max_slope {max_slope:g}°")
        runtime_s, _ = run_pipeline(args.param_path, args.max_slope)

    log_path = REPO_ROOT / "data" / "log" / f"{part}.log"
    if log_path.is_file():
        stage_times = parse_stage_times(log_path.read_text(encoding="utf-8"))

    init_taichi("cpu")
    # This machine, in both branches. With --skip-pipeline the toolpath comes
    # from `data/toolpath/`, which is gitignored local scratch, so it was
    # produced here by an earlier invocation. Only --reanalyse reads a toolpath
    # that may have come from elsewhere, and that path carries the old record
    # forward instead.
    report = measure(
        part,
        max_slope,
        float(params["deposition_width"]),
        runtime_s,
        stage_times,
        pipeline_provenance=prov.fingerprint(),
    )

    if not args.skip_pipeline:
        archived = archive_toolpath(part, max_slope)
        if archived is not None:
            print(f"Archived the toolpath to {archived}")

    destination = report_path(part, max_slope)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_progress(report)

    verdict = report["verdict"]
    worst = verdict["worst_effective_deg"]
    print(f"\n{part} at max_slope {max_slope:g}°")
    print(f"  worst effective overhang : {worst:.1f}°" if not math.isnan(worst)
          else "  worst effective overhang : not measured")
    print(f"  unsupported near overhang: "
          f"{report['metrics']['unsupported_fraction_near_overhangs'] * 100:.2f}%")
    print(f"  max tilt used            : {report['metrics']['max_tool_tilt_deg']:.2f}°")
    print(f"  printable                : {verdict['printable']}")
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
