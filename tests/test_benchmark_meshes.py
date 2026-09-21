"""Tests for the benchmark mesh generators (build plan task P0.7).

The central property is the overhang angle: `ramp60` must present a surface a
printer would see as a 60-degree overhang, or every measurement built on it in
P0.8 and P2.5 is meaningless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a dev dependency")

from atom import benchmark_meshes as bm

#: Faces whose centroid sits below this height are resting on the bed, so they
#: are supported rather than overhanging. Matches how `overhang_metrics` will
#: treat first-layer deposition.
BED_CONTACT_MM = 0.5


def downward_face_groups(mesh, min_height: float = BED_CONTACT_MM) -> dict[float, float]:
    """Total area per distinct overhang angle, for downward faces above the bed.

    The overhang angle of a face with downward normal ``n``, seen from a
    vertical build direction, is ``90 - degrees(arccos(n . -z))``.
    """
    normals = mesh.face_normals
    centres = mesh.triangles_center
    downward = (normals[:, 2] < -1e-6) & (centres[:, 2] > min_height)

    angles = 90.0 - np.degrees(
        np.arccos(np.clip(-normals[downward][:, 2], -1.0, 1.0))
    )
    groups: dict[float, float] = {}
    for angle, area in zip(angles, mesh.area_faces[downward]):
        key = round(float(angle), 1)
        groups[key] = groups.get(key, 0.0) + float(area)
    return groups


ALL_PARTS = [
    ("box", lambda d: bm.make_box(**d)),
    *[
        (f"ramp{a}", (lambda a: lambda d: bm.make_ramp(a, **d))(a))
        for a in bm.RAMP_ANGLES
    ],
    ("tshape", lambda d: bm.make_tshape(**d)),
    (
        "twin_domes",
        lambda d: bm.make_twin_domes(
            base_width=d["length"], depth=d["depth"], base_height=d["height"] * 0.4
        ),
    ),
]


@pytest.mark.parametrize("name,builder", ALL_PARTS, ids=[n for n, _ in ALL_PARTS])
@pytest.mark.parametrize("size", list(bm.SIZE_PRESETS))
def test_part_is_a_valid_printable_solid(name, builder, size):
    """Every part, at every size: watertight, positive volume, sitting on z=0."""
    mesh = builder(bm.default_dimensions(bm.SIZE_PRESETS[size]))

    assert mesh.is_watertight, f"{name} at size {size} is not watertight"
    assert mesh.is_winding_consistent, f"{name} at size {size} has mixed winding"
    assert mesh.volume > 0.0, f"{name} at size {size} has non-positive volume"
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-9), (
        f"{name} at size {size} does not sit on the bed"
    )
    assert mesh.bounds[0][0] == pytest.approx(0.0, abs=1e-9)
    assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("angle", bm.RAMP_ANGLES)
def test_ramp_has_exactly_one_overhang_at_the_requested_angle(angle):
    """`ramp<A>` presents one overhanging surface, at A degrees from vertical."""
    mesh = bm.make_ramp(angle, **bm.default_dimensions(100.0))
    groups = downward_face_groups(mesh)

    assert len(groups) == 1, (
        f"ramp{angle} should have exactly one overhang surface, found "
        f"{len(groups)}: {groups}"
    )
    measured = next(iter(groups))
    assert measured == pytest.approx(float(angle), abs=0.1)
    assert groups[measured] > 0.0


def test_ramp_overhang_area_is_the_full_face():
    """The overhang spans the part's depth and the face's slant length."""
    dims = bm.default_dimensions(100.0)
    angle = 60.0
    mesh = bm.make_ramp(angle, **dims)

    run = dims["length"] * 0.30                  # the default run_fraction
    rise = run / math.tan(math.radians(angle))
    slant = math.hypot(run, rise)
    expected_area = slant * dims["depth"]

    area = sum(downward_face_groups(mesh).values())
    assert area == pytest.approx(expected_area, rel=1e-6)


def test_box_has_no_overhang():
    """The regression part must present nothing a slicer would call an overhang."""
    mesh = bm.make_box(**bm.default_dimensions(100.0))
    assert downward_face_groups(mesh) == {}


@pytest.mark.parametrize("underside", [90.0, 70.0, 45.0])
def test_tshape_underside_angle_is_honoured(underside):
    mesh = bm.make_tshape(
        underside_angle_deg=underside, **bm.default_dimensions(100.0)
    )
    groups = downward_face_groups(mesh)

    assert len(groups) == 1, f"expected one underside angle, got {groups}"
    assert next(iter(groups)) == pytest.approx(underside, abs=0.1)


def test_twin_domes_volume_matches_the_analytic_solid():
    """Base box plus two hemispheres, to within the height-field resolution."""
    base_width, depth, base_height = 100.0, 45.0, 24.0
    radius = base_width * 0.20

    mesh = bm.make_twin_domes(
        base_width=base_width, depth=depth, base_height=base_height
    )
    expected = base_width * depth * base_height + 2.0 * (2.0 / 3.0) * math.pi * radius**3

    assert mesh.volume == pytest.approx(expected, rel=2e-3)
    assert mesh.bounds[1][2] == pytest.approx(base_height + radius, rel=1e-3)


def test_sizes_scale_volume_as_the_cube_of_the_principal_dimension():
    """Doubling the principal dimension must multiply volume by about eight.

    This is what makes the runtime estimates in `tools/make_benchmarks.py`
    meaningful: `order_atoms` scales with atom count, and atom count with
    volume.
    """
    small = bm.make_ramp(60, **bm.default_dimensions(50.0))
    large = bm.make_ramp(60, **bm.default_dimensions(100.0))
    assert large.volume / small.volume == pytest.approx(8.0, rel=1e-6)


def test_ramp_rejects_an_angle_that_will_not_fit():
    """A shallow overhang rises too far for a short part, and must say so."""
    with pytest.raises(ValueError, match="needing height"):
        bm.make_ramp(45, length=100.0, depth=45.0, height=20.0)


@pytest.mark.parametrize("angle", [-1.0, 0.0, 91.0])
def test_ramp_rejects_angles_outside_the_valid_range(angle):
    with pytest.raises(ValueError, match="angle_deg"):
        bm.make_ramp(angle, **bm.default_dimensions(100.0))


def test_tshape_rejects_an_underside_that_cannot_fit():
    with pytest.raises(ValueError, match="which needs height"):
        bm.make_tshape(length=100.0, depth=45.0, height=20.0, underside_angle_deg=30.0)


def test_twin_domes_rejects_domes_that_do_not_fit_the_depth():
    with pytest.raises(ValueError, match="do not fit"):
        bm.make_twin_domes(base_width=100.0, depth=20.0, base_height=10.0)


def test_triangulation_handles_a_non_convex_profile():
    """The T-shape profile is non-convex, so fan triangulation would be wrong."""
    profile = np.array(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.0, 3.0],
         [2.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]]
    )
    triangles = bm._triangulate_simple_polygon(profile)

    assert len(triangles) == len(profile) - 2
    area = sum(
        abs(bm._cross_2d(profile[b] - profile[a], profile[c] - profile[a])) / 2.0
        for a, b, c in triangles
    )
    assert area == pytest.approx(7.0)  # 3x3 square minus the 1x2 notch


# --------------------------------------------------------------------------
# Repository integrity
#
# data/mesh/.gitignore ignores *.stl and whitelists meshes by name, so a newly
# generated mesh is skipped by `git add` without any warning. That leaves the
# parameter file committed and the mesh it names missing for everyone else, and
# the failure only surfaces when someone tries to run the pipeline. These tests
# check the committed state rather than the working tree.
# --------------------------------------------------------------------------


def _committed_files(repo_root, path_prefix: str) -> set[str]:
    """Paths committed at HEAD under `path_prefix`.

    Deliberately `git ls-tree HEAD` rather than `git ls-files`: the latter
    reports the index, so a file staged with `git add` but never committed
    would look tracked. Only a commit survives a push, a clone, or a new
    machine.
    """
    import subprocess

    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", path_prefix],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout, or git is unavailable")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_every_parameter_file_has_a_committed_mesh(repo_root):
    """A committed param JSON whose mesh is not committed breaks the pipeline."""
    import json

    committed_meshes = _committed_files(repo_root, "data/mesh/")
    missing = []

    for param_path in sorted((repo_root / "data" / "param").glob("*.json")):
        solid_name = json.loads(param_path.read_text(encoding="utf-8"))["solid_name"]
        expected = f"data/mesh/{solid_name}.stl"
        if expected not in committed_meshes:
            missing.append(f"{param_path.name} -> {expected}")

    assert not missing, (
        "These parameter files name a mesh that is not committed at HEAD:\n  "
        + "\n  ".join(missing)
        + "\n\ndata/mesh/.gitignore ignores *.stl and whitelists by name; add a "
        "matching '!' rule for any new mesh."
    )


def test_generated_benchmark_meshes_are_not_gitignored(repo_root):
    """Every part x size combination the generator produces must be committable."""
    import subprocess

    names = [
        f"{part}_{size}.stl"
        for part in ("box", "tshape", "twin_domes", *[f"ramp{a}" for a in bm.RAMP_ANGLES])
        for size in bm.SIZE_PRESETS
    ]
    paths = [f"data/mesh/{name}" for name in names]

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    # check-ignore exits 0 when at least one path IS ignored, 1 when none are.
    ignored = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "These benchmark meshes would be silently skipped by `git add`:\n  "
        + "\n  ".join(ignored)
    )
