"""Tests for the overhang-aware field (build plan P2.2, `atom.orientation_field`).

All on the CPU, on one exact 70-degree ramp a few millimetres long
(`atom.analytic_sdf`), so they run anywhere. The stage itself runs in a child
process where the test compares against it: `atom.kinematics3z` and
`atom.toolpath3` fix their constants at import, and the stage re-initialises
Taichi.

A field costs some 20 s on the CPU whatever the grid's size (Taichi compiles
the aligner's kernels afresh for every new field), so the tests share three
module-scoped fields rather than computing their own (five, with P2.4's
ramp-in and the edges of P2-19 each switched off once). They stay in the
default (unit) tier by the operator's choice, 2026-09-30.

70 degrees, at a 30-degree budget: the rule asks 70 - 45 + 2 = 27 degrees
toward the overhang, inside the budget; the infill's inner surface faces 20
degrees away, also inside it, so stock follows it (plan_corrections P2-3).

No `from __future__ import annotations`: this module drives Taichi kernels.
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from atom import analytic_sdf, grid3
from atom.overhang_field import OverhangSettings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOSITION_WIDTH = 0.9
#: A ramp 12 mm long: big enough for the infill's 1.8 mm shell to leave a
#: hollow core, small enough for a second on the CPU.
LENGTH, DEPTH, HEIGHT = 12.0, 5.4, 7.2
ANGLE = 70
MAX_SLOPE = 30.0


@pytest.fixture(scope="module")
def cell(ti_cpu):
    import atom.fff3

    return float(atom.fff3.cell_sides_length_from_deposition_width_kernel(DEPOSITION_WIDTH))


def _sdf(array, cell):
    import taichi as ti

    import atom.solid3

    sdf = atom.solid3.SDF()
    sdf.grid = grid3.Grid()
    sdf.grid.cell_3dcount = np.array(array.shape)
    sdf.grid.origin = np.zeros(3)
    sdf.grid.cell_sides_length = cell
    sdf.sdf = ti.field(dtype=ti.f32, shape=array.shape)
    sdf.sdf.from_numpy(array)
    return sdf


def _infilled(array, cell):
    """`tools/sdf_to_isdf.py`'s infill on ``array``: period 8, shell 2, gyroid."""
    import taichi as ti

    import atom.fff3

    source = _sdf(array, cell)
    target = _sdf(array, cell)
    atom.fff3.sdf_generate_infill(source.sdf, target.sdf, cell, ti.math.vec3(0), 8, 2, True)
    return target


@pytest.fixture(scope="module")
def ramp(cell):
    """``(solid, geometry)``: the ramp's exact SDF as an array, and its shape."""
    return analytic_sdf.ramp_sdf(ANGLE, LENGTH, DEPTH, HEIGHT, cell)


@pytest.fixture(scope="module")
def solid(ramp, cell):
    return _sdf(ramp[0], cell)


@pytest.fixture(scope="module")
def infilled(ramp, cell):
    return _infilled(ramp[0], cell)


def _init(sdf, cell, max_slope_deg, rule_on=True):
    """`init_overhang_aware` alone: ``(direction, mark)`` as arrays."""
    import taichi as ti

    from atom import orientation_field as of

    shape = tuple(int(n) for n in sdf.grid.cell_3dcount)
    direction = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    state = ti.field(dtype=ti.u32, shape=shape)
    mark = ti.field(dtype=ti.u8, shape=shape)
    settings = OverhangSettings()
    budget = of.tilt_budget_rad(max_slope_deg)
    of.init_overhang_aware(
        sdf.sdf, cell, budget,
        math.radians(settings.max_overhang_deg) if rule_on else of._OVERHANG_OFF,
        math.radians(settings.margin_deg) if rule_on else 0.0,
        budget, direction, state, mark,
    )
    return direction.to_numpy(), mark.to_numpy()


@pytest.fixture(scope="module")
def stock_field(infilled):
    """Upstream's field on the infilled ramp."""
    from atom import orientation_field as of

    return of.compute_direction_field(infilled, MAX_SLOPE)


@pytest.fixture(scope="module")
def aware_field(infilled, solid):
    """The pipeline's overhang-aware run: rule on, on the solid SDF, held (the default)."""
    from atom import orientation_field as of

    return of.compute_direction_field(infilled, MAX_SLOPE, overhang=OverhangSettings(), solid_sdf=solid)


@pytest.fixture(scope="module")
def loose_field(infilled, solid):
    """``aware_field`` without holding: the final passes smooth the overhang constraints."""
    from atom import orientation_field as of

    return of.compute_direction_field(
        infilled, MAX_SLOPE, overhang=OverhangSettings(), solid_sdf=solid, hold_overhang=False
    )


@pytest.fixture(scope="module")
def no_ramp_field(infilled, solid):
    """``aware_field`` without the ramp-in (build plan P2.4)."""
    from atom import orientation_field as of

    return of.compute_direction_field(
        infilled, MAX_SLOPE, overhang=OverhangSettings(), solid_sdf=solid, ramp=None
    )


@pytest.fixture(scope="module")
def no_edges_field(infilled, solid):
    """``aware_field`` without carrying the constraint to the overhang's edges (P2-19)."""
    from atom import orientation_field as of

    return of.compute_direction_field(
        infilled, MAX_SLOPE, overhang=OverhangSettings(), solid_sdf=solid, edges=False
    )


def _overhang_marks(result):
    from atom import orientation_field as of

    return (result.mark == of.MARK_OVERHANG) | (result.mark == of.MARK_OVERHANG_CAPPED)


def _same_field(a, b):
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def _cartesian(spherical):
    theta, phi = spherical[..., 0], spherical[..., 1]
    return np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=-1
    )


def _next_to_underside(solid, geometry, cell, direction):
    """Field directions in cells inside the part within 0.9 mm of the underside."""
    shape = solid.shape
    xs = (np.arange(shape[0]) + 0.5) * cell
    zs = (np.arange(shape[2]) + 0.5) * cell
    X = np.broadcast_to(xs[:, None, None], shape)
    Z = np.broadcast_to(zs[None, None, :], shape)
    n = geometry["normal"]
    above = -((X - geometry["column_width"]) * n[0] + (Z - geometry["base_height"]) * n[2])
    along = (X - geometry["column_width"]) / geometry["run"]
    near = (solid < 0) & (along > 0.15) & (along < 0.85) & (above > 0) & (above <= 0.9)
    return _cartesian(direction[near]), n


# --------------------------------------------------------------------------
# With the rule off, nothing changes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("infill", [False, True])
def test_with_the_rule_off_the_kernel_is_upstreams(ti_cpu, cell, solid, infilled, infill):
    """`init_overhang_aware`, rule off, equals `fff3.init_spherical_direction_field_from_sdf`."""
    ti = ti_cpu
    import atom.fff3

    sdf = infilled if infill else solid
    shape = tuple(int(n) for n in sdf.grid.cell_3dcount)
    stock_dir = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    stock_state = ti.field(dtype=ti.u32, shape=shape)
    atom.fff3.init_spherical_direction_field_from_sdf(sdf.sdf, cell, 0, stock_dir, stock_state)

    # The tool sets fff3.CEIL_MAX_ANGLE to the budget; `_init` passes the budget.
    direction, mark = _init(sdf, cell, math.degrees(atom.fff3.CEIL_MAX_ANGLE), rule_on=False)
    _same_field(stock_dir.to_numpy(), direction)
    assert not mark.any()


@pytest.fixture(scope="module")
def stage_runs(ti_cpu, solid, infilled, tmp_path_factory):
    """`tools/compute_tool_orientations.py` itself, in child processes, on the ramp."""
    work = tmp_path_factory.mktemp("stage")
    solid.save(str(work / "solid.npz"))
    infilled.save(str(work / "infilled.npz"))

    env = dict(os.environ, ATOM_TI_ARCH="cpu", PYTHONIOENCODING="utf-8")
    env.pop("ATOM_TI_ARCH_LOG", None)
    env.pop("ATOM_MACHINE", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH")]))
    tool = os.path.join(REPO_ROOT, "tools", "compute_tool_orientations.py")
    infilled_path = str(work / "infilled.npz")
    slope = ["--maxslope", f"{MAX_SLOPE:g}"]

    def run(name, *arguments):
        log = work / f"{name}.log"
        result = subprocess.run(
            [sys.executable, tool, infilled_path, str(work / f"{name}.npz"), *slope, *arguments,
             "--logpath", str(log)],
            env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=300,
        )
        return result, log, work / f"{name}.npz"

    return {
        "stock": run("stock"),
        "aware": run("aware", "--overhang_aware", "--solid_sdf", str(work / "solid.npz")),
        "loose": run("loose", "--overhang_aware", "--no_hold_overhang", "--solid_sdf", str(work / "solid.npz")),
        "misused": run("misused", "--max_overhang", "50"),
        "misused_hold": run("misused_hold", "--no_hold_overhang"),
    }


def test_the_driver_reproduces_the_stage_bit_for_bit(stage_runs, stock_field):
    """`compute_direction_field` with its options off is upstream's stage 4."""
    result, _, output = stage_runs["stock"]
    assert result.returncode == 0, result.stderr
    stage = np.load(output)
    _same_field(stage["direction"], stock_field.field.direction.to_numpy())
    assert np.array_equal(stage["state"], stock_field.field.state.to_numpy())


@pytest.mark.parametrize("name,fixture,hold", [("aware", "aware_field", "on"), ("loose", "loose_field", "off")])
def test_the_stage_runs_the_overhang_aware_field(stage_runs, request, name, fixture, hold):
    """Hold is on by default; ``--no_hold_overhang`` (what `atomize.py` writes for
    ``"hold_overhang": false``) switches it off."""
    field = request.getfixturevalue(fixture)
    result, log, output = stage_runs[name]
    assert result.returncode == 0, result.stderr
    text = log.read_text(encoding="utf-8")
    assert (
        f"Overhang constraints: {field.overhang_cells} cells, "
        f"{field.overhang_capped} of them capped"
    ) in text
    assert f"hold {hold})" in text
    assert f"Flat overhangs: {field.flat_cells} cells leaning outward along the smoothed direction" in text
    assert (
        f"Overhang edges: {field.edge_cells} cells given the nearest overhang cell's constraint, "
        f"{field.corrected_cells} overhang cells by an edge given the nearest core cell's"
    ) in text
    assert f"Ramp-in: {field.ramp.cells} cells below {field.ramp.walks_used} overhang cells" in text
    assert "Field computed on the solid SDF" in text
    assert "Direction computation took" in text  # the line the reports time
    _same_field(np.load(output)["direction"], field.field.direction.to_numpy())


@pytest.mark.parametrize("name", ["misused", "misused_hold"])
def test_rule_options_without_the_rule_are_refused(stage_runs, name):
    result, _, output = stage_runs[name]
    assert result.returncode != 0
    assert "need --overhang_aware" in result.stderr + result.stdout
    assert not output.exists()


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_overhang_cells_lean_27_degrees_toward_a_70_degree_overhang(aware_field, ramp):
    """The kernel asks what `overhang_field.overhang_target` asks: 70 - 45 + 2, toward."""
    from atom import orientation_field as of
    from atom.overhang_field import overhang_target

    solid, geometry = ramp
    target = overhang_target(geometry["normal"], MAX_SLOPE)
    assert (target.tilt_deg, target.azimuth_deg) == pytest.approx((ANGLE - 45 + 2, 0.0))
    marked = _overhang_marks(aware_field)
    assert aware_field.overhang_cells == int(marked.sum()) > 100
    assert aware_field.overhang_capped == int((aware_field.mark == of.MARK_OVERHANG_CAPPED).sum())
    assert aware_field.overhang_capped < 0.1 * aware_field.overhang_cells
    theta = np.degrees(aware_field.initial[..., 0][marked])
    phi = np.degrees(aware_field.initial[..., 1][marked])
    np.testing.assert_allclose(np.median(theta), target.tilt_deg, atol=0.2)
    np.testing.assert_allclose(np.median(phi), target.azimuth_deg, atol=0.5)
    # Only the outer band: within one layer height (0.45 mm) of the surface.
    assert (solid[marked] > -0.45).all()


def test_the_tilt_budget_caps_the_constraint(solid, cell):
    """At a 10-degree budget the rule's 27 degrees is capped to 10.

    Not every cell: where the underside blends into the column, the surface
    is gentler and asks for less than the budget.
    """
    from atom import orientation_field as of

    direction, mark = _init(solid, cell, 10.0)
    theta = np.degrees(direction[..., 0])
    capped = mark == of.MARK_OVERHANG_CAPPED
    uncapped = mark == of.MARK_OVERHANG
    assert capped.sum() > 0.9 * (capped | uncapped).sum() > 100
    np.testing.assert_allclose(theta[capped], 10.0, atol=1e-4)
    assert (theta[uncapped] <= 10.0 + 1e-4).all()


def test_the_solid_sdf_keeps_the_rule_off_the_infills_inner_surfaces(aware_field, infilled, cell, ramp):
    """plan_corrections P2-8 (a): without it, the hollow's undersides are 'overhangs' too."""
    from atom import orientation_field as of

    solid, _ = ramp
    assert (solid[_overhang_marks(aware_field)] > -0.45).all()
    _, mark_on_infilled = _init(infilled, cell, MAX_SLOPE)
    assert (solid[mark_on_infilled != of.MARK_NONE] < -1.0).any()
    # Masked exactly where the infilled SDF is outside, as upstream masks.
    final = aware_field.field.direction.to_numpy()
    assert np.array_equal(np.isnan(final[..., 0]), infilled.sdf.to_numpy() >= 0)


def test_holding_keeps_the_overhang_constraints_exact(aware_field, loose_field):
    from atom import orientation_field as of

    marked = _overhang_marks(aware_field)
    assert np.array_equal(marked, _overhang_marks(loose_field))
    np.testing.assert_array_equal(aware_field.field.direction.to_numpy()[marked], aware_field.initial[marked])
    assert not np.array_equal(loose_field.field.direction.to_numpy()[marked], loose_field.initial[marked])


def test_the_field_turns_from_away_to_toward_the_overhang(stock_field, aware_field, ramp, cell):
    """P2.2's sign check at the field: stock with infill leans away, overhang-aware toward."""
    solid, geometry = ramp
    for result, sign in ((stock_field, -1.0), (aware_field, 1.0)):
        d, n = _next_to_underside(solid, geometry, cell, result.field.direction.to_numpy())
        lean = d @ np.array([n[0], n[1], 0.0]) / math.hypot(n[0], n[1])
        effective = 90 - np.degrees(np.arccos(np.clip(d @ -n, -1, 1)))
        assert sign * lean.mean() > 0.2
        if sign > 0:
            assert effective.mean() < 46.0
        else:
            assert effective.mean() > 80.0


# --------------------------------------------------------------------------
# The ramp-in (build plan P2.4)
# --------------------------------------------------------------------------


def test_the_ramp_in_constrains_the_column_below_the_corner(aware_field, ramp, cell):
    """Below where the column's wall turns into the overhang, leaning its way.

    The corner is 1.8 mm above the bed: 27 degrees at 3 per mm would need 9 mm,
    so the ramp is steepened to fit, by the operator's decision, and the
    overhang keeps its 27 degrees. Never as steep as copying the tilt into the
    wall below the corner would make it (P2-19: 40 degrees per mm).
    """
    from atom import orientation_field as of

    solid, geometry = ramp
    r = aware_field.ramp
    ramp_cells = aware_field.mark == of.MARK_RAMP
    assert r.cells == int(ramp_cells.sum()) > 0
    assert r.walks_steepened > 0 and 3.0 < r.steepest_rate_deg_per_mm < 20.0

    index = np.argwhere(ramp_cells)
    x = (index[:, 0] + 0.5) * cell
    z = (index[:, 2] + 0.5) * cell
    assert (solid[ramp_cells] < 0).all()
    below_corner = (x < geometry["column_width"] + 0.5) & (z < geometry["base_height"] + 0.5)
    # Most are in the column below the corner. The rest are under the edge
    # cells at the tip (P2-19), in the material printed just before them.
    assert below_corner.mean() > 0.8
    assert (below_corner | (x > LENGTH - 1.0)).all()
    tilt = np.degrees(aware_field.initial[..., 0][ramp_cells])
    phi = np.degrees(aware_field.initial[..., 1][ramp_cells])
    assert (tilt > 0).all() and (tilt < ANGLE - 45 + 2).all()
    # The overhang's way. Next to the side walls the normals blend, and the
    # edge cells there (P2-19) pass a few degrees of that on: within 10.
    np.testing.assert_allclose(phi, 0.0, atol=10.0)
    assert aware_field.overhang_capped == 0


def test_the_ramp_in_raises_the_tilt_below_the_corner(aware_field, no_ramp_field, ramp, cell):
    """In the column just below and before the corner, the field leans further.

    On this ramp (and on `ramp60_xs`) the overhang's own first strip hardly
    changes, since its held constraints already give it nearly the full tilt;
    what the ramp-in changes is the material printed just before it
    (plan_corrections P2-13).
    """
    from atom import orientation_field as of

    assert no_ramp_field.ramp is None
    assert not (no_ramp_field.mark == of.MARK_RAMP).any()
    solid, geometry = ramp
    shape = solid.shape
    X = np.broadcast_to(((np.arange(shape[0]) + 0.5) * cell)[:, None, None], shape)
    Z = np.broadcast_to(((np.arange(shape[2]) + 0.5) * cell)[None, None, :], shape)
    below_corner = (
        (solid < 0)
        & (X < geometry["column_width"]) & (X > geometry["column_width"] - 1.0)
        & (Z > geometry["base_height"] - 1.0) & (Z < geometry["base_height"] + 0.5)
    )
    assert below_corner.sum() > 50

    def tilt(result):
        return np.degrees(result.field.direction.to_numpy()[..., 0][below_corner]).mean()

    assert tilt(aware_field) > tilt(no_ramp_field) + 0.5


# --------------------------------------------------------------------------
# The overhang's edges (plan_corrections P2-19)
# --------------------------------------------------------------------------


def _underside_band(ramp, cell, near_side_mm=None):
    """Boundary cells within one layer height above the underside; optionally only near the side walls."""
    solid, geometry = ramp
    shape = solid.shape
    X = np.broadcast_to(((np.arange(shape[0]) + 0.5) * cell)[:, None, None], shape)
    Y = np.broadcast_to(((np.arange(shape[1]) + 0.5) * cell)[None, :, None], shape)
    Z = np.broadcast_to(((np.arange(shape[2]) + 0.5) * cell)[None, None, :], shape)
    n = geometry["normal"]
    above = -((X - geometry["column_width"]) * n[0] + (Z - geometry["base_height"]) * n[2])
    along = (X - geometry["column_width"]) / geometry["run"]
    band = (solid < 0) & (solid > -0.45) & (above > 0) & (above <= 0.45) & (along > 0.1) & (along < 0.9)
    if near_side_mm is not None:
        band &= np.minimum(Y, DEPTH - Y) < near_side_mm
    return band


def test_the_edges_next_to_the_side_walls_get_the_constraint(aware_field, no_edges_field, ramp, cell):
    from atom import orientation_field as of

    edge = aware_field.mark == of.MARK_OVERHANG_EDGE
    assert aware_field.edge_cells == int(edge.sum()) > 0
    assert no_edges_field.edge_cells == 0 and not (no_edges_field.mark == of.MARK_OVERHANG_EDGE).any()
    near_walls = _underside_band(ramp, cell, near_side_mm=0.5)
    constrained = lambda r: (r.mark[near_walls] != of.MARK_NONE).mean()  # noqa: E731
    assert constrained(aware_field) > 0.95 > constrained(no_edges_field)
    # Held like the overhang cells they copy.
    np.testing.assert_array_equal(aware_field.field.direction.to_numpy()[edge], aware_field.initial[edge])


def test_the_field_leans_further_next_to_the_side_walls(aware_field, no_edges_field, ramp, cell):
    near_walls = _underside_band(ramp, cell, near_side_mm=0.5)
    tilt = lambda r: np.degrees(r.field.direction.to_numpy()[..., 0][near_walls]).mean()  # noqa: E731
    assert tilt(aware_field) > tilt(no_edges_field) + 0.5
