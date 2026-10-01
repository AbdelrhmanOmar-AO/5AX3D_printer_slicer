"""The reachable-tilt map (build plan P2.3), on the reference profile.

Small grids keep each build to a fraction of a second; the kinematics are the
vendored `kinematics3z.inverse`, on the CPU.
"""

import numpy as np
import pytest

from atom import reachability as r

#: The tilt the golden cube actually uses (build plan P2.3's test).
GOLDEN_CUBE_TILT_DEG = 5.53


@pytest.fixture
def reference(ti_cpu, monkeypatch):
    monkeypatch.delenv("ATOM_MACHINE", raising=False)
    from atom import machine_profile

    return machine_profile.load_profile("reference")


def test_the_bed_centre_near_the_bed_reaches_the_golden_cubes_tilt(reference):
    m = r.build_map(reference, x_mm=[140, 150, 160], y_mm=[140, 150, 160], z_mm=[0, 10])
    centre = [[150.0, 146.5, 2.0]]
    for azimuth in np.arange(0, 360, 15.0):
        assert m.max_tilt_here(centre, azimuth)[0] >= GOLDEN_CUBE_TILT_DEG
        assert m.max_tilt_here(centre, azimuth, without_lift=True)[0] >= GOLDEN_CUBE_TILT_DEG


def test_a_point_outside_travel_reaches_nothing(reference):
    m = r.build_map(reference, x_mm=[-60.0, 150.0, 400.0], y_mm=[146.5], z_mm=[10.0])
    table = m.max_tilt_deg()
    assert np.isnan(table[0]).all() and np.isnan(table[2]).all()
    assert np.isfinite(table[1]).all()
    # Off the grid altogether: nothing is known to be reachable.
    assert np.isnan(m.max_tilt_here([[150.0, 146.5, 500.0]], 0.0)[0])


def test_the_max_tilt_is_monotonic(reference):
    """Every tilt up to a cell's max is reachable in that azimuth, at the travel's edge too."""
    m = r.build_map(reference, x_mm=[0.0, 10.0, 150.0, 290.0, 300.0], y_mm=[0.0, 146.5, 293.0], z_mm=[0.0, 100.0, 200.0])
    for without_lift, ok in ((False, m.reachable), (True, m.reachable_without_lift)):
        table = m.max_tilt_deg(without_lift)
        steps = np.where(np.isnan(table), -1, np.rint(table / 2.5)).astype(int)
        for index in np.ndindex(table.shape):
            assert ok[index[:3] + (slice(0, steps[index] + 1), index[3])].all()


def test_the_lift_is_converged(reference):
    """Raised by the stored lift, the kinematics ask for no more (plan_corrections P1-9)."""
    m = r.build_map(reference, x_mm=[150.0], y_mm=[146.5], z_mm=[2.0])
    lifted = np.argwhere(np.nan_to_num(m.lift_mm, nan=0.0) > 0)
    assert len(lifted) > 0
    points = np.array([[m.x_mm[i], m.y_mm[j], m.z_mm[k] + m.lift_mm[i, j, k, t, a]] for i, j, k, t, a in lifted])
    fan = r.directions(m.tilts_deg, m.azimuths_deg)
    dirs = np.array([fan[t, a] for *_, t, a in lifted])
    _, more = r._solve(points, dirs)
    assert np.nanmax(more) <= 2 * r.LIFT_CONVERGED_MM


def test_the_reference_clearance_proxy_rejects_nothing_the_kinematics_accept(reference):
    """Its gantry is the plane the kinematics test the corners against."""
    m = r.build_map(reference, x_mm=[50.0, 150.0], y_mm=[146.5], z_mm=[2.0, 60.0])
    assert np.array_equal(m.clear, np.isfinite(m.lift_mm))
    assert m.clearance_model == "reference"


def test_the_cache_round_trips(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(r, "CACHE_DIR", tmp_path)
    built = r.build_map(reference, x_mm=[150.0], y_mm=[146.5], z_mm=[2.0])
    built.save(r.cache_path("reference"))
    loaded = r.ReachabilityMap.load(r.cache_path("reference"))
    assert np.array_equal(loaded.lift_mm, built.lift_mm, equal_nan=True)
    assert np.array_equal(loaded.clear, built.clear)
    assert loaded.profile_name == "reference" and np.array_equal(loaded.tilts_deg, built.tilts_deg)
    # load_or_build takes the cached file rather than building the whole bed.
    assert r.load_or_build(reference).lift_mm.shape == built.lift_mm.shape


def _hand_made_map(table):
    """A map whose largest reachable tilt is ``table[x, y, z, azimuth]`` exactly."""
    tilts = np.arange(0.0, 30.01, 2.5)
    lift = np.where(tilts[None, None, None, :, None] <= table[:, :, :, None, :], 0.0, np.nan).astype(np.float32)
    return r.ReachabilityMap(
        profile_name="test", x_mm=np.array([0.0, 10.0]), y_mm=np.array([0.0]), z_mm=np.array([0.0]),
        tilts_deg=tilts, azimuths_deg=np.arange(4) * 90.0, lift_mm=lift,
        clear=np.ones(lift.shape, dtype=bool), clearance_model="none",
    )


def test_the_lookup_interpolates_takes_the_smaller_azimuth_and_rounds_down():
    table = np.zeros((2, 1, 1, 4))
    table[0, 0, 0] = [20.0, 30.0, 30.0, 30.0]
    table[1, 0, 0] = [10.0, 30.0, 30.0, 30.0]
    m = _hand_made_map(table)
    # Halfway in x at azimuth 0: (20 + 10) / 2 = 15.
    assert m.max_tilt_here([[5.0, 0.0, 0.0]], 0.0)[0] == 15.0
    # A quarter of the way: 17.5 exactly; a little further, 16.25, rounds down to 15.
    assert m.max_tilt_here([[2.5, 0.0, 0.0]], 0.0)[0] == 17.5
    assert m.max_tilt_here([[3.5, 0.0, 0.0]], 0.0)[0] == 15.0
    # Between azimuths 0 and 90 the smaller one governs.
    assert m.max_tilt_here([[0.0, 0.0, 0.0]], 45.0)[0] == 20.0
    # The part's placement is added before the lookup.
    assert m.max_tilt_here([[0.0, 0.0, 0.0]], 0.0, offset_mm=(10.0, 0.0, 0.0))[0] == 10.0


def test_a_cell_that_reaches_nothing_poisons_its_neighbourhood():
    table = np.full((2, 1, 1, 4), 30.0)
    table[1, 0, 0, 0] = np.nan
    m = _hand_made_map(np.nan_to_num(table, nan=-1.0))
    assert np.isnan(m.max_tilt_deg()[1, 0, 0, 0])
    assert np.isnan(m.max_tilt_here([[5.0, 0.0, 0.0]], 0.0)[0])
    assert m.max_tilt_here([[0.0, 0.0, 0.0]], 0.0)[0] == 30.0


def test_the_viewer_prints_the_worst_direction_per_grid_point():
    import reachability_map

    table = np.zeros((2, 1, 1, 4))
    table[0, 0, 0] = [20.0, 30.0, 30.0, 30.0]
    table[1, 0, 0] = [-1.0, 30.0, 30.0, 30.0]  # reaches nothing toward +x
    lines = reachability_map.bed_table(_hand_made_map(table), 0.0, without_lift=False, every=1)
    assert lines[2].split() == ["x", "0", "20"] and lines[3].split() == ["x", "10", "-"]
