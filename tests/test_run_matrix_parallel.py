"""Tests for the parallel matrix runner (`tools/run_matrix_parallel.py`).

The runner exists because the matrix is 48 independent runs and the lab machine
has 36 cores. The thing that makes it dangerous is that Atomizer writes the same
`data/` paths for every run of a part regardless of `max_slope`, so two
concurrent runs of one part would silently overwrite each other — correction 4.7
again, on the numbers the project is judged by.

So the tests concentrate on isolation, on not losing finished work, and on the
queue actually being a queue. Nothing here runs the real pipeline; `run_job` is
stubbed, which is the only way to test the orchestration at all.
"""

import json
import threading
import time

import pytest

import overhang_report as orep
import run_matrix_parallel as rmp

REPO_ROOT_MESH = rmp.REPO_ROOT / "data" / "mesh"


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_every_combination_appears_once():
    jobs = rmp.build_jobs(["xs", "s"], parts=("ramp45", "ramp60"), slopes=(7.0, 30.0))
    keys = [(j["part"], j["slope"]) for j in jobs]

    assert len(keys) == 8
    assert len(set(keys)) == 8
    assert ("ramp45_xs", 7.0) in keys
    assert ("ramp60_s", 30.0) in keys


def test_the_longest_jobs_are_scheduled_first():
    """Otherwise one worker grinds through the slowest part while the rest idle."""
    jobs = rmp.build_jobs(["xs", "s"])
    estimates = [j["estimate_s"] for j in jobs if j["estimate_s"]]
    assert estimates == sorted(estimates, reverse=True)


def test_a_job_with_no_estimate_is_scheduled_before_the_known_ones():
    """An unmeasured job could be the long one. Assuming it is short is the
    mistake that leaves a worker running alone at the end."""
    jobs = rmp.build_jobs(["xs", "s"], parts=("ramp45", "nonexistent_part"))
    first_unknown = next(i for i, j in enumerate(jobs) if not j["estimate_s"])
    last_known = max(
        (i for i, j in enumerate(jobs) if j["estimate_s"]), default=-1
    )
    assert first_unknown < last_known or last_known == -1


def test_resume_drops_what_is_already_current(monkeypatch):
    """Pretending to be the machine that wrote the committed reports.

    This used to read the reports and assume they counted as done, which quietly
    depended on whose machine ran the test: the committed 48 were measured on the
    operator's laptop, so the assertion held there and nowhere else. Now that
    resume compares machines, the test has to say which machine it is.
    """
    reports = orep.load_reports(quiet=True)
    assert reports, "the committed baseline reports are missing"
    theirs = rmp.report_machine_key(reports[0])
    assert theirs is not None, "the committed reports should carry provenance"
    monkeypatch.setattr(rmp, "this_machine_key", lambda: theirs)

    everything = rmp.build_jobs(["xs"])
    resumed = rmp.build_jobs(["xs"], resume=True)
    assert len(everything) == 24
    assert len(resumed) < len(everything)


def test_resume_keeps_everything_when_another_machine_wrote_the_reports(monkeypatch):
    """The lab machine's case: the committed reports are the laptop's, so all 24
    combinations still need running here."""
    monkeypatch.setattr(rmp, "this_machine_key",
                        lambda: ("somewhere-else", "stock mix", "1.7.4", "reference"))
    assert len(rmp.build_jobs(["xs"], resume=True)) == 24


# --------------------------------------------------------------------------
# Resume must know which machine wrote a report
# --------------------------------------------------------------------------


def _provenance(machine, arch="stock mix", taichi="1.7.4", profile="reference"):
    """A provenance block complete enough for `prov.is_known` to accept it."""
    return {
        "source": "pipeline",
        "machine": machine,
        "os": "Windows-10",
        "python": "3.10.21",
        "taichi": taichi,
        "ti_arch_setting": arch,
        "stage_arches": {"order_atoms": "x64"},
        "machine_profile": profile,
        "git_commit": "abc123",
        "code_modified": False,
        "recorded_utc": "2026-09-27T00:00:00Z",
        "metrics_version": orep.METRICS_VERSION,
        "scored": {},
        "parallel_workers": 8,
        "taichi_cpu_threads": None,
    }


def _report(part, slope, provenance):
    return {
        "part": part,
        "max_slope_deg": slope,
        "metrics_version": orep.METRICS_VERSION,
        "provenance": provenance,
        "runtime": {"total_s": 100.0, "stages": {}},
    }


def test_resume_skips_a_run_this_machine_already_did(monkeypatch):
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0,
                                                     _provenance("labpc"))])
    jobs = rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,), resume=True)
    assert jobs == []


def test_resume_re_runs_a_report_from_another_machine(monkeypatch):
    """The failure this fixes.

    Half the lab machine's 48-run matrix failed, leaving the laptop's committed
    reports in place. They were current, so `--resume` counted them done and
    would have run nothing — the 24 that needed re-running were invisible to it.
    """
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0,
                                                     _provenance("AbdoYasser"))])
    jobs = rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,), resume=True)
    assert [(j["part"], j["slope"]) for j in jobs] == [("ramp45_xs", 7.0)]


def test_a_renamed_machine_is_still_the_same_machine(monkeypatch):
    """Case only: `platform.node()` casing varies, the machine does not."""
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0,
                                                     _provenance("LabPC"))])
    assert rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,),
                          resume=True) == []


def test_resume_re_runs_when_the_backend_was_forced(monkeypatch):
    """Correction 3.9: a forced backend is a different computation."""
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0,
                                                     _provenance("labpc",
                                                                 arch="cpu"))])
    assert len(rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,),
                              resume=True)) == 1


def test_resume_re_runs_a_report_with_no_provenance(monkeypatch):
    """Unknown is not "mine" — the 48 reports predating P1.7 must not be assumed."""
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0, None)])
    assert len(rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,),
                              resume=True)) == 1


def test_resume_re_runs_a_skip_pipeline_report(monkeypatch):
    """--skip-pipeline records no origin for the toolpath, so it is not evidence
    that this machine produced it."""
    monkeypatch.setattr(rmp, "this_machine_key", lambda: ("labpc", "stock mix",
                                                          "1.7.4", "reference"))
    block = dict(_provenance("labpc"), source="skip-pipeline")
    monkeypatch.setattr(orep, "load_reports",
                        lambda quiet=False: [_report("ramp45_xs", 7.0, block)])
    assert len(rmp.build_jobs(["xs"], parts=("ramp45",), slopes=(7.0,),
                              resume=True)) == 1


def test_the_machine_key_matches_how_provenance_builds_it(monkeypatch):
    """The two must not drift: resume compares a key it builds itself against one
    `atom.provenance.collect` built."""
    monkeypatch.delenv("ATOM_TI_ARCH", raising=False)
    monkeypatch.delenv("ATOM_MACHINE", raising=False)
    key = rmp.this_machine_key()
    assert key[1] == rmp.prov.STOCK_MIX
    assert key[3] == "reference"

    monkeypatch.setenv("ATOM_TI_ARCH", "CPU")
    monkeypatch.setenv("ATOM_MACHINE", "ours")
    key = rmp.this_machine_key()
    assert key[1] == "cpu", "the setting is lower-cased, as collect() does"
    assert key[3] == "ours"


# --------------------------------------------------------------------------
# Worker count against RAM
# --------------------------------------------------------------------------


def test_the_worker_limit_reproduces_the_observed_failure():
    """Calibration, not a model: 8 workers worked on 64 GiB and 16 did not."""
    assert rmp.workers_ram_allows(64.0) == 8
    assert rmp.workers_ram_allows(64.0) < 16


def test_the_worker_limit_scales_with_ram():
    assert rmp.workers_ram_allows(128.0) == 16
    assert rmp.workers_ram_allows(16.0) == 2


def test_at_least_one_worker_is_always_allowed():
    assert rmp.workers_ram_allows(1.0) == 1


def test_unknown_ram_gives_no_limit(monkeypatch):
    """A warning must never stop a run just because RAM could not be read."""
    monkeypatch.setattr(rmp, "total_ram_gib", lambda: None)
    assert rmp.workers_ram_allows() is None


def test_total_ram_is_readable_here():
    ram = rmp.total_ram_gib()
    assert ram is None or ram > 0.5


def test_the_dry_run_warns_when_workers_exceed_ram(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rmp, "total_ram_gib", lambda: 64.0)
    rmp.main(["--sizes", "xs", "--workers", "16", "--dry-run",
              "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "--workers 8" in out
    assert "1073741824" in out, "the warning should quote the error it prevents"


def test_the_dry_run_does_not_warn_within_the_limit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rmp, "total_ram_gib", lambda: 64.0)
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    assert "WARNING" not in capsys.readouterr().out


def test_projection_is_bounded_below_by_the_longest_job():
    """No job can be split, so one long job sets the floor however many workers."""
    jobs = [
        {"part": "a", "slope": 7.0, "estimate_s": 3600.0},
        {"part": "b", "slope": 7.0, "estimate_s": 60.0},
    ]
    assert rmp.projected_seconds(jobs, workers=100) == pytest.approx(3600.0)


def test_projection_divides_the_work_when_jobs_are_even():
    jobs = [{"part": str(i), "slope": 7.0, "estimate_s": 100.0} for i in range(10)]
    assert rmp.projected_seconds(jobs, workers=10) == pytest.approx(100.0)
    assert rmp.projected_seconds(jobs, workers=5) == pytest.approx(200.0)


def test_the_warmup_is_the_shortest_job_not_the_longest(monkeypatch, tmp_path, capsys):
    """The warm-up runs alone, so its length is pure serial time.

    Taking the first of a longest-first list meant taking the *longest*: on the
    full matrix that is `ramp90_s` at 1:27:31, spent alone with 15 workers idle.
    The first real run paid 13:21 of it. Any job fills the cache equally well.
    """
    ran = []

    def fake_run_job(worker, job, log_dir, threads=None):
        ran.append((job["part"], job["slope"]))
        return True, 0.01, tmp_path / "l.log"

    monkeypatch.setattr(rmp, "run_job", fake_run_job)
    monkeypatch.setattr(rmp, "collect",
                        lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)
    monkeypatch.setattr(orep, "load_reports", lambda quiet=False: [])
    monkeypatch.setattr(rmp, "create_worker",
                        lambda root, i, overwrite=False: tmp_path / f"w{i}")
    monkeypatch.setattr(rmp, "seed_cache", lambda source, workers: len(workers) - 1)

    jobs = [
        {"part": "slow", "slope": 7.0, "estimate_s": 3600.0},
        {"part": "middling", "slope": 7.0, "estimate_s": 600.0},
        {"part": "quick", "slope": 7.0, "estimate_s": 60.0},
    ]
    monkeypatch.setattr(rmp, "build_jobs", lambda *a, **k: list(jobs))

    rmp.main(["--sizes", "xs", "--workers", "3", "--root", str(tmp_path / "w"),
              "--keep-workers"])

    assert ran, "nothing ran"
    assert ran[0][0] == "quick", (
        f"the warm-up took {ran[0][0]!r}; it must take the shortest job"
    )
    assert sorted(part for part, _ in ran) == ["middling", "quick", "slow"], (
        "every job must still run exactly once, warm-up included"
    )


def test_choose_warmup_takes_the_shortest():
    jobs = [
        {"part": "slow", "slope": 7.0, "estimate_s": 3600.0},
        {"part": "quick", "slope": 7.0, "estimate_s": 60.0},
        {"part": "middling", "slope": 7.0, "estimate_s": 600.0},
    ]
    assert jobs[rmp.choose_warmup(jobs)]["part"] == "quick"


def test_an_unmeasured_job_is_not_chosen_as_the_warmup():
    """It might be the long one, and the warm-up is the worst place to find out."""
    jobs = [
        {"part": "unknown", "slope": 7.0, "estimate_s": 0.0},
        {"part": "quick", "slope": 7.0, "estimate_s": 60.0},
    ]
    assert jobs[rmp.choose_warmup(jobs)]["part"] == "quick"


def test_choose_warmup_on_an_empty_queue():
    assert rmp.choose_warmup([]) is None


def test_the_dry_run_names_the_warmup_job(tmp_path, capsys):
    """A dry run that predicts nothing about the warm-up is worse than useless:
    the operator was told to check it and had nothing to look at."""
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "Warm-up (runs alone" in out
    assert "SHORTEST" in out


def test_the_dry_run_and_the_real_run_agree_on_the_warmup(monkeypatch, tmp_path):
    """They share `choose_warmup` rather than each deciding. A dry run that
    predicts a different warm-up from the real one is worse than no dry run."""
    ran = []
    monkeypatch.setattr(rmp, "run_job",
                        lambda w, j, d, t=None: (ran.append(j["part"]),
                                                 (True, 0.01, tmp_path / "l.log"))[1])
    monkeypatch.setattr(rmp, "collect",
                        lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)
    monkeypatch.setattr(orep, "load_reports", lambda quiet=False: [])
    monkeypatch.setattr(rmp, "create_worker",
                        lambda root, i, overwrite=False: tmp_path / f"w{i}")
    monkeypatch.setattr(rmp, "seed_cache", lambda s, w: len(w) - 1)

    jobs = [
        {"part": "slow", "slope": 7.0, "estimate_s": 3600.0},
        {"part": "quick", "slope": 7.0, "estimate_s": 60.0},
        {"part": "middling", "slope": 7.0, "estimate_s": 600.0},
    ]
    monkeypatch.setattr(rmp, "build_jobs", lambda *a, **k: list(jobs))
    predicted = jobs[rmp.choose_warmup(jobs)]["part"]

    rmp.main(["--sizes", "xs", "--workers", "3", "--root", str(tmp_path / "w"),
              "--keep-workers"])

    assert ran[0] == predicted


def test_the_two_measured_efficiencies_bracket_each_other():
    """Measured twice at 8 workers: 65 % on short jobs, 79 % on long ones."""
    assert 0 < rmp.EFFICIENCY_SHORT_JOBS < rmp.EFFICIENCY_LONG_JOBS < 1.0
    assert rmp.MEASURED_EFFICIENCY == rmp.EFFICIENCY_SHORT_JOBS, (
        "the single-number fallback must be the conservative end"
    )


def test_the_estimate_is_reported_as_a_range(tmp_path, capsys):
    """Two measurements, so two figures. Interpolating between them by job mix
    would be a model, and models of this machine have lost to measurements of it
    twice (4.19)."""
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "79%-65% efficiency" in out
    assert " to " in out
    assert "expect this" in out


def test_a_short_job_mix_is_named_as_the_slower_end(tmp_path, capsys):
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    assert "mostly short jobs" in capsys.readouterr().out


def test_a_long_job_mix_is_named_as_the_faster_end(tmp_path, capsys):
    rmp.main(["--sizes", "xs", "s", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    assert "mostly long jobs" in capsys.readouterr().out


def test_an_explicit_efficiency_replaces_the_range(tmp_path, capsys):
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--efficiency", "0.5", "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "at 50% efficiency" in out
    assert "79%-65%" not in out


def test_the_measured_efficiency_makes_the_estimate_larger():
    """The ideal was 2.4x optimistic against the first real run, so both are
    reported and the realistic one is the headline."""
    jobs = [{"part": str(i), "slope": 7.0, "estimate_s": 100.0} for i in range(8)]
    ideal = rmp.projected_seconds(jobs, workers=8)
    likely = rmp.projected_seconds(jobs, workers=8,
                                   efficiency=rmp.MEASURED_EFFICIENCY)
    assert likely > ideal
    assert likely == pytest.approx(100.0 / rmp.MEASURED_EFFICIENCY)


def test_the_longest_job_still_floors_the_estimate_with_efficiency():
    """No job can be split, however inefficient the pool."""
    jobs = [{"part": "a", "slope": 7.0, "estimate_s": 3600.0}]
    assert rmp.projected_seconds(jobs, 100, efficiency=0.1) == pytest.approx(3600.0)


def test_a_zero_efficiency_does_not_divide_by_zero():
    """Eight jobs, not one: a single job would floor the answer at its own
    length and hide whether the division happened at all."""
    jobs = [{"part": str(i), "slope": 7.0, "estimate_s": 100.0} for i in range(8)]
    assert rmp.projected_seconds(jobs, 4, efficiency=0.0) == pytest.approx(200.0)


def test_the_dry_run_prints_both_estimates(tmp_path, capsys):
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "lower bound" in out
    assert "expect this" in out, "the realistic estimate must be the headline"


def test_projection_is_unknown_without_previous_runtimes():
    jobs = [{"part": "a", "slope": 7.0, "estimate_s": 0.0}]
    assert rmp.projected_seconds(jobs, workers=4) is None
    assert rmp.format_duration(None) == "unknown"


def test_a_ratio_scales_the_projection():
    """The lab machine is 1.21x slower per run (handoff section 3)."""
    jobs = [{"part": "a", "slope": 7.0, "estimate_s": 100.0}]
    assert rmp.projected_seconds(jobs, 1, ratio=1.21) == pytest.approx(121.0)


# --------------------------------------------------------------------------
# Worker isolation — the reason this tool is shaped the way it is
# --------------------------------------------------------------------------


def test_a_worker_has_its_own_copy_of_every_shared_output_path(tmp_path):
    """The whole design rests on this. If two workers shared `data/sdf`, two
    concurrent runs of one part would overwrite each other's intermediates."""
    a = rmp.create_worker(tmp_path, 0)
    b = rmp.create_worker(tmp_path, 1)

    assert a != b
    for name in rmp.DATA_OUTPUTS:
        assert (a / "data" / name).is_dir()
        assert (b / "data" / name).is_dir()
        assert (a / "data" / name) != (b / "data" / name)
    assert (a / "ticache") != (b / "ticache")


def test_a_worker_carries_the_inputs_it_needs(tmp_path):
    worker = rmp.create_worker(tmp_path, 0)
    assert list((worker / "data" / "mesh").glob("*.stl")), "no meshes to slice"
    assert list((worker / "data" / "param").glob("*.json")), "no parameters"
    assert (worker / "tools" / "overhang_report.py").is_file()
    assert (worker / "src" / "atom" / "overhang_metrics.py").is_file()
    assert (worker / "config" / "machines" / "reference.json").is_file()


def test_a_worker_gets_inputs_only_not_leftovers(tmp_path, monkeypatch):
    """A worker must not inherit another run's intermediates.

    `data/mesh/` accumulates `.obj` files that Blender writes during a run. If
    Blender then fails inside a worker, a stale `.obj` copied in would let the
    rest of the pipeline carry on with the WRONG geometry and write a plausible
    but wrong report. Found because a worker copy measured 53 MB on the
    operator's machine against 21 MB on a clean tree.
    """
    stale = REPO_ROOT_MESH / "zz_stale_worker_probe.obj"
    stale.write_text("# not an input\n", encoding="utf-8")
    try:
        worker = rmp.create_worker(tmp_path, 0)
        assert list((worker / "data" / "mesh").glob("*.obj")) == [], (
            "a worker inherited a Blender intermediate"
        )
        assert list((worker / "data" / "mesh").glob("*.stl")), "the meshes are gone"
    finally:
        stale.unlink(missing_ok=True)


def test_a_worker_gets_the_default_tangent_image(tmp_path):
    """The bug that broke the first parallel run, pinned.

    `tools/compute_tangents.py` loads `data/image/0.png` in its *else* branch,
    so every run without an explicit `--top_lines` needs it. Enumerating inputs
    by hand missed it, stage 6 died in the worker, and `atomize.py`'s ignored
    exit codes turned that into twelve failures.
    """
    worker = rmp.create_worker(tmp_path, 0)
    assert (worker / "data" / "image" / "0.png").is_file(), (
        "compute_tangents.py's default tangent field is missing from the worker"
    )


def test_inputs_come_from_git_rather_than_a_hand_written_list(tmp_path):
    """Everything a run generates is gitignored, so tracked-under-data is
    exactly input — and it stays correct as stages change, which two hand-written
    lists did not."""
    paths = rmp.data_input_paths()
    assert paths, "no inputs found at all"
    names = {str(p).replace("\\", "/") for p in paths}
    assert "data/image/0.png" in names
    assert any(n.endswith(".stl") for n in names)
    assert any(n.endswith(".json") for n in names)
    assert not any(n.endswith(".obj") for n in names), (
        "a Blender intermediate is not an input"
    )
    assert not any(n.endswith(".gitignore") for n in names)


def test_a_worker_has_no_git_directory(tmp_path):
    """So a worker tree can never be committed from, and 16 copies of history
    are not paid for."""
    worker = rmp.create_worker(tmp_path, 0)
    assert not (worker / ".git").exists()


def test_a_worker_starts_with_no_results(tmp_path):
    """A worker inheriting the committed reports could make a skipped run look
    complete — correction 4.7's failure, one level up."""
    worker = rmp.create_worker(tmp_path, 0)
    assert list((worker / "reports" / "baseline_overhang").glob("*.json")) == []
    assert list((worker / "reports" / "toolpaths").glob("*.npz")) == []


def test_creating_a_worker_twice_leaves_it_alone(tmp_path):
    worker = rmp.create_worker(tmp_path, 0)
    (worker / "data" / "sdf" / "marker.npz").write_bytes(b"x")
    again = rmp.create_worker(tmp_path, 0)
    assert again == worker
    assert (worker / "data" / "sdf" / "marker.npz").is_file()


def test_fresh_workers_rebuilds_from_scratch(tmp_path):
    worker = rmp.create_worker(tmp_path, 0)
    (worker / "data" / "sdf" / "marker.npz").write_bytes(b"x")
    rmp.create_worker(tmp_path, 0, overwrite=True)
    assert not (worker / "data" / "sdf" / "marker.npz").exists()


def test_the_measured_tree_size_is_plausible():
    """Measured, not guessed: an earlier guess of 600 MB was thirty times out."""
    size = rmp.tree_megabytes()
    assert 1 < size < 200, f"{size} MB does not look like a copy of this tree"


# --------------------------------------------------------------------------
# Cache seeding
# --------------------------------------------------------------------------


def test_a_warmed_cache_is_copied_to_every_other_worker(tmp_path):
    """A cold cache inflates the GPU stages five to sevenfold, so sixteen empty
    caches would pay that sixteen times."""
    workers = [rmp.create_worker(tmp_path, i) for i in range(3)]
    (workers[0] / "ticache" / "kernel.bin").write_bytes(b"compiled")

    assert rmp.seed_cache(workers[0], workers) == 2
    for worker in workers[1:]:
        assert (worker / "ticache" / "kernel.bin").read_bytes() == b"compiled"


def test_seeding_an_empty_cache_does_nothing(tmp_path):
    workers = [rmp.create_worker(tmp_path, i) for i in range(2)]
    assert rmp.seed_cache(workers[0], workers) == 0


def test_seeding_does_not_copy_a_worker_onto_itself(tmp_path):
    workers = [rmp.create_worker(tmp_path, 0)]
    (workers[0] / "ticache" / "kernel.bin").write_bytes(b"compiled")
    assert rmp.seed_cache(workers[0], workers) == 0
    assert (workers[0] / "ticache" / "kernel.bin").is_file()


# --------------------------------------------------------------------------
# Collection — never lose a finished run
# --------------------------------------------------------------------------


def _stub_report(part, slope):
    return {
        "schema_version": orep.SCHEMA_VERSION,
        "metrics_version": orep.METRICS_VERSION,
        "part": part,
        "max_slope_deg": slope,
        "runtime": {"total_s": 12.0, "stages": {}},
        "metrics": {
            "max_tool_tilt_deg": 1.0,
            "unsupported_fraction_near_overhangs": 0.002,
            "surfaces": [],
        },
        "verdict": {"printable": True, "worst_effective_deg": 45.0, "assessable": True},
        "provenance": {"pipeline": None, "scored": None},
    }


def test_collect_brings_back_the_report_and_the_archived_toolpath(
    tmp_path, monkeypatch
):
    worker = rmp.create_worker(tmp_path / "workers", 0)
    job = {"part": "ramp45_xs", "slope": 7.0}
    name = "ramp45_xs_ms7"
    (worker / "reports" / "baseline_overhang" / f"{name}.json").write_text(
        json.dumps(_stub_report("ramp45_xs", 7.0)), encoding="utf-8"
    )
    (worker / "reports" / "toolpaths" / f"{name}.npz").write_bytes(b"npz")

    monkeypatch.setattr(orep, "REPORT_DIR", tmp_path / "out")
    monkeypatch.setattr(orep, "TOOLPATH_ARCHIVE", tmp_path / "archive")

    report = rmp.collect(worker, job)

    assert report is not None and report["part"] == "ramp45_xs"
    assert (tmp_path / "out" / f"{name}.json").is_file()
    assert (tmp_path / "archive" / f"{name}.npz").read_bytes() == b"npz"


def test_collect_reports_nothing_when_the_worker_wrote_nothing(tmp_path, monkeypatch):
    worker = rmp.create_worker(tmp_path / "workers", 0)
    monkeypatch.setattr(orep, "REPORT_DIR", tmp_path / "out")
    assert rmp.collect(worker, {"part": "ramp45_xs", "slope": 7.0}) is None


def test_a_missing_archive_does_not_lose_the_report(tmp_path, monkeypatch):
    """The report is the artifact; the archive only saves a re-slice later."""
    worker = rmp.create_worker(tmp_path / "workers", 0)
    (worker / "reports" / "baseline_overhang" / "ramp45_xs_ms7.json").write_text(
        json.dumps(_stub_report("ramp45_xs", 7.0)), encoding="utf-8"
    )
    monkeypatch.setattr(orep, "REPORT_DIR", tmp_path / "out")
    monkeypatch.setattr(orep, "TOOLPATH_ARCHIVE", tmp_path / "archive")

    assert rmp.collect(worker, {"part": "ramp45_xs", "slope": 7.0}) is not None


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


@pytest.fixture
def stub_run(monkeypatch, tmp_path):
    """Replace the real slicing with a recorder, so orchestration is testable."""
    calls = []
    lock = threading.Lock()

    def fake_run_job(worker, job, log_dir, threads=None):
        with lock:
            calls.append((str(worker), job["part"], job["slope"]))
        time.sleep(0.01)
        return True, 0.01, tmp_path / "fake.log"

    def fake_collect(worker, job):
        return _stub_report(job["part"], job["slope"])

    monkeypatch.setattr(rmp, "run_job", fake_run_job)
    monkeypatch.setattr(rmp, "collect", fake_collect)
    monkeypatch.setattr(orep, "append_progress", lambda report: None)
    return calls


def test_every_job_runs_exactly_once(stub_run, tmp_path):
    jobs = [{"part": f"p{i}", "slope": 7.0, "estimate_s": 1.0} for i in range(20)]
    workers = [tmp_path / f"w{i}" for i in range(4)]

    completed, failed = rmp.run_matrix(jobs, workers, tmp_path)

    assert failed == []
    assert len(completed) == 20
    assert sorted(part for _, part, _ in stub_run) == sorted(
        j["part"] for j in jobs
    )


def test_work_is_spread_across_the_workers(stub_run, tmp_path):
    jobs = [{"part": f"p{i}", "slope": 7.0, "estimate_s": 1.0} for i in range(40)]
    workers = [tmp_path / f"w{i}" for i in range(4)]

    rmp.run_matrix(jobs, workers, tmp_path)

    used = {worker for worker, _, _ in stub_run}
    assert len(used) == 4, "a queue that leaves workers unused is not a queue"


def test_no_two_jobs_share_a_worker_at_the_same_time(monkeypatch, tmp_path):
    """The isolation guarantee, checked rather than assumed: a worker tree must
    never host two runs at once, or they overwrite each other's `data/`."""
    active = {}
    overlaps = []
    lock = threading.Lock()

    def fake_run_job(worker, job, log_dir, threads=None):
        key = str(worker)
        with lock:
            if key in active:
                overlaps.append((key, active[key], job["part"]))
            active[key] = job["part"]
        time.sleep(0.02)
        with lock:
            active.pop(key, None)
        return True, 0.02, tmp_path / "fake.log"

    monkeypatch.setattr(rmp, "run_job", fake_run_job)
    monkeypatch.setattr(rmp, "collect",
                        lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)

    jobs = [{"part": f"p{i}", "slope": 7.0, "estimate_s": 1.0} for i in range(30)]
    rmp.run_matrix(jobs, [tmp_path / f"w{i}" for i in range(5)], tmp_path)

    assert overlaps == [], f"two runs shared a worker tree: {overlaps}"


def test_a_failing_run_is_reported_and_does_not_stop_the_others(
    monkeypatch, tmp_path
):
    def fake_run_job(worker, job, log_dir, threads=None):
        return job["part"] != "bad", 0.01, tmp_path / f"{job['part']}.log"

    monkeypatch.setattr(rmp, "run_job", fake_run_job)
    monkeypatch.setattr(rmp, "collect",
                        lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)

    jobs = [{"part": p, "slope": 7.0, "estimate_s": 1.0}
            for p in ("good1", "bad", "good2")]
    completed, failed = rmp.run_matrix(jobs, [tmp_path / "w0", tmp_path / "w1"],
                                       tmp_path)

    assert len(completed) == 2
    assert len(failed) == 1
    assert failed[0]["part"] == "bad"
    assert "log" in failed[0]


def test_a_run_that_exits_zero_but_writes_nothing_counts_as_failed(
    monkeypatch, tmp_path
):
    """Silent non-production is worse than a crash: it would leave a gap that
    --resume then treats as never attempted, forever."""
    monkeypatch.setattr(rmp, "run_job",
                        lambda w, j, d, t=None: (True, 0.01, tmp_path / "l.log"))
    monkeypatch.setattr(rmp, "collect", lambda w, j: None)
    monkeypatch.setattr(orep, "append_progress", lambda report: None)

    completed, failed = rmp.run_matrix(
        [{"part": "p", "slope": 7.0, "estimate_s": 1.0}], [tmp_path / "w0"], tmp_path
    )

    assert completed == []
    assert len(failed) == 1
    assert "wrote no report" in failed[0]["error"]


def test_results_are_collected_as_each_job_finishes(monkeypatch, tmp_path):
    """Not at the end. An interrupted matrix must keep what it finished — the
    lesson of correction 4.7, which cost 12 hours."""
    collected_at = []
    total = 6

    def fake_collect(worker, job):
        collected_at.append(len(collected_at))
        return _stub_report(job["part"], job["slope"])

    monkeypatch.setattr(rmp, "run_job",
                        lambda w, j, d, t=None: (True, 0.01, tmp_path / "l.log"))
    monkeypatch.setattr(rmp, "collect", fake_collect)
    monkeypatch.setattr(orep, "append_progress", lambda report: None)

    seen = []
    rmp.run_matrix(
        [{"part": f"p{i}", "slope": 7.0, "estimate_s": 1.0} for i in range(total)],
        [tmp_path / "w0", tmp_path / "w1"], tmp_path,
        on_event=lambda kind, payload: seen.append((kind, len(collected_at))),
    )

    dones = [count for kind, count in seen if kind == "done"]
    assert dones == list(range(1, total + 1)), (
        "each finished job must be collected before the next event, not batched"
    )


def test_more_workers_than_jobs_is_not_an_error(stub_run, tmp_path):
    jobs = [{"part": "only", "slope": 7.0, "estimate_s": 1.0}]
    completed, failed = rmp.run_matrix(
        jobs, [tmp_path / f"w{i}" for i in range(4)], tmp_path
    )
    assert len(completed) == 1 and failed == []


def test_an_empty_job_list_does_nothing_quietly(stub_run, tmp_path):
    assert rmp.run_matrix([], [tmp_path / "w0"], tmp_path) == ([], [])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_dry_run_creates_nothing(tmp_path, capsys):
    root = tmp_path / "workers"
    code = rmp.main(["--sizes", "xs", "--workers", "4", "--dry-run",
                     "--root", str(root)])

    assert code == 0
    assert not root.exists(), "--dry-run must not create worker trees"
    out = capsys.readouterr().out
    assert "nothing was created or run" in out
    assert "Planned order" in out


def test_dry_run_states_the_projection_as_a_lower_bound(tmp_path, capsys):
    rmp.main(["--sizes", "xs", "--workers", "8", "--dry-run",
              "--root", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "lower bound" in out, (
        "the projection ignores contention and must not be presented as a promise"
    )


def test_nothing_to_do_is_reported_rather_than_run(tmp_path, capsys, monkeypatch):
    """As the machine that wrote the committed reports — see the note above."""
    reports = orep.load_reports(quiet=True)
    assert reports, "the committed baseline reports are missing"
    monkeypatch.setattr(rmp, "this_machine_key",
                        lambda: rmp.report_machine_key(reports[0]))

    code = rmp.main(["--sizes", "xs", "--resume", "--dry-run",
                     "--root", str(tmp_path / "w")])
    assert code == 0
    assert "already has a current result" in capsys.readouterr().out


# --------------------------------------------------------------------------
# What a worker's report records about how it ran (P1.7 follow-up)
# --------------------------------------------------------------------------


def test_a_worker_is_told_the_commit_and_the_pool_size(monkeypatch, tmp_path):
    """The worker tree has no .git, and its runtime depends on how many ran at once."""
    from atom import provenance as prov

    seen = {}

    def fake_run(command, cwd, env, stdout, stderr):
        seen.update(env)
        return type("Result", (), {"returncode": 0})()

    # The parent's git state is worked out before any job; seed it, so the
    # stubbed subprocess.run below is only ever the worker's run.
    monkeypatch.setattr(rmp, "_PARENT_GIT_ENV", {
        prov.GIT_COMMIT_ENV_VAR: "b" * 40, prov.CODE_MODIFIED_ENV_VAR: "false",
    })
    monkeypatch.setattr(rmp.subprocess, "run", fake_run)
    job = {"part": "ramp45_xs", "slope": 7.0, "estimate_s": 0.0, "pool_size": 8}
    rmp.run_job(tmp_path, job, tmp_path)

    assert seen[prov.GIT_COMMIT_ENV_VAR] == "b" * 40
    assert seen[prov.CODE_MODIFIED_ENV_VAR] == "false"
    assert seen[prov.PARALLEL_WORKERS_ENV_VAR] == "8"


def test_the_parent_commit_is_the_repositorys(monkeypatch):
    from atom import provenance as prov

    monkeypatch.delenv(prov.GIT_COMMIT_ENV_VAR, raising=False)
    monkeypatch.setattr(rmp, "_PARENT_GIT_ENV", None)
    commit, modified = prov.git_state(rmp.REPO_ROOT)
    env = rmp.parent_git_env()
    assert env[prov.GIT_COMMIT_ENV_VAR] == (commit or "")
    assert env[prov.CODE_MODIFIED_ENV_VAR] == {True: "true", False: "false"}.get(modified, "")


def test_the_pool_size_is_the_number_of_workers(monkeypatch, tmp_path):
    """The warm-up runs in a pool of one; everything after in the full pool."""
    pools = []

    def fake_run_job(worker, job, log_dir, threads=None):
        pools.append(job["pool_size"])
        return True, 0.01, tmp_path / "l.log"

    monkeypatch.setattr(rmp, "run_job", fake_run_job)
    monkeypatch.setattr(rmp, "collect", lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)
    monkeypatch.setattr(orep, "load_reports", lambda quiet=False: [])
    monkeypatch.setattr(rmp, "create_worker",
                        lambda root, i, overwrite=False: tmp_path / f"w{i}")
    monkeypatch.setattr(rmp, "seed_cache", lambda source, workers: len(workers) - 1)
    jobs = [{"part": f"p{i}", "slope": 7.0, "estimate_s": 60.0 * (i + 1)} for i in range(4)]
    monkeypatch.setattr(rmp, "build_jobs", lambda *a, **k: [dict(j) for j in jobs])

    rmp.main(["--sizes", "xs", "--workers", "3", "--root", str(tmp_path / "w"),
              "--keep-workers"])

    assert pools == [1, 3, 3, 3]


def test_a_mixed_table_is_announced_on_screen(monkeypatch, tmp_path, capsys):
    """The warning is in the summary file too, but must not be found only there."""
    monkeypatch.setattr(rmp, "run_job", lambda w, j, l, t=None: (True, 0.01, tmp_path / "l.log"))
    monkeypatch.setattr(rmp, "collect", lambda w, j: _stub_report(j["part"], j["slope"]))
    monkeypatch.setattr(orep, "append_progress", lambda report: None)
    monkeypatch.setattr(
        orep, "load_reports",
        lambda quiet=False, field_only=False, overhang_aware=False: (
            [] if field_only or overhang_aware is not False else [{"part": "a"}, {"part": "b"}]
        ),
    )
    monkeypatch.setattr(
        orep, "summarize", lambda reports, field_only_reports=(), aware_reports=(): "table\n"
    )
    monkeypatch.setattr(orep, "SUMMARY_PATH", tmp_path / "summary.md")
    monkeypatch.setattr(orep, "provenance_warning", lambda reports: "WARNING: this table mixes 2 provenance groups")
    monkeypatch.setattr(rmp, "create_worker",
                        lambda root, i, overwrite=False: tmp_path / f"w{i}")
    monkeypatch.setattr(rmp, "seed_cache", lambda source, workers: len(workers) - 1)
    monkeypatch.setattr(rmp, "build_jobs",
                        lambda *a, **k: [{"part": "p", "slope": 7.0, "estimate_s": 60.0}])

    rmp.main(["--sizes", "xs", "--workers", "1", "--root", str(tmp_path / "w"),
              "--keep-workers"])

    assert "WARNING: this table mixes 2 provenance groups" in capsys.readouterr().out
