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


def test_resume_drops_what_is_already_current():
    everything = rmp.build_jobs(["xs"])
    resumed = rmp.build_jobs(["xs"], resume=True)
    assert len(everything) == 24
    assert len(resumed) < len(everything)


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


def test_nothing_to_do_is_reported_rather_than_run(tmp_path, capsys):
    code = rmp.main(["--sizes", "xs", "--resume", "--dry-run",
                     "--root", str(tmp_path / "w")])
    assert code == 0
    assert "already has a current result" in capsys.readouterr().out
