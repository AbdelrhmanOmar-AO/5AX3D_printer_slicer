"""Run the baseline matrix across many CPU cores at once (build plan P0.8).

Why
---
`order_atoms` is 86 % of runtime and is a sequential loop, so no machine makes
one run much faster — the 36-core lab machine is 1.21x *slower* per run than the
operator's laptop (`docs/handoff.md` section 3). But the matrix is **48
independent runs**, and 36 cores can work on many at once. Measured from the 48
committed runtimes: 26.4 hours serially there against roughly 1.7 hours at 16
workers.

The constraint that shapes this
-------------------------------
Every run of a given part writes the same `data/` paths whatever its
`max_slope` (see `tools/overhang_report.py`'s docstring). **Two concurrent runs
of one part would silently overwrite each other's intermediates** — the same
class of failure as correction 4.7, and silent corruption of the numbers the
project is judged by.

Partitioning by part *and size* would avoid it, since those differ by
`solid_name`, but it leaves a 3.7-hour critical path: `ramp80_s` and `ramp90_s`
need their three slopes in sequence while the fastest worker finishes in 21
minutes. So instead each worker gets **its own copy of the working tree**. The
tree is 35 MB, so sixteen copies is 560 MB, and then nothing whatsoever is
shared: all 48 runs are independent and a queue keeps every worker busy.

Two things that fail quietly, and how they are handled
-----------------------------------------------------
**Taichi's kernel cache.** A single interrupted run has already produced
`Lock .../ticache.lock failed`, so sixteen processes sharing one cache directory
would contend. Each worker gets its own via `TI_OFFLINE_CACHE_FILE_PATH`. But an
*empty* cache is expensive — measured on the lab machine, a cold run's GPU
stages cost 49.9 / 56.6 / 45.7 / 151.9 s against 8.7 / 10.5 / 16.0 / 30.7 s warm
— so sixteen empty caches would pay that sixteen times, perhaps 80 minutes.
One job therefore runs first, alone, and its warmed cache is copied to every
worker before the rest start.

**Losing finished work.** Results are collected back into the real `reports/`
**as each job finishes**, not at the end, so an interruption keeps everything
completed up to that point. That is the lesson of correction 4.7, which cost
12 hours.

Usage
-----
    python tools/run_matrix_parallel.py --sizes xs s --workers 16
    python tools/run_matrix_parallel.py --sizes xs --workers 8 --resume
    python tools/run_matrix_parallel.py --sizes xs s --workers 16 --dry-run

`--dry-run` prints the plan, the disk estimate and the projected wall-clock
without running or copying anything. Worth doing first on a new machine.

No `from __future__ import annotations`: this imports `overhang_report`, which
drives Taichi kernels (corrections 3.2 and 4.10).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import overhang_report as orep  # noqa: E402

#: Where worker copies live by default. Deliberately outside the repository, so
#: a stray worker tree can never be committed, and on the same volume so copying
#: is cheap.
DEFAULT_WORKER_ROOT = REPO_ROOT.parent / "5ax3d_workers"

#: Copied into each worker. Everything the pipeline reads, and nothing it does
#: not: no `.git` (hundreds of megabytes over 16 copies, and a worker never
#: commits), no archived toolpaths, no previous reports.
WORKER_CONTENT = ("src", "tools", "config", "scripts", "pyproject.toml")

#: Fallback list of `data/` inputs, used only when git cannot be asked.
#: `data_input_paths` is the real answer; see why there.
FALLBACK_DATA_INPUTS = {
    "mesh": ("*.stl",),
    "param": ("*.json",),
    "image": ("*.png",),
}


def data_input_paths():
    """Every tracked file under `data/` — the pipeline's inputs, by definition.

    **Not a hand-written list.** Everything a run generates is gitignored (each
    `data/*/` has its own `.gitignore`), so "tracked under `data/`" is exactly
    "input", and it stays correct as stages change.

    Enumerating them by hand failed twice in one afternoon:

    * copying whole folders dragged in Blender's `.obj` intermediates, so a
      worker could start with another run's geometry;
    * narrowing to `*.stl` and `*.json` then dropped `data/image/0.png`, and
      **every** run needs it — `tools/compute_tangents.py` hardcodes it as the
      default target-tangent field when no `--top_lines` is given. Stage 6 died
      in every worker, and because `atomize.py` ignores stage exit codes it
      produced twelve more failures behind it.

    `git ls-files` rather than `git ls-tree HEAD` here, deliberately, and
    despite correction 4.11: that correction is about "has this been committed",
    where the index lies. This asks "is this an input", and a newly added input
    that has not been committed yet is still an input a worker needs.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", "data"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
        if result.returncode == 0:
            paths = [
                Path(line.strip()) for line in result.stdout.splitlines()
                if line.strip() and not line.strip().endswith(".gitignore")
            ]
            if paths:
                return paths
    except (OSError, subprocess.SubprocessError):
        pass

    paths = []
    for name, patterns in FALLBACK_DATA_INPUTS.items():
        for pattern in patterns:
            paths += [
                path.relative_to(REPO_ROOT)
                for path in (REPO_ROOT / "data" / name).glob(pattern)
            ]
    return paths

#: Output subdirectories of `data/`, created empty in each worker. `image` is
#: absent because it holds committed inputs and the input copy creates it.
DATA_OUTPUTS = (
    "basis", "direction", "frame", "gcode", "log", "phasor",
    "point_normal", "sdf", "toolpath", "toolpath_planner", "triphasor",
)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def build_jobs(sizes, parts=orep.MATRIX_PARTS, slopes=orep.MATRIX_SLOPES,
               resume=False):
    """Every (part, slope) to run, in longest-first order.

    Longest first matters: with a queue, starting the slow jobs last leaves one
    worker grinding through `ramp90_s` while fifteen sit idle. Duration is
    estimated from the committed reports when they exist, so the ordering
    improves as the study accumulates data and degrades gracefully to the given
    order when it has none.

    ``resume`` drops combinations that already hold a current result, using the
    same metrics-version test as ``--status`` so a stale report counts as
    missing (correction 4.7).
    """
    known = {}
    for report in orep.load_reports(quiet=True):
        total = (report.get("runtime") or {}).get("total_s") or 0.0
        if total:
            known[(report["part"], report["max_slope_deg"])] = total

    current = set()
    if resume:
        current = {
            (r["part"], r["max_slope_deg"])
            for r in orep.load_reports(quiet=True)
            if r.get("metrics_version") == orep.METRICS_VERSION
        }

    jobs = []
    for size in sizes:
        for part in parts:
            for slope in slopes:
                key = (f"{part}_{size}", float(slope))
                if key in current:
                    continue
                jobs.append({
                    "part": key[0],
                    "slope": key[1],
                    "estimate_s": known.get(key, 0.0),
                })

    # Unknown durations sort first: an unmeasured job could be the long one, and
    # guessing it is short is the mistake that leaves a worker grinding alone.
    jobs.sort(key=lambda j: (-1e9 if not j["estimate_s"] else -j["estimate_s"]))
    return jobs


def projected_seconds(jobs, workers, ratio=1.0):
    """Ideal wall-clock: the greater of total work over workers, and the longest job.

    No job can be split, so one job longer than the average worker's share sets
    the floor. Ignores contention between workers, so it is a lower bound and is
    reported as one.
    """
    estimates = [j["estimate_s"] * ratio for j in jobs if j["estimate_s"]]
    if not estimates:
        return None
    return max(sum(estimates) / max(workers, 1), max(estimates))


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------
# Worker trees
# --------------------------------------------------------------------------


def tree_megabytes():
    """Measured size of one worker copy, for the disk line in the plan.

    Measured rather than guessed: the answer is about 20 MB, and an earlier
    guess of 600 MB was off by thirty times. The stage outputs a worker then
    writes are the real consumer, and this does not pretend to know them.
    """
    total = 0
    for name in WORKER_CONTENT:
        source = REPO_ROOT / name
        if source.is_file():
            total += source.stat().st_size
        elif source.is_dir():
            total += sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    for relative in data_input_paths():
        source = REPO_ROOT / relative
        if source.is_file():
            total += source.stat().st_size
    return total / 1e6


def create_worker(root, index, overwrite=False):
    """Make one worker's working tree and return its path.

    A copy rather than a git clone: no `.git` means a worker cannot be committed
    from by accident, and 16 copies of history would cost far more than 16 copies
    of the files.
    """
    worker = Path(root) / f"worker-{index:02d}"
    if worker.exists():
        if not overwrite:
            return worker
        shutil.rmtree(worker)
    worker.mkdir(parents=True)

    for name in WORKER_CONTENT:
        source = REPO_ROOT / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, worker / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, worker / name)

    for relative in data_input_paths():
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        target = worker / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for name in DATA_OUTPUTS:
        (worker / "data" / name).mkdir(parents=True, exist_ok=True)
    (worker / "reports" / "baseline_overhang").mkdir(parents=True, exist_ok=True)
    (worker / "reports" / "toolpaths").mkdir(parents=True, exist_ok=True)
    (worker / "ticache").mkdir(parents=True, exist_ok=True)

    return worker


def seed_cache(source_worker, workers):
    """Copy a warmed Taichi cache into every other worker.

    Returns how many were seeded. See the module docstring: a cold cache inflates
    the GPU stages five to sevenfold, and sixteen cold caches would pay it
    sixteen times.
    """
    source = Path(source_worker) / "ticache"
    if not source.is_dir() or not any(source.iterdir()):
        return 0

    seeded = 0
    for worker in workers:
        target = Path(worker) / "ticache"
        if Path(worker) == Path(source_worker):
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        seeded += 1
    return seeded


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_job(worker, job, log_dir, threads=None):
    """Run one (part, slope) inside a worker tree. Returns (ok, seconds, log_path).

    The subprocess is given explicit UTF-8 and `PYTHONIOENCODING`: without them
    the reader thread decodes with the locale codec, which is cp1252 on Windows,
    and one non-ASCII byte raises inside `subprocess` and destroys the output of
    a stage that genuinely failed (correction 3.10).
    """
    worker = Path(worker)
    param = f"data/param/{job['part']}.json"
    log_path = Path(log_dir) / f"{job['part']}_ms{job['slope']:g}.log"

    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        TI_OFFLINE_CACHE_FILE_PATH=str(worker / "ticache"),
    )
    # Each worker is its own process tree; capping Taichi's threads stops 16
    # workers each trying to use every core.
    if threads:
        env["TI_CPU_MAX_NUM_THREADS"] = str(threads)

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"# {job['part']} @ max_slope {job['slope']:g}\n")
        handle.write(f"# worker: {worker}\n")
        handle.write(f"# started: {datetime.now(timezone.utc).isoformat()}\n\n")
        handle.flush()
        result = subprocess.run(
            [sys.executable, "tools/overhang_report.py", param,
             "--max-slope", f"{job['slope']:g}"],
            cwd=str(worker), env=env,
            stdout=handle, stderr=subprocess.STDOUT,
        )
    return result.returncode == 0, time.monotonic() - started, log_path


def collect(worker, job):
    """Bring one finished job's artifacts back into the real `reports/`.

    Done per job rather than at the end: an interrupted matrix must keep
    everything already completed (correction 4.7 cost 12 hours by not).
    Returns the report dict, or None if the worker produced none.
    """
    worker = Path(worker)
    name = f"{job['part']}_ms{job['slope']:g}"

    source = worker / "reports" / "baseline_overhang" / f"{name}.json"
    if not source.is_file():
        return None
    orep.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, orep.REPORT_DIR / f"{name}.json")

    archive = worker / "reports" / "toolpaths" / f"{name}.npz"
    if archive.is_file():
        orep.TOOLPATH_ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, orep.TOOLPATH_ARCHIVE / f"{name}.npz")

    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_matrix(jobs, workers, log_dir, threads=None, on_event=None):
    """Work through `jobs` across `workers` worker trees.

    A queue, not a fixed partition: workers pull the next job as they free up, so
    a worker that draws short jobs does not sit idle while another grinds. Returns
    (completed, failed).

    `on_event(kind, payload)` is called for "start", "done" and "fail" so the
    caller owns all display and all writes to the real `reports/`. Collection
    happens on the calling thread for exactly that reason: `append_progress`
    appends to one file, and sixteen threads appending to the crash trail is the
    ragged-CSV problem again.
    """
    queue = list(jobs)
    lock = threading.Lock()
    completed, failed = [], []

    def take():
        with lock:
            return queue.pop(0) if queue else None

    def finish(job, ok, seconds, log_path, worker):
        with lock:
            if ok:
                report = collect(worker, job)
                if report is None:
                    ok = False
                    job["error"] = "the run exited 0 but wrote no report"
                else:
                    orep.append_progress(report)
                    completed.append(job)
            if not ok:
                job.setdefault("error", f"exit code non-zero; see {log_path}")
                job["log"] = str(log_path)
                failed.append(job)
            if on_event:
                on_event("done" if ok else "fail",
                         dict(job, seconds=seconds, worker=str(worker)))

    def work(worker):
        while True:
            job = take()
            if job is None:
                return
            if on_event:
                on_event("start", dict(job, worker=str(worker)))
            ok, seconds, log_path = run_job(worker, job, log_dir, threads)
            finish(job, ok, seconds, log_path, worker)

    threads_running = [
        threading.Thread(target=work, args=(worker,), daemon=True)
        for worker in workers
    ]
    for thread in threads_running:
        thread.start()
    for thread in threads_running:
        thread.join()

    return completed, failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the P0.8 baseline matrix across many cores at once. Each worker "
            "gets its own copy of the working tree, so no two runs can overwrite "
            "each other's intermediates."
        ),
    )
    parser.add_argument(
        "--sizes", nargs="+", default=["s"], choices=["xs", "s", "m", "l"],
        help="Benchmark part sizes to run (default: s).",
    )
    parser.add_argument(
        "--parts", nargs="+", default=list(orep.MATRIX_PARTS),
        help="Part names without the size suffix.",
    )
    parser.add_argument(
        "--slopes", nargs="+", type=float, default=list(orep.MATRIX_SLOPES),
        help="max_slope values in degrees.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Concurrent workers. Default: half the logical processors, which "
            "leaves each worker about two — roughly all the sequential planner "
            "can use — and leaves the machine usable."
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help="Cap each worker's Taichi CPU threads (default: unset, Taichi decides).",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_WORKER_ROOT,
        help=f"Where worker trees live (default: {DEFAULT_WORKER_ROOT}).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip combinations that already hold a current result.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan, disk estimate and projected time; change nothing.",
    )
    parser.add_argument(
        "--fresh-workers", action="store_true",
        help="Delete and rebuild existing worker trees.",
    )
    parser.add_argument(
        "--keep-workers", action="store_true",
        help="Leave worker trees in place afterwards (default: delete them).",
    )
    parser.add_argument(
        "--no-seed-cache", action="store_true",
        help=(
            "Do not run one job first to warm a Taichi cache. Faster to start "
            "and much slower overall; see the module docstring."
        ),
    )
    args = parser.parse_args(argv)

    jobs = build_jobs(args.sizes, args.parts, args.slopes, resume=args.resume)
    if not jobs:
        print("Nothing to run: every combination already has a current result.")
        return 0

    workers_wanted = args.workers or max(1, (os.cpu_count() or 2) // 2)
    workers_wanted = min(workers_wanted, len(jobs))

    ideal = projected_seconds(jobs, workers_wanted)
    timed = sum(1 for j in jobs if j["estimate_s"])
    serial = sum(j["estimate_s"] for j in jobs)

    print(f"Matrix: {len(jobs)} run(s) across {workers_wanted} worker(s)")
    print(f"Sizes: {', '.join(args.sizes)}   Slopes: "
          f"{', '.join(f'{s:g}' for s in args.slopes)} deg")
    if timed:
        print(f"Serial estimate (from {timed} previous run(s)): "
              f"{format_duration(serial)}")
        print(f"Parallel estimate, ignoring contention: {format_duration(ideal)} "
              "(a lower bound)")
    else:
        print("No previous runtimes, so no estimate. The first run will supply them.")
    print(f"Worker trees: {args.root}")
    print(f"Disk: {workers_wanted} x {tree_megabytes():.0f} MB of worker copies, "
          "plus each worker's stage outputs (`data/sdf`, `data/toolpath` and the "
          "rest), which are the bulk and are not estimated here")

    if args.dry_run:
        print("\n--dry-run: nothing was created or run. Planned order:")
        for index, job in enumerate(jobs, 1):
            estimate = (format_duration(job["estimate_s"])
                        if job["estimate_s"] else "unknown")
            print(f"  {index:3d}. {job['part']:16} @ {job['slope']:5g} deg  ~{estimate}")
        return 0

    log_dir = REPO_ROOT / "reports" / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPreparing {workers_wanted} worker tree(s)...")
    trees = [
        create_worker(args.root, index, overwrite=args.fresh_workers)
        for index in range(workers_wanted)
    ]

    started_at = time.monotonic()
    done_count = [0]
    total = len(jobs)

    def on_event(kind, payload):
        if kind == "start":
            return
        done_count[0] += 1
        elapsed = time.monotonic() - started_at
        mark = "ok  " if kind == "done" else "FAIL"
        print(f"[{done_count[0]:3d}/{total}] {mark} {payload['part']:16} "
              f"@ {payload['slope']:5g} deg  "
              f"{format_duration(payload['seconds'])}  "
              f"(elapsed {format_duration(elapsed)})", flush=True)
        if kind == "fail":
            print(f"         {payload.get('error', '')}", flush=True)

    # One job alone first, to fill a cache worth copying.
    if not args.no_seed_cache and len(trees) > 1:
        first = jobs[0]
        print(f"\nWarming the Taichi cache on one run: {first['part']} "
              f"@ {first['slope']:g} deg")
        completed, failed = run_matrix([first], trees[:1], log_dir, args.threads,
                                       on_event)
        if failed:
            print("\nThe warm-up run failed. Stopping rather than starting "
                  f"{workers_wanted} workers that would all fail the same way.")
            print(f"Log: {failed[0].get('log')}")
            return 1
        seeded = seed_cache(trees[0], trees)
        print(f"Copied the warmed cache to {seeded} worker(s)")
        jobs = jobs[1:]
        first_completed, first_failed = completed, failed
    else:
        first_completed, first_failed = [], []

    print(f"\nRunning {len(jobs)} run(s) across {len(trees)} worker(s)\n")
    completed, failed = run_matrix(jobs, trees, log_dir, args.threads, on_event)
    completed += first_completed
    failed += first_failed

    elapsed = time.monotonic() - started_at
    print(f"\nFinished in {format_duration(elapsed)}: "
          f"{len(completed)} completed, {len(failed)} failed")

    if failed:
        print("\nFailed runs (their logs are under reports/worker_logs/):")
        for job in failed:
            print(f"  {job['part']} @ {job['slope']:g} deg — "
                  f"{job.get('error', 'unknown')}")

    reports = orep.load_reports()
    if reports:
        orep.SUMMARY_PATH.write_text(orep.summarize(reports), encoding="utf-8")
        print(f"\nWrote {orep.SUMMARY_PATH} from {len(reports)} report(s)")

    if not args.keep_workers:
        print(f"Removing worker trees under {args.root}")
        shutil.rmtree(args.root, ignore_errors=True)
    else:
        print(f"Worker trees kept at {args.root}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
