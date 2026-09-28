# The first baseline: the operator's laptop, one run at a time

This folder keeps the 48 stock-Atomizer overhang reports **as measured on the
operator's laptop** (host `AbdoYasser`, AMD Ryzen 5 5600H, CUDA; stock backend
mix; reference profile), run serially by `scripts/run_baseline_matrix.ps1` on
2026-09-22/23, and `summary.md`, the table `--summarize` built from them.

**It is not the reference any more.** From pull request #12 (2026-09-28) the
committed baseline in `reports/baseline_overhang/` is the lab machine's
(`cad-p07-2065-9`), and every comparison against it, P2.5 included, runs there
(`docs/plan_corrections.md` 7a, P1-16). No tool reads this folder, so it can
never be pooled into that table by accident.

It is kept for two reasons:

1. **Runtime.** These are the project's only serial timings of the full matrix.
   The lab set mixes 16, 8 and 1 workers, and a run is about 1.4x slower under
   8-way load, so its runtimes cannot stand in for these. The finding that
   `order_atoms` cost grows as **points^1.5** (`docs/handoff.md` 7b) comes from
   this set, and needs it to stay traceable.
2. **Robustness.** Measured on a different machine, CPU and GPU, the lab set
   gives **0 verdict changes in 48**, the worst effective overhang moves by at
   most **0.008 degrees** and the tilt used by at most **0.21 degrees**; only the
   unsupported fraction near overhangs moves: by +0.5 to +1.3 percentage
   points on 7 cells (all upward), and by less than half a point, either way,
   on 25 more.
   This folder is the other half of that comparison.

Provenance: every report carries a `provenance` block with `source:
backfilled` (the values were filled in after the runs, from known facts, by
`tools/backfill_provenance.py`; its per-stage backends were then confirmed by a
real laptop run, P1-7) and `parallel_workers: 1`.
