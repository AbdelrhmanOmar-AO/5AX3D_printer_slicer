# Overhang baseline: stock Atomizer

Produced by `tools/overhang_report.py --summarize` (build plan P0.8).

Each cell is **worst effective overhang angle / unsupported deposition near overhangs / maximum tilt actually used**.

A part counts as printable when the worst effective overhang stays at or below 45° *and* unsupported deposition near the overhangs is under 1%. Those thresholds are placeholders pending gate D0.

`n/m` means not measured: no deposition was found near that surface.

Measured on: machine cad-p07-2065-9, backend stock mix (cuda + x64), Taichi 1.7.4, profile reference. All 48 runs are comparable.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | ✅ 45° / 0.1% / 0.7° | ✅ 45° / 0.1% / 0.8° | ❌ 51° / 2.9% / 6.3° |
| `ramp45_xs` | ✅ 45° / 0.2% / 0.6° | ✅ 45° / 0.0% / 0.6° | ❌ 51° / 2.7% / 5.9° |
| `ramp50_s` | ❌ 50° / 4.3% / 0.7° | ❌ 50° / 4.4% / 0.8° | ❌ 67° / 8.5% / 17.1° |
| `ramp50_xs` | ❌ 50° / 3.3% / 0.6° | ❌ 50° / 2.9% / 0.6° | ❌ 58° / 4.0% / 7.5° |
| `ramp60_s` | ❌ 60° / 5.2% / 0.7° | ❌ 65° / 4.9% / 4.8° | ❌ 90° / 26.6% / 29.7° |
| `ramp60_xs` | ❌ 60° / 5.4% / 0.6° | ❌ 65° / 7.9% / 5.1° | ❌ 89° / 21.2% / 29.1° |
| `ramp70_s` | ❌ 70° / 9.8% / 0.7° | ❌ 80° / 17.1% / 9.7° | ❌ 90° / 27.2% / 20.6° |
| `ramp70_xs` | ❌ 70° / 8.6% / 0.6° | ❌ 78° / 13.2% / 8.3° | ❌ 90° / 22.6% / 20.4° |
| `ramp80_s` | ❌ 83° / 17.7% / 3.0° | ❌ 90° / 26.1% / 10.2° | ❌ 90° / 24.8% / 14.6° |
| `ramp80_xs` | ❌ 84° / 16.8% / 3.6° | ❌ 90° / 21.4% / 10.2° | ❌ 90° / 21.3% / 14.1° |
| `ramp90_s` | ❌ 90° / 24.9% / 1.0° | ❌ 90° / 24.5% / 4.3° | ❌ 90° / 25.6% / 11.9° |
| `ramp90_xs` | ❌ 90° / 19.7% / 1.1° | ❌ 90° / 20.4% / 4.6° | ❌ 90° / 19.7% / 11.1° |
| `tshape_s` | ❌ 90° / 24.7% / 1.3° | ❌ 90° / 24.4% / 5.1° | ❌ 90° / 24.6% / 10.2° |
| `tshape_xs` | ❌ 90° / 20.9% / 1.5° | ❌ 90° / 21.2% / 4.5° | ❌ 90° / 21.3% / 11.1° |
| `twin_domes_s` | – no overhang / tilt 2.6° | – no overhang / tilt 8.7° | – no overhang / tilt 23.0° |
| `twin_domes_xs` | – no overhang / tilt 1.9° | – no overhang / tilt 6.2° | – no overhang / tilt 19.3° |

## Conclusion

Stock Atomizer printed overhangs up to **45°** from vertical within the thresholds. The shallowest ramp it failed was 45°, so the limit lies between those two angles. Across all runs it used between 4% and 99% of the tilt budget it was given, which says whether raising `max_slope` alone would achieve anything. Raising `max_slope` from 7° to 30° is the comparison to read across each row: the field constrains only the first layer and low-curvature top surfaces, so a larger budget need not produce more tilt where an overhang actually is. These are the numbers the overhang-aware field is measured against in P2.5, on the same parts with the same thresholds.

## Stock vs overhang-aware (build plan P2.5)

Each cell is **stock → overhang-aware** for the same part and max_slope: worst effective overhang angle / unsupported deposition near overhangs. Overhang-aware runs: `overhang_report.py --overhang-aware`, or `run_matrix_parallel.py --overhang-aware` for the matrix. `—` means that side has no run.

Stock and overhang-aware runs share one provenance group: machine cad-p07-2065-9, backend stock mix (cuda + x64), Taichi 1.7.4, profile reference. Comparable.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | ✅ 45° / 0.1% → ✅ 44° / 0.0% | ✅ 45° / 0.1% → ✅ 44° / 0.0% | ❌ 51° / 2.9% → ✅ 44° / 0.1% |
| `ramp45_xs` | ✅ 45° / 0.2% → ✅ 44° / 0.1% | ✅ 45° / 0.0% → ✅ 44° / 0.1% | ❌ 51° / 2.7% → ✅ 44° / 0.2% |
| `ramp50_s` | ❌ 50° / 4.3% → ❌ 45° / 1.6% | ❌ 50° / 4.4% → ❌ 45° / 2.0% | ❌ 67° / 8.5% → ❌ 45° / 1.8% |
| `ramp50_xs` | ❌ 50° / 3.3% → ❌ 45° / 2.1% | ❌ 50° / 2.9% → ❌ 45° / 1.6% | ❌ 58° / 4.0% → ❌ 45° / 1.8% |
| `ramp60_s` | ❌ 60° / 5.2% → ❌ 53° / 2.0% | ❌ 65° / 4.9% → ❌ 51° / 1.4% | ❌ 90° / 26.6% → ❌ 49° / 0.8% |
| `ramp60_xs` | ❌ 60° / 5.4% → ❌ 54° / 2.4% | ❌ 65° / 7.9% → ❌ 50° / 0.9% | ❌ 89° / 21.2% → ❌ 46° / 1.7% |
| `ramp70_s` | ❌ 70° / 9.8% → ❌ 64° / 3.5% | ❌ 80° / 17.1% → ❌ 56° / 4.4% | ❌ 90° / 27.2% → ❌ 47° / 0.7% |
| `ramp70_xs` | ❌ 70° / 8.6% → ❌ 65° / 5.1% | ❌ 78° / 13.2% → ❌ 59° / 1.7% | ❌ 90° / 22.6% → ❌ 50° / 1.3% |
| `ramp80_s` | ❌ 83° / 17.7% → ❌ 73° / 12.7% | ❌ 90° / 26.1% → ❌ 66° / 12.2% | ❌ 90° / 24.8% → ❌ 58° / 1.5% |
| `ramp80_xs` | ❌ 84° / 16.8% → ❌ 74° / 11.1% | ❌ 90° / 21.4% → ❌ 66° / 7.0% | ❌ 90° / 21.3% → ❌ 61° / 2.0% |
| `ramp90_s` | ❌ 90° / 24.9% → ❌ 90° / 23.5% | ❌ 90° / 24.5% → ❌ 89° / 24.0% | — |
| `ramp90_xs` | ❌ 90° / 19.7% → ❌ 90° / 18.4% | ❌ 90° / 20.4% → ❌ 89° / 18.3% | — |
| `tshape_s` | ❌ 90° / 24.7% → ❌ 90° / 23.9% | ❌ 90° / 24.4% → ❌ 89° / 23.9% | — |
| `tshape_xs` | ❌ 90° / 20.9% → ❌ 90° / 21.1% | ❌ 90° / 21.2% → ❌ 89° / 20.4% | — |
| `twin_domes_s` | – no overhang → – no overhang | – no overhang → – no overhang | – no overhang → – no overhang |
| `twin_domes_xs` | – no overhang → – no overhang | – no overhang → – no overhang | – no overhang → – no overhang |

Printable, over the 44 part and slope pair(s) with both runs: stock 4, overhang-aware 6.

### Top-surface quality

Each cell is **stock → overhang-aware**: the share of the top layer's deposition points printed within 2° of the top surface's normal (`overhang_metrics.top_surface_quality`; top surfaces are the faces within max_slope of level, upstream's ceilings). P2.5 asks for overhang-aware ≥ stock − 5 points; ⚠️ marks a pair that misses it. `n/m`: no top surface measured; `not scored`: the report predates the measure (`--reanalyse` adds it).

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp45_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp90_s` | not scored → not scored | not scored → not scored | — |
| `ramp90_xs` | not scored → not scored | not scored → not scored | — |
| `tshape_s` | not scored → not scored | not scored → not scored | — |
| `tshape_xs` | not scored → not scored | not scored → not scored | — |
| `twin_domes_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `twin_domes_xs` | not scored → not scored | not scored → not scored | not scored → not scored |

Pairs within 5 points of stock or better: 0 of 0. 44 pair(s) not scored yet.

### Tilt rate

Each cell is **stock → overhang-aware**: the largest turn of the tool within any 1 mm of continuous printing, in degrees per mm, and in brackets the share of printing moves where the 1 mm of printing from there turns faster than 3°/mm (the gate D3 placeholder; `overhang_metrics.tilt_rate`). Reported, not judged: P2.5 sets no limit on it.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp45_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp90_s` | not scored → not scored | not scored → not scored | — |
| `ramp90_xs` | not scored → not scored | not scored → not scored | — |
| `tshape_s` | not scored → not scored | not scored → not scored | — |
| `tshape_xs` | not scored → not scored | not scored → not scored | — |
| `twin_domes_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `twin_domes_xs` | not scored → not scored | not scored → not scored | not scored → not scored |

### Reach and platform

Each cell is **stock → overhang-aware**: the platform `add_platform` prints under the part so the tilted bed clears the gantry (`0 mm`: none), or ⚠️ the number of toolpath points the machine cannot reach at any lift (`atom.machine_reach`, the toolpath tesselated as the pipeline does). P2.5 asks for no new points out of reach. A run whose points are out of reach stops at `add_platform`, so it has no report and does not appear here; `other profile`: made on another machine profile than the one scoring it.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp45_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp90_s` | not scored → not scored | not scored → not scored | — |
| `ramp90_xs` | not scored → not scored | not scored → not scored | — |
| `tshape_s` | not scored → not scored | not scored → not scored | — |
| `tshape_xs` | not scored → not scored | not scored → not scored | — |
| `twin_domes_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `twin_domes_xs` | not scored → not scored | not scored → not scored | not scored → not scored |

Overhang-aware runs with points out of reach: 0.

### Layer thickness

Each cell is **stock → overhang-aware**: the share of deposition points whose layer is thinner than 0.5x / thicker than 1.5x the nominal (P3.2's placeholders), the thickness measured from the toolpath's geometry, not its `height` (`atom.layer_thickness`; plan_corrections P2-25). A thick one is mostly a bead with the bead under it missing: a one-layer gap. The reports also give these near the overhangs.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp45_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp50_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp60_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp70_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp80_xs` | not scored → not scored | not scored → not scored | not scored → not scored |
| `ramp90_s` | not scored → not scored | not scored → not scored | — |
| `ramp90_xs` | not scored → not scored | not scored → not scored | — |
| `tshape_s` | not scored → not scored | not scored → not scored | — |
| `tshape_xs` | not scored → not scored | not scored → not scored | — |
| `twin_domes_s` | not scored → not scored | not scored → not scored | not scored → not scored |
| `twin_domes_xs` | not scored → not scored | not scored → not scored | not scored → not scored |

## Runtime

How the pipeline's cost scaled with part size. `order_atoms` is the stage that dominates, and it runs on the CPU.

| Part | max_slope | Volume mm³ | Toolpath points | order_atoms min | Total min | Run |
|---|---|---|---|---|---|---|
| `ramp45_xs` | 7° | 6,196 | 34,724 | 5.8 | 14.5 | alone |
| `ramp45_xs` | 15° | 6,196 | 36,835 | 6.2 | 15.2 | alone |
| `ramp45_xs` | 30° | 6,196 | 35,706 | 6.4 | 11.8 | 8 workers |
| `ramp50_xs` | 7° | 6,284 | 35,756 | 6.3 | 9.4 | 16 workers |
| `ramp50_xs` | 15° | 6,284 | 35,846 | 6.3 | 11.6 | 16 workers |
| `ramp50_xs` | 30° | 6,284 | 36,508 | 6.3 | 8.9 | 16 workers |
| `ramp60_xs` | 7° | 6,428 | 36,308 | 6.4 | 9.1 | 16 workers |
| `ramp60_xs` | 15° | 6,428 | 37,826 | 6.5 | 12.0 | 16 workers |
| `ramp60_xs` | 30° | 6,428 | 38,882 | 6.7 | 12.6 | 16 workers |
| `ramp70_xs` | 7° | 6,544 | 38,228 | 6.8 | 9.9 | 16 workers |
| `ramp70_xs` | 15° | 6,544 | 37,598 | 6.8 | 12.7 | 16 workers |
| `ramp70_xs` | 30° | 6,544 | 40,234 | 6.8 | 12.8 | 16 workers |
| `ramp80_xs` | 7° | 6,647 | 38,941 | 6.9 | 12.6 | 8 workers |
| `ramp80_xs` | 15° | 6,647 | 39,427 | 7.0 | 12.9 | 16 workers |
| `ramp80_xs` | 30° | 6,647 | 38,920 | 7.0 | 13.1 | 16 workers |
| `ramp90_xs` | 7° | 6,743 | 39,393 | 7.1 | 12.8 | 8 workers |
| `ramp90_xs` | 15° | 6,743 | 38,876 | 7.2 | 13.4 | 16 workers |
| `ramp90_xs` | 30° | 6,743 | 39,694 | 7.4 | 13.1 | 8 workers |
| `tshape_xs` | 7° | 4,738 | 28,440 | 5.4 | 7.8 | 16 workers |
| `tshape_xs` | 15° | 4,738 | 28,188 | 5.5 | 8.0 | 16 workers |
| `tshape_xs` | 30° | 4,738 | 28,957 | 5.5 | 8.0 | 16 workers |
| `twin_domes_xs` | 7° | 3,820 | 22,618 | 4.9 | 14.0 | 8 workers |
| `twin_domes_xs` | 15° | 3,820 | 23,079 | 4.6 | 13.9 | 16 workers |
| `twin_domes_xs` | 30° | 3,820 | 22,914 | 4.8 | 13.9 | 8 workers |
| `ramp45_s` | 7° | 28,688 | 142,930 | 52.5 | 63.5 | 8 workers |
| `ramp45_s` | 15° | 28,688 | 141,944 | 51.7 | 62.7 | 8 workers |
| `ramp45_s` | 30° | 28,688 | 147,096 | 48.4 | 60.3 | 16 workers |
| `ramp50_s` | 7° | 29,095 | 145,265 | 52.8 | 63.4 | 8 workers |
| `ramp50_s` | 15° | 29,095 | 143,317 | 53.5 | 64.1 | 8 workers |
| `ramp50_s` | 30° | 29,095 | 161,553 | 55.1 | 67.1 | 16 workers |
| `ramp60_s` | 7° | 29,757 | 150,155 | 54.2 | 64.9 | 8 workers |
| `ramp60_s` | 15° | 29,757 | 148,555 | 54.0 | 66.1 | 16 workers |
| `ramp60_s` | 30° | 29,757 | 166,094 | 58.4 | 69.1 | 8 workers |
| `ramp70_s` | 7° | 30,297 | 149,295 | 58.8 | 70.8 | 16 workers |
| `ramp70_s` | 15° | 30,297 | 153,463 | 61.1 | 71.8 | 8 workers |
| `ramp70_s` | 30° | 30,297 | 161,865 | 59.3 | 71.3 | 16 workers |
| `ramp80_s` | 7° | 30,772 | 158,159 | 62.6 | 73.4 | 8 workers |
| `ramp80_s` | 15° | 30,772 | 155,801 | 60.6 | 64.5 | 8 workers |
| `ramp80_s` | 30° | 30,772 | 159,366 | 59.7 | 66.3 | 8 workers |
| `ramp90_s` | 7° | 31,219 | 156,198 | 62.6 | 66.5 | 8 workers |
| `ramp90_s` | 15° | 31,219 | 156,792 | 61.1 | 73.2 | 16 workers |
| `ramp90_s` | 30° | 31,219 | 156,438 | 60.5 | 67.1 | 8 workers |
| `tshape_s` | 7° | 21,938 | 114,348 | 36.1 | 39.5 | 8 workers |
| `tshape_s` | 15° | 21,938 | 118,796 | 38.7 | 50.3 | 16 workers |
| `tshape_s` | 30° | 21,938 | 115,615 | 39.0 | 42.4 | 8 workers |
| `twin_domes_s` | 7° | 17,693 | 97,219 | 38.0 | 48.2 | 8 workers |
| `twin_domes_s` | 15° | 17,693 | 96,216 | 38.1 | 48.1 | 8 workers |
| `twin_domes_s` | 30° | 17,693 | 99,290 | 35.8 | 45.9 | 8 workers |

**Not all of these runs had the machine to itself** (the Run column). A run sharing it with others is slower by contention, 1.47x at 8 workers on the lab machine, so compare times only between runs made the same way, and do not fit a scaling curve across them.
