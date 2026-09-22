# Overhang baseline: stock Atomizer

Produced by `tools/overhang_report.py --summarize` (build plan P0.8).

Each cell is **worst effective overhang angle / unsupported deposition near overhangs / maximum tilt actually used**.

A part counts as printable when the worst effective overhang stays at or below 45° *and* unsupported deposition near the overhangs is under 1%. Those thresholds are placeholders pending gate D0.

`n/m` means not measured: no deposition was found near that surface.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | ❌ 45° / 27.3% / 0.7° | ❌ 45° / 27.3% / 0.8° | ❌ 50° / 32.4% / 6.3° |
| `ramp45_xs` | ❌ 45° / 28.7% / 0.6° | ❌ 45° / 28.7% / 0.6° | ❌ 50° / 35.2% / 5.9° |
| `ramp50_s` | ❌ 50° / 29.2% / 0.7° | ❌ 50° / 26.3% / 0.8° | ❌ 65° / 35.7% / 17.1° |
| `ramp50_xs` | ❌ 50° / 32.3% / 0.6° | ❌ 50° / 31.6% / 0.6° | ❌ 57° / 38.6% / 7.5° |
| `ramp60_s` | ❌ 60° / 27.5% / 0.7° | ❌ 64° / 28.7% / 4.8° | ❌ 90° / 43.4% / 29.7° |
| `ramp60_xs` | ❌ 60° / 35.9% / 0.6° | ❌ 65° / 41.7% / 5.1° | ❌ 89° / 42.9% / 29.1° |
| `ramp70_s` | ❌ 70° / 45.7% / 0.7° | ❌ 79° / 41.2% / 9.7° | ❌ 90° / 46.2% / 20.6° |
| `ramp70_xs` | ❌ 70° / 42.1% / 0.6° | ❌ 78° / 41.9% / 8.3° | ❌ 90° / 41.3% / 20.4° |
| `ramp80_s` | ❌ 83° / 41.4% / 3.0° | ❌ 90° / 42.9% / 10.2° | ❌ 90° / 44.9% / 14.6° |
| `ramp80_xs` | ❌ 84° / 43.6% / 3.6° | ❌ 90° / 44.1% / 10.2° | ❌ 90° / 44.8% / 14.1° |
| `ramp90_s` | ❌ 90° / 45.3% / 1.0° | ❌ 90° / 44.1% / 4.3° | ❌ 90° / 45.4% / 11.9° |
| `ramp90_xs` | ❌ 90° / 44.7% / 1.1° | ❌ 90° / 45.2% / 4.6° | ❌ 90° / 45.7% / 11.1° |
| `tshape_s` | ❌ 90° / 41.9% / 1.3° | ❌ 90° / 41.3% / 5.1° | ❌ 90° / 40.8% / 10.2° |
| `tshape_xs` | ❌ 90° / 41.9% / 1.5° | ❌ 90° / 42.2% / 4.5° | ❌ 90° / 42.3% / 11.1° |
| `twin_domes_s` | ❌ n/m / n/m / 2.6° | ❌ n/m / n/m / 8.7° | ❌ n/m / n/m / 22.7° |
| `twin_domes_xs` | ❌ n/m / n/m / 1.9° | ❌ n/m / n/m / 6.2° | ❌ n/m / n/m / 19.3° |

## Conclusion

Stock Atomizer did not print any ramp in the family within the thresholds, so its unsupported overhang limit is below the shallowest angle measured (45°). Across all runs it used between 4% and 99% of the tilt budget it was given, which says whether raising `max_slope` alone would achieve anything. Raising `max_slope` from 7° to 30° is the comparison to read across each row: the field constrains only the first layer and low-curvature top surfaces, so a larger budget need not produce more tilt where an overhang actually is. These are the numbers the overhang-aware field is measured against in P2.5, on the same parts with the same thresholds.

## Runtime

How the pipeline's cost scaled with part size. `order_atoms` is the stage that dominates, and it runs on the CPU.

| Part | max_slope | Volume mm³ | Toolpath points | order_atoms min | Total min |
|---|---|---|---|---|---|
| `ramp45_xs` | 7° | 6,196 | 34,559 | 4.5 | 11.0 |
| `ramp45_xs` | 15° | 6,196 | 37,041 | 5.0 | 8.9 |
| `ramp45_xs` | 30° | 6,196 | 35,074 | 4.6 | 8.4 |
| `ramp50_xs` | 7° | 6,284 | 35,756 | 4.6 | 6.5 |
| `ramp50_xs` | 15° | 6,284 | 35,777 | 4.9 | 7.0 |
| `ramp50_xs` | 30° | 6,284 | 36,937 | 4.5 | 6.3 |
| `ramp60_xs` | 7° | 6,428 | 36,308 | 4.7 | 6.5 |
| `ramp60_xs` | 15° | 6,428 | 37,826 | 5.0 | 7.0 |
| `ramp60_xs` | 30° | 6,428 | 39,019 | 5.0 | 6.9 |
| `ramp70_xs` | 7° | 6,544 | 38,337 | 5.0 | 6.9 |
| `ramp70_xs` | 15° | 6,544 | 37,598 | 5.0 | 6.8 |
| `ramp70_xs` | 30° | 6,544 | 39,600 | 5.0 | 6.8 |
| `ramp80_xs` | 7° | 6,647 | 39,068 | 5.3 | 7.3 |
| `ramp80_xs` | 15° | 6,647 | 39,373 | 5.0 | 6.9 |
| `ramp80_xs` | 30° | 6,647 | 39,311 | 5.2 | 7.1 |
| `ramp90_xs` | 7° | 6,743 | 38,803 | 5.1 | 7.0 |
| `ramp90_xs` | 15° | 6,743 | 38,876 | 5.3 | 7.2 |
| `ramp90_xs` | 30° | 6,743 | 39,694 | 5.3 | 7.2 |
| `tshape_xs` | 7° | 4,738 | 28,440 | 3.9 | 5.7 |
| `tshape_xs` | 15° | 4,738 | 27,952 | 4.0 | 5.8 |
| `tshape_xs` | 30° | 4,738 | 28,480 | 3.8 | 5.6 |
| `twin_domes_xs` | 7° | 3,820 | 23,018 | 3.6 | 10.3 |
| `twin_domes_xs` | 15° | 3,820 | 23,057 | 3.4 | 7.1 |
| `twin_domes_xs` | 30° | 3,820 | 23,370 | 3.5 | 7.3 |
| `ramp45_s` | 7° | 28,688 | 144,168 | 38.4 | 46.2 |
| `ramp45_s` | 15° | 28,688 | 145,883 | 38.3 | 43.4 |
| `ramp45_s` | 30° | 28,688 | 146,076 | 39.9 | 44.8 |
| `ramp50_s` | 7° | 29,095 | 147,848 | 39.0 | 42.1 |
| `ramp50_s` | 15° | 29,095 | 146,767 | 36.9 | 40.0 |
| `ramp50_s` | 30° | 29,095 | 161,911 | 40.0 | 43.1 |
| `ramp60_s` | 7° | 29,757 | 150,772 | 37.6 | 40.7 |
| `ramp60_s` | 15° | 29,757 | 148,977 | 41.7 | 44.8 |
| `ramp60_s` | 30° | 29,757 | 161,944 | 41.7 | 44.9 |
| `ramp70_s` | 7° | 30,297 | 151,020 | 41.9 | 45.0 |
| `ramp70_s` | 15° | 30,297 | 156,495 | 42.7 | 45.8 |
| `ramp70_s` | 30° | 30,297 | 161,905 | 48.0 | 51.2 |
| `ramp80_s` | 7° | 30,772 | 155,958 | 44.6 | 47.8 |
| `ramp80_s` | 15° | 30,772 | 158,121 | 45.2 | 48.3 |
| `ramp80_s` | 30° | 30,772 | 158,577 | 44.5 | 47.7 |
| `ramp90_s` | 7° | 31,219 | 153,905 | 42.7 | 45.8 |
| `ramp90_s` | 15° | 31,219 | 159,338 | 44.7 | 47.9 |
| `ramp90_s` | 30° | 31,219 | 156,283 | 46.1 | 49.3 |
| `tshape_s` | 7° | 21,938 | 113,725 | 27.5 | 30.2 |
| `tshape_s` | 15° | 21,938 | 116,822 | 29.5 | 32.3 |
| `tshape_s` | 30° | 21,938 | 117,831 | 27.6 | 30.4 |
| `twin_domes_s` | 7° | 17,693 | 96,261 | 30.5 | 37.9 |
| `twin_domes_s` | 15° | 17,693 | 95,936 | 33.3 | 37.9 |
| `twin_domes_s` | 30° | 17,693 | 98,708 | 26.1 | 30.7 |
