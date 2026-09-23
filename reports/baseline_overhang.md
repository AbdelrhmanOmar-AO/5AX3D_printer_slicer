# Overhang baseline: stock Atomizer

Produced by `tools/overhang_report.py --summarize` (build plan P0.8).

Each cell is **worst effective overhang angle / unsupported deposition near overhangs / maximum tilt actually used**.

A part counts as printable when the worst effective overhang stays at or below 45° *and* unsupported deposition near the overhangs is under 1%. Those thresholds are placeholders pending gate D0.

`n/m` means not measured: no deposition was found near that surface.

| Part | max_slope 7° | max_slope 15° | max_slope 30° |
|---|---|---|---|
| `ramp45_s` | ✅ 45° / 0.2% / 0.7° | ✅ 45° / 0.2% / 0.8° | ❌ 51° / 3.1% / 6.3° |
| `ramp45_xs` | ✅ 45° / 0.2% / 0.6° | ✅ 45° / 0.0% / 0.6° | ❌ 51° / 2.7% / 5.9° |
| `ramp50_s` | ❌ 50° / 4.6% / 0.7° | ❌ 50° / 4.1% / 0.8° | ❌ 67° / 8.4% / 17.1° |
| `ramp50_xs` | ❌ 50° / 3.3% / 0.6° | ❌ 50° / 2.9% / 0.6° | ❌ 58° / 4.2% / 7.5° |
| `ramp60_s` | ❌ 60° / 4.7% / 0.7° | ❌ 65° / 4.9% / 4.8° | ❌ 90° / 26.8% / 29.7° |
| `ramp60_xs` | ❌ 60° / 5.4% / 0.6° | ❌ 65° / 7.8% / 5.1° | ❌ 89° / 20.6% / 29.1° |
| `ramp70_s` | ❌ 70° / 9.8% / 0.7° | ❌ 80° / 16.5% / 9.7° | ❌ 90° / 26.9% / 20.6° |
| `ramp70_xs` | ❌ 70° / 8.5% / 0.6° | ❌ 78° / 13.0% / 8.3° | ❌ 90° / 22.6% / 20.4° |
| `ramp80_s` | ❌ 83° / 17.8% / 3.0° | ❌ 90° / 25.8% / 10.2° | ❌ 90° / 24.9% / 14.6° |
| `ramp80_xs` | ❌ 84° / 16.9% / 3.6° | ❌ 90° / 21.0% / 10.2° | ❌ 90° / 20.8% / 14.1° |
| `ramp90_s` | ❌ 90° / 24.7% / 1.0° | ❌ 90° / 24.6% / 4.3° | ❌ 90° / 25.5% / 11.9° |
| `ramp90_xs` | ❌ 90° / 19.2% / 1.1° | ❌ 90° / 20.3% / 4.6° | ❌ 90° / 19.6% / 11.1° |
| `tshape_s` | ❌ 90° / 24.4% / 1.3° | ❌ 90° / 24.4% / 5.1° | ❌ 90° / 24.5% / 10.2° |
| `tshape_xs` | ❌ 90° / 19.7% / 1.5° | ❌ 90° / 20.1% / 4.5° | ❌ 90° / 20.0% / 11.1° |
| `twin_domes_s` | ❌ n/m / n/m / 2.6° | ❌ n/m / n/m / 8.7° | ❌ n/m / n/m / 22.7° |
| `twin_domes_xs` | ❌ n/m / n/m / 1.9° | ❌ n/m / n/m / 6.2° | ❌ n/m / n/m / 19.3° |

## Conclusion

Stock Atomizer printed overhangs up to **45°** from vertical within the thresholds. The shallowest ramp it failed was 45°, so the limit lies between those two angles. Across all runs it used between 4% and 99% of the tilt budget it was given, which says whether raising `max_slope` alone would achieve anything. Raising `max_slope` from 7° to 30° is the comparison to read across each row: the field constrains only the first layer and low-curvature top surfaces, so a larger budget need not produce more tilt where an overhang actually is. These are the numbers the overhang-aware field is measured against in P2.5, on the same parts with the same thresholds.

## Runtime

How the pipeline's cost scaled with part size. `order_atoms` is the stage that dominates, and it runs on the CPU.

| Part | max_slope | Volume mm³ | Toolpath points | order_atoms min | Total min |
|---|---|---|---|---|---|
| `ramp45_xs` | 7° | 6,196 | 34,559 | 3.8 | 5.7 |
| `ramp45_xs` | 15° | 6,196 | 37,041 | 4.6 | 8.4 |
| `ramp45_xs` | 30° | 6,196 | 35,074 | 4.6 | 8.3 |
| `ramp50_xs` | 7° | 6,284 | 35,756 | 5.0 | 6.9 |
| `ramp50_xs` | 15° | 6,284 | 35,777 | 4.5 | 6.4 |
| `ramp50_xs` | 30° | 6,284 | 36,937 | 4.5 | 6.3 |
| `ramp60_xs` | 7° | 6,428 | 36,308 | 4.8 | 6.7 |
| `ramp60_xs` | 15° | 6,428 | 37,826 | 4.8 | 6.6 |
| `ramp60_xs` | 30° | 6,428 | 39,019 | 5.0 | 6.8 |
| `ramp70_xs` | 7° | 6,544 | 38,337 | 5.2 | 7.1 |
| `ramp70_xs` | 15° | 6,544 | 37,598 | 5.0 | 6.8 |
| `ramp70_xs` | 30° | 6,544 | 39,600 | 5.0 | 6.9 |
| `ramp80_xs` | 7° | 6,647 | 39,068 | 5.5 | 7.5 |
| `ramp80_xs` | 15° | 6,647 | 39,373 | 5.0 | 6.9 |
| `ramp80_xs` | 30° | 6,647 | 39,311 | 5.2 | 7.1 |
| `ramp90_xs` | 7° | 6,743 | 38,803 | 5.4 | 7.4 |
| `ramp90_xs` | 15° | 6,743 | 38,876 | 5.3 | 7.2 |
| `ramp90_xs` | 30° | 6,743 | 39,694 | 5.5 | 7.5 |
| `tshape_xs` | 7° | 4,738 | 28,440 | 3.9 | 5.8 |
| `tshape_xs` | 15° | 4,738 | 27,952 | 4.0 | 5.8 |
| `tshape_xs` | 30° | 4,738 | 28,480 | 4.0 | 5.9 |
| `twin_domes_xs` | 7° | 3,820 | 23,018 | 3.4 | 9.6 |
| `twin_domes_xs` | 15° | 3,820 | 23,057 | 3.4 | 7.1 |
| `twin_domes_xs` | 30° | 3,820 | 23,370 | 3.9 | 7.8 |
| `ramp45_s` | 7° | 28,688 | 144,168 | 41.8 | 49.7 |
| `ramp45_s` | 15° | 28,688 | 145,883 | 38.9 | 43.8 |
| `ramp45_s` | 30° | 28,688 | 146,076 | 47.4 | 52.4 |
| `ramp50_s` | 7° | 29,095 | 147,848 | 39.7 | 42.8 |
| `ramp50_s` | 15° | 29,095 | 146,767 | 37.4 | 40.5 |
| `ramp50_s` | 30° | 29,095 | 161,911 | 53.4 | 56.5 |
| `ramp60_s` | 7° | 29,757 | 150,772 | 38.2 | 41.4 |
| `ramp60_s` | 15° | 29,757 | 148,977 | 41.7 | 44.8 |
| `ramp60_s` | 30° | 29,757 | 161,944 | 46.1 | 49.8 |
| `ramp70_s` | 7° | 30,297 | 151,020 | 42.4 | 45.6 |
| `ramp70_s` | 15° | 30,297 | 156,495 | 42.8 | 46.0 |
| `ramp70_s` | 30° | 30,297 | 161,905 | 53.3 | 57.2 |
| `ramp80_s` | 7° | 30,772 | 155,958 | 45.2 | 48.3 |
| `ramp80_s` | 15° | 30,772 | 158,121 | 61.7 | 65.0 |
| `ramp80_s` | 30° | 30,772 | 158,577 | 65.2 | 69.6 |
| `ramp90_s` | 7° | 31,219 | 153,905 | 43.4 | 46.6 |
| `ramp90_s` | 15° | 31,219 | 159,338 | 45.3 | 48.5 |
| `ramp90_s` | 30° | 31,219 | 156,283 | 84.1 | 87.5 |
| `tshape_s` | 7° | 21,938 | 113,725 | 27.9 | 30.7 |
| `tshape_s` | 15° | 21,938 | 116,822 | 28.8 | 31.6 |
| `tshape_s` | 30° | 21,938 | 117,831 | 27.0 | 29.8 |
| `twin_domes_s` | 7° | 17,693 | 96,261 | 30.9 | 38.2 |
| `twin_domes_s` | 15° | 17,693 | 95,936 | 37.8 | 41.1 |
| `twin_domes_s` | 30° | 17,693 | 98,708 | 26.1 | 30.7 |
