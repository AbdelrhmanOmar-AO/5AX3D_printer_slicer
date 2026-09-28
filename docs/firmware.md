# Firmware: what the printer must do with the slicer's G-code

Gate E1 (build plan §3, task P1.5): which firmware the printer runs, how the three
bed screws appear to it, what the 3Z macros do, and how the bed is levelled.
Answered by the operator on **2026-09-28**. This file records the answers, what
the G-code asks of the printer, and what is still open.

**State:** the slicer side is done. The macro files themselves are **not written
yet**, by the operator's decision: their contents are not published, and they
depend on our control board's wiring. They follow when the authors' files or our
board are available (build plan P6.3).

---

## 1. The answers

| Question | Answer | Consequence for the slicer |
|---|---|---|
| Firmware | **RepRapFirmware** (RRF) | The `rrf` dialect, upstream's, is the one we use. `klipper` stays unimplemented and raises |
| Axis letters | **X, Y, Z, U, V**, as RRF allows | Unchanged: what Atomizer already writes. X and Y move the CoreXY head; Z, U and V are the three bed screws |
| What `enable3Z` / `disable3Z` do | Specified below (section 3) from the G-code; **file contents not yet available** | None: the G-code only calls them |
| Homing and bed levelling | **Not decided.** `G32` is kept | None until decided |

**Axis letters.** RRF has no fixed letters for axes beyond X, Y and Z: `M584`
creates them and names them, and the extra letters start U, V, W, A, B, C.
U and V are the conventional first two, and nearly every multi-Z RRF setup
uses them. RRF 2.x created U automatically when V was created; RRF 3.x
creates only the axes asked for.

Which screw each letter drives is fixed by the kinematics: `z0 → Z`,
`z1 → U`, `z2 → V` (`docs/conventions.md` section 4). Z drives the screw under
ball 0, U the one under ball 1, V the one under ball 2, with the balls
numbered as in the machine profile (`ball_2dpos_0/1/2`). **Which physical screw
that is on our machine** depends on the bed geometry and the board's wiring;
that is part of P6 (gates M1, E1).

**Tests pin the slicer side** (`tests/test_gcode_templates.py`, section "Gate
E1"): the axis letters, the order of `G32` and the macros, and that this file
names what the G-code calls. Changing any of them means changing the printer's
configuration too.

## 2. What the G-code does, in order

Header (`atom.gcode_templates`, byte-identical to upstream with the default
temperatures):

| Line | Meaning in RRF | Why it is there |
|---|---|---|
| `G21`, `G90` | millimetres, absolute coordinates | |
| `M190 S55` | heat the bed and wait | temperature from the part's JSON (`bed_temp`), default 55 |
| `G32` | run `/sys/bed.g`: home and level the bed | the three screws are still **one Z axis** here |
| `M104 S210`, `M109 S210` | heat the nozzle and wait | `nozzle_temp`, default 210 |
| `T0`, `M82` | select the tool, absolute extrusion | |
| `M98 P"/macros/enable3Z.g"` | run the enable macro | split the screws into Z, U, V |
| `M400` | wait until all moves have finished | nothing moves before the macro is done |
| `G1 Z76.3 U76.3 V76.3 F500` and the purge lines | first moves of U and V | all three equal: the bed is level |

Every move after that is `G1 X.. Y.. Z.. U.. V.. E.. F..`, with Z, U and V
generally **different**: that difference is the bed's tilt.

Footer: extruder retract, heaters and fan off, then
`M98 P"/macros/disable3Z.g"` and `M400`, and nothing after.

## 3. What the two macros must do

This is a **specification** taken from the G-code and the kinematics, not a
copy of the authors' files, which are not published (section 5).

**`/macros/enable3Z.g`** is called after `G32` has levelled the bed. When it
returns:

1. Z, U and V are **independent axes**, each driving one screw: Z the screw
   under ball 0, U under ball 1, V under ball 2.
2. The three axes **agree on height**: each one's current position equals the
   height Z had before the split, in millimetres, with the same zero and the
   same direction. The G-code assumes that equal Z, U and V mean a level bed at
   that height (`docs/conventions.md` section 4; the paper assumes "the bed is
   level when homed"). If U or V started from an unknown or different
   position, the very first purge move would tilt the bed. For example, `G92`
   can set a position without moving.
3. U and V have their **own motion limits**, set to what the screws can do:
   maximum speed, acceleration, jerk and travel (in RRF: `M203`, `M201`,
   `M566`, `M208`), plus motor current and microstepping as for Z. The
   G-code keeps every screw within `[0, max_z_axis]` of the machine profile
   (the validator's `AXIS_RANGE` rule).
4. It **does not move** the bed. The G-code moves next, from the positions
   in (2).

**`/macros/disable3Z.g`** is called at the very end, after the heaters are
off. When it returns:

1. The three screws are **one Z axis** again, so that the next `G32` and any
   homing work as for a normal print.
2. U and V are no longer in use (removed, or hidden from the user
   interface).

**Speed.** RepRapFirmware slows a move so that no axis exceeds its `M203`
maximum. The G-code asks for up to `travel_feedrate` (3000 mm/min on the
reference machine) on moves that also change Z, U and V; the paper's printer
limits its Z axes to **1900 mm/min**, so fast tilting moves will simply run
slower than asked. That is a speed question, not a correctness one: extrusion
stays tied to the move.

## 4. Still open

| # | Question | Needed for |
|---|---|---|
| O-1 | **The macro files themselves.** Source: the authors (section 5), or written for our board | printing (P6.3, P7) |
| O-2 | **Our board and wiring**: which driver each screw motor is on. Every `M584` line depends on it | the macros (P6.3) |
| O-3 | **Homing and bed levelling.** `G32` runs `/sys/bed.g`, which on a three-screw bed normally uses `M671` (screw positions) and probes the bed to level it. Our screw positions come from gate M1 | the header's `G32` line, and `bed.g` (P6) |
| O-4 | **What `disable3Z` does with a tilted bed.** A print can end with the bed tilted. Joining the screws as they stand leaves it tilted until the next `G32` levels it; the macro could instead level it first by moving U and V to Z's height. Either works if every print starts with `G32` | the macros (P6.3) |

## 5. Sources

* **Cocco, Garner, Belle, Zanni and Chermain, "Towards Accessible Non-Planar
  FFF Using Triple Z-Axis Kinematics", ACM Symposium on Computational
  Fabrication (SCF '25), 2025.** doi:10.1145/3745778.3766652. The printer
  Atomizer was tested on. What it says, read 2026-09-28:
  * a **RatRig V-Core 3.1** with **RepRapFirmware 3.6.0** (section 4);
  * "RepRap firmware can be configured to interpret 5-axis G-code without any
    modifications" (section 4.6): configuration only, no firmware changes;
  * "the bed is level when homed, which can be ensured by using the machine's
    bed-leveling feature prior to each print" (section 3);
  * the Z axes are limited to **1900 mm/min** (section 4.8);
  * it recommends at most **1 mm and 1 degree** between G-code points, a
    worst-case error of 0.05 mm and 0.01 degrees from the firmware's
    straight-line interpolation of machine coordinates (section 4.5);
  * the ball positions, ball height, build area and 30-degree tilt match our
    `reference` machine profile exactly. Two rail angles are given in the
    opposite direction (150.11 and -90 degrees against -29.89 and 90). The
    kinematics treat each rail as a line: `kinematics3z.inverse` built with the
    paper's angles gives the same screw heights as ours, to the last digit, on
    200 random poses tilted up to 25 degrees (checked 2026-09-28).
  * It does **not** give the macros, the `M584` lines or the axis letters.
    Its supplementary material is listed as the Python kinematics and a video.
* **Atomizer's README** says only that the printer "is described in another
  article" (the paper above).
* **RepRapFirmware documentation** on `M584` (creating and naming axes), macros
  (`M98`) and bed levelling with several independent Z motors (`G32`,
  `bed.g`, `M671`): docs.duet3d.com.

To close O-1 from the authors: ask Xavier Chermain or Giovanni Cocco, or open
an issue on `github.com/xavierchermain/atomizer`, for their `config.g`,
`enable3Z.g`, `disable3Z.g` and `bed.g`. Those will use their board's driver
numbers, so ours still come from O-2.
