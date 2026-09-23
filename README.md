# Atomizer: Beyond Non-Planar Slicing for Fused Filament Fabrication

![Representative toolpath visualization](data/image/doc/representative_image.png)

This repository contains the implementation of **Atomizer**, a novel toolpath generation algorithm for fused filament fabrication (FFF) 3D printing. Atomizer introduces the concept of *atoms* as a replacement for traditional slices. This method enables collision-free toolpaths that tightly conform to complex geometries and offer new fabrication capabilities.

- [Project page](https://xavierchermain.github.io/publications/atomizer)
- [Article (preprint)](https://drive.google.com/file/d/1P4MiYd7Qz1rJdqrq3-CEuzC4vQ2PIMeO/view?usp=sharing)
- [Supplemental Video](https://youtu.be/d9SBcfkywqA)

*[Xavier Chermain](https://xavierchermain.github.io), [Giovanni Cocco](https://github.com/iota97), [Cédric Zanni](https://members.loria.fr/CZanni/), Eric Garner, [Pierre-Alexandre Hugron](https://www.linkedin.com/in/pierre-alexandre-hugron-b7b22552), and [Sylvain Lefebvre](https://www.antexel.com/sylefeb/research)*

[Université de Lorraine](https://www.univ-lorraine.fr/en/univ-lorraine/), [CNRS](https://www.cnrs.fr/en), [Inria](https://www.inria.fr/en), [Loria](https://www.loria.fr/en/)

[Computer Graphics Forum](https://doi.org/10.1111/cgf.70189) ([Proceedings of the Symposium on Geometry Processing](https://sgp2025.my.canva.site/program-page-sgp)), 2025

**SGP Honorable Mention**

Replicability Stamp: [http://www.replicabilitystamp.org#https-github-com-xavierchermain-atomizer](http://www.replicabilitystamp.org#https-github-com-xavierchermain-atomizer)

[![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-xavierchermain-atomizer)

## Key Features

* Toolpath generation using frames (i.e., atoms) instead of slices
* Control of the deposition direction and tool orientation
* Collision-free toolpath ordering
* Support for anisotropic appearance on curved surfaces

## Replicating One Result (for the Impatient)

This set of PowerShell commands reproduces the colored mesh of Fig. 17 (top left) from the article. You’ll need Windows 10 or newer and Blender to visualize it.

Open **PowerShell as Administrator**, then run:
```ps1
winget source update
.\install_git_blender_miniconda.ps1
```
Close **and reopen** PowerShell. Then run:
```ps1
git clone https://github.com/xavierchermain/atomizer
cd atomizer
conda create --name atomizer python=3.10
conda activate atomizer
pip install -e .
python tools/atomize.py data/param/tubes.json
python tools/toolpath_to_ply.py data/toolpath/tubes_smoothed.npz data/mesh/tubes.ply
blender --python see_toolpath_with_blender.py -- "data/mesh/tubes.ply"
```

**Notes**

* The script `install_git_blender_miniconda.ps1` sets up Git, Blender, and Miniconda if needed. You have to download the script [install_git_blender_miniconda.ps1](./install_git_blender_miniconda.ps1).
* `pip install -e .` installs the local library in editable mode inside the `atomizer` conda env
* If `blender` isn’t on your `PATH`, use the full path to `blender.exe`.
* If something goes wrong, please read the following.

## Installation

The code compiles, runs, and is maintained on Windows 10 or newer. It has also been tested on Linux and macOS, but we do not provide installation guidance for those platforms. The following installation instructions target Windows 10 or newer.

### Dependencies

To install the dependencies (Git, Blender, and Miniconda), we provide a script that automatically downloads and installs all of them:
```ps1
.\install_git_blender_miniconda.ps1
```
The script checks whether the softwares are already installed; if not, it installs them and adds them to the `PATH`. Alternatively, you can install them manually by following the instructions on the respective software websites.

- **Blender** must be installed and added to your executable path. It is used to remesh the model (see [`tools/process_for_atomizer.py`](tools/process_for_atomizer.py)) for the atomizer. We tested with Blender 4.4 and 4.5.
- **Git and Miniconda** are optional dependencies if you are already comfortable with code cloning, Python environments, and modules.


### Clone the Repository

```
git clone https://github.com/xavierchermain/atomizer
```

### Install Python and Module Dependencies

This implementation is written in Python 3.10. Supported versions are Python >= 3.7 and < 3.11, as required by the [Taichi](https://www.taichi-lang.org/) module dependency. The implementation also relies on the local library [`src/atom`](src/atom). Please refer to [`pyproject.toml`](pyproject.toml) for the complete list of Python module dependencies.

#### Conda

We recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to install a specific version of Python, the dependencies, and the local library [`src/atom`](src/atom):

```
conda create --name atomizer python=3.10
conda activate atomizer
pip install -e .
```

### Development setup (this fork)

Contributors to the 5-axis work should install the dev dependencies as well and
use the setup script, which creates the environment, installs everything, and
reports the Python / Blender / CUDA versions that later tasks depend on:

```powershell
.\scripts\setup_laptop.ps1
```

#### "Terms of Service have not been accepted" from conda

Recent Miniconda releases refuse to install from Anaconda's own channels
(`pkgs/main`, `pkgs/r`, `pkgs/msys2`) until their Terms of Service are accepted,
failing with `CondaToSNonInteractiveError`. `setup_laptop.ps1` avoids this by
creating the environment from **conda-forge**, which has no such prompt and no
licence restriction for larger organisations. It provides the same Python 3.10,
and every other dependency is installed by pip afterwards, so the slicer is
unaffected.

To use Anaconda's channels instead, accept their terms first:

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
.\scripts\setup_laptop.ps1 -Channel defaults
```

#### "conda"/"blender" is not recognized

Neither the Miniconda nor the Blender installer puts itself on `PATH`, so both
commands fail even after a successful install (and after a reboot). Fix both at
once:

```powershell
.\scripts\fix_tool_paths.ps1
```

It locates each tool, appends it to your **user** `PATH` (no Administrator
needed), runs `conda init powershell` so `conda activate` works, and prints
what is still missing. Close and reopen PowerShell afterwards — `PATH` changes
only reach newly-started shells.

If several Blender versions are installed it prefers a version the pipeline has
been verified against, because
[`tools/process_for_atomizer.py`](tools/process_for_atomizer.py) drives the
`bpy` API directly and those calls change across major Blender releases.

| Version | Status |
|---|---|
| 4.4, 4.5 | Tested upstream |
| 5.2.1 LTS | Verified here on the calibration cube (STL import, voxel remesh, smooth modifier, OBJ export) |

Blender is invoked exactly once in the pipeline, by
[`tools/atomize.py`](tools/atomize.py), so checking that one stage covers every
use of it.

To point at a specific install:

```powershell
.\scripts\fix_tool_paths.ps1 -BlenderPath "C:\Program Files\Blender Foundation\Blender 4.5"
```

You can check the Blender stage on its own, without running the whole
pipeline, with:

```powershell
blender -b -P tools/process_for_atomizer.py -- data/mesh/calibration_cube.stl data/mesh/calibration_cube.obj 0.084375
```

It should finish without a Python traceback and write
`data/mesh/calibration_cube.obj`.

#### "running scripts is disabled on this system"

Windows blocks PowerShell scripts by default, so the first script you run fails
with `UnauthorizedAccess` / `PSSecurityException`. Allow scripts for your own
user account, once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`RemoteSigned` permits scripts stored on your own disk (anything from
`git clone`) while still requiring a signature on files downloaded from the
internet. It is per-user and does not need Administrator.

For a script downloaded with a browser or `curl` — such as
`install_git_blender_miniconda.ps1` — either unblock it with
`Unblock-File .\install_git_blender_miniconda.ps1`, or allow scripts for just
the current window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

`-Scope Process` lasts only until you close that PowerShell window.

### Choosing the Taichi backend

Each pipeline stage has its own default backend (see
[`src/atom/ti_env.py`](src/atom/ti_env.py)). `ATOM_TI_ARCH` overrides all of
them at once:

```powershell
$env:ATOM_TI_ARCH = "cuda"      # force every stage onto CUDA
$env:ATOM_TI_ARCH = "cpu"       # force every stage onto the CPU
Remove-Item Env:\ATOM_TI_ARCH   # back to each stage's own default
```

Note that Taichi does **not** fail when a requested GPU backend is unavailable:
it logs a warning and falls back to the CPU, running far slower. Check what it
actually started on rather than assuming.

### Project documents

| Document | What it holds |
|---|---|
| [`docs/plan_corrections.md`](docs/plan_corrections.md) | Everything in the build plan that is wrong or deliberately departed from. **Read before trusting the plan on a specific fact.** |
| [`docs/handoff.md`](docs/handoff.md) | State of the work, environment facts and conventions, written to be read with no prior context. |
| [`tests/golden/baseline.md`](tests/golden/baseline.md) | The frozen "before" record: baseline commit, environment, stage timings, determinism result. |

### Choosing a machine

Machine constants (bed geometry, travel limits, tilt limit, feed rates) live in
JSON profiles under [`config/machines/`](config/machines), not in the code. The
`ATOM_MACHINE` environment variable picks one:

```powershell
$env:ATOM_MACHINE = "reference"   # the default
```

| Profile | Status | What it is |
|---|---|---|
| `reference` | verified | The upstream Atomizer machine. Values copied exactly from the original constants block; the golden test runs against it, and it stays in the test matrix permanently. |
| `ours` | **PLACEHOLDER** | Our printer. Every number is copied from `reference` and none has been measured. Loading it warns on stderr and raises `PlaceholderProfileWarning`. |

Fill in `ours.json` only once the mechanical team supplies real measurements
(gates M1 and M2), then set its `status` to `verified`.

#### Noise during a pipeline run

Three things look alarming and are not:

* `conda.exe : ... NativeCommandError` — PowerShell reports anything written to
  stderr as an error record. Progress bars go to stderr.
* `ΓûêΓûÄ` in place of a progress bar — the console code page is not UTF-8.
  `scripts/run_baseline_matrix.ps1` sets it; other scripts do not.
* `Lock C:/taichi_cache/ticache/ticache.lock failed` — Taichi could not lock its
  compiled-kernel cache, so it recompiles instead of reusing. Results are
  unaffected, but every stage pays the compilation cost again. Clear it with
  `ti cache clean -p C:/taichi_cache/ticache`, or delete the folder; an
  interrupted run can leave the lock behind.

#### "No module named ..." after a `git pull`

Dependencies are added as the work progresses, and a `git pull` brings the new
`pyproject.toml` without installing anything. If a test suddenly fails to import
a module, refresh the environment:

```powershell
pip install -e ".[dev]"
```

Re-running `.\scripts\setup_laptop.ps1` does the same thing and re-checks
Blender and CUDA as well; it reuses the existing conda environment rather than
rebuilding it.

Note that a missing dependency shows up as a *collection error* that stops the
whole run (`Interrupted: 1 error during collection`), not as a single failing
test, so one absent package hides the state of everything else.

### Running the tests

```powershell
pytest -m unit                              # fast, CPU only, no GPU needed
.\scripts\run_pipeline_tests.ps1           # real pipeline stages (GPU + Blender)
.\scripts\run_pipeline_tests.ps1 -Benchmark  # full benchmark parts (slow)
```

The `pipeline` and `benchmark` tiers are skipped unless explicitly requested,
so `pytest` on a machine without a GPU still gives a meaningful result. See
[`tests/conftest.py`](tests/conftest.py) for the tiers and fixtures.

## Usage

### Atomize

```
python tools/atomize.py data/param/<json_filename>.json
```


The parameters in the JSON file are:

* `solid_name`: the filename of the STL representing the 3D solid, without the extension. The STL file must be located in the `data/mesh/` folder and have the `.stl` extension.
* `deposition_width`: the deposition width in millimeters. The layer height is half the deposition width in our implementation.
* `max_slope`: the maximum tilting angle of the tool, in degrees.
* `top_lines` and `bottom_lines` (optional): paths to single-channel, 8-bit-per-pixel PNG files representing the target tangent directions for the top and bottom surfaces, respectively. The mapping is $\[0, 255] \leftarrow \[−\pi/2, \pi/2]\$. The orientation defines a 2D line in the xy-plane. The 2D line field is defined on the upper face of the solid’s bounding box and is planarly projected onto the top and bottom surfaces along the z-axis.
* `ortho_to_wall` (optional): if true, forces the tool orientation to be parallel to the boundary. By default, this is false, as enabling this feature causes many tool orientation changes that are detrimental to surface quality.
* `infill` (optional): if true, a gyroid pattern infills the solid.

The inputs and outputs are:

* **STL input:** `data/mesh/<json_filename>.stl`
* **Toolpath output:** `data/toolpath/<json_filename>.npz`
* **Log file:** `data/log/<json_filename>.log`

Refer to the log file to see all the individual computation and visualization commands.

Add the `--warmup` option to exclude the compilation time from the computation time reported in the log file. **Caution:** this causes each step of the pipeline to run twice, as Taichi uses just-in-time compilation.

#### Example

```
python tools/atomize.py data/param/tubes.json
```

### Preview like a slicer (this fork)

`tools/visualize_5ax.py` opens a 3D window in the style of a slicer preview.
It shows the toolpath as lines and has these controls:

**Install the window once** into the conda environment (it is a desktop app
built with Qt):

```powershell
conda activate atomizer
conda install -c conda-forge pyside6 pyvistaqt
```

Without it the tool still works, in a simpler "classic" window with the
controls drawn over the 3D view (`--classic` picks that one on purpose).

The window has a side panel on the left, the 3D view, and a timeline along
the bottom:

- a timeline that scrubs through the print in the printer's order, with
  jump-to-start, step back, play/pause, step forward and jump-to-end buttons;
- a Z-height clip for looking inside the part;
- a dropdown that colours the lines by tilt, tilt direction, bead width or
  height, feed rate, unsupported points (the P0.8 metric) or a shell/infill
  guess;
- the part's STL drawn over the lines, and a cone showing the nozzle's tilt at
  the current point;
- a playback speed slider (labelled with the time the whole print takes);
- camera presets (Iso, Top, Front, Side) and "Save image";
- a "Current point" card: position, tilt, bead size, screw values;
- a **machine view** that shows the printer the way it moves: the nozzle stays
  vertical and the bed tilts beneath it. The bed's pose comes from the three
  screw values (Z, U, V), and the gantry level is drawn too. The bed turns red
  if a corner would hit the gantry or a point is out of reach.

It reads a toolpath `.npz` or the final G-code:

```powershell
python tools/visualize_5ax.py data/toolpath/ramp60_xs_smoothed.npz
python tools/visualize_5ax.py reports/toolpaths/ramp60_s_ms30.npz --mode azimuth
python tools/visualize_5ax.py data/gcode/ramp60_xs.gcode
```

**Mouse:** left-drag rotates, scroll zooms, shift+drag pans.

**Keys:** Left/Right step one point, `,` and `.` step 1 %, Home/End jump to
the start/end, Space plays and pauses. `--machine-view` starts in the machine
view. Hover over any control for a tooltip.

**Screenshot:** `--screenshot out.png` saves a picture without opening a
window. Run `--help` for every option, and see the tool's docstring for which
file to open when.

### Check a toolpath for collisions (this fork, build plan P4)

`tools/check_motion_safety.py` checks what the printer would hit:

- **nozzle** (P4.3): at every point, is material printed earlier inside the
  nozzle? The nozzle is a 40-degree cone up to the gantry.
- **swept** (P4.2): every machine state *between* the points too, for moves
  that turn the tool and for travel: the part printed so far and the bed
  corners against the gantry, and the nozzle against printed material.

```powershell
python tools/check_motion_safety.py data/toolpath/ramp60_xs_platform.npz
python tools/check_motion_safety.py "reports/toolpaths/*_xs_*.npz" --json reports/motion_safety/xs.json
```

It prints one line per file and exits 1 if anything collides. On a toolpath
from before `add_platform` (`_smoothed`, or the P0.8 archive), bed-corner hits
are mostly the lift the platform stage adds; check the `_platform` toolpath.
Axis ranges and the tilt limit between points are not checked yet (they come
from the G-code validator, P1.4).

In the viewer, the colour mode **Collisions (P4)** runs the same checks and
marks each flagged point with a red dot; the current-point card says what was
hit and how deep, and in the machine view the bed turns red there.

### Visualize

After using the atomizer, you can visualize the generated `<toolpath>` with:

```
python tools/visualize_toolpath.py data/toolpath/<toolpath>.npz
```
Use the `WASD` keys and the right mouse button to move the camera. Press `H` to hide the travel paths.

Check the log file generated by the atomizer tool to see all individual visualization commands.

#### Example

```
python tools/visualize_toolpath.py data/toolpath/tubes_smoothed.npz
```

#### Export Toolpath Mesh

![Toolpath mesh](data/image/doc/toolpath_mesh.png)

```
python tools/toolpath_to_ply.py data/toolpath/<toolpath>.npz data/mesh/<toolpath>.ply
```

The generated mesh `<toolpath>.ply` can be visualized in Blender.
```
blender --python see_toolpath_with_blender.py -- "data/mesh/<toolpath>.ply"
```

The script displays the toolpath relative length using the Turbo colormap. 

To do it manually, you need to add a material to the mesh. Then, in the **Material** tab, do the following:

1. Select **Base Color**.
2. Choose **Color Attribute**.

To visualize the vertex color attribute:

1. Select **Shading** (shortcut **z**).
2. Choose **Material Preview** (shortcut **2**).

##### Example

```
python tools/toolpath_to_ply.py data/toolpath/tubes_smoothed.npz data/mesh/tubes.ply
blender --python see_toolpath_with_blender.py -- "data/mesh/tubes.ply"
```

### Commands Manuals

Tool manuals are accessible by typing `--help`, e.g.,
```
python tools/atomize.py --help
```

## Repository Structure

```
atomizer/
├── tools/                             # The main tools
│    │── atomize.py                    # Main pipeline entry point
│    │── bpn_to_sdf.py                 # SDF generation and voxelization
│    │── compute_tool_orientations.py  # Tool orientation field optimization
│    │── compute_tangents.py           # Deposition tangent field optimization
│    │── align_atoms.py                # Tricosine field optimization
│    │── extract_explicit_atoms.py     # Atom extraction
│    │── order_atoms.py                # Atom ordering
│    │── visualize_bpn_sdf.py          # Visualize SDF
│    │── visualize_bpn_df.py           # Visualize tool orientation field
│    │── visualize_bases.py            # Visualize deposition tangent field
│    │── visualize_implicit_atoms.py   # Visualize tricosine field
│    │── visualize_explicit_atoms.py   # Visualize explicit atoms
│    │── visualize_toolpath.py         # Visualize generated toolpath
│    └── ...                           # Other tools
│── data/                              # Input and output data
│    │── mesh/                         # Input STLs
│    │── param/                        # Input parameter JSON files
│    │── toolpath/                     # Output toolpaths
│    └── ...                           # Other data
├── src/atom/                          # Local library with core functionalities
├── experiment/                        # For experimenting with core functionalities
├── generate_results.ps1               # Script to generate toolpaths
└── README.md
```

## 3D Printer

Atomizer was tested on a custom 5-axis printer with independently controlled Z-axis screws. The customization is intended for experts and is described in another article : [https://xavierchermain.github.io/publications/threez](https://xavierchermain.github.io/publications/threez). For now, the code generates the sequence of positions, each associated with a tool orientation and a travel type (deposition or no deposition). The generated toolpaths are available in [`data/toolpath`](data/toolpath). To print, you must use the inverse kinematics model of your machine. The G-code generated for our machine is located in [`data/gcode`](data/gcode). Note that files ending with `_craftware.gcode` can be opened with Craftware Legacy to inspect the toolpath (the travel moves represent the tool orientations).

## Citation

If you use this code in your research, please cite:

```
@article{Chermain2025Atomizer,
author = {Chermain, Xavier and Cocco, Giovanni and Zanni, Cédric and Garner, Eric and Hugron, Pierre-Alexandre and Lefebvre, Sylvain},
title = {{Atomizer: Beyond Non-Planar Slicing for Fused Filament Fabrication}},
journal = {Computer Graphics Forum (Proceedings of the Symposium on Geometry Processing)},
year = {2025},
doi = {10.1111/cgf.70189},
volume = {44},
number = {5},
}
```

## License

The source code is under the BSD 3-Clause "New" or "Rivised" license. See
[LICENSE](LICENSE) for more details.
