import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import taichi as ti

from atom.ti_env import init_taichi

import atom.fff3
from atom import machine_profile
from atom.gcode_templates import temperature_arguments
from atom.infill_options import infill_arguments
from atom.overhang_field import overhang_arguments

init_taichi("cpu", offline_cache_cleaning_policy="never")


class Parameters:
    def __init__(self):
        self.deposition_width: float = None
        self.solid_name: str = None
        self.max_slope: float = None
        self.degree_angle_max_diff: float = None
        self.smoothing_iter_count: int = None
        self.ortho_to_wall: bool = None
        self.infill: bool = None

        self.log_path: str = None
        self.stl_path: str = None
        self.obj_path: str = None
        self.bpn_path: str = None
        self.direction_path: str = None
        self.basis_path: str = None
        self.phasor_path: str = None
        self.triphasor_path: str = None
        self.frame_path: str = None
        self.toolpath_path: str = None
        self.smoothed_toolpath_path: str = None
        self.smoothed_tesselated_toolpath_path: str = None
        self.gcode_path: str = None

    def load(self, path: str):
        with open(path) as file:
            data = file.read()
            param_dict = json.loads(data)

        self.deposition_width = param_dict["deposition_width"]
        self.solid_name = param_dict["solid_name"]
        # This fork (operator, 2026-09-30): without "max_slope", the tilt
        # budget is the active machine profile's limit (ATOM_MACHINE). Every
        # upstream parameter file gives it, and then nothing changes.
        self.max_slope = param_dict.get("max_slope")
        if self.max_slope is None:
            self.max_slope = machine_profile.load_profile().max_tilt_angle_deg

        self.ortho_to_wall = param_dict.get("ortho_to_wall")
        if self.ortho_to_wall is None:
            self.ortho_to_wall = False

        self.all_up = param_dict.get("all_up")
        if self.all_up is None:
            self.all_up = False

        self.infill = param_dict.get("infill")
        if self.infill is None:
            self.infill = False

        self.top_lines = param_dict.get("top_lines")
        self.bottom_lines = param_dict.get("bottom_lines")

        # Build plan P1.2: optional temperatures in deg C. Absent keys keep
        # upstream's header (bed 55, nozzle 210) byte for byte.
        self.bed_temp = param_dict.get("bed_temp")
        self.nozzle_temp = param_dict.get("nozzle_temp")
        # Checked now, not at the G-code stage an hour later.
        self.temperature_arguments = temperature_arguments(self.bed_temp, self.nozzle_temp)

        # Build plan P1.3: optional infill settings, in deposition widths.
        # Absent keys keep upstream's 8 and 2, and the upstream command.
        self.infill_period = param_dict.get("infill_period")
        self.shell_thickness = param_dict.get("shell_thickness")
        self.infill_arguments = infill_arguments(self.infill_period, self.shell_thickness)

        # Build plan P2.2: the optional overhang-aware field. Absent keys keep
        # upstream's field and upstream's commands.
        self.overhang_aware = param_dict.get("overhang_aware") is True
        self.overhang_arguments = overhang_arguments(
            param_dict.get("overhang_aware"),
            param_dict.get("max_overhang_deg"),
            param_dict.get("overhang_margin_deg"),
            param_dict.get("hold_overhang"),
            param_dict.get("ramp_in"),
            param_dict.get("max_tilt_rate_deg_per_mm"),
            param_dict.get("overhang_edges"),
            param_dict.get("flat_overhangs_outward"),
        )
        if self.overhang_aware and (self.ortho_to_wall or self.all_up):
            raise ValueError('"overhang_aware" cannot be combined with "ortho_to_wall" or "all_up"')

        self.log_path = f"data/log/{self.solid_name}.log"
        self.stl_path = f"data/mesh/{self.solid_name}.stl"
        self.obj_path = f"data/mesh/{self.solid_name}.obj"
        self.bpn_path = f"data/point_normal/{self.solid_name}.npz"
        self.sdf_path = f"data/sdf/{self.solid_name}.npz"
        # Build plan P2.2, plan_corrections P2-8 (a): an overhang-aware part
        # with infill keeps its SDF from before infill here.
        self.solid_sdf_path = f"data/sdf/{self.solid_name}_solid.npz"
        self.direction_path = f"data/direction/{self.solid_name}.npz"
        self.basis_path = f"data/basis/{self.solid_name}.npz"
        self.phasor_path = f"data/phasor/{self.solid_name}.npz"
        self.triphasor_path = f"data/triphasor/{self.solid_name}.npz"
        self.frame_path = f"data/frame/{self.solid_name}.npz"
        self.toolpath_path = f"data/toolpath/{self.solid_name}.npz"
        self.smoothed_toolpath_path = f"data/toolpath/{self.solid_name}_smoothed.npz"
        self.platform_toolpath_path = f"data/toolpath/{self.solid_name}_platform.npz"
        self.craftware_gcode_path = f"data/gcode/{self.solid_name}_craftware.gcode"
        self.smoothed_tesselated_toolpath_path = (
            f"data/toolpath/{self.solid_name}_smoothed_tesselated.npz"
        )
        self.gcode_path = f"data/gcode/{self.solid_name}.gcode"

        self.smoothing_iter_count = 8
        self.degree_angle_max_diff = 1.0


# --------------------------------------------------------------------------
# The stages (build plan P5.1a)
#
# Upstream built every stage's command inside `__main__` and ran them one
# after another. They are built here instead, so a caller can run part of the
# pipeline: P2.0's field-only evaluation stops after the atoms are extracted,
# and P5.1's driver resumes from a later stage. The command line behaves
# exactly as before; `tests/test_atomize_stages.py` checks every command,
# printed line and log line against a record of the script before this change.
# --------------------------------------------------------------------------

#: Every stage, in the order it runs, named after the tool that implements it
#: (`tools/<name>.py`). `sdf_to_isdf` runs only for a part with infill.
STAGE_NAMES = (
    "process_for_atomizer",
    "obj_to_bpn",
    "bpn_to_sdf",
    "sdf_to_isdf",
    "compute_tool_orientations",
    "sdf_df_to_layers",
    "compute_tangents",
    "align_atoms",
    "extract_explicit_atoms",
    "order_atoms",
    "smooth_toolpath_point",
    "tesselate_toolpath_orientations",
    "add_platform",
    "toolpath_to_gcode",
    "ratrig_to_craftware",
)

#: The stages the log lists under "Pipeline commands". Upstream never listed
#: the infill stage or the last four, and the log is kept as it was
#: (plan_corrections P1-13): do not read it as the list of what ran.
LOGGED_STAGE_NAMES = (
    "process_for_atomizer",
    "obj_to_bpn",
    "bpn_to_sdf",
    "compute_tool_orientations",
    "sdf_df_to_layers",
    "compute_tangents",
    "align_atoms",
    "extract_explicit_atoms",
    "order_atoms",
    "smooth_toolpath_point",
)


@dataclass(frozen=True)
class Stage:
    """One stage of the pipeline and the shell command that runs it."""

    #: The tool that runs the stage, `tools/<name>.py`; one of `STAGE_NAMES`.
    name: str
    command: str
    #: Printed before the stage. Consecutive stages sharing one step (10 and
    #: 11 each run several tools) print it once.
    step: str
    #: Written to the log as "# <heading>" before the step, or None for the
    #: steps upstream did not log (1 to 3).
    log_heading: Optional[str] = None
    #: Whether `--warmup` runs the stage once beforehand without `--logpath`,
    #: so Taichi's compilation is not counted in the logged time.
    warmup: bool = False


def pipeline_geometry(deposition_width: float):
    """``(cell_sides_length, layer_height)`` in mm for a deposition width in mm.

    Computed by Atomizer's own Taichi functions, in 32-bit floats, exactly as
    upstream did; the layer height reaches `add_platform` as
    ``0.44999998807907104`` for 0.9 mm beads, not 0.45.
    """
    cell_sides_length = atom.fff3.cell_sides_length_from_deposition_width_kernel(
        deposition_width
    )
    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        cell_sides_length
    )
    return cell_sides_length, layer_height


def build_stage_commands(params: Parameters) -> List[Stage]:
    """Every stage for the part ``params`` describes, in the order they run.

    The commands are upstream's, character for character.
    """
    cell_sides_length, layer_height = pipeline_geometry(params.deposition_width)

    # Build plan P2.2, plan_corrections P2-8 (a): an overhang-aware part with
    # infill writes its SDF from before infill to its own file, the infill
    # stage reads it from there, and the field is computed on it. Otherwise
    # upstream's single file, and upstream's commands exactly.
    keep_solid_sdf = params.overhang_aware and params.infill
    bpn_sdf_path = params.solid_sdf_path if keep_solid_sdf else params.sdf_path

    process_mesh_command = f"blender -b -P tools/process_for_atomizer.py -- {params.stl_path} {params.obj_path} {cell_sides_length * 0.5:.6f}"
    obj_to_bpn_cmd = f"python tools/obj_to_bpn.py {params.obj_path} {params.bpn_path}"
    bpn_to_sdf_cmd = f"python tools/bpn_to_sdf.py {params.bpn_path} {bpn_sdf_path} {params.deposition_width:.2f}"
    sdf_to_isdf_cmd = f"python tools/sdf_to_isdf.py {params.bpn_path} {bpn_sdf_path} {params.sdf_path} no_gui=True{params.infill_arguments}"

    compute_tool_orientations_cmd = f"python tools/compute_tool_orientations.py {params.sdf_path} {params.direction_path} --maxslope {params.max_slope}"
    if params.ortho_to_wall:
        compute_tool_orientations_cmd += " --ortho_to_wall"
    if params.all_up:
        compute_tool_orientations_cmd += " --allup"
    compute_tool_orientations_cmd += params.overhang_arguments
    if keep_solid_sdf:
        compute_tool_orientations_cmd += f" --solid_sdf {params.solid_sdf_path}"
    compute_tool_orientations_cmd += f" --logpath {params.log_path}"

    sdf_df_to_layers_cmd = f"python tools/sdf_df_to_layers.py {params.sdf_path} {params.direction_path} {params.phasor_path} --maxslope {params.max_slope}"
    if params.all_up:
        sdf_df_to_layers_cmd += " --allup"
    sdf_df_to_layers_cmd += f" --logpath {params.log_path}"

    compute_tangents_cmd = f"python tools/compute_tangents.py {params.sdf_path} {params.direction_path} {params.basis_path} --maxslope {params.max_slope}"
    if params.top_lines is not None:
        compute_tangents_cmd += f" --top_lines {params.top_lines}"
    if params.bottom_lines is not None:
        compute_tangents_cmd += f" --bottom_lines {params.bottom_lines}"
    compute_tangents_cmd += f" --logpath {params.log_path}"

    align_atoms_cmd = f"python tools/align_atoms.py {params.sdf_path} {params.phasor_path} {params.basis_path} {params.triphasor_path} --maxslope {params.max_slope} --logpath {params.log_path}"
    extract_explicit_atoms_cmd = f"python tools/extract_explicit_atoms.py {params.sdf_path} {params.triphasor_path} {params.frame_path} --logpath {params.log_path}"
    order_atoms_cmd = f"python tools/order_atoms.py {params.sdf_path} {params.frame_path} {params.toolpath_path} --logpath {params.log_path}"
    smooth_toolpath_point_cmd = f"python tools/smooth_toolpath_point.py {params.toolpath_path} {params.smoothed_toolpath_path} {params.smoothing_iter_count}"
    tesselate_toolpath_orientations_cmd = f"python tools/tesselate_toolpath_orientations.py {params.smoothed_toolpath_path} {params.smoothed_tesselated_toolpath_path} {params.degree_angle_max_diff}"
    add_platform_cmd = f"python tools/add_platform.py {params.smoothed_tesselated_toolpath_path} {params.platform_toolpath_path} {params.deposition_width} {layer_height}"
    toolpath_to_gcode_cmd = f"python tools/toolpath_to_gcode.py {params.platform_toolpath_path} {params.gcode_path}{params.temperature_arguments}"
    ratrig_to_craftware_cmd = f"python tools/ratrig_to_craftware.py {params.gcode_path} {params.craftware_gcode_path}"

    stages = [
        Stage("process_for_atomizer", process_mesh_command, "Step 1 Starting: Remesh"),
        Stage("obj_to_bpn", obj_to_bpn_cmd, "Step 2 Starting: Mesh to Point Normal Cloud"),
        Stage("bpn_to_sdf", bpn_to_sdf_cmd, "Step 3 Starting: Point Normal Cloud to SDF"),
    ]
    if params.infill:
        stages.append(
            Stage("sdf_to_isdf", sdf_to_isdf_cmd, "Step 3 Bis Starting: SDF to SDF with Infill")
        )

    step_10 = "Step 10 Starting: Smooth, Tesselate and Add a Platform"
    log_10 = "Smooth, Tesselate and Add a Platform"
    step_11 = "Step 11 Starting: G-Code Generation"
    log_11 = "G-Code Generation"
    stages += [
        Stage(
            "compute_tool_orientations",
            compute_tool_orientations_cmd,
            "Step 4 Starting: Direction Field Computation",
            "Direction Field Computation",
            warmup=True,
        ),
        Stage(
            "sdf_df_to_layers",
            sdf_df_to_layers_cmd,
            "Step 5 Starting: Implicit Layers Computation",
            "Implicit Layers Computation",
            warmup=True,
        ),
        Stage(
            "compute_tangents",
            compute_tangents_cmd,
            "Step 6 Starting: Tangents Computation",
            "Tangents Computation",
            warmup=True,
        ),
        Stage(
            "align_atoms",
            align_atoms_cmd,
            "Step 7 Starting: Atoms alignment",
            "Atoms alignment",
            warmup=True,
        ),
        Stage(
            "extract_explicit_atoms",
            extract_explicit_atoms_cmd,
            "Step 8 Starting: Implicit to Explicit Atoms",
            "Implicit to Explicit Atoms",
            warmup=True,
        ),
        Stage(
            "order_atoms",
            order_atoms_cmd,
            "Step 9 Starting: Order Atoms",
            "Order Atoms",
            warmup=True,
        ),
        Stage("smooth_toolpath_point", smooth_toolpath_point_cmd, step_10, log_10),
        Stage("tesselate_toolpath_orientations", tesselate_toolpath_orientations_cmd, step_10, log_10),
        Stage("add_platform", add_platform_cmd, step_10, log_10),
        Stage("toolpath_to_gcode", toolpath_to_gcode_cmd, step_11, log_11),
        Stage("ratrig_to_craftware", ratrig_to_craftware_cmd, step_11, log_11),
    ]
    return stages


def check_stage_names(names: Iterable[str]) -> None:
    """Raise `ValueError` naming any entry that is not in `STAGE_NAMES`."""
    unknown = sorted(set(names) - set(STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage name(s): {', '.join(unknown)}. "
            f"Valid names, in order: {', '.join(STAGE_NAMES)}."
        )


def stages_through(last: str) -> List[str]:
    """The names of every stage up to and including ``last``, in order."""
    check_stage_names([last])
    return list(STAGE_NAMES[: STAGE_NAMES.index(last) + 1])


def select_stages(
    stages: List[Stage],
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
) -> List[Stage]:
    """The stages named in ``only`` (all when None), less those in ``skip``.

    The pipeline's order is kept whatever order the names come in. A name in
    `STAGE_NAMES` that the part does not use (`sdf_to_isdf` without infill) is
    simply absent; a name that is not a stage at all raises `ValueError`.
    """
    only = None if only is None else list(only)
    skip = [] if skip is None else list(skip)
    check_stage_names((only or []) + skip)
    wanted = set(STAGE_NAMES if only is None else only) - set(skip)
    return [stage for stage in stages if stage.name in wanted]


def run_stages(
    params: Parameters,
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
    warmup: bool = False,
    stages: Optional[List[Stage]] = None,
) -> List[str]:
    """Run the selected stages for ``params`` in order; return their names.

    ``only`` and ``skip`` take names from `STAGE_NAMES` (see `select_stages`).
    ``stages`` defaults to `build_stage_commands(params)`.

    Each stage runs through `os.system`, as upstream's did, and **its exit
    code is not checked** (plan_corrections 4.16): a failed stage is followed
    by the next one regardless. `tools/overhang_report.py` checks each
    stage's output afterwards; a caller that runs stages some other way must
    do the same.
    """
    stages = build_stage_commands(params) if stages is None else stages
    selected = select_stages(stages, only, skip)

    announced = None
    for stage in selected:
        if stage.step != announced:
            announced = stage.step
            print(f"\n{stage.step}\n")
            if stage.log_heading is not None:
                with open(params.log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n# {stage.log_heading}\n\n")
        if warmup and stage.warmup:
            print("Warmup")
            os.system(stage.command.partition(" --logpath")[0])
        os.system(stage.command)

    return [stage.name for stage in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atomize and order atoms for fused filament fabrication. The input is a 3D solid and the output is a toolpath composed of 3D positions, each associated with a basis. The 3D solid path and the command parameters (deposition width, maximum tilting angle for the tool) are given with a json file. This command is composed of several sub-commands, first processing the mesh to convert it to a SDF, then atomizing the 3D solid, and finally ordering atoms to create the toolpath. The output toolpath is located in the `data/toolpath` folder, and a log file is generated in the `data/log` folder, giving the list of sub-commands that are runned, and the list of commands to visualize all the generated data. "
    )
    parser.add_argument(
        "json_path",
        help="The path to the JSON file contening the command parameters. The parameters are: - `solid_name`: the filename of the STL representing the 3D solid, without the extension. The STL has to be in the `data/mesh/` folder, and its extension has to be `.stl`; - `deposition_width`: the deposition width in millimeter. The layer height is half the deposition width; - `max_slope`: the maximum tilting angle for the tool in degrees; - `top_lines` and `bottom_lines` (optionals): the paths to a 1 channel, 8 bits per pixel, PNG file representing the target tangent directions, for the top and bottom surfaces, respectively. The mapping is: [0, 255] -> [-pi/2, pi/2]. The orientation represents a 2D line in the xy-plane. The 2D line field is on the upper face of the solid bounding box and then planarly projected onto the top and bottom surfaces along the z-axis; - `ortho_to_wall` (optional): if true, force the tool orientation to be parallel to the boundary. By default is false, as this feature creates a lot of tool orientation changes that are detrimental to the surface quality.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="To avoid including compilation time in the computation time, run each sub-command twice to cache the compilation.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGE_NAMES,
        default=None,
        metavar="STAGE",
        help="Run the pipeline only up to and including this stage, then stop (this fork, build plan P5.1a). Build plan P2.0 stops after extract_explicit_atoms, to measure the orientation field without the slow ordering stage. Stages: "
        + ", ".join(STAGE_NAMES)
        + ".",
    )

    args = parser.parse_args()
    json_path = args.json_path
    warmup = args.warmup

    # DEBUG
    # json_path = "data/param/calibration_slope_1.json"
    # warmup = True

    params = Parameters()
    params.load(json_path)

    log_file = open(params.log_path, "w", encoding="utf-8")

    cell_sides_length, layer_height = pipeline_geometry(params.deposition_width)

    str_to_print = "# Parameters"
    print(str_to_print)
    log_file.write(str_to_print + "\n\n")

    str_to_print = f"- Solid name: {params.solid_name}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- STL input: {params.stl_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- GCode output: {params.gcode_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Log file: {params.log_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Deposition width: {params.deposition_width:.2f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Layer height: {layer_height:.2f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Cell sides length: {cell_sides_length:.3f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Machine max slope angle: {params.max_slope:.1f} degrees"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    stages = build_stage_commands(params)
    command_of = {stage.name: stage.command for stage in stages}

    visualize_bpn_cmd = f"python tools/visualize_bpn.py {params.bpn_path}"
    visualize_bpn_sdf_cmd = (
        f"python tools/visualize_bpn_sdf.py {params.bpn_path} {params.sdf_path}"
    )
    visualize_bpn_df_cmd = (
        f"python tools/visualize_bpn_df.py {params.bpn_path} {params.direction_path}"
    )
    visualize_bpn_layers_cmd = (
        f"python tools/visualize_bpn_layers.py {params.bpn_path} {params.phasor_path}"
    )
    visualize_bases_cmd = (
        f"python tools/visualize_bases.py {params.bpn_path} {params.basis_path}"
    )
    visualize_implicit_atoms_cmd = f"python tools/visualize_implicit_atoms.py {params.bpn_path} {params.triphasor_path}"
    visualize_explicit_atoms_cmd = f"python tools/visualize_explicit_atoms.py {params.frame_path} {layer_height:.3f}"
    visualize_toolpath_cmd = (
        f"python tools/visualize_toolpath.py {params.smoothed_toolpath_path}"
    )

    log_file.write("\n# Pipeline commands\n\n")
    for name in LOGGED_STAGE_NAMES:
        log_file.write(command_of[name] + "\n")

    log_file.write("\n# Visualize commands\n\n")
    log_file.write(visualize_bpn_cmd + "\n")
    log_file.write(visualize_bpn_sdf_cmd + "\n")
    log_file.write(visualize_bpn_df_cmd + "\n")
    log_file.write(visualize_bpn_layers_cmd + "\n")
    log_file.write(visualize_bases_cmd + "\n")
    log_file.write(visualize_implicit_atoms_cmd + "\n")
    log_file.write(visualize_explicit_atoms_cmd + "\n")
    log_file.write(visualize_toolpath_cmd + "\n")

    log_file.close()

    only = None if args.stop_after is None else stages_through(args.stop_after)
    run_stages(params, only=only, warmup=warmup, stages=stages)

    if args.stop_after is not None:
        str_to_print = (
            f"\nStopped after {args.stop_after} (--stop-after); "
            "the later stages did not run.\n"
        )
        print(str_to_print)
        with open(params.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(str_to_print + "\n")
