import argparse
import sys
import time

import numpy as np
import taichi as ti

from atom.ti_env import init_taichi
from tqdm import tqdm

import atom.fff3
import atom.frame3
import atom.solid3
import atom.toolpath3
from atom.bvh import BVH

init_taichi("cpu", offline_cache_cleaning_policy="never", kernel_profiler=True)


def order_atoms():
    parser = argparse.ArgumentParser(
        description="Order the atoms to create a valid toolpath for non-planar 3D printing."
    )
    parser.add_argument(
        "sdf_path", help="The path to the input signed distance field (SDF)"
    )
    parser.add_argument(
        "frame_path",
        help="The path to the input unordered set of atoms, i.e., a set of frames.",
    )
    parser.add_argument("toolpath_path", help="The path to the output toolpath.")
    parser.add_argument(
        "--logpath",
        type=str,
        help="Path to the log file.",
    )

    args = parser.parse_args()

    sdf_path = args.sdf_path
    frame_path = args.frame_path
    toolpath_path = args.toolpath_path

    log_path = args.logpath

    if log_path is not None:
        log_file = open(log_path, "a", encoding="utf-8")

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)

    frame_set = atom.frame3.Set()
    frame_set.load(frame_path)
    p_min, p_max = frame_set.get_aabb()
    p_heighest = frame_set.get_highest_point()
    side_length = p_max - p_min
    domain_diag = np.linalg.norm(side_length)
    atom_count = frame_set.phi_t.shape[0]

    str_to_print = f"Atom count: {atom_count}"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    # Create BVH
    bvh = BVH(auto_rebuild=True)
    bvh.from_frame_set(frame_set)

    triperiod = atom.fff3.triperiod_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    ).to_numpy()

    neighborhood_radius = atom.fff3.neighborhood_radius_kernel(
        sdf.grid.cell_sides_length
    )
    str_to_print = f"Neighborhood radius: {neighborhood_radius:.4f}"
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")

    toolpath_planner = atom.toolpath3.ToolpathPlanner()
    toolpath_planner.init(
        frame_set, triperiod, neighborhood_radius, domain_diag, bvh, p_heighest
    )
    # Flag walls, but not used in this implementation
    atom.fff3.init_frame_set_wall_state(
        sdf.sdf,
        toolpath_planner.frame_set.point,
        sdf.grid.cell_sides_length,
        toolpath_planner.frame_data_set.state,
    )

    is_finished = False

    t0 = time.perf_counter()

    with tqdm(total=atom_count, desc="Ordering", unit="step") as pbar:
        # Open the file in write mode ('w')
        # with open("benchy.log", "w") as file:
        while not is_finished:
            # file.write(f"iteration_number: {toolpath_planner.iteration_number}\n")

            if toolpath_planner.iteration_number == 0:
                toolpath_planner.init_non_supporting_set()
            else:
                if not toolpath_planner.backpropagate:
                    toolpath_planner.update_non_supporting_set()

            # file.write(
            #     f"set size: {toolpath_planner.non_supporting_frame_set.size}\n"
            # )

            # toolpath_planner.init_non_supporting_state()
            # toolpath_planner.update_relative_state_and_distance()
            # toolpath_planner.init_unaccessibility()

            toolpath_planner.init_non_supporting_state_fast_track_option()

            if not toolpath_planner.fast_track:
                toolpath_planner.init_atom_in_a_cycle()

            # toolpath_planner.compute_cost()
            # next_index, next_travel_type = toolpath_planner.find_best_next()

            next_index, next_travel_type = (
                toolpath_planner.compute_cost_and_find_best_next()
            )
            if next_index == -1 and toolpath_planner.backpropagate:
                if toolpath_planner.iteration_number == atom_count:
                    toolpath_np = atom.toolpath3.Toolpath()
                    toolpath_np.from_numpy(toolpath_planner.toolpath.to_numpy())
                    toolpath_planner.toolpath = toolpath_np
                    toolpath_planner.toolpath.set_constant_deposition_width_and_height(
                        triperiod[1], triperiod[2]
                    )
                    toolpath_planner.toolpath.flip()
                    # Uplift by half a layer
                    toolpath_planner.toolpath.uplift(triperiod[2] * 0.5)
                    toolpath_planner.toolpath.save(toolpath_path)

                    is_finished = True
                else:
                    sys.exit(
                        f"No more valid next atom, even with the backpropagate, but there is still atoms in the set. Maybe the set of non supporting atoms is corrupted and you should use toolpath_planner.init_non_supporting_set() to reinitialize it. Current iteration number: {toolpath_planner.iteration_number}. Or maybe two adjacent atoms are constraining each other."
                    )

            if next_index != -1:
                next_is_depo = next_travel_type == atom.toolpath3.TRAVEL_TYPE_DEPOSITION

                next_is_not_depo = not next_is_depo
                current_is_not_depo = not (
                    toolpath_planner.current_travel_type
                    == atom.toolpath3.TRAVEL_TYPE_DEPOSITION
                )
                is_one_atom_length_trajectory = next_is_not_depo and current_is_not_depo

                # backpropagate; Next is depo;
                # 00: fast_track: false; update_toolpath: false; backpropagate: true

                # 01 or
                # 10 or
                # 11: fast_track: true; update_toolpath: true; backpropagate: false

                backpropagate_required = (
                    not toolpath_planner.backpropagate
                    and (not next_is_depo or next_index == -1)
                    and not toolpath_planner.iteration_number == 0
                )

                # DEBUG
                # backpropagate_required = False

                if backpropagate_required:
                    toolpath_planner.fast_track = False
                    toolpath_planner.backpropagate = True
                else:
                    if next_is_not_depo:
                        next_is_reachable_by_constraint_index = 0
                        next_list = []
                        while next_is_reachable_by_constraint_index != -1:
                            next_list.append(next_index)
                            next_is_reachable_by_constraint_index = (
                                toolpath_planner.is_atom_i_reachable_by_a_constraint(
                                    next_index
                                )
                            )
                            if next_is_reachable_by_constraint_index != -1:
                                next_index, next_travel_type = (
                                    toolpath_planner.find_best_next_constrainer(
                                        next_is_reachable_by_constraint_index
                                    )
                                )
                                # If there is a circular constraint
                                if next_index in next_list:
                                    break

                    toolpath_planner.fast_track = True
                    # DEBUG
                    # toolpath_planner.fast_track = False
                    if (
                        not is_one_atom_length_trajectory
                        or toolpath_planner.toolpath_point_count_previous == 0
                    ):
                        toolpath_planner.update_toolpath(next_index, next_travel_type)
                    else:
                        # Get back to the previous point
                        toolpath_planner.toolpath.point_count = (
                            toolpath_planner.toolpath_point_count_previous
                        )
                        toolpath_planner.current_frame_index = (
                            toolpath_planner.previous_frame_index
                        )
                        toolpath_planner.current_point = toolpath_planner.previous_point
                        toolpath_planner.update_toolpath(next_index, next_travel_type)
                    toolpath_planner.backpropagate = False
                    pbar.update()
            else:
                # For an unknow reason, the set of non supporting can be unexact,
                # most probably because the update_non_supporing_set can miss one atom.
                toolpath_planner.init_non_supporting_set()
                toolpath_planner.fast_track = False
                toolpath_planner.backpropagate = True

    t1 = time.perf_counter()

    ti.profiler.print_kernel_profiler_info()

    duration = t1 - t0

    str_to_print = f"Toolpath planner took {duration:.1f} seconds."
    print(str_to_print)
    if log_path is not None:
        log_file.write(str_to_print + "\n")
        log_file.close()

    ti.reset()


if __name__ == "__main__":
    order_atoms()
