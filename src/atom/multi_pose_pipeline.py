"""Orchestrates Atomizer's existing pipeline stages once per pose group.

Status: placeholder — Phase 3, sub-phase 3 of the implementation plan.

For each pose group produced by `atom.pose_assign`, this module will:
  1. Rotate that sub-volume's BPN (`atom.solid3.BoundaryPointNormal` — point
     and normal arrays, a cheap matrix multiply) by the inverse pose
     rotation, then regenerate a fresh axis-aligned SDF for it via
     `atom.solid3.SDF.create_from_bpn()`. Atomizer has no operation that
     rotates an existing SDF grid in place — its SDF is a fixed grid built
     directly from a BPN — so regenerating is the correct, already-supported
     path, not extra work, and it only runs once per *assigned* pose, not
     once per candidate pose in the search grid.
  2. Run Atomizer's existing, UNMODIFIED pipeline stages on it as if the
     pose's local "up" were global up: compute_tool_orientations ->
     sdf_df_to_layers -> compute_tangents -> align_atoms ->
     extract_explicit_atoms -> order_atoms.
  3. Rotate the resulting frames/toolpath back into the machine's global
     frame and tag every point with its pose id.

This replaces the single top-level call `tools/atomize.py` makes today,
for parts that need more than one pose.

Not implemented yet.
"""


def run_multi_pose(build_groups, params):
    """Run the stock pipeline once per (pose, region) build group and return
    a single combined, pose-tagged toolpath.

    Placeholder.
    """
    raise NotImplementedError("Phase 3, sub-phase 3 — see implementation plan")
