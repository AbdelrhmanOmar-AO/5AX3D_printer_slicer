"""Top-level driver for multi-pose (reorientation) parts.

Status: placeholder — Phase 4 of the implementation plan.

Parallel to the existing `tools/atomize.py`, but routes through
`atom.reorient` -> `atom.pose_assign` -> `atom.multi_pose_pipeline` ->
`tools/toolpath_to_gcode.py` instead of a single flat pipeline call, for
parts that need more than one bed pose to avoid supports.

Not implemented yet. `tools/atomize.py` (vendored from upstream Atomizer,
unmodified) remains the correct entry point for single-pose parts.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Phase 4 — see implementation plan. Use tools/atomize.py for "
        "single-pose parts in the meantime."
    )
