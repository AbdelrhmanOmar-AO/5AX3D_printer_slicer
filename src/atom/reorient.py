"""Candidate bed-tilt pose sampling and printability testing.

Status: placeholder — Phase 3, sub-phase 1 of the implementation plan
(see the "5-Axis Slicer: Implementation Plan" doc, section "Phase 3").

This module will:
  1. Sample a discretized grid of candidate two-axis bed tilts, bounded by
     `kinematics3z.MAX_TILT_ANGLE_DEG`.
  2. For each candidate pose, rotate the part's boundary normals
     (`atom.solid3.BoundaryPointNormal`) by the inverse pose rotation.
  3. Test the rotated normals against Atomizer's existing printability
     thresholds (`atom.fff3.CEIL_MAX_ANGLE`, `atom.fff3.MACHINE_MAX_SLOPE_ANGLE`)
     to flag which surface regions become printable under that pose.

Not implemented yet. Nothing here should be imported by the rest of the
pipeline until this note is removed.
"""

from . import fff3, kinematics3z  # noqa: F401  (imported for the constants above)


def sample_candidate_poses(max_tilt_deg: float = None, step_deg: float = 5.0):
    """Return a list of (tilt_x_deg, tilt_y_deg) candidate bed poses.

    Placeholder. `max_tilt_deg` defaults to `kinematics3z.MAX_TILT_ANGLE_DEG`
    once implemented.
    """
    raise NotImplementedError("Phase 3, sub-phase 1 — see implementation plan")


def printable_mask_under_pose(bpn, pose):
    """Return a per-point boolean mask of which boundary points are printable
    (normal within the upper-hemisphere cone) if the bed were tilted to `pose`.

    Placeholder.
    """
    raise NotImplementedError("Phase 3, sub-phase 1 — see implementation plan")
