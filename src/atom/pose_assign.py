"""Region-to-pose assignment and build sequencing.

Status: placeholder — Phase 3, sub-phase 2 of the implementation plan.

This module will assign each surface/volume region (flagged by
`atom.reorient`) to one candidate bed pose such that:
  - every region is covered by at least one pose that makes it printable,
  - the number of distinct poses used is minimized (each pose switch costs
    real print time),
  - a region assigned to a pose is only reachable from material already
    printed — its own base rests on the bed or on a lower, already-assigned
    region (a discretized version of Dai et al. 2018's "supported from
    below" idea).

Scope note (from the 2026-09-18 architecture review): this is a constrained
set-cover-with-partial-order problem, which is NP-hard-flavored for
arbitrary geometry. For the two benchmark parts this project targets, start
with a greedy heuristic (try poses in a fixed preference order; assign each
unresolved region to the first pose that both makes it printable and
satisfies the "supported from below" constraint) rather than a general
solver. Also decide and document what happens if greedy can't cover a
region on either benchmark part — there is currently no fallback, and
Atomizer's own lattice-support generator is explicitly out of scope.

Not implemented yet.
"""


def assign_regions_to_poses(regions, candidate_poses):
    """Return an ordered list of (pose, region_ids) build groups.

    Placeholder.
    """
    raise NotImplementedError("Phase 3, sub-phase 2 — see implementation plan")
