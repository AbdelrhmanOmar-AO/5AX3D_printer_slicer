"""Round-trip test for the triple-Z kinematics module.

Status: placeholder — Phase 1 of the implementation plan.

Once our machine's constants are set in `src/atom/kinematics3z.py` (Phase 0),
this test should:
  1. Pick a sample (x, y, z0, z1, z2) within our axis limits.
  2. Run `KinematicTaichi(...).forward(...)` to get a position + normal.
  3. Run `KinematicTaichi(...).inverse(...)` to recover (z0, z1, z2).
  4. Assert the round trip matches within a small tolerance.

Not implemented yet — needs our real machine constants first (Phase 0).
"""

import pytest


@pytest.mark.skip(reason="Phase 0 (our machine constants) not done yet — see implementation plan")
def test_forward_inverse_round_trip():
    raise NotImplementedError
