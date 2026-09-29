"""The atoms as the orientation field left them, before any print order (build plan P2.0).

`tools/extract_explicit_atoms.py` writes ``data/frame/<part>.npz`` through
`atom.frame3.Field.save_active_frame_set`: one row per atom that survived
extraction, with no grid and no order::

    point    (M, 3) float32  mm, part frame: the atom's centre
    normal   (M, 2) float32  radians, spherical [theta, phi]: its tool orientation
    phi_t    (M,)   float32  radians: its deposition tangent, as an angle in
                             the atom's tangent plane

``normal`` is the orientation field at the atom's cell, carried unchanged
through the basis, triphasor and extraction stages. `order_atoms` then copies
it, unchanged, into the toolpath's ``tool_orientation``
(`toolpath3.toolpath_planner_update_toolpath`), and `smooth_toolpath_point`
moves positions only. So every deposition point of ``<part>_smoothed.npz``
carries exactly one atom's orientation, and the metrics that need only
positions and orientations can be measured here, before the ordering stage
that takes 86 % of a run.

What cannot be measured here is anything that needs a print order:
unsupported deposition. Two smaller differences from a full run:

* every atom counts, while a full run's deposition mask leaves out the first
  atom of each deposition run, which is reached by a travel move (about 2 % of
  the points on the golden cube);
* positions are the atoms' own, before smoothing moves them along the path.

`FrameAtoms` has the attributes `atom.overhang_metrics` reads from a toolpath
(``point``, ``tool_orientation``, ``travel_type``, ``point_count``), with every
atom a deposition point, so the same functions measure both.

numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The arrays `frame3.Field.to_active_frame_set` writes.
FRAME_KEYS = ("point", "normal", "phi_t")

#: `toolpath3.TRAVEL_TYPE_DEPOSITION`, repeated so this module needs no Taichi.
TRAVEL_TYPE_DEPOSITION = 0


@dataclass(frozen=True)
class FrameAtoms:
    """Extracted atoms, readable by `atom.overhang_metrics` like a toolpath.

    ``tool_orientation`` is the frame file's ``normal``, renamed to the name
    the toolpath and the metrics use for the same quantity.
    """

    #: ``(M, 3)`` atom centres, mm, part frame.
    point: np.ndarray
    #: ``(M, 2)`` spherical ``[theta, phi]`` in radians; ``theta`` is the tilt.
    tool_orientation: np.ndarray
    #: ``(M,)`` deposition tangent angles, radians.
    phi_t: np.ndarray
    #: Rows left out because their position or orientation was not finite.
    dropped_non_finite: int = 0

    @property
    def point_count(self) -> int:
        return int(len(self.point))

    @property
    def travel_type(self) -> np.ndarray:
        """Every atom deposits: there are no travel moves before ordering."""
        return np.full(self.point_count, TRAVEL_TYPE_DEPOSITION, dtype=np.int32)


def load_frame_atoms(path) -> FrameAtoms:
    """Read ``data/frame/<part>.npz`` as written by `extract_explicit_atoms`.

    Raises `ValueError` when the file is not in that format (a missing array,
    wrong shapes, rows that do not match, or no atoms at all). Rows whose
    position or orientation is not finite are left out and counted in
    ``dropped_non_finite`` rather than raised on: the writer already drops
    atoms without a position, so any found here are worth reporting, but not
    worth losing a run over.
    """
    path = Path(path)
    with np.load(path) as data:
        missing = [key for key in FRAME_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"{path} is not an atom frame file: no {', '.join(missing)} "
                f"(has {', '.join(sorted(data.files))})."
            )
        point = np.asarray(data["point"])
        normal = np.asarray(data["normal"])
        phi_t = np.asarray(data["phi_t"])

    count = len(point)
    if point.ndim != 2 or point.shape[1] != 3:
        raise ValueError(f"{path}: point has shape {point.shape}, expected (M, 3).")
    if normal.shape != (count, 2):
        raise ValueError(
            f"{path}: normal has shape {normal.shape}, expected ({count}, 2) "
            "spherical [theta, phi] per atom."
        )
    if phi_t.shape != (count,):
        raise ValueError(f"{path}: phi_t has shape {phi_t.shape}, expected ({count},).")

    finite = np.isfinite(point).all(axis=1) & np.isfinite(normal).all(axis=1)
    if not finite.any():
        raise ValueError(f"{path} holds no atom with a finite position and orientation.")

    return FrameAtoms(
        point=point[finite],
        tool_orientation=normal[finite],
        phi_t=phi_t[finite],
        dropped_non_finite=int(count - np.count_nonzero(finite)),
    )
