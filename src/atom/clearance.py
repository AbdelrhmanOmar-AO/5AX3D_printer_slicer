"""Where the bed and the printed part must never go (build plan P4.1).

A clearance model describes the machine's solid parts around the nozzle, the
region the tilted bed and the part printed so far must stay out of. The swept
check (P4.2, `atom.tilt_motion_check`) moves the bed through every machine
state and asks the model how close it comes; the reachability map (P2.3) will
ask the same question of candidate tilts.

numpy only: no Taichi, no machine kinematics. The caller puts the points in
the world frame first (`atom.bed_motion`).

Frames
------
World frame
    Fixed to the machine, as in `atom.bed_motion`: +Z up, and the nozzle tip
    at ``(X, Y, 0)``, where X and Y are the head's machine coordinates. The
    nozzle never moves vertically on this printer (the bed does), so
    ``z = 0`` is always the nozzle tip's level.
Head frame
    The world frame shifted so the nozzle tip is at the origin:
    ``p_head = p_world - (X, Y, 0)``.

Every method takes world-frame points plus the head position ``head_xy``
(X, Y). With the default ``head_xy = (0, 0)`` the points are simply taken to
be in the head frame already.

Signed distance
---------------
Each model reports, per point, a signed distance to each of its solid bodies:
positive outside (the clearance, in mm), negative inside (the penetration
depth), zero on the surface. A point exactly on a surface is touching, not
colliding. The distances are exact Euclidean distances, not bounds, so a
reported clearance is the true gap.

Models
------
`ReferenceClearance`
    The proxy the build plan prescribes until gate M3 supplies the real
    envelope: the nozzle as a cone opening upward from the tip, half-angle
    ``NOZZLE_HALF_ANGLE_DEG`` (Atomizer's own ``toolpath3.NOZZLE_CONE_ANGLE /
    2``), up to ``nozzle_to_gantry``; and the gantry as the whole half-space
    above that height. The gantry half-space is the same one
    `kinematics3z.inverse` tests the bed corners against.
`BoxesClearance`
    Axis-aligned boxes read from ``config/machines/<name>.clearance.json``.
    Each box is fixed to the frame or follows the head in X and/or Y. The file
    format is defined here; the real numbers are gate M3 and arrive in P6.2.
    No such file exists yet.

`load_clearance(profile)` returns the boxes when the profile has a clearance
file and the reference proxy otherwise.

Units: millimetres and degrees at every public boundary.
"""

from __future__ import annotations  # no Taichi kernels in this module

import json
import sys
import warnings
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from . import machine_profile

#: Half-angle of the nozzle cone, degrees: `toolpath3.NOZZLE_CONE_ANGLE / 2`
#: (80 / 2). Repeated here because `toolpath3` imports Taichi at module level;
#: `tests/test_clearance.py` checks the two agree.
NOZZLE_HALF_ANGLE_DEG = 40.0

#: Suffix of a profile's clearance file under `machine_profile.CONFIG_DIR`.
CLEARANCE_SUFFIX = ".clearance.json"

#: Axes a box may follow. The head moves in X and Y; the bed carries its own
#: motion, so no machine part moves with the screws.
VALID_MOVES_WITH = frozenset({"x", "y"})


# --------------------------------------------------------------------------
# The protocol every model follows
# --------------------------------------------------------------------------


@runtime_checkable
class ClearanceModel(Protocol):
    """What the safety checks need from a clearance model.

    ``bodies`` names the solid parts, in the column order of
    `body_distances`. All methods take world-frame points ``(N, 3)`` mm and
    the head position ``head_xy`` (see the module docstring).
    """

    bodies: tuple[str, ...]

    def body_distances(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        """``(N, B)`` signed distance from each point to each body, mm."""
        ...

    def signed_distance(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        """``(N,)`` signed distance to the nearest body, mm."""
        ...

    def min_clearance(self, points, head_xy=(0.0, 0.0)) -> float:
        """Smallest signed distance over all points, mm; ``inf`` for none."""
        ...

    def violations(self, points, head_xy=(0.0, 0.0), margin_mm: float = 0.0) -> np.ndarray:
        """Indices of points closer than ``margin_mm`` to any body."""
        ...

    def describe(self) -> dict:
        """The model and its parameters, for a report's provenance."""
        ...


class _BodyModel:
    """Shared arithmetic: everything follows from `body_distances`."""

    bodies: tuple[str, ...] = ()

    def body_distances(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        raise NotImplementedError

    def signed_distance(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        distances = self.body_distances(points, head_xy)
        if distances.shape[0] == 0:
            return np.zeros(0)
        return distances.min(axis=1)

    def nearest_body(self, points, head_xy=(0.0, 0.0)):
        """``(distance (N,), body index (N,))``: the closest body per point."""
        distances = self.body_distances(points, head_xy)
        if distances.shape[0] == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        index = np.argmin(distances, axis=1)
        return distances[np.arange(len(index)), index], index

    def min_clearance(self, points, head_xy=(0.0, 0.0)) -> float:
        distance = self.signed_distance(points, head_xy)
        return float(distance.min()) if distance.size else float("inf")

    def violations(self, points, head_xy=(0.0, 0.0), margin_mm: float = 0.0) -> np.ndarray:
        """Indices of points whose signed distance is below ``margin_mm``.

        With the default margin of 0 only points strictly inside a body count;
        a positive margin also flags points nearer than that to one.
        """
        return np.flatnonzero(self.signed_distance(points, head_xy) < margin_mm)


def _head_frame(points, head_xy) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    head = np.asarray(head_xy, dtype=np.float64).reshape(2)
    return points - np.array([head[0], head[1], 0.0])


# --------------------------------------------------------------------------
# The reference proxy: a nozzle cone and a gantry half-space
# --------------------------------------------------------------------------


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from 2D points ``(N, 2)`` to the segment ``a``-``b``."""
    ab = b - a
    ap = p - a
    t = np.clip(ap @ ab / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(ap - t[:, None] * ab, axis=1)


def cone_signed_distance(points_head, half_angle_deg: float, height_mm: float) -> np.ndarray:
    """Exact signed distance to a solid upright cone, mm.

    The cone has its apex at the head-frame origin, opens along +Z with the
    given half-angle, and is capped flat at ``z = height_mm``. It is a solid
    of revolution, so the distance is found in the half-plane through the
    axis, ``(r, z)`` with ``r >= 0``, against the triangle ``(0, 0)``,
    ``(R, H)``, ``(0, H)`` where ``R = H tan(half_angle)``. For ``r >= 0`` the
    nearest boundary is always the slanted side or the top, never the axis.
    """
    points_head = np.asarray(points_head, dtype=np.float64).reshape(-1, 3)
    radius = np.hypot(points_head[:, 0], points_head[:, 1])
    return cone_signed_distance_axial(radius, points_head[:, 2], half_angle_deg, height_mm)


def cone_signed_distance_axial(radial, axial, half_angle_deg: float, height_mm: float) -> np.ndarray:
    """`cone_signed_distance` for points already given relative to the axis.

    ``axial`` is each point's distance along the cone's axis from the apex
    (negative behind it), ``radial`` its distance from the axis (>= 0), both
    mm. This is how the nozzle-vs-material check (P4.3) uses the same cone
    along a tilted tool direction.
    """
    radial = np.asarray(radial, dtype=np.float64).ravel()
    axial = np.asarray(axial, dtype=np.float64).ravel()
    planar = np.column_stack([radial, axial])

    tan_half = np.tan(np.radians(half_angle_deg))
    apex = np.array([0.0, 0.0])
    rim = np.array([height_mm * tan_half, height_mm])
    top_centre = np.array([0.0, height_mm])

    distance = np.minimum(
        _segment_distance(planar, apex, rim),
        _segment_distance(planar, rim, top_centre),
    )
    inside = (axial <= height_mm) & (radial <= axial * tan_half)
    return np.where(inside, -distance, distance)


class ReferenceClearance(_BodyModel):
    """Nozzle cone plus gantry half-space, sized from a machine profile.

    Parameters
    ----------
    profile
        A `machine_profile.MachineProfile`. Only ``nozzle_to_gantry`` (mm) is
        read: the cone's height and the gantry's level above the nozzle tip.
    half_angle_deg
        The nozzle cone's half-angle. Defaults to Atomizer's own value, the
        one `order_atoms` plans its nozzle-collision avoidance with.

    Bodies
    ------
    ``nozzle``
        The cone, apex at the tip, opening upward, capped at the gantry level.
    ``gantry``
        Everything at or above ``nozzle_to_gantry``, over the whole XY plane.
        A proxy: the real gantry has finite extent (gate M3).
    """

    bodies = ("nozzle", "gantry")

    def __init__(self, profile, half_angle_deg: float = NOZZLE_HALF_ANGLE_DEG):
        if not 0.0 < half_angle_deg < 90.0:
            raise ValueError(f"half_angle_deg must be in (0, 90), got {half_angle_deg}")
        height = float(profile.nozzle_to_gantry)
        if height <= 0.0:
            raise ValueError(f"nozzle_to_gantry must be positive, got {height}")
        self.profile_name = profile.name
        self.half_angle_deg = float(half_angle_deg)
        self.gantry_height_mm = height

    def body_distances(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        local = _head_frame(points, head_xy)
        nozzle = cone_signed_distance(local, self.half_angle_deg, self.gantry_height_mm)
        gantry = self.gantry_height_mm - local[:, 2]
        return np.column_stack([nozzle, gantry])

    def describe(self) -> dict:
        return {
            "model": "reference",
            "profile": self.profile_name,
            "nozzle_half_angle_deg": self.half_angle_deg,
            "gantry_height_mm": self.gantry_height_mm,
        }


# --------------------------------------------------------------------------
# Boxes from a clearance file (format now, numbers at gate M3)
# --------------------------------------------------------------------------


class BoxesClearance(_BodyModel):
    """Axis-aligned boxes, each fixed to the frame or following the head.

    Parameters
    ----------
    boxes
        A sequence of dicts: ``name`` (unique), ``min`` and ``max`` (three
        numbers each, mm) and ``moves_with`` (a list drawn from ``"x"``,
        ``"y"``). Coordinates are world-frame coordinates **with the head at
        X = Y = 0**; a box that moves with ``x`` is shifted by the head's X,
        one with ``y`` by its Y. So a hotend box (``["x", "y"]``) is given
        relative to the nozzle tip, a frame member (``[]``) in machine
        coordinates, and a beam that rides on Y (``["y"]``) in machine X and
        tip-relative Y.
    name
        The machine profile this envelope belongs to.
    status
        ``"verified"`` or ``"PLACEHOLDER"``, as for machine profiles.
    """

    def __init__(self, boxes: Sequence[dict], name: str = "", status: str = "verified"):
        _validate_boxes(boxes, source=name or "boxes")
        self.profile_name = name
        self.status = status
        self.bodies = tuple(str(box["name"]) for box in boxes)
        self._low = np.array([box["min"] for box in boxes], dtype=np.float64)
        self._high = np.array([box["max"] for box in boxes], dtype=np.float64)
        self._follows = np.array(
            [[axis in box.get("moves_with", []) for axis in ("x", "y")] for box in boxes],
            dtype=np.float64,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BoxesClearance":
        """Load ``<name>.clearance.json``. Warns loudly if it is a placeholder."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("name", "status", "boxes"):
            if key not in data:
                raise ValueError(f"{path}: missing required key {key!r}")
        if data["status"] not in machine_profile.VALID_STATUS:
            raise ValueError(
                f"{path}: status must be one of {sorted(machine_profile.VALID_STATUS)}, "
                f"got {data['status']!r}"
            )
        model = cls(data["boxes"], name=str(data["name"]), status=str(data["status"]))
        if data["status"] == machine_profile.STATUS_PLACEHOLDER:
            _warn_placeholder(path)
        return model

    def body_distances(self, points, head_xy=(0.0, 0.0)) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        head = np.asarray(head_xy, dtype=np.float64).reshape(2)
        shift = np.zeros((len(self.bodies), 3))
        shift[:, :2] = self._follows * head
        low, high = self._low + shift, self._high + shift

        centre = 0.5 * (low + high)
        half = 0.5 * (high - low)
        # Exact box SDF: q is the per-axis excess over the half-extent.
        q = np.abs(points[:, None, :] - centre[None]) - half[None]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
        inside = np.minimum(q.max(axis=2), 0.0)
        return outside + inside

    def describe(self) -> dict:
        return {
            "model": "boxes",
            "profile": self.profile_name,
            "status": self.status,
            "boxes": [
                {
                    "name": name,
                    "min": low.tolist(),
                    "max": high.tolist(),
                    "moves_with": [a for a, f in zip(("x", "y"), follows) if f],
                }
                for name, low, high, follows in zip(
                    self.bodies, self._low, self._high, self._follows
                )
            ],
        }


def _validate_boxes(boxes, source: str) -> None:
    if not isinstance(boxes, (list, tuple)) or not boxes:
        raise ValueError(f"{source}: 'boxes' must be a non-empty list")
    names = set()
    for index, box in enumerate(boxes):
        where = f"{source}: box {index}"
        for key in ("name", "min", "max"):
            if key not in box:
                raise ValueError(f"{where}: missing key {key!r}")
        name = str(box["name"])
        if name in names:
            raise ValueError(f"{where}: duplicate name {name!r}")
        names.add(name)
        low = np.asarray(box["min"], dtype=np.float64)
        high = np.asarray(box["max"], dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError(f"{where} ({name}): 'min' and 'max' need three numbers each")
        if not np.all(low < high):
            raise ValueError(f"{where} ({name}): every 'min' must be below its 'max'")
        moves = box.get("moves_with", [])
        if not isinstance(moves, list) or not set(moves) <= VALID_MOVES_WITH:
            raise ValueError(
                f"{where} ({name}): 'moves_with' must be a list drawn from "
                f"{sorted(VALID_MOVES_WITH)}, got {moves!r}"
            )


def _warn_placeholder(path: Path) -> None:
    message = (
        f"Clearance file {path} is marked {machine_profile.STATUS_PLACEHOLDER}: "
        "its boxes have not been measured. Safety checks against it describe a "
        "machine that does not exist. See gate M3."
    )
    warnings.warn(message, machine_profile.PlaceholderProfileWarning, stacklevel=3)
    print("=" * 78, file=sys.stderr)
    print(f"WARNING: {message}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)


# --------------------------------------------------------------------------
# Choosing a model
# --------------------------------------------------------------------------


def clearance_path(profile) -> Path:
    """Where a profile's clearance file lives, whether or not it exists."""
    return machine_profile.CONFIG_DIR / f"{profile.name}{CLEARANCE_SUFFIX}"


def load_clearance(profile=None, path: str | Path | None = None) -> ClearanceModel:
    """The clearance model for a machine.

    Uses ``path`` if given, else ``config/machines/<profile>.clearance.json``
    if it exists, else `ReferenceClearance` built from the profile: the proxy
    the build plan prescribes until gate M3. ``profile`` defaults to the
    active one (``ATOM_MACHINE``).
    """
    profile = machine_profile.load_profile() if profile is None else profile
    candidate = Path(path) if path is not None else clearance_path(profile)
    if path is None and not candidate.is_file():
        return ReferenceClearance(profile)
    model = BoxesClearance.from_json(candidate)
    if model.profile_name != profile.name:
        raise ValueError(
            f"{candidate} describes machine {model.profile_name!r}, "
            f"but profile {profile.name!r} is active"
        )
    return model
