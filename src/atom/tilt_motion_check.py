"""The swept check: every machine state between the toolpath points (build plan P4.2).

`kinematics3z.inverse` checks each toolpath point on its own, and P4.3 checks
the nozzle at each point against the material printed before it. Neither sees
what happens *between* points. The firmware moves all five axes linearly from
one state to the next, so during a move that changes the tilt, and during a
travel, the bed passes through states nobody has checked. This module walks
those states and asks, for each one:

* ``part_vs_<body>``: does the part printed so far reach a machine body
  (the gantry, for the reference proxy)?
* ``bed_vs_<body>``: does a bed corner?
* ``nozzle_vs_material``: is printed material inside the nozzle cone? (This
  is P4.3's test, applied between the points; P4.3 covers the points.)
* ``nozzle_below_bed``: is the nozzle tip below the bed surface?

Not yet checked here: axis ranges and the tilt limit between points. By the
operator's decision (2026-09-23) those come from the G-code validator (P1.4)
once it is merged into ``main``, so the rules are written once. Every report
lists them under ``not_checked``.

Method
------
1. Solve the toolpath with `contracts.from_toolpath(..., center_on_bed=True)`,
   so the machine states are those the written G-code will command (hazard 7).
   The bed frame is then the re-centred part frame.
2. States: every reachable toolpath point, plus interior states of each move
   that changes the tool direction by more than ``check_tilt_step_deg``, and
   of every travel. A move from state ``a`` to ``b`` is split into ``n`` equal
   steps, ``m(s) = (1 - s) a + s b`` for ``s = k / n`` (linear in the five
   axes, as the firmware moves), with ``n`` set by the direction change and,
   for travel, the distance. Moves touching an unreachable point are skipped
   and counted.
3. Each state's bed pose comes from the vendored forward kinematics through
   `atom.bed_motion` (three `forward` probes), and the nozzle tip and axis in
   the bed frame from the same pose.
4. The part printed so far is represented two ways, by the operator's choice:
   * **against the machine bodies** (the gantry): the vertices of its convex
     hull, rebuilt every ``hull_every`` material points, plus the raw material
     laid since. A convex solid's highest point is a vertex, so against the
     flat gantry this is exact, not an approximation.
   * **against the nozzle**: the real material points (optionally thinned),
     via `nozzle_material_check.cone_hits`. The hull would fill hollows and
     report the nozzle inside material whenever it works in one.

Material timing follows P4.3: during move ``i`` (from point ``i - 1`` to
``i``) the material from moves up to ``i - 1`` exists; move ``i``'s own bead is
under the nozzle.

Units: millimetres and degrees at every public boundary. The kinematics run in
float32, so heights carry about 1e-3 mm of rounding; ``tolerance_mm`` absorbs
it.
"""

from __future__ import annotations  # no Taichi kernels are defined in this module

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import bed_motion, clearance, contracts, machine_profile
from . import nozzle_material_check as nm
from . import overhang_metrics as om

#: Checks this module does not perform yet, and where they will come from.
NOT_CHECKED = {
    "axis_range": "P1.4 G-code validator, once merged into main",
    "tilt_limit": "P1.4 G-code validator, once merged into main",
}

#: What each kind of violation means, for reports.
KINDS = {
    "part": "the part printed so far reaches a machine body",
    "bed": "a bed corner reaches a machine body",
    "nozzle_vs_material": "printed material is inside the nozzle cone",
    "nozzle_below_bed": "the nozzle tip is below the bed surface",
}


@dataclass
class SweptCheckSettings:
    """Parameters of the swept check. Lengths in mm, angles in degrees."""

    #: Interior states are placed so the tool direction changes by at most
    #: this much between them (build plan P4.2's default).
    check_tilt_step_deg: float = 0.5
    #: Travel moves are also split so the nozzle moves at most this far
    #: between states: about half a bead width.
    travel_step_mm: float = 0.5
    #: Material points between rebuilds of the part's convex hull.
    hull_every: int = 500
    #: A violation must be deeper than this; absorbs float32 rounding.
    tolerance_mm: float = 0.01
    #: Voxel size for thinning material in the nozzle check; None = every point.
    material_subsample_mm: float | None = None


@dataclass
class States:
    """The machine states the check visits, in print order."""

    #: ``(S, 5)`` X, Y, Z, U, V.
    machine: np.ndarray
    #: ``(S,)`` the move each state belongs to: move ``i`` runs from point
    #: ``i - 1`` to point ``i``; the state at point ``i`` itself has move ``i``.
    move: np.ndarray
    #: ``(S,)`` position along the move, in (0, 1]; 1 is the toolpath point.
    fraction: np.ndarray

    @property
    def interior(self) -> np.ndarray:
        return self.fraction < 1.0

    @property
    def count(self) -> int:
        return len(self.move)


@dataclass
class SweptResult:
    """What `check` found. One violation row per (move, kind, body).

    ``violations`` rows hold: ``move`` (toolpath index the move ends at),
    ``fraction`` (where along it the worst state is), ``kind``, ``body``,
    ``clearance_mm`` (signed; negative = inside), ``point_bed`` (the
    offending point in the bed frame, mm), ``tilt_deg`` (at that state) and
    ``deposit`` (whether the move prints).
    """

    points: int
    states: int
    interior_states: int
    interpolated_moves: int
    skipped_moves: int
    min_part_clearance_mm: float
    min_bed_clearance_mm: float
    violations: list = field(default_factory=list)
    settings: SweptCheckSettings = None
    model: dict = None

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self, limit: int | None = 100) -> dict:
        worst = sorted(self.violations, key=lambda row: row["clearance_mm"])
        listed = worst if limit is None else worst[:limit]
        by_kind = {}
        for row in self.violations:
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        return {
            "check": "swept",
            "ok": self.ok,
            "points": self.points,
            "states": self.states,
            "interior_states": self.interior_states,
            "interpolated_moves": self.interpolated_moves,
            "skipped_moves_unreachable": self.skipped_moves,
            "min_part_clearance_mm": _finite(self.min_part_clearance_mm),
            "min_bed_clearance_mm": _finite(self.min_bed_clearance_mm),
            "violations": len(self.violations),
            "violations_by_kind": by_kind,
            "not_checked": dict(NOT_CHECKED),
            "settings": asdict(self.settings) if self.settings else None,
            "clearance_model": self.model,
            "worst": [_rounded(row) for row in listed],
        }


def _finite(value):
    return None if not np.isfinite(value) else round(float(value), 4)


def _rounded(row: dict) -> dict:
    out = dict(row)
    out["clearance_mm"] = round(float(row["clearance_mm"]), 4)
    out["fraction"] = round(float(row["fraction"]), 4)
    out["tilt_deg"] = round(float(row["tilt_deg"]), 3)
    out["point_bed"] = [round(float(v), 4) for v in row["point_bed"]]
    return out


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, row by row, degrees (robust near zero)."""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))


def interior_counts(points, directions, deposit, valid, settings: SweptCheckSettings):
    """Steps per move: ``n[i]`` for the move to point ``i`` (0 = not split).

    A move is split into ``n >= 2`` steps when it changes the tool direction
    by more than ``check_tilt_step_deg`` or is a travel longer than
    ``travel_step_mm``; ``n - 1`` interior states result. Moves touching an
    unreachable point are not split (they are skipped).
    """
    count = len(points)
    steps = np.zeros(count, dtype=np.int64)
    if count < 2:
        return steps
    turn = angle_between_deg(directions[:-1], directions[1:])
    length = np.linalg.norm(points[1:] - points[:-1], axis=1)
    by_turn = np.ceil(turn / settings.check_tilt_step_deg - 1e-9).astype(np.int64)
    by_travel = np.where(deposit[1:], 0,
                         np.ceil(length / settings.travel_step_mm - 1e-9).astype(np.int64))
    n = np.maximum(by_turn, by_travel)
    reachable = valid[:-1] & valid[1:]
    steps[1:] = np.where(reachable & (n >= 2), n, 0)
    return steps


def build_states(machine, valid, steps) -> States:
    """Every reachable point's state, plus the interior states of split moves."""
    moves, fractions, rows = [], [], []
    for i in np.flatnonzero(steps):
        n = int(steps[i])
        s = np.arange(1, n) / n
        moves.append(np.full(n - 1, i))
        fractions.append(s)
        rows.append((1.0 - s)[:, None] * machine[i - 1] + s[:, None] * machine[i])
    reachable = np.flatnonzero(valid)
    moves.append(reachable)
    fractions.append(np.ones(len(reachable)))
    rows.append(machine[reachable])

    move = np.concatenate(moves) if moves else np.zeros(0, np.int64)
    fraction = np.concatenate(fractions) if fractions else np.zeros(0)
    state = np.vstack(rows) if rows else np.zeros((0, 5))
    order = np.lexsort((fraction, move))
    return States(machine=state[order], move=move[order].astype(np.int64),
                  fraction=fraction[order])


def forward(machine: np.ndarray):
    """Nozzle tip positions and build directions in the bed frame, from
    machine states, via the vendored `kinematics3z.forward`. Requires Taichi
    to be initialised (`atom.ti_env.init_taichi`)."""
    from . import kinematics3z

    machine = np.ascontiguousarray(np.asarray(machine, dtype=np.float32))
    points = np.zeros((len(machine), 3), dtype=np.float32)
    orientation = np.zeros((len(machine), 2), dtype=np.float32)
    if len(machine):
        kinematics3z.toolpath_to_cartesian_toolpath(machine, points, orientation)
    return points.astype(np.float64), om.spherical_to_cartesian(orientation.astype(np.float64))


def poses(machine: np.ndarray):
    """Bed poses ``(R, t)`` for machine states (`atom.bed_motion`) and the
    nozzle tip in the bed frame."""
    machine = np.asarray(machine, dtype=np.float64)
    probed, _ = forward(bed_motion.probe_states(machine))
    rotation, translation = bed_motion.poses_from_probes(machine, probed)
    return rotation, translation, probed[: len(machine)]


# --------------------------------------------------------------------------
# The part printed so far, as hull vertices plus recent material
# --------------------------------------------------------------------------


def _hull_vertices(points: np.ndarray) -> np.ndarray:
    """Indices of the convex hull's vertices, or all points if it is flat."""
    if len(points) < 5:
        return np.arange(len(points))
    try:
        return ConvexHull(points, qhull_options="QJ").vertices
    except (QhullError, ValueError):
        return np.arange(len(points))


class _PartProxy:
    """The material visible at each time, reduced to hull vertices + recent points."""

    def __init__(self, points, material_time, hull_every):
        self.points = points
        known = np.flatnonzero(material_time != nm._NEVER)
        self.order = known[np.argsort(material_time[known], kind="stable")]
        self.times = material_time[self.order]
        self.every = max(1, int(hull_every))
        self._hulls = {0: np.zeros(0, dtype=np.int64)}

    def visible_count(self, time):
        return np.searchsorted(self.times, time, side="right")

    def hull(self, epoch: int) -> np.ndarray:
        """Point indices of the hull vertices of the first ``epoch * every``
        material points, built on the previous epoch's vertices."""
        if epoch not in self._hulls:
            previous = self.hull(epoch - 1)
            new = self.order[(epoch - 1) * self.every: epoch * self.every]
            candidates = np.concatenate([previous, new])
            self._hulls[epoch] = candidates[_hull_vertices(self.points[candidates])]
        return self._hulls[epoch]


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def check(machine_toolpath: contracts.MachineToolpath, profile=None, model=None,
          settings: SweptCheckSettings | None = None) -> SweptResult:
    """Walk every machine state and test it. See the module docstring.

    Parameters
    ----------
    machine_toolpath
        From `contracts.from_toolpath(..., center_on_bed=True)`: its ``point``
        is then the bed frame the poses are in.
    profile
        The machine; defaults to the active one. Sizes the bed and nozzle.
    model
        A `clearance.ClearanceModel`; defaults to `clearance.load_clearance`.
    settings
        See `SweptCheckSettings`.

    Requires Taichi to be initialised: the forward kinematics run for every
    state.
    """
    settings = settings or SweptCheckSettings()
    profile = machine_profile.load_profile() if profile is None else profile
    model = clearance.load_clearance(profile) if model is None else model
    mt = machine_toolpath

    points = np.asarray(mt.point, dtype=np.float64)
    directions = om.spherical_to_cartesian(np.asarray(mt.tool_orientation))
    deposit = np.asarray(mt.travel_type) == om.TRAVEL_TYPE_DEPOSITION
    valid = np.asarray(mt.valid, dtype=bool) & np.isfinite(mt.machine).all(axis=1)
    count = len(points)

    steps = interior_counts(points, directions, deposit, valid, settings)
    states = build_states(np.asarray(mt.machine, dtype=np.float64), valid, steps)
    skipped = int(np.count_nonzero(~(valid[:-1] & valid[1:]))) if count > 1 else 0

    rotation, translation, tip = poses(states.machine)
    axis = bed_motion.realised_tool_direction(rotation)
    head = states.machine[:, :2]
    tilt = om.tilt_from_vertical_deg(axis)
    time = nm.material_time(deposit)
    nozzle_time = states.move - 1

    found = []
    threshold = -float(settings.tolerance_mm)
    all_bodies = list(model.bodies)
    # The nozzle is checked against real material below, never the hull.
    machine_bodies = [name for name in all_bodies if name != "nozzle"]

    def record_worst(state_index, distance, where, kind, bodies):
        """Keep, per state and body, the deepest of ``distance (S, P, B)``."""
        flat = distance.reshape(len(state_index), -1, len(bodies))
        deepest = flat.argmin(axis=1)
        value = np.take_along_axis(flat, deepest[:, None, :], axis=1)[:, 0, :]
        for s, b in zip(*np.nonzero(value < threshold)):
            found.append((int(state_index[s]), kind, bodies[b],
                          float(value[s, b]), where[s, deepest[s, b]]))
        return float(value.min()) if value.size else np.inf

    # -- bed corners against every body -----------------------------------
    corners = bed_motion.bed_corners(profile)
    min_bed = np.inf
    for chunk in _chunks(states.count, _STATE_CHUNK):
        world = _to_world(corners, rotation[chunk], translation[chunk])
        distance = model.body_distances(
            world.reshape(-1, 3), np.repeat(head[chunk], len(corners), axis=0), all_bodies
        ).reshape(len(chunk), len(corners), -1)
        where = np.broadcast_to(corners, (len(chunk),) + corners.shape)
        min_bed = min(min_bed, record_worst(chunk, distance, where, "bed", all_bodies))

    # -- the part so far against the machine bodies ------------------------
    proxy = _PartProxy(points, time, settings.hull_every)
    visible = proxy.visible_count(nozzle_time)
    epoch = visible // proxy.every
    min_part = np.inf
    for e in np.unique(epoch):
        hull = proxy.hull(int(e))
        in_epoch = np.flatnonzero(epoch == e)
        for members in (in_epoch[c] for c in _chunks(len(in_epoch), _STATE_CHUNK)):
            recent = proxy.order[int(e) * proxy.every: int(visible[members].max())]
            candidates = np.concatenate([hull, recent])
            if not len(candidates) or not machine_bodies:
                continue
            world = _to_world(points[candidates], rotation[members], translation[members])
            distance = model.body_distances(
                world.reshape(-1, 3), np.repeat(head[members], len(candidates), axis=0),
                machine_bodies,
            ).reshape(len(members), len(candidates), -1)
            # Recent material only exists once it has been printed.
            unseen = np.zeros((len(members), len(candidates)), dtype=bool)
            unseen[:, len(hull):] = time[recent][None, :] > nozzle_time[members][:, None]
            distance[unseen] = np.inf
            where = np.broadcast_to(points[candidates], (len(members), len(candidates), 3))
            min_part = min(min_part, record_worst(members, distance, where, "part",
                                                  machine_bodies))

    # -- between the points: nozzle vs material, nozzle vs bed -------------
    interior = np.flatnonzero(states.interior)
    if len(interior):
        nozzle_settings = nm.NozzleCheckSettings.for_profile(
            profile, subsample_mm=settings.material_subsample_mm,
            tolerance_mm=settings.tolerance_mm)
        which, blocker, depth, _ = nm.cone_hits(
            tip[interior] + nozzle_settings.apex_offset_mm * axis[interior],
            axis[interior], nozzle_time[interior], points, time, nozzle_settings)
        if len(which):
            order = np.lexsort((-depth, which))
            first = np.ones(len(order), dtype=bool)
            first[1:] = which[order][1:] != which[order][:-1]
            for w, k, d in zip(which[order][first], blocker[order][first],
                               depth[order][first]):
                found.append((int(interior[w]), "nozzle_vs_material", "nozzle",
                              -float(d), points[k]))

        for s in interior[tip[interior, 2] < threshold]:
            found.append((int(s), "nozzle_below_bed", "bed", float(tip[s, 2]), tip[s]))

    return SweptResult(
        points=count,
        states=states.count,
        interior_states=int(len(interior)),
        interpolated_moves=int(np.count_nonzero(steps)),
        skipped_moves=skipped,
        min_part_clearance_mm=min_part,
        min_bed_clearance_mm=min_bed,
        violations=_worst_per_move(found, states, tilt, deposit),
        settings=settings,
        model=model.describe(),
    )


#: States scored per batch, to bound memory.
_STATE_CHUNK = 2048


def _to_world(points, rotation, translation):
    """Bed-frame points ``(P, 3)`` under poses ``(S, 3, 3)``, ``(S, 3)``: ``(S, P, 3)``."""
    return np.matmul(points[None], np.transpose(rotation, (0, 2, 1))) + translation[:, None, :]


def _chunks(count: int, size: int):
    for start in range(0, count, size):
        yield np.arange(start, min(start + size, count))


def _worst_per_move(found, states: States, tilt, deposit) -> list:
    """One row per (move, kind, body): the deepest state of that move."""
    worst = {}
    for state_index, kind, body, clearance_mm, point in found:
        key = (int(states.move[state_index]), kind, body)
        if key not in worst or clearance_mm < worst[key][2]:
            worst[key] = (state_index, point, clearance_mm)
    rows = []
    for (move, kind, body), (state_index, point, clearance_mm) in sorted(worst.items()):
        rows.append({
            "move": move,
            "fraction": float(states.fraction[state_index]),
            "kind": kind,
            "body": body,
            "clearance_mm": clearance_mm,
            "point_bed": np.asarray(point, dtype=np.float64).tolist(),
            "tilt_deg": float(tilt[state_index]),
            "deposit": bool(deposit[move]),
        })
    return rows


def check_toolpath(toolpath, profile=None, model=None, **overrides) -> SweptResult:
    """`check` on a `toolpath3.Toolpath` (or the same arrays): solves it
    re-centred on the bed first, as `toolpath_to_gcode` does. Requires
    Taichi to be initialised."""
    profile = machine_profile.load_profile() if profile is None else profile
    solved = contracts.from_toolpath(toolpath, profile, center_on_bed=True)
    return check(solved, profile, model, SweptCheckSettings(**overrides))
