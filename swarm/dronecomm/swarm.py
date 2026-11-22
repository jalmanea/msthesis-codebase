"""Each access drone as an agent that moves and re-points itself to raise its own utility.

Physics, admission and the per-drone search are those of dronecomm.joint (journal Sec. sec:joint).
That solver is already block-coordinate ascent on the global served share: one drone moves, the
others fixed. Read as a game, it is best-response dynamics with the identical-interest utility
U_j = S (every drone values every user) and full information. This module changes the utility.

Association is user-driven: every user attaches to the drone it receives most power from (cell
selection by RSRP), so no operator is needed and an agent cannot assign users to itself.

Utilities of agent j at a candidate pose v = (x, y, z, tx, ty), others fixed. All are admitted
demand (a share of all points); a utility's band use breaks ties (less is better).
    selfish        demand admitted on j's own cell
    window(rho)    demand admitted on the users of j's window: those within horizontal distance rho
                   of j's position when it decides, and those j serves then. Inside one best
                   response this ranks moves exactly as the marginal contribution of j truncated to
                   the window (wonderful-life utility): the two differ by the demand served in the
                   window without j, which j's move does not change.
    global         demand admitted on all users (rho = inf; the team game of dronecomm.joint)

Information (`info_m`): None means the agent predicts a candidate with the true simulator, so only
what it values is local. A radius r means its model is the simulator restricted to the users it can
observe: those within r of it and those served by any drone within r of it (its own and its
neighbours' cells, whose state crosses the inter-drone link). For those users the restriction is
exact while the others stand still: their reports give the power of every drone they hear (RSRP),
so association and SINR follow. What the agent misses is every user it cannot observe: one it
would pull into its cell, or one it would interfere with. Its utility is computed on the observed
users (window and selfish alike); `global` with finite r is the window of the observed users.

Self-sizing fleet (`run_fleet`, `FleetRules`): each drone advertises where its unadmitted demand is
(`beacons`); every agent's search also tries relocating there (exploration), lands when its own
marginal contribution is small, and calls a reserve drone when its cell's unserved demand is large.

Dynamics:
    sequential     one agent at a time in a random order each round (Gauss-Seidel)
    simultaneous   every agent best-responds to the same state, then each applies its move with
                   probability `inertia` (Jacobi; inertia = 1 is fully synchronous)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .joint import (STANDOFF_AZ, STANDOFF_KEEP, STANDOFF_SHORT, T_MAX, Deployment, Physics, _better,
                    _compass, admit, band_need, boresight, cap_tilt, min_enclosing_circle, rx_power,
                    sinr_db)

UTILITIES = ("selfish", "window", "global")
EXPLORE_KEEP = 3                       # beacon starts climbed in one best response
BEACON_CELL_M = 250.0                  # grid on which a drone bins its unserved demand
BEACONS_PER_DRONE = 2
BEACONS_TRIED = 6                      # strongest beacons an agent tries


def strongest(power: np.ndarray) -> np.ndarray:
    """Every user attaches to the drone it receives most power from (ties: lowest index)."""
    return np.argmax(power, axis=0)


def evaluate(phys: Physics, power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(association, admitted share per point, band used per point) of a power matrix (n, M)."""
    n, m = power.shape
    assoc = strongest(power)
    s = power[assoc, np.arange(m)]
    s_db = sinr_db(phys, s, power.sum(axis=0))
    adm, _ = admit(phys, s_db, assoc, n)
    w = band_need(phys, s_db)
    return assoc, adm, np.where(adm > 0.0, adm * np.where(np.isfinite(w), w, 0.0), 0.0)


def served(phys: Physics, users: np.ndarray, dep: Deployment) -> float:
    """Global served share of a deployment under strongest-cell association."""
    return float(np.mean(evaluate(phys, rx_power(phys, dep.pos, dep.bore, users))[1]))


R80_PER_SIGMA = math.sqrt(-2.0 * math.log(0.2))     # 80th percentile of a 2-D error over its per-axis sigma


@dataclass(frozen=True)
class PredictionError:
    """Errors in an agent's model of its candidate poses; the outcome is scored without them.

    Each decision the agent sees every user displaced by a 2-D Gaussian error whose 80th percentile is
    pos_r80_m (the form of 3GPP positioning targets), and predicts its own received power at each user
    off by an independent log-normal gain of gain_sigma_db (shadowing its map does not know). The
    other drones' power at each user stays exact: it comes from the users' reports (RSRP). Errors are
    redrawn at every decision.
    """

    pos_r80_m: float = 0.0
    gain_sigma_db: float = 0.0
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or f"err{self.pos_r80_m:.0f}m{self.gain_sigma_db:g}dB"

    def draw(self, users: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(users as the agent sees them, its linear gain error per user)."""
        seen = users.copy()
        seen[:, :2] += rng.normal(0.0, self.pos_r80_m / R80_PER_SIGMA, (users.shape[0], 2))
        return seen, 10.0 ** (rng.normal(0.0, self.gain_sigma_db, users.shape[0]) / 10.0)


@dataclass(frozen=True)
class Utility:
    kind: str                  # one of UTILITIES
    rho_m: float = math.inf    # window radius (kind 'window')

    def __post_init__(self):
        if self.kind not in UTILITIES:
            raise ValueError(self.kind)

    @property
    def label(self) -> str:
        return f"window{self.rho_m:.0f}" if self.kind == "window" else self.kind


class AgentMove:
    """Agent j's utility as j alone moves (others fixed): a callable on poses for joint._compass."""

    def __init__(self, phys: Physics, users: np.ndarray, power: np.ndarray, j: int, util: Utility,
                 here: np.ndarray, info_m: float | None = None, pos: np.ndarray | None = None,
                 err: PredictionError | None = None, rng: np.random.Generator | None = None):
        self.phys, self.j, self.util = phys, j, util
        self.m = users.shape[0]
        gain = None
        if err is not None:               # the agent's view: displaced users, an unknown gain on its own link
            users, gain = err.draw(users, rng)
        self.view = users
        assoc0 = strongest(power)
        dist = np.hypot(users[:, 0] - here[0], users[:, 1] - here[1])
        if info_m is None:
            vis = np.ones(self.m, dtype=bool)
        else:
            near_drones = np.hypot(pos[:, 0] - here[0], pos[:, 1] - here[1]) <= info_m
            near_drones[j] = True
            vis = (dist <= info_m) | near_drones[assoc0]
        self.cols = np.nonzero(vis)[0]
        self.users, self.power = users[self.cols], power[:, self.cols].copy()
        self.gain = None if gain is None else gain[self.cols]
        if util.kind == "window":
            self.mask = ((dist <= util.rho_m) | (assoc0 == j))[self.cols]
        elif util.kind == "global" and info_m is not None:
            self.mask = np.ones(self.cols.size, dtype=bool)
        else:
            self.mask = None
        self.evals = 0

    def row(self, v: np.ndarray) -> np.ndarray:
        p = rx_power(self.phys, v[None, :3], boresight(v[None, 3:5]), self.users)[0]
        return p if self.gain is None else p * self.gain

    def __call__(self, v: np.ndarray) -> tuple[float, float]:
        self.evals += 1
        if self.cols.size == 0:                 # observes no user: every pose is worth nothing to it
            return 0.0, 0.0
        self.power[self.j] = self.row(v)
        assoc, adm, band = evaluate(self.phys, self.power)
        if self.util.kind == "selfish":
            sel = assoc == self.j
        elif self.mask is not None:
            sel = self.mask
        else:
            return float(np.sum(adm)) / self.m, float(np.sum(band))
        return float(np.sum(adm[sel])) / self.m, float(np.sum(band[sel]))


def best_response(phys: Physics, users: np.ndarray, dep: Deployment, power: np.ndarray, j: int,
                  util: Utility, pointing: bool = True, budget: int = 200,
                  max_step_m: float | None = None, info_m: float | None = None,
                  beacons: np.ndarray | None = None, n_active: int | None = None,
                  err: PredictionError | None = None, rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, tuple, tuple, int]:
    """Agent j's search, as in joint.optimise but on its own utility: (pose, utility before, after, evals).

    max_step_m bounds the horizontal move of one decision (speed x decision period); None = anywhere.
    info_m: the agent's information radius (module docstring); before/after are its own predictions.
    beacons: (K, 2) points of unserved demand to try relocating to (exploration; `run_fleet`), nadir at
    the current altitude, each with a short climb; the EXPLORE_KEEP best starts are climbed.
    n_active: drones in the air, for the initial step size (default: every drone of `dep`).
    err: the agent's prediction errors (PredictionError, drawn from rng); before/after are then its
    erroneous predictions, and it seeds its search from its own users as it sees them.
    """
    n = dep.pos.shape[0] if n_active is None else n_active
    v0 = np.concatenate([dep.pos[j], dep.t[j]])
    lo = np.array([0.0, 0.0, phys.z_min, -T_MAX, -T_MAX])
    hi = np.array([phys.area_m, phys.area_m, phys.z_max, T_MAX, T_MAX])
    if max_step_m is not None:
        lo[:2] = np.maximum(lo[:2], v0[:2] - max_step_m)
        hi[:2] = np.minimum(hi[:2], v0[:2] + max_step_m)
    dims = [0, 1, 2, 3, 4] if pointing else [0, 1, 2]
    step0 = np.array([phys.area_m / math.sqrt(n) / 4.0] * 2 + [2.0 * phys.z_step, 0.2, 0.2])
    if max_step_m is not None:
        step0[:2] = np.minimum(step0[:2], max_step_m / 2.0)
    step_min = np.array([2.0, 2.0, 1.0, 0.01, 0.01])
    mv = AgentMove(phys, users, power, j, util, v0, info_m, dep.pos, err, rng)
    before = mv(v0)
    v, best = v0.copy(), before
    mine = strongest(power) == j
    if np.any(mine):
        c, rad = min_enclosing_circle(mv.view[mine], seed=int(j))
        cand = v.copy()
        cand[:2] = np.clip(c, lo[:2], hi[:2])
        sc = mv(cand)
        if _better(sc, best):
            v, best = cand, sc
        if pointing:
            stand = []
            for dist, az in product((0.5 * rad, rad, 2.0 * rad), STANDOFF_AZ):
                cand = v.copy()
                cand[:2] = np.clip(c - dist * np.array([math.cos(az), math.sin(az)]), lo[:2], hi[:2])
                cand[3:5] = (c - cand[:2]) / cand[2]
                stand.append((mv(cap_tilt(cand)), cand))
            stand.sort(key=lambda q: (-q[0][0], q[0][1]))
            short = [_compass(mv, s, sc, dims, step0, step_min, lo, hi, STANDOFF_SHORT)
                     for s, sc in [(v, best)] + [(cand, sc) for sc, cand in stand[:STANDOFF_KEEP]]]
            v, best = short[0]
            for vv, sc in short[1:]:
                if _better(sc, best):
                    v, best = vv, sc
    if beacons is not None and len(beacons):
        far = []
        for b in beacons:
            cand = np.array([*np.clip(b, lo[:2], hi[:2]), v[2], 0.0, 0.0])
            far.append((mv(cand), cand))
        far.sort(key=lambda q: (-q[0][0], q[0][1]))
        for sc0, cand in far[:EXPLORE_KEEP]:
            vv, sc = _compass(mv, cand, sc0, dims, step0, step_min, lo, hi, STANDOFF_SHORT)
            if _better(sc, best):
                v, best = vv, sc
    v, best = _compass(mv, v, best, dims, step0, step_min, lo, hi, budget)
    return v, before, best, mv.evals


BRUTE_TILTS_DEG = (15.0, 30.0, 45.0, 60.0)    # brute_response's pointing grid, besides nadir
BRUTE_AZIMUTHS = 8
BRUTE_POLISH = 10                              # best grid points climbed


def brute_response(phys: Physics, users: np.ndarray, dep: Deployment, power: np.ndarray, j: int,
                   util: Utility, pointing: bool = True, xy_step_m: float = 50.0, budget: int = 200,
                   info_m: float | None = None) -> tuple[np.ndarray, tuple, tuple, int]:
    """Agent j's best response by exhaustive grid, to check `best_response`: (pose, before, after, evals).

    Every pose on an xy grid of xy_step_m over the whole area x the altitude grid x (nadir and
    BRUTE_TILTS_DEG at BRUTE_AZIMUTHS azimuths if pointing), then the compass search of
    `best_response` from the BRUTE_POLISH best grid points and from the current pose.
    """
    v0 = np.concatenate([dep.pos[j], dep.t[j]])
    lo = np.array([0.0, 0.0, phys.z_min, -T_MAX, -T_MAX])
    hi = np.array([phys.area_m, phys.area_m, phys.z_max, T_MAX, T_MAX])
    dims = [0, 1, 2, 3, 4] if pointing else [0, 1, 2]
    step0 = np.array([xy_step_m / 2.0] * 2 + [phys.z_step, 0.1, 0.1])
    step_min = np.array([2.0, 2.0, 1.0, 0.01, 0.01])
    mv = AgentMove(phys, users, power, j, util, v0, info_m, dep.pos)
    before = mv(v0)
    ts = [np.zeros(2)]
    if pointing:
        ts += [math.tan(math.radians(a)) * np.array([math.cos(b), math.sin(b)])
               for a in BRUTE_TILTS_DEG for b in np.linspace(0.0, 2.0 * np.pi, BRUTE_AZIMUTHS, endpoint=False)]
    g = np.arange(0.0, phys.area_m + 1e-9, xy_step_m)
    top = []                                    # (score, pose) of the BRUTE_POLISH best, unsorted
    for x, y, z in product(g, g, phys.altitude_grid()):
        for t in ts:
            cand = np.array([x, y, z, t[0], t[1]])
            sc = mv(cand)
            if len(top) < BRUTE_POLISH:
                top.append((sc, cand))
            else:
                worst = min(range(BRUTE_POLISH), key=lambda i: (top[i][0][0], -top[i][0][1]))
                if _better(sc, top[worst][0]):
                    top[worst] = (sc, cand)
    v, best = v0, before
    for sc0, cand in [(before, v0)] + top:
        vv, sc = _compass(mv, cand, sc0, dims, step0, step_min, lo, hi, budget)
        if _better(sc, best):
            v, best = vv, sc
    return v, before, best, mv.evals


@dataclass
class RunResult:
    dep: Deployment
    served: float                       # global served share at the end
    rounds: int
    converged: bool                     # a full round in which no agent gained more than tol
    evals: int
    time_s: float
    history: list = field(default_factory=list)      # global served share after each round
    moves: list = field(default_factory=list)        # agents that moved in each round


def run(phys: Physics, users: np.ndarray, start: Deployment, util: Utility, dynamics: str = "sequential",
        pointing: bool = True, max_rounds: int = 20, tol: float = 1e-4, inertia: float = 1.0,
        max_step_m: float | None = None, budget: int = 200, seed: int = 0,
        info_m: float | None = None, err: PredictionError | None = None) -> RunResult:
    """Best-response dynamics of every agent on `util` from `start` (see module docstring).

    err: every agent predicts with PredictionError err, drawn from a stream of its own so that the
    dynamics' random order and inertia match the run without errors.
    """
    if dynamics not in ("sequential", "simultaneous"):
        raise ValueError(dynamics)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    err_rng = None if err is None else np.random.default_rng([seed, 1])
    dep = start.copy()
    n = dep.pos.shape[0]
    power = rx_power(phys, dep.pos, dep.bore, users)
    history = [float(np.mean(evaluate(phys, power)[1]))]
    moves, evals, converged, rounds = [], 0, False, 0
    for rounds in range(1, max_rounds + 1):
        moved = []
        if dynamics == "sequential":
            for j in rng.permutation(n):
                v, before, after, e = best_response(phys, users, dep, power, int(j), util, pointing,
                                                    budget, max_step_m, info_m, err=err, rng=err_rng)
                evals += e
                if after[0] > before[0] + tol:
                    dep.pos[j], dep.t[j] = v[:3], v[3:5]
                    power[j] = rx_power(phys, v[None, :3], boresight(v[None, 3:5]), users)[0]
                    moved.append(int(j))
        else:
            plans, wanted = [], 0
            for j in range(n):
                v, before, after, e = best_response(phys, users, dep, power, j, util, pointing,
                                                    budget, max_step_m, info_m, err=err, rng=err_rng)
                evals += e
                if after[0] > before[0] + tol:
                    wanted += 1
                    if rng.random() < inertia:
                        plans.append((j, v))
            for j, v in plans:
                dep.pos[j], dep.t[j] = v[:3], v[3:5]
                moved.append(j)
            power = rx_power(phys, dep.pos, dep.bore, users)
        history.append(float(np.mean(evaluate(phys, power)[1])))
        moves.append(moved)
        if not moved if dynamics == "sequential" else not wanted:
            converged = True
            break
    return RunResult(dep, history[-1], rounds, converged, evals, time.perf_counter() - t0, history, moves)


def depot_start(phys: Physics, n: int, depot_xy: np.ndarray, seed: int, spread_m: float = 30.0) -> Deployment:
    """Every drone launched from one depot: nadir, lowest altitude, jittered so cells are distinct."""
    rng = np.random.default_rng(seed)
    xy = np.clip(depot_xy + rng.uniform(-spread_m, spread_m, (n, 2)), 0.0, phys.area_m)
    return Deployment(np.column_stack([xy, np.full(n, phys.z_min)]), np.zeros((n, 2)))


# ── Self-sizing fleet: exploration, landing, calling ─────────────────


def kappa(phys: Physics, n_points: int) -> float:
    """One drone's peak capacity as a share of all demand: B se_max / (r_min lam M)."""
    return phys.bandwidth_hz * phys.se_max / (phys.r_min_bps * phys.lam * n_points)


def beacons(users: np.ndarray, assoc: np.ndarray, adm: np.ndarray, active: np.ndarray,
            cell_m: float = BEACON_CELL_M, per_drone: int = BEACONS_PER_DRONE) -> tuple[np.ndarray, np.ndarray]:
    """What each drone advertises: its unserved demand, binned on a grid, as (centroids (K, 2), mass (K,)).

    A drone knows which of its users it could not admit (GBR admission control), so the beacons
    need no information beyond each cell's own. Mass is in points (1 - admitted share, summed).
    Returns the per_drone heaviest bins of every active drone, heaviest first.
    """
    miss = 1.0 - adm
    cent, mass = [], []
    for j in np.nonzero(active)[0]:
        sel = (assoc == j) & (miss > 1e-9)
        if not np.any(sel):
            continue
        xy, w = users[sel, :2], miss[sel]
        keys = np.floor(xy / cell_m).astype(int)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        m = np.bincount(inv, weights=w)
        for b in np.argsort(-m)[:per_drone]:
            pts = inv == b
            cent.append(np.average(xy[pts], axis=0, weights=w[pts]))
            mass.append(m[b])
    if not cent:
        return np.zeros((0, 2)), np.zeros(0)
    order = np.argsort(-np.array(mass))
    return np.array(cent)[order], np.array(mass)[order]


@dataclass(frozen=True)
class FleetRules:
    """Exploration and the land / call rules (thresholds in units of kappa, one drone's peak capacity).

    land: an agent lands when its marginal contribution, measured by its own utility (the utility
          with it in the air minus the same utility with it landed), is below land_alpha * kappa.
          For `selfish` that is the demand its own cell serves.
    call: an agent launches a reserve drone at its heaviest beacon when its own cell's unserved
          demand exceeds call_alpha * kappa and the reserve is not empty. With call_area_m, it counts
          the unserved demand of every user within call_area_m of it (whichever drone they attach to)
          and sends the reserve to the heaviest beacon of the drones within that radius: unserved
          demand spread thinly over several cells can then add up to a call.
          With call_check, the caller first predicts the reserve drone's marginal contribution at the
          beacon (by the caller's utility kind, nadir at the caller's altitude) and calls only if it
          clears the landing threshold, so it does not call a drone that would land at once.
    hysteresis: with call_patience = K, an agent calls only once the call condition has held on K
          consecutive turns of its own; with land_grace = H, a called drone may not land in the H
          rounds after its launch, so it has time to move before it is judged.
    None disables a rule. A drone launched in a round acts from the next round on.
    """

    explore: bool = True
    land_alpha: float | None = None
    call_alpha: float | None = None
    call_area_m: float | None = None
    call_check: bool = False
    call_patience: int = 1
    land_grace: int = 0

    @property
    def label(self) -> str:
        f = lambda a: "off" if a is None else f"{a:g}"
        area = "" if self.call_area_m is None else f"_area{self.call_area_m:.0f}"
        chk = "_chk" if self.call_check else ""
        hyst = (f"_pat{self.call_patience}" if self.call_patience > 1 else "") + \
               (f"_gr{self.land_grace}" if self.land_grace > 0 else "")
        return f"{'x' if self.explore else 'nox'}_land{f(self.land_alpha)}_call{f(self.call_alpha)}{area}{chk}{hyst}"


def contribution(mv: AgentMove, v: np.ndarray) -> float:
    """Agent mv.j's marginal contribution at pose v by its own utility (its utility in the air minus landed)."""
    up = mv(v)[0]
    keep = mv.power[mv.j].copy()
    mv.power[mv.j] = 0.0
    assoc, adm, _ = evaluate(mv.phys, mv.power)
    if mv.util.kind == "selfish":
        down = 0.0
    elif mv.mask is not None:
        down = float(np.sum(adm[mv.mask])) / mv.m
    else:
        down = float(np.sum(adm)) / mv.m
    mv.power[mv.j] = keep
    return up - down


def call_target(users: np.ndarray, pos: np.ndarray, active: np.ndarray, assoc: np.ndarray, adm: np.ndarray,
                j: int, rules: FleetRules, k_cap: float) -> np.ndarray | None:
    """Where agent j sends a reserve drone under `rules` (FleetRules, call), or None if it does not call."""
    if rules.call_area_m is None:
        mine, near = assoc == j, np.arange(len(active)) == j
    else:
        mine = np.hypot(users[:, 0] - pos[j, 0], users[:, 1] - pos[j, 1]) <= rules.call_area_m
        near = active & (np.hypot(*(pos[:, :2] - pos[j, :2]).T) <= rules.call_area_m)
    if float(np.sum(1.0 - adm[mine])) / users.shape[0] <= rules.call_alpha * k_cap:
        return None
    c = beacons(users, assoc, adm, near, per_drone=1)[0]
    return c[0] if len(c) else None


@dataclass
class FleetResult:
    dep: Deployment                     # every drone, landed ones included
    active: np.ndarray                  # in the air at the end
    served: float
    rounds: int
    converged: bool                     # a round with no move, landing or call
    evals: int
    time_s: float
    history: list = field(default_factory=list)      # (served share, drones in the air) after each round
    events: list = field(default_factory=list)       # (round, 'land' | 'call', drone)


def run_fleet(phys: Physics, users: np.ndarray, start: Deployment, n_start: int, util: Utility,
              rules: FleetRules, pointing: bool = True, max_rounds: int = 40, tol: float = 1e-4,
              budget: int = 200, seed: int = 0, depot_xy: np.ndarray | None = None) -> FleetResult:
    """Sequential best response with exploration and the land / call rules (exact prediction).

    start holds every drone the operator owns; the first n_start are in the air, the rest wait at
    depot_xy (default: where start puts them). Landed drones rejoin the reserve and may be called again.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dep = start.copy()
    total = dep.pos.shape[0]
    active = np.zeros(total, dtype=bool)
    active[:n_start] = True
    if depot_xy is not None:
        dep.pos[~active, :2] = depot_xy
    k_cap = kappa(phys, users.shape[0])

    def power_of(dep, active):
        p = rx_power(phys, dep.pos, dep.bore, users)
        p[~active] = 0.0
        return p

    power = power_of(dep, active)
    history = [(float(np.mean(evaluate(phys, power)[1])), int(active.sum()))]
    events, evals, converged, rounds = [], 0, False, 0
    launched = np.full(total, -total - rules.land_grace)       # round each drone was last called
    streak = np.zeros(total, dtype=int)                        # consecutive turns the call condition held
    for rounds in range(1, max_rounds + 1):
        changed = False
        for j in rng.permutation(np.nonzero(active)[0]):
            j = int(j)
            if not active[j]:
                continue
            bc = None
            if rules.explore:
                assoc, adm, _ = evaluate(phys, power)
                bc = beacons(users, assoc, adm, active)[0][:BEACONS_TRIED]
            v, before, after, e = best_response(phys, users, dep, power, j, util, pointing, budget,
                                                beacons=bc, n_active=int(active.sum()))
            evals += e
            if after[0] > before[0] + tol:
                dep.pos[j], dep.t[j] = v[:3], v[3:5]
                power[j] = rx_power(phys, v[None, :3], boresight(v[None, 3:5]), users)[0]
                changed = True
            here = np.concatenate([dep.pos[j], dep.t[j]])
            if rules.land_alpha is not None and active.sum() > 1 and rounds - launched[j] > rules.land_grace:
                mv = AgentMove(phys, users, power, j, util, dep.pos[j])
                if contribution(mv, here) < rules.land_alpha * k_cap:
                    active[j] = False
                    power[j] = 0.0
                    streak[j] = 0
                    events.append((rounds, "land", j))
                    changed = True
                    continue
            if rules.call_alpha is not None and not active.all():
                c = call_target(users, dep.pos, active, *evaluate(phys, power)[:2], j, rules, k_cap)
                r = int(np.nonzero(~active)[0][0])
                streak[j] = 0 if c is None else streak[j] + 1
                if c is not None and streak[j] < rules.call_patience:
                    c = None
                if c is not None:
                    v = np.array([*np.clip(c, 0.0, phys.area_m), dep.pos[j, 2], 0.0, 0.0])
                    if rules.call_check and rules.land_alpha is not None:
                        trial = power.copy()
                        trial[r] = rx_power(phys, v[None, :3], boresight(v[None, 3:5]), users)[0]
                        mv = AgentMove(phys, users, trial, r, util, v[:3])
                        if contribution(mv, v) < rules.land_alpha * k_cap:
                            c = None
                if c is not None:
                    dep.pos[r], dep.t[r] = v[:3], v[3:5]
                    active[r] = True
                    launched[r], streak[j], streak[r] = rounds, 0, 0
                    power[r] = rx_power(phys, dep.pos[r][None], dep.bore[r][None], users)[0]
                    events.append((rounds, "call", r))
                    changed = True
        history.append((float(np.mean(evaluate(phys, power)[1])), int(active.sum())))
        # a call waiting out its patience, or a drone still in its grace, is not a fixed point yet
        pending = (not active.all() and bool(np.any(streak[active] > 0))) or \
                  bool(np.any(active & (rounds - launched <= rules.land_grace)))
        if not changed and not pending:
            converged = True
            break
    return FleetResult(dep, active, history[-1][0], rounds, converged, evals, time.perf_counter() - t0,
                       history, events)
