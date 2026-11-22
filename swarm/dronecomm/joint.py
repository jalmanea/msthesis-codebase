"""Joint placement and antenna pointing of N access drones (journal Sec. sec:joint).

Input: ground user points u_k (M, 3), each standing for lam active users.
Decision, per drone j: position p_j = (x, y, z) and access boresight b_j, a unit
vector in the lower hemisphere. Objective: the served share of active demand
under guaranteed-bit-rate (GBR) admission at r_min, the metric of Sec. sec:capacity.

    maximise   S(p, b, a) = mean_k admitted_k
    subject to 0 <= x_j, y_j <= L,  z_min <= z_j <= z_max,  b_j,z < 0

with a the user-drone association: `nearest` (3-D distance, the paper's rule),
`strongest` (received power, users' cell selection; dronecomm.swarm) or `free`
(chosen by the operator; a linear program per geometry, see `lp_association`). Interference is full-buffer: every drone transmits on the
whole band, so a user's SINR towards drone j does not depend on who else is
associated with j.

The boresight is parametrised by t = (tx, ty), b = (tx, ty, -1) / |(tx, ty, -1)|:
nadir is t = 0, the tilt from nadir is atan|t|, and there is no wrap-around or
singularity at nadir as with (tilt, azimuth).

The physics is written out here rather than imported (antenna pattern of
TR 38.901 form, Al-Hourani air-to-ground path loss, truncated Shannon spectral
efficiency of TR 36.942, GBR admission best SINR first); `scripts/run_joint.py
--stage check` verifies it against dronecomm.interference / dronecomm.capacity.

Method (`optimise`): from a starting deployment, rounds of
    1. association (nearest or strongest, or the LP for `free`),
    2. per drone, the minimum-enclosing-circle centre of the users it serves as a
       candidate, then a compass search over (x, y, z) or (x, y, z, tx, ty),
       others fixed, on the exact served share (ties: less band used); with
       pointing free, short climbs first from the current point and the six
       best standoff starts (positions off the enclosing circle with the beam
       aimed at its centre), then the full climb from the best of them,
until a round gains nothing. Only exact improvements are accepted, so the
result never serves less than the starting deployment.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .channel import ENVIRONMENTS
from .config import Config

C_LIGHT = 3.0e8
T_MAX = math.tan(math.radians(80.0))   # largest tilt the search may reach
STANDOFF_AZ = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
STANDOFF_KEEP = 6                      # standoff starts climbed besides the current point
STANDOFF_SHORT = 60                    # evaluations of each short climb


@dataclass(frozen=True)
class Physics:
    """Every number the objective needs, from a Config and the number of user points."""

    p_tx_w: float
    g_max_dbi: float
    beamwidth_rad: float
    sla_db: float
    los_a: float
    los_b: float
    eta_los_db: float
    eta_nlos_db: float
    f_c: float
    noise_w: float
    bandwidth_hz: float
    se_alpha: float
    se_sinr_min_db: float
    se_max: float
    r_min_bps: float
    lam: float
    area_m: float
    z_min: float
    z_max: float
    z_step: float

    @classmethod
    def from_config(cls, cfg: Config, n_points: int) -> Physics:
        if cfg.capacity.max_users_per_cell is not None:
            raise ValueError("joint placement does not model a per-cell user cap")
        net, ant, cap, env = cfg.network, cfg.antenna, cfg.capacity, ENVIRONMENTS[cfg.channel.environment]
        return cls(
            p_tx_w=10.0 ** ((net.p_tx_dl_dbm - 30.0) / 10.0),
            g_max_dbi=ant.dl_g_max_dbi, beamwidth_rad=math.radians(ant.dl_beamwidth_deg),
            sla_db=ant.dl_sla_db, los_a=env.a, los_b=env.b,
            eta_los_db=env.eta_los_db, eta_nlos_db=env.eta_nlos_db, f_c=cfg.channel.carrier_freq_hz,
            noise_w=10.0 ** ((net.noise_dbm_hz - 30.0) / 10.0) * net.bandwidth_hz,
            bandwidth_hz=net.bandwidth_hz, se_alpha=cap.se_alpha, se_sinr_min_db=cap.se_sinr_min_db,
            se_max=cap.se_max_bps_hz, r_min_bps=cfg.demand.r_min_bps,
            lam=cfg.demand.active_per_point(n_points), area_m=cfg.scenario.area_size_m,
            z_min=net.z_min_m, z_max=net.z_max_m, z_step=net.z_step_m)

    def altitude_grid(self) -> np.ndarray:
        n = int(round((self.z_max - self.z_min) / self.z_step)) + 1
        return self.z_min + self.z_step * np.arange(n)


# ── Physics ──────────────────────────────────────────────────────────


def cap_tilt(v: np.ndarray) -> np.ndarray:
    """Scale the tilt coordinates v[3:5] back to the largest tilt if they exceed it (in place)."""
    r = math.hypot(v[3], v[4])
    if r > T_MAX:
        v[3:5] *= T_MAX / r
    return v


def boresight(t: np.ndarray) -> np.ndarray:
    """Unit boresight(s) from tilt coordinates t = (tx, ty), shape (..., 2) -> (..., 3)."""
    t = np.asarray(t, dtype=np.float64)
    v = np.concatenate([t, -np.ones(t.shape[:-1] + (1,))], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tilt_deg(t: np.ndarray) -> np.ndarray:
    """Tilt from nadir [deg] of tilt coordinates t, shape (..., 2)."""
    return np.degrees(np.arctan(np.linalg.norm(np.asarray(t, dtype=np.float64), axis=-1)))


def rx_power(phys: Physics, pos: np.ndarray, bore: np.ndarray, users: np.ndarray) -> np.ndarray:
    """Received power [W] at every user from every drone, shape (n, M)."""
    pos = np.atleast_2d(pos)
    bore = np.atleast_2d(bore)
    diff = users[None, :, :] - pos[:, None, :]                       # drone -> user
    dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-6)
    cos = np.clip(np.einsum("nmk,nk->nm", diff, bore) / dist, -1.0, 1.0)
    theta = np.arccos(cos)
    gain_db = phys.g_max_dbi - np.minimum(12.0 * (theta / phys.beamwidth_rad) ** 2, phys.sla_db)
    horiz = np.maximum(np.hypot(diff[..., 0], diff[..., 1]), 1e-6)
    elev_deg = np.degrees(np.arctan2(np.abs(diff[..., 2]), horiz))
    p_los = 1.0 / (1.0 + phys.los_a * np.exp(-phys.los_b * (elev_deg - phys.los_a)))
    pl_db = (20.0 * np.log10(4.0 * np.pi * dist * phys.f_c / C_LIGHT)
             + p_los * phys.eta_los_db + (1.0 - p_los) * phys.eta_nlos_db)
    return phys.p_tx_w * 10.0 ** (gain_db / 10.0) * 10.0 ** (-pl_db / 10.0)


def sinr_db(phys: Physics, signal: np.ndarray, total: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(signal / (phys.noise_w + (total - signal)), 1e-30))


def spectral_efficiency(phys: Physics, s_db: np.ndarray) -> np.ndarray:
    """Truncated Shannon bound [bit/s/Hz] (TR 36.942 A.1)."""
    se = phys.se_alpha * np.log2(1.0 + 10.0 ** (s_db / 10.0))
    return np.where(s_db < phys.se_sinr_min_db, 0.0, np.minimum(se, phys.se_max))


def band_need(phys: Physics, s_db: np.ndarray) -> np.ndarray:
    """Band share one point's lam active users need at r_min; inf if one user cannot fit."""
    se = spectral_efficiency(phys, s_db)
    with np.errstate(divide="ignore"):
        per_user = np.where(se > 0.0, phys.r_min_bps / (phys.bandwidth_hz * se), np.inf)
    return np.where(per_user > 1.0, np.inf, phys.lam * per_user)


def admit(phys: Physics, s_db: np.ndarray, assoc: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """GBR admission, best SINR first per drone: (admitted share per point, band used in total)."""
    w = band_need(phys, s_db)
    order = np.lexsort((-s_db, assoc))
    a, ws = assoc[order], w[order]
    fin = np.isfinite(ws)
    wf = np.where(fin, ws, 0.0)
    csum = np.cumsum(wf)
    counts = np.bincount(assoc, minlength=n)
    starts = np.cumsum(counts) - counts
    offset = np.where(starts > 0, csum[np.maximum(starts - 1, 0)], 0.0)[a]
    used_before = csum - wf - offset
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(fin, np.clip((1.0 - used_before) / ws, 0.0, 1.0), 0.0)
    out = np.empty_like(frac)
    out[order] = frac
    return out, float(np.sum(frac * wf))


# ── Geometry helpers ─────────────────────────────────────────────────


def min_enclosing_circle(pts: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Smallest circle containing 2-D points (Welzl, randomised incremental, expected O(n))."""
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    if p.shape[0] == 0:
        raise ValueError("no points")
    p = p[np.random.default_rng(seed).permutation(p.shape[0])]
    eps = 1e-9

    def inside(q, c, r):
        return np.hypot(*(q - c)) <= r + eps

    c, r = p[0].copy(), 0.0
    for i in range(1, p.shape[0]):
        if inside(p[i], c, r):
            continue
        c, r = p[i].copy(), 0.0
        for j in range(i):
            if inside(p[j], c, r):
                continue
            c, r = (p[i] + p[j]) / 2.0, np.hypot(*(p[i] - p[j])) / 2.0
            for k in range(j):
                if not inside(p[k], c, r):
                    c, r = _circle3(p[i], p[j], p[k])
    return c, float(r)


def _circle3(a, b, c) -> tuple[np.ndarray, float]:
    """Circumcircle of three points; for (near-)collinear points, the circle on the farthest pair."""
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        pairs = [(a, b), (a, c), (b, c)]
        u, v = max(pairs, key=lambda q: np.hypot(*(q[0] - q[1])))
        return (u + v) / 2.0, float(np.hypot(*(u - v)) / 2.0)
    sa, sb, sc = a @ a, b @ b, c @ c
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    o = np.array([ux, uy])
    return o, float(max(np.hypot(*(o - q)) for q in (a, b, c)))


def kmeans_pp(xy: np.ndarray, n: int, seed: int, iters: int = 50) -> np.ndarray:
    """Lloyd's algorithm from a k-means++ start (equal point weights)."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    cen = [xy[rng.integers(xy.shape[0])]]
    d2 = np.sum((xy - cen[0]) ** 2, axis=1)
    for _ in range(1, n):
        cen.append(xy[rng.choice(xy.shape[0], p=d2 / d2.sum())] if d2.sum() > 0 else xy[rng.integers(xy.shape[0])])
        d2 = np.minimum(d2, np.sum((xy - cen[-1]) ** 2, axis=1))
    cen = np.array(cen)
    for _ in range(iters):
        lab = np.argmin(np.sum((xy[:, None, :] - cen[None]) ** 2, axis=-1), axis=1)
        new = np.array([xy[lab == j].mean(axis=0) if np.any(lab == j) else cen[j] for j in range(n)])
        if np.allclose(new, cen):
            break
        cen = new
    return cen


# ── State and evaluation ─────────────────────────────────────────────


def nearest(pos: np.ndarray, users: np.ndarray) -> np.ndarray:
    """Nearest drone in 3-D distance (ties: lowest index)."""
    return np.argmin(np.linalg.norm(pos[:, None, :] - users[None, :, :], axis=-1), axis=0)


@dataclass
class Deployment:
    pos: np.ndarray            # (N, 3)
    t: np.ndarray              # (N, 2) tilt coordinates; zeros = nadir

    def copy(self) -> Deployment:
        return Deployment(self.pos.copy(), self.t.copy())

    @property
    def bore(self) -> np.ndarray:
        return boresight(self.t)


def score(phys: Physics, users: np.ndarray, dep: Deployment, assoc: np.ndarray) -> tuple[float, float]:
    """(served share, band used) of a deployment under an association, from scratch."""
    p = rx_power(phys, dep.pos, dep.bore, users)
    cols = np.arange(users.shape[0])
    s = p[assoc, cols]
    adm, used = admit(phys, sinr_db(phys, s, p.sum(axis=0)), assoc, dep.pos.shape[0])
    return float(np.mean(adm)), used


def lp_association(phys: Physics, power: np.ndarray) -> tuple[np.ndarray, float]:
    """Association maximising admitted demand for a fixed geometry.

    Interference does not depend on the association, so user k's band need at
    drone j, c_kj, is fixed; the fractional problem
        max sum x_kj  s.t.  sum_j x_kj <= 1,  sum_k c_kj x_kj <= 1,  0 <= x <= 1
    is a linear program whose value bounds the served share of every
    association of this geometry from above. Each user is then rounded to the
    drone carrying most of it. Returns (association, LP bound on served share);
    users with no feasible drone keep -1.
    """
    n, m = power.shape
    total = power.sum(axis=0)
    need = band_need(phys, sinr_db(phys, power, total[None, :]))           # (n, m)
    j_idx, k_idx = np.nonzero(np.isfinite(need))
    assoc = np.full(m, -1)
    if j_idx.size == 0:
        return assoc, 0.0
    c = need[j_idx, k_idx]
    e = np.arange(j_idx.size)
    a_ub = coo_matrix((np.concatenate([np.ones(e.size), c]),
                       (np.concatenate([k_idx, m + j_idx]), np.concatenate([e, e]))),
                      shape=(m + n, e.size)).tocsr()
    res = linprog(-np.ones(e.size), A_ub=a_ub, b_ub=np.ones(m + n), bounds=(0.0, 1.0), method="highs")
    if res.status != 0:
        raise RuntimeError(f"association LP failed: {res.message}")
    x = res.x
    order = np.lexsort((-power[j_idx, k_idx], -x))[::-1]       # ascending x: the last write per user wins
    keep = order[x[order] > 1e-9]
    assoc[k_idx[keep]] = j_idx[keep]
    return assoc, float(-res.fun / m)


def associate(phys: Physics, users: np.ndarray, dep: Deployment, mode: str,
              keep: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """(association, LP bound or nan). `free` keeps the best of LP rounding, nearest, strongest and `keep`."""
    near = nearest(dep.pos, users)
    if mode == "nearest":
        return near, float("nan")
    power = rx_power(phys, dep.pos, dep.bore, users)
    if mode == "strongest":
        return np.argmax(power, axis=0), float("nan")
    if mode != "free":
        raise ValueError(mode)
    lp, bound = lp_association(phys, power)
    lp = np.where(lp >= 0, lp, near)
    cands = [lp, near, np.argmax(power, axis=0)] + ([] if keep is None else [keep])
    scores = [score(phys, users, dep, a) for a in cands]
    best = max(range(len(cands)), key=lambda i: (scores[i][0], -scores[i][1]))
    return cands[best], bound


class _DroneMove:
    """Served share as one drone moves, every other drone and the association fixed (O(M) per move)."""

    def __init__(self, phys: Physics, users: np.ndarray, dep: Deployment, power: np.ndarray,
                 assoc: np.ndarray | None, j: int, rule: str = "nearest"):
        self.phys, self.users, self.j = phys, users, j
        self.n, self.m = dep.pos.shape[0], users.shape[0]
        cols = np.arange(self.m)
        self.total_others = power.sum(axis=0) - power[j]
        self.rule = rule
        if assoc is None and rule == "strongest":         # strongest: recompute j's side of the rule
            p = power.copy()
            p[j] = -np.inf
            self.idx_o = np.argmax(p, axis=0)
            self.sig_o = p[self.idx_o, cols]
            self.assoc = None
        elif assoc is None:                                # nearest: recompute j's side of the rule
            d = np.linalg.norm(dep.pos[:, None, :] - users[None, :, :], axis=-1)
            d[j] = np.inf
            self.idx_o = np.argmin(d, axis=0) if self.n > 1 else np.zeros(self.m, dtype=int)
            self.d_o = d[self.idx_o, cols]
            self.sig_o = power[self.idx_o, cols] if self.n > 1 else np.zeros(self.m)
            self.assoc = None
        else:
            self.assoc = assoc
            self.sig_fixed = power[assoc, cols]
            self.mine = assoc == j
        self.evals = 0

    def row(self, v: np.ndarray) -> np.ndarray:
        return rx_power(self.phys, v[None, :3], boresight(v[None, 3:5]), self.users)[0]

    def __call__(self, v: np.ndarray) -> tuple[float, float]:
        self.evals += 1
        r = self.row(v)
        total = self.total_others + r
        if self.assoc is None:
            if self.rule == "strongest":
                take = (r > self.sig_o) | ((r == self.sig_o) & (self.j < self.idx_o))
            else:
                dj = np.linalg.norm(self.users - v[:3], axis=-1)
                take = (dj < self.d_o) | ((dj == self.d_o) & (self.j < self.idx_o))
            assoc = np.where(take, self.j, self.idx_o)
            sig = np.where(take, r, np.maximum(self.sig_o, 0.0))
        else:
            assoc = self.assoc
            sig = np.where(self.mine, r, self.sig_fixed)
        adm, used = admit(self.phys, sinr_db(self.phys, sig, total), assoc, self.n)
        return float(np.mean(adm)), used


def _compass(mv: _DroneMove, v: np.ndarray, best: tuple[float, float], dims: list[int], step0: np.ndarray,
             step_min: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: int) -> tuple[np.ndarray, tuple]:
    """Compass search from v: first improving step per coordinate, halve all steps when none improves."""
    step = step0.copy()
    stop = mv.evals + budget
    while mv.evals < stop:
        moved = False
        for d in dims:
            if step[d] < step_min[d]:
                continue
            for sign in (1.0, -1.0):
                cand = v.copy()
                cand[d] = np.clip(v[d] + sign * step[d], lo[d], hi[d])
                if d >= 3:
                    cap_tilt(cand)
                if np.array_equal(cand, v):
                    continue
                sc = mv(cand)
                if _better(sc, best):
                    v, best, moved = cand, sc, True
                    break
        if not moved:
            step = step / 2.0
            if all(step[d] < step_min[d] for d in dims):
                break
    return v, best


def _better(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Served share first; at equal served share, less band used."""
    return a[0] > b[0] + 1e-12 or (a[0] >= b[0] - 1e-12 and a[1] < b[1] - 1e-9)


@dataclass
class OptimiseResult:
    dep: Deployment
    assoc: np.ndarray
    served: float
    lp_bound: float            # LP bound on any association of the final geometry (nan for nearest)
    rounds: int
    evals: int
    time_s: float
    history: list = field(default_factory=list)


def optimise(phys: Physics, users: np.ndarray, start: Deployment, assoc_mode: str = "free",
             move: str = "xyzt", max_rounds: int = 8, min_gain: float = 1e-4, seed: int = 0,
             max_evals_per_drone: int = 200, start_assoc: np.ndarray | None = None) -> OptimiseResult:
    """Rounds of association and per-drone compass search (see module docstring).

    move: '' (association only), 'xyz' (positions and altitudes, nadir), 'xyzt' (and pointing).
    start_assoc: an association of `start` to keep if it beats the fresh ones (free mode), so a search
    continued from another's result never serves less than that result.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dep = start.copy()
    n = dep.pos.shape[0]
    lo = np.array([0.0, 0.0, phys.z_min, -T_MAX, -T_MAX])
    hi = np.array([phys.area_m, phys.area_m, phys.z_max, T_MAX, T_MAX])
    dims = {"": [], "xyz": [0, 1, 2], "xyzt": [0, 1, 2, 3, 4]}[move]
    dynamic = assoc_mode in ("nearest", "strongest")         # the association follows the geometry
    pointing = move == "xyzt"
    step0 = np.array([phys.area_m / math.sqrt(n) / 4.0] * 2 + [2.0 * phys.z_step, 0.2, 0.2])
    step_min = np.array([2.0, 2.0, 1.0, 0.01, 0.01])
    assoc, bound = associate(phys, users, dep, assoc_mode, start_assoc)
    cur = score(phys, users, dep, assoc)
    history = [cur[0]]
    evals, rounds = 0, 0
    for rounds in range(1, max_rounds + 1):
        start_served = cur[0]
        if dims:
            power = rx_power(phys, dep.pos, dep.bore, users)
            for j in rng.permutation(n):
                mv = _DroneMove(phys, users, dep, power, None if dynamic else assoc, j, assoc_mode)
                v = np.concatenate([dep.pos[j], dep.t[j]])
                best = mv(v)
                # candidate: the minimum-enclosing-circle centre of the points j serves
                mine = (assoc == j) if assoc_mode != "nearest" else (nearest(dep.pos, users) == j)
                if np.any(mine):
                    c, _ = min_enclosing_circle(users[mine], seed=int(j))
                    cand = v.copy()
                    cand[:2] = np.clip(c, lo[:2], hi[:2])
                    sc = mv(cand)
                    if _better(sc, best):
                        v, best = cand, sc
                if pointing and np.any(mine):
                    # standoff starts: off the cell's enclosing circle, beam aimed at its centre
                    # (a nadir beam lights only z tan(theta_3dB / 2), so the best basin may lie outside)
                    c, rad = min_enclosing_circle(users[mine], seed=int(j))
                    stand = []
                    for dist, az in product((0.5 * rad, rad, 2.0 * rad), STANDOFF_AZ):
                        cand = v.copy()
                        cand[:2] = np.clip(c - dist * np.array([math.cos(az), math.sin(az)]), lo[:2], hi[:2])
                        cand[3:5] = (c - cand[:2]) / cand[2]
                        stand.append((mv(cap_tilt(cand)), cand))
                    stand.sort(key=lambda q: (-q[0][0], q[0][1]))
                    # successive halving: a short climb from each start, the full climb from the leader
                    # (a start's own score ranks basins poorly)
                    short = [_compass(mv, v0, s0, dims, step0, step_min, lo, hi, STANDOFF_SHORT)
                             for v0, s0 in [(v, best)] + [(cand, sc) for sc, cand in stand[:STANDOFF_KEEP]]]
                    v0, s0 = short[0]
                    for vv, sc in short[1:]:
                        if _better(sc, s0):
                            v0, s0 = vv, sc
                    v, best = _compass(mv, v0, s0, dims, step0, step_min, lo, hi, max_evals_per_drone)
                else:
                    v, best = _compass(mv, v, best, dims, step0, step_min, lo, hi, max_evals_per_drone)
                evals += mv.evals
                if not np.array_equal(v, np.concatenate([dep.pos[j], dep.t[j]])):
                    dep.pos[j], dep.t[j] = v[:3], v[3:5]
                    power[j] = mv.row(v)
                    if assoc_mode == "nearest":
                        assoc = nearest(dep.pos, users)
                    elif assoc_mode == "strongest":
                        assoc = np.argmax(power, axis=0)
            cur = score(phys, users, dep, assoc)
        new_assoc, bound = associate(phys, users, dep, assoc_mode)
        new = score(phys, users, dep, new_assoc)
        if _better(new, cur) or dynamic:
            assoc, cur = new_assoc, new
        history.append(cur[0])
        if not dims or cur[0] - start_served < min_gain:
            break
    return OptimiseResult(dep, assoc, cur[0], bound, rounds, evals, time.perf_counter() - t0, history)


def sweep_start(phys: Physics, users: np.ndarray, xy: np.ndarray) -> Deployment:
    """Nadir drones at the xy given, all at the common grid altitude serving the most (nearest association)."""
    best = None
    for z in phys.altitude_grid():
        dep = Deployment(np.column_stack([xy, np.full(xy.shape[0], z)]), np.zeros((xy.shape[0], 2)))
        sc = score(phys, users, dep, nearest(dep.pos, users))
        if best is None or _better(sc, best[0]):
            best = (sc, dep)
    return best[1]
