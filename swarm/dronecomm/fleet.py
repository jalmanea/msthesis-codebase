"""Fleet sizing under the load-aware service objective.

For a K-means placement of N drones, the altitude methods of the ICC paper are
re-targeted from coverage (SINR >= 3 dB) to the share of active demand served:
admitted at r_min by guaranteed-bit-rate admission (capacity.admit_gbr). The
round-robin served share (capacity.serve) and the end-to-end served share
(admitted traffic that the backhaul tree and gateway deliver, dronecomm.backhaul)
are reported for the chosen deployment but not optimised.

    kmeans  every drone at config.network.altitude_m
    sweep   one common grid altitude, the one serving the most demand
    greedy  coordinate ascent from the sweep, one drone at a time, accepting
            only strict gains in served demand

Ties go to higher coverage, then to the lower altitude. `service_curve`
evaluates these for a list of fleet sizes, and `fleet_size` reads off the
smallest N reaching a target.

Each drone's received-power row depends only on its own altitude, so it is
precomputed once per grid altitude (as heuristic._AltitudeGridCoverage does
for coverage): moving one drone then costs O(M) per candidate altitude.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import backhaul
from .capacity import admit_gbr, serve, served_share
from .config import Config
from .heuristic import _kmeans_centroids
from .interference import downlink_power_matrix
from .optimize import _build_models
from .sinr import dbm_to_watts, linear_to_db

METHODS = ("kmeans", "sweep", "greedy")


class GridService:
    """Served share and coverage of every altitude vector drawn from a fixed grid."""

    def __init__(self, config: Config, user_positions: np.ndarray,
                 centroids_xy: np.ndarray, grid: np.ndarray):
        net, cap, dem = config.network, config.capacity, config.demand
        dl_antenna, self._bh_antenna, channel = _build_models(config)
        self._channel, self._config, self._xy = channel, config, centroids_xy
        n, m = centroids_xy.shape[0], user_positions.shape[0]
        self.n, self.m, self.grid = n, m, np.asarray(grid, dtype=float)
        self.power = np.stack([
            downlink_power_matrix(
                drone_positions=np.column_stack([centroids_xy, np.full(n, a)]),
                dl_tilt_rad=np.zeros(n), dl_azimuth_rad=np.zeros(n),
                user_positions=user_positions, dl_antenna=dl_antenna,
                channel=channel, p_tx_dl_w=dbm_to_watts(net.p_tx_dl_dbm))
            for a in self.grid
        ])                                                         # (A, N, M)
        dxy2 = np.sum((centroids_xy[:, None, :] - user_positions[None, :, :2]) ** 2, axis=-1)
        dz = self.grid[:, None, None] - user_positions[None, None, :, 2]
        self.dist = np.sqrt(dxy2[None, :, :] + dz ** 2)            # (A, N, M)
        self.noise_w = dbm_to_watts(net.noise_dbm_hz) * net.bandwidth_hz
        self.threshold_db = net.sinr_threshold_db
        self.bandwidth_hz = net.bandwidth_hz
        self.lam = dem.active_per_point(m)
        self.r_min = dem.r_min_bps
        self.se = (cap.se_alpha, cap.se_sinr_min_db, cap.se_max_bps_hz)
        self.cap = cap.max_users_per_cell
        self.n_evaluations = 0

    def _sinr_db(self, signal, total) -> np.ndarray:
        return linear_to_db(signal / (self.noise_w + (total - signal)))

    def _score(self, signal, total, assoc) -> tuple[float, float]:
        self.n_evaluations += 1
        sinr_db = self._sinr_db(signal, total)
        admitted = admit_gbr(sinr_db, assoc, self.n, self.bandwidth_hz, self.r_min, self.lam,
                             *self.se, self.cap)
        return float(np.mean(admitted)), float(np.mean(sinr_db >= self.threshold_db))

    def _rows(self, k: np.ndarray):
        cols = np.arange(self.m)
        p = self.power[k, np.arange(self.n)]
        assoc = np.argmin(self.dist[k, np.arange(self.n)], axis=0)
        return p[assoc, cols], p.sum(axis=0), assoc

    def evaluate(self, k: np.ndarray) -> tuple[float, float]:
        """(served share, coverage) with drone j at grid altitude k[j]."""
        return self._score(*self._rows(k))

    def end_to_end_served(self, k: np.ndarray, topology: str | None = None) -> tuple[float, int]:
        """(share of active demand admitted AND delivered through backhaul and gateway, relays needed)."""
        signal, total, assoc = self._rows(k)
        admitted = admit_gbr(self._sinr_db(signal, total), assoc, self.n, self.bandwidth_hz, self.r_min,
                             self.lam, *self.se, self.cap)
        users = self.lam * np.bincount(assoc, weights=admitted, minlength=self.n)
        positions = np.column_stack([self._xy, self.grid[k]])
        share, relays = backhaul.end_to_end(positions, users * self.r_min, self._config, self._bh_antenna,
                                            self._channel, topology)
        return float(np.sum(users * share) / (self.lam * self.m)), relays

    def round_robin_served(self, k: np.ndarray) -> float:
        """Served share if each cell shared its band equally instead (not counted as an evaluation)."""
        signal, total, assoc = self._rows(k)
        rate, admitted = serve(self._sinr_db(signal, total), assoc, self.n, self.bandwidth_hz,
                               self.lam, *self.se, self.cap)
        return served_share(rate, admitted, self.r_min)

    def evaluate_moving_one(self, k: np.ndarray, j: int) -> np.ndarray:
        """(A, 2) array: (served, coverage) for every grid altitude of drone j, others fixed."""
        cols = np.arange(self.m)
        out = np.empty((self.grid.size, 2))
        if self.n == 1:
            for a in range(self.grid.size):
                s = self.power[a, 0]
                out[a] = self._score(s, s, np.zeros(self.m, dtype=int))
            return out
        p = self.power[k, np.arange(self.n)]
        d = self.dist[k, np.arange(self.n)].copy()
        total_others = p.sum(axis=0) - p[j]
        d[j] = np.inf
        idx_o = np.argmin(d, axis=0)
        d_o = d[idx_o, cols]
        sig_o = p[idx_o, cols]
        for a in range(self.grid.size):
            dj = self.dist[a, j]
            take = (dj < d_o) | ((dj == d_o) & (j < idx_o))          # argmin's tie rule
            assoc = np.where(take, j, idx_o)
            signal = np.where(take, self.power[a, j], sig_o)
            out[a] = self._score(signal, total_others + self.power[a, j], assoc)
        return out


def _better(a: tuple, b: tuple) -> bool:
    """(served, coverage) a strictly better than b, served first."""
    return a[0] > b[0] + 1e-12 or (abs(a[0] - b[0]) <= 1e-12 and a[1] > b[1] + 1e-12)


def sweep(gs: GridService) -> np.ndarray:
    """Grid index of the common altitude serving the most demand (ties: coverage, then lower)."""
    best_a, best = 0, None
    for a in range(gs.grid.size):
        score = gs.evaluate(np.full(gs.n, a))
        if best is None or _better(score, best):
            best_a, best = a, score
    return np.full(gs.n, best_a)


def greedy(gs: GridService, k0: np.ndarray, max_passes: int = 20) -> tuple[np.ndarray, int]:
    """Coordinate ascent from k0 over drones in index order; returns (k, passes)."""
    k = k0.copy()
    current = gs.evaluate(k)
    for p in range(1, max_passes + 1):
        changed = False
        for j in range(gs.n):
            scores = gs.evaluate_moving_one(k, j)
            a = int(k[j])
            for cand in range(gs.grid.size):                        # lower altitudes first
                if _better(tuple(scores[cand]), tuple(scores[a])):
                    a = cand
            if a != k[j] and _better(tuple(scores[a]), current):
                k[j], current, changed = a, tuple(scores[a]), True
        if not changed:
            return k, p
    return k, max_passes


@dataclass(frozen=True)
class ServiceResult:
    n_drones: int
    method: str
    served: float          # share of active demand admitted at r_min (GBR admission)
    coverage: float        # share of points with SINR >= threshold
    served_round_robin: float  # share served if cells shared their band equally
    end_to_end: dict           # topology -> (share admitted and delivered, relay drones needed)
    altitudes_m: np.ndarray
    n_evaluations: int
    build_time_s: float


def service_curve(config: Config, user_positions: np.ndarray, fleet_sizes, seed: int,
                  methods=METHODS, topologies=backhaul.TOPOLOGIES) -> list[ServiceResult]:
    """Served share and coverage of each method at each fleet size (K-means placement, same seed)."""
    grid = config.network.altitude_grid()
    fixed = np.flatnonzero(np.isclose(grid, config.network.altitude_m))
    if "kmeans" in methods and fixed.size == 0:
        raise ValueError(f"altitude_m {config.network.altitude_m} m is not on the altitude grid")
    out = []
    for n in fleet_sizes:
        t0 = time.perf_counter()
        gs = GridService(config, user_positions, _kmeans_centroids(user_positions, n, seed), grid)
        t_setup = time.perf_counter() - t0
        k_sweep = None
        for method in methods:
            t0, e0 = time.perf_counter(), gs.n_evaluations
            if method == "kmeans":
                k = np.full(n, fixed[0])
            elif method == "sweep":
                k = k_sweep = sweep(gs)
            elif method == "greedy":
                k = greedy(gs, k_sweep if k_sweep is not None else sweep(gs))[0]
            else:
                raise ValueError(method)
            served, cov = gs.evaluate(k)
            e2e = {t: gs.end_to_end_served(k, t) for t in topologies}
            out.append(ServiceResult(n, method, served, cov, gs.round_robin_served(k), e2e, grid[k],
                                     gs.n_evaluations - e0, t_setup + time.perf_counter() - t0))
    return out


def fleet_size(curve: list[ServiceResult], method: str, target: float) -> int | None:
    """Smallest evaluated N whose served share reaches the target, or None."""
    ok = [r.n_drones for r in curve if r.method == method and r.served >= target - 1e-12]
    return min(ok) if ok else None
