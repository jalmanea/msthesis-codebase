"""Drone-to-gateway transport: how much admitted traffic actually reaches the gateway.

Topology (BackhaulConfig.topology):
  mst_corner   minimum spanning tree over the drones' 3-D
               positions, rooted at the drone nearest the area origin;
  mst_central  the same tree rooted at the best gateway among the drones nearest
               the load-weighted centre of the fleet;
  balanced     a tree grown outward from the gateway (best among the same
               candidates): each drone, nearest first, joins the least-loaded
               branch through a drone with a free backhaul interface, preferring
               links within range, then the nearest such drone.
The gateway's ground link (satellite terminal or tether) is separate from its
drone-to-drone interfaces.

Link capacity (BackhaulConfig.link_model):
  fixed  every link carries fixed_link_bps (a dedicated 60 GHz mesh); a drone
         with more links than `interfaces` time-shares, so a link gets
         min(1, I / deg(u), I / deg(parent)) of the rate; with half_duplex and a
         two-way tier, each link carries both directions of every call in turn,
         so half its capacity is left for the direction the tree carries;
  sinr   every drone transmits on one shared backhaul channel (worst case:
         all at once); the link into drone j from its parent has
         SINR = P[parent, j] / (noise + sum of P[k, j], k != parent), with the
         backhaul antenna gains at both ends and air-to-air free-space loss,
         and capacity B_bh x truncated-Shannon(SINR) (TR 36.942 A.1);
  none   unconstrained.
A link longer than link_range_m is bridged by ceil(d / R) - 1 relay drones on
the straight line; they carry no users and the bottleneck rate of the chain
equals one link's, so they add to the fleet count only. The gateway's ground
link carries at most gateway_bps (None: unconstrained).

Traffic: drone j's own load is the guaranteed bit rate of the users it
admitted. A link carries its subtree's load; an overloaded link throttles
everything through it proportionally (own and children's traffic alike), and
the delivered share of drone j's load is the product of these ratios on its
path to the gateway, gateway link included.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .capacity import truncated_shannon_se
from .interference import backhaul_interference_matrix
from .sinr import dbm_to_watts

TOPOLOGIES = ("mst_corner", "mst_central", "balanced")
N_GATEWAY_CANDIDATES = 8


def gateway_index(drone_positions: np.ndarray) -> int:
    """The drone nearest the (0, 0) corner of the area (the corner MST's gateway)."""
    return int(np.argmin(np.linalg.norm(drone_positions[:, :2], axis=1)))


def _pairwise(drone_positions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(drone_positions[:, None, :] - drone_positions[None, :, :], axis=-1)


def mst_tree(drone_positions: np.ndarray, gateway: int) -> tuple[np.ndarray, np.ndarray]:
    """(parent, BFS order from the gateway) of the 3-D MST; parent[gateway] = -1.

    Coincident drones get a 1 mm edge, so the tree always spans every drone.
    """
    n = drone_positions.shape[0]
    if n == 1:
        return np.array([-1]), np.array([0])
    dist = np.where(np.eye(n, dtype=bool), 0.0, np.maximum(_pairwise(drone_positions), 1e-3))
    mst = minimum_spanning_tree(csr_matrix(dist))
    order, pred = breadth_first_order(mst + mst.T, gateway, directed=False, return_predecessors=True)
    parent = np.where(pred < 0, -1, pred)
    parent[gateway] = -1
    return parent, order


def tree_order(parent: np.ndarray) -> np.ndarray:
    """BFS order from the root of a parent array."""
    children: dict[int, list[int]] = {}
    for j, p in enumerate(parent):
        if p >= 0:
            children.setdefault(int(p), []).append(j)
    order = [int(np.flatnonzero(parent < 0)[0])]
    for j in order:
        order.extend(children.get(j, []))
    return np.array(order)


def balanced_tree(drone_positions: np.ndarray, load_bps: np.ndarray, gateway: int,
                  interfaces: int | None, link_range_m: float | None) -> np.ndarray:
    """Parent array of a load-balanced, interface-aware tree grown from the gateway."""
    n = drone_positions.shape[0]
    dist = _pairwise(drone_positions)
    lim = np.inf if interfaces is None else interfaces
    parent = np.full(n, -1)
    deg = np.zeros(n)
    branch = np.full(n, -1)
    branch_load: dict[int, float] = {}
    in_tree = np.zeros(n, dtype=bool)
    in_tree[gateway] = True
    for u in np.argsort(dist[gateway], kind="stable"):
        if u == gateway:
            continue
        pool = np.flatnonzero(in_tree)
        free = pool[deg[pool] < lim]
        pool = free if free.size else pool
        if link_range_m is not None:
            near = pool[dist[u, pool] <= link_range_m]
            pool = near if near.size else pool
        score = np.array([load_bps[u] if v == gateway else branch_load[branch[v]] + load_bps[u] for v in pool])
        best = np.lexsort((dist[u, pool], score))[0]
        v = int(pool[best])
        parent[u] = v
        deg[u] += 1
        deg[v] += 1
        branch[u] = u if v == gateway else branch[v]
        branch_load[branch[u]] = branch_load.get(branch[u], 0.0) + load_bps[u]
        in_tree[u] = True
    return parent


def relays_needed(drone_positions: np.ndarray, parent: np.ndarray, link_range_m: float | None) -> int:
    """Relay drones that bridge links longer than the range."""
    if link_range_m is None:
        return 0
    kids = np.flatnonzero(parent >= 0)
    d = np.linalg.norm(drone_positions[kids] - drone_positions[parent[kids]], axis=1)
    return int(np.sum(np.maximum(np.ceil(d / link_range_m) - 1, 0)))


def orientations(drone_positions: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backhaul tilt (from nadir) and azimuth: each drone at its parent, the gateway horizontal."""
    tilt = np.full(parent.size, np.pi / 2)
    az = np.zeros(parent.size)
    kids = np.flatnonzero(parent >= 0)
    v = drone_positions[parent[kids]] - drone_positions[kids]
    r = np.linalg.norm(v, axis=1)
    tilt[kids] = np.arccos(np.clip(-v[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
    az[kids] = np.arctan2(v[:, 1], v[:, 0])
    return tilt, az


def link_capacity(drone_positions: np.ndarray, parent: np.ndarray, config, bh_antenna, channel) -> np.ndarray:
    """Capacity [bit/s] of the link from parent[j] into j (inf for the gateway and for link_model none)."""
    bh = config.backhaul
    cap = np.full(parent.size, np.inf)
    kids = np.flatnonzero(parent >= 0)
    if bh.link_model == "none" or kids.size == 0:
        return cap
    if bh.link_model == "fixed":
        share = np.ones(kids.size)
        if bh.interfaces is not None:
            deg = np.bincount(parent[kids], minlength=parent.size).astype(float)
            deg[kids] += 1.0                                        # each non-gateway drone's uplink
            share = np.minimum(1.0, np.minimum(bh.interfaces / deg[kids], bh.interfaces / deg[parent[kids]]))
        cap[kids] = bh.fixed_link_bps * share * _direction_share(bh)
        return cap
    if bh.link_model != "sinr":
        raise ValueError(f"unknown backhaul link_model {bh.link_model!r}")
    tilt, az = orientations(drone_positions, parent)
    p = backhaul_interference_matrix(drone_positions, tilt, az, bh_antenna, channel,
                                     dbm_to_watts(config.network.p_tx_bh_dbm))   # p[k, i]: k at i
    noise = dbm_to_watts(config.network.noise_psd_dbm_hz + bh.noise_figure_db) * bh.bandwidth_hz
    signal = p[parent[kids], kids]
    interference = p[:, kids].sum(axis=0) - signal
    sinr_db = 10.0 * np.log10(signal / (noise + interference))
    cs = config.capacity
    cap[kids] = (bh.bandwidth_hz * truncated_shannon_se(sinr_db, cs.se_alpha, cs.se_sinr_min_db, cs.se_max_bps_hz)
                 * _direction_share(bh))
    return cap


def _direction_share(bh) -> float:
    """Share of a link's capacity left for one direction: half when a half-duplex link carries a two-way
    tier (the same rate flows back), all of it otherwise."""
    return 0.5 if (bh.half_duplex and bh.two_way) else 1.0


def delivered_share(load_bps: np.ndarray, parent: np.ndarray, order: np.ndarray,
                    link_cap_bps: np.ndarray, gateway_bps: float | None) -> np.ndarray:
    """Share of each drone's own load that reaches the gateway, with proportional throttling."""
    n = parent.size
    gateway = int(order[0])
    carried = np.zeros(n)
    ratio = np.ones(n)
    inflow = np.asarray(load_bps, dtype=float).copy()
    for j in order[::-1]:                                   # leaves first
        cap = link_cap_bps[j] if j != gateway else (np.inf if gateway_bps is None else gateway_bps)
        carried[j] = min(inflow[j], cap)
        ratio[j] = carried[j] / inflow[j] if inflow[j] > 0 else 1.0
        if parent[j] >= 0:
            inflow[parent[j]] += carried[j]
    share = np.ones(n)
    for j in order:                                         # gateway first
        share[j] = ratio[j] * (share[parent[j]] if parent[j] >= 0 else 1.0)
    return share


def _gateway_candidates(drone_positions: np.ndarray, load_bps: np.ndarray) -> np.ndarray:
    w = np.asarray(load_bps, dtype=float)
    centre = (w @ drone_positions[:, :2] / w.sum()) if w.sum() > 0 else drone_positions[:, :2].mean(axis=0)
    d = np.linalg.norm(drone_positions[:, :2] - centre, axis=1)
    return np.argsort(d, kind="stable")[:N_GATEWAY_CANDIDATES]


def end_to_end(drone_positions: np.ndarray, load_bps: np.ndarray, config, bh_antenna, channel,
               topology: str | None = None) -> tuple[np.ndarray, int]:
    """(delivered share of each drone's load, relay drones needed) under config.backhaul."""
    bh = config.backhaul
    topology = topology or bh.topology
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown backhaul topology {topology!r}")
    if topology == "mst_corner":
        candidates = [gateway_index(drone_positions)]
    else:
        candidates = _gateway_candidates(drone_positions, load_bps)
    best = None
    for g in candidates:
        if topology == "balanced":
            parent = balanced_tree(drone_positions, load_bps, int(g), bh.interfaces, bh.link_range_m)
            order = tree_order(parent)
        else:
            parent, order = mst_tree(drone_positions, int(g))
        cap = link_capacity(drone_positions, parent, config, bh_antenna, channel)
        share = delivered_share(load_bps, parent, order, cap, bh.gateway_bps)
        relays = relays_needed(drone_positions, parent, bh.link_range_m)
        key = (float(np.dot(load_bps, share)), -relays)
        if best is None or key > best[0]:
            best = (key, share, relays)
    return best[1], best[2]
