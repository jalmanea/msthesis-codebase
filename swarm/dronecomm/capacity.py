"""Load-aware service: who gets a rate above the service minimum.

Coverage only asks whether a user's SINR clears a threshold. Capacity also
asks how many active users share the drone that serves it. Every simulated
user point stands for lambda active users (DemandConfig.active_per_point:
people per point x device share x activity factor), so drone j carries
lambda * n_j active users:

    se_u   = truncated Shannon spectral efficiency of SINR_u      [bit/s/Hz]
    rate_u = B * se_u / max(1, active users admitted on u's drone)  [bit/s]

Spectral efficiency follows the attenuated and truncated Shannon bound of
3GPP TR 36.942 Annex A.1 (DL: alpha = 0.6, SINR_min = -10 dB, 4.4 bit/s/Hz).
Each drone shares its band equally (round robin) among its admitted active
users. With a per-cell cap, users are admitted in decreasing SINR order until
the cap, so a point may be admitted in part; the rest get nothing. A user is
served if admitted with rate >= r_min (`serve`, `served_share`).

Guaranteed-bit-rate admission (`admit_gbr`) is the service model the fleet
search uses: a cell admits active users in decreasing SINR order while the
resources they need fit, user u needing the share r_min / (B * se_u) of the
band; users below SINR_min, or needing more than the whole band, are never
admitted. A point stands for lambda users, so it may be admitted in part. Admitting the cheapest users
first maximises the number admitted. It replaces round robin's all-or-nothing
cliff (every user of a cell gets the same share, so a cell that cannot give
all of them r_min serves none) with what radio admission control does for
GBR bearers (voice, video calls).
"""

from __future__ import annotations

import numpy as np


def truncated_shannon_se(
    sinr_db: np.ndarray,
    alpha: float = 0.6,
    sinr_min_db: float = -10.0,
    se_max: float = 4.4,
) -> np.ndarray:
    """Spectral efficiency [bit/s/Hz] from SINR, TR 36.942 Annex A.1.

    0 below SINR_min, alpha * log2(1 + SINR) above it, capped at se_max.
    """
    sinr_db = np.asarray(sinr_db, dtype=np.float64)
    se = alpha * np.log2(1.0 + 10.0 ** (sinr_db / 10.0))
    return np.where(sinr_db < sinr_min_db, 0.0, np.minimum(se, se_max))


def serve(
    sinr_db: np.ndarray,
    association: np.ndarray,
    n_drones: int,
    bandwidth_hz: float,
    active_per_point: float = 1.0,
    alpha: float = 0.6,
    sinr_min_db: float = -10.0,
    se_max: float = 4.4,
    max_users_per_cell: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per point: rate [bit/s] of an admitted active user, and the admitted share in [0, 1]."""
    sinr_db = np.asarray(sinr_db, dtype=np.float64)
    association = np.asarray(association)
    lam = float(active_per_point)
    counts = np.bincount(association, minlength=n_drones)
    load = lam * counts
    if max_users_per_cell is None:
        admitted = np.ones(sinr_db.shape)
    else:
        load = np.minimum(load, max_users_per_cell)
        order = np.lexsort((-sinr_db, association))          # by cell, best SINR first
        starts = np.cumsum(counts) - counts
        rank = np.arange(order.size) - starts[association[order]]
        admitted = np.empty(sinr_db.shape)
        admitted[order] = np.clip((max_users_per_cell - lam * rank) / lam, 0.0, 1.0)
    se = truncated_shannon_se(sinr_db, alpha, sinr_min_db, se_max)
    rate = bandwidth_hz * se / np.maximum(load[association], 1.0)
    return rate, admitted


def admit_gbr(
    sinr_db: np.ndarray,
    association: np.ndarray,
    n_drones: int,
    bandwidth_hz: float,
    r_min_bps: float,
    active_per_point: float = 1.0,
    alpha: float = 0.6,
    sinr_min_db: float = -10.0,
    se_max: float = 4.4,
    max_users_per_cell: float | None = None,
) -> np.ndarray:
    """Per point: share of its active users admitted at r_min (GBR admission, best SINR first)."""
    sinr_db = np.asarray(sinr_db, dtype=np.float64)
    association = np.asarray(association)
    lam = float(active_per_point)
    se = truncated_shannon_se(sinr_db, alpha, sinr_min_db, se_max)
    with np.errstate(divide="ignore"):
        need = np.where(se > 0.0, r_min_bps / (bandwidth_hz * se), np.inf)   # band share per user
    need = np.where(need > 1.0, np.inf, need)          # one user needing more than the band never fits
    order = np.lexsort((-sinr_db, association))
    a_sorted = association[order]
    w = lam * need[order]                                                    # band share per point
    counts = np.bincount(association, minlength=n_drones)
    starts = np.cumsum(counts) - counts
    csum = np.cumsum(np.where(np.isfinite(w), w, 0.0))
    before = csum - np.where(np.isfinite(w), w, 0.0)
    cell_offset = np.where(starts > 0, csum[np.maximum(starts - 1, 0)], 0.0)[a_sorted]
    used_before = before - cell_offset
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(w), np.clip((1.0 - used_before) / w, 0.0, 1.0), 0.0)
    if max_users_per_cell is not None:
        rank = np.arange(order.size) - starts[a_sorted]
        frac = np.minimum(frac, np.clip((max_users_per_cell - lam * rank) / lam, 0.0, 1.0))
    admitted = np.empty(sinr_db.shape)
    admitted[order] = frac
    return admitted


def served_share(rate_bps: np.ndarray, admitted: np.ndarray, r_min_bps: float) -> float:
    """Share of active demand admitted with a rate of at least r_min (points weigh equally)."""
    return float(np.mean(np.asarray(admitted) * (np.asarray(rate_bps) >= r_min_bps)))


def user_sinr_db(scenario, config, dl_antenna, channel) -> tuple[np.ndarray, np.ndarray]:
    """Per-user downlink SINR [dB] and serving drone, as in optimize._evaluate_single."""
    from .interference import downlink_power_matrix
    from .sinr import compute_sinr, dbm_to_watts, nearest_drone_association

    net = config.network
    power = downlink_power_matrix(
        drone_positions=scenario.drone_positions, dl_tilt_rad=scenario.dl_tilt_rad,
        dl_azimuth_rad=scenario.dl_azimuth_rad, user_positions=scenario.user_positions,
        dl_antenna=dl_antenna, channel=channel, p_tx_dl_w=dbm_to_watts(net.p_tx_dl_dbm))
    assoc = nearest_drone_association(scenario.drone_positions, scenario.user_positions)
    res = compute_sinr(power, assoc, noise_psd_dbm_hz=net.noise_dbm_hz,
                       bandwidth_hz=net.bandwidth_hz)
    return res.sinr_db, assoc


def capacity_summary(scenario, config, dl_antenna, channel) -> dict:
    """Served share and rate percentiles under config.capacity and config.demand."""
    cap, dem = config.capacity, config.demand
    sinr_db, assoc = user_sinr_db(scenario, config, dl_antenna, channel)
    rate, admitted = serve(sinr_db, assoc, scenario.drone_positions.shape[0],
                           config.network.bandwidth_hz, dem.active_per_point(sinr_db.size),
                           cap.se_alpha, cap.se_sinr_min_db, cap.se_max_bps_hz,
                           cap.max_users_per_cell)
    eff = rate * (admitted > 0)
    return {
        "served_pct": 100.0 * served_share(rate, admitted, dem.r_min_bps),
        "rate_median_kbps": float(np.median(eff)) / 1e3,
        "rate_p10_kbps": float(np.percentile(eff, 10)) / 1e3,
        "max_cell_load": int(np.bincount(assoc).max()),
    }
