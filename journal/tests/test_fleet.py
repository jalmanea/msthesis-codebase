"""Tests for fleet sizing under the service objective."""

from dataclasses import replace

import numpy as np
import pytest

from dronecomm.capacity import admit_gbr, capacity_summary, user_sinr_db
from dronecomm.config import Config
from dronecomm.fleet import GridService, fleet_size, greedy, service_curve, sweep
from dronecomm.heuristic import _kmeans_centroids, deploy_kmeans_baseline
from dronecomm.optimize import _build_models, _evaluate_single
from dronecomm.scenario import generate_users_hotspot


@pytest.fixture
def setup():
    base = Config.from_yaml("configs/grounded.yaml")
    cfg = replace(base, network=replace(base.network, n_drones=6),
                  demand=replace(base.demand, population=20000.0))
    users = generate_users_hotspot(rng=np.random.default_rng(5))
    cen = _kmeans_centroids(users, 6, 5)
    return cfg, users, GridService(cfg, users, cen, cfg.network.altitude_grid())


def test_grid_service_matches_full_pipeline(setup):
    cfg, users, gs = setup
    a = int(np.flatnonzero(gs.grid == 120.0)[0])
    k = np.full(gs.n, a)
    served, cov = gs.evaluate(k)
    sc = deploy_kmeans_baseline(cfg, users, seed=5)
    dl, bh, ch = _build_models(cfg)
    assert cov == pytest.approx(_evaluate_single(sc, cfg, dl, bh, ch).coverage_fraction)
    assert 100.0 * gs.round_robin_served(k) == pytest.approx(capacity_summary(sc, cfg, dl, ch)["served_pct"])
    sinr_db, assoc = user_sinr_db(sc, cfg, dl, ch)
    dem, cap = cfg.demand, cfg.capacity
    gbr = admit_gbr(sinr_db, assoc, gs.n, cfg.network.bandwidth_hz, dem.r_min_bps,
                    dem.active_per_point(users.shape[0]), cap.se_alpha, cap.se_sinr_min_db,
                    cap.se_max_bps_hz, cap.max_users_per_cell)
    assert served == pytest.approx(float(np.mean(gbr)))


def test_moving_one_matches_full_evaluation(setup):
    _, _, gs = setup
    k = np.array([0, 3, 7, 2, 5, 1])
    scores = gs.evaluate_moving_one(k, 2)
    for a in range(gs.grid.size):
        kk = k.copy()
        kk[2] = a
        np.testing.assert_allclose(scores[a], gs.evaluate(kk))


def test_greedy_never_below_sweep(setup):
    _, _, gs = setup
    k0 = sweep(gs)
    s0 = gs.evaluate(k0)
    k, _ = greedy(gs, k0)
    assert gs.evaluate(k)[0] >= s0[0]
    assert all(gs.evaluate(np.full(gs.n, a))[0] <= s0[0] + 1e-12 for a in range(gs.grid.size))


def test_more_demand_serves_a_smaller_share(setup):
    cfg, users, _ = setup
    cen = _kmeans_centroids(users, 6, 5)
    shares = []
    for pop in (1000.0, 20000.0, 200000.0):
        c = replace(cfg, demand=replace(cfg.demand, population=pop, r_min_bps=384e3))
        gs = GridService(c, users, cen, c.network.altitude_grid())
        shares.append(gs.evaluate(np.full(6, 0))[0])
    assert shares[0] >= shares[1] >= shares[2] and shares[0] > shares[2]


def test_service_curve_and_fleet_size(setup):
    cfg, users, _ = setup
    curve = service_curve(cfg, users, [1, 3, 6], seed=5)
    assert {(r.n_drones, r.method) for r in curve} == {(n, m) for n in (1, 3, 6) for m in ("kmeans", "sweep", "greedy")}
    for n in (1, 3, 6):
        by = {r.method: r.served for r in curve if r.n_drones == n}
        assert by["greedy"] >= by["sweep"] >= 0.0
    best = max(r.served for r in curve)
    assert fleet_size(curve, "greedy", best) is not None
    assert fleet_size(curve, "greedy", 1.01) is None
