"""Tests for the drone-to-gateway transport model."""

from dataclasses import replace

import numpy as np
import pytest

from dronecomm import backhaul
from dronecomm.config import Config
from dronecomm.fleet import GridService
from dronecomm.geometry import boresight_vector
from dronecomm.heuristic import _kmeans_centroids
from dronecomm.optimize import _build_models
from dronecomm.scenario import generate_users_hotspot


def test_mst_spans_all_drones_even_coincident_ones():
    pos = np.array([[0, 0, 100], [100, 0, 100], [100, 0, 100], [300, 0, 100]], float)
    parent, order = backhaul.mst_tree(pos, 0)
    assert parent[0] == -1 and sorted(order) == [0, 1, 2, 3]
    assert np.all(parent[1:] >= 0)


def test_children_point_at_their_parent():
    pos = np.array([[0, 0, 100], [200, 50, 80], [400, 300, 120]], float)
    parent, _ = backhaul.mst_tree(pos, 0)
    tilt, az = backhaul.orientations(pos, parent)
    b = boresight_vector(tilt, az)
    for j in (1, 2):
        v = pos[parent[j]] - pos[j]
        assert b[j] @ (v / np.linalg.norm(v)) == pytest.approx(1.0)


def test_proportional_throttling_along_a_chain():
    # gateway 0 <- 1 <- 2; loads 10, 20, 30; link into 2 carries 15, into 1 carries 25, gateway 30
    parent = np.array([-1, 0, 1])
    order = np.array([0, 1, 2])
    share = backhaul.delivered_share(np.array([10.0, 20.0, 30.0]), parent, order,
                                     np.array([np.inf, 25.0, 15.0]), 30.0)
    # link 2: 30 -> 15 (0.5); link 1: 20 + 15 = 35 -> 25 (5/7); gateway: 10 + 25 = 35 -> 30 (6/7)
    np.testing.assert_allclose(share, [6 / 7, 5 / 7 * 6 / 7, 0.5 * 5 / 7 * 6 / 7])
    delivered = np.array([10.0, 20.0, 30.0]) @ share
    assert delivered == pytest.approx(30.0)


def test_unconstrained_backhaul_delivers_everything():
    share = backhaul.delivered_share(np.array([5.0, 7.0]), np.array([-1, 0]), np.array([0, 1]),
                                     np.array([np.inf, np.inf]), None)
    np.testing.assert_allclose(share, 1.0)


@pytest.fixture
def grid_service():
    def make(**bh):
        base = Config.from_yaml("configs/grounded.yaml")
        cfg = replace(base, network=replace(base.network, n_drones=8),
                      demand=replace(base.demand, population=20000.0),
                      backhaul=replace(base.backhaul, **bh))
        users = generate_users_hotspot(rng=np.random.default_rng(9))
        return GridService(cfg, users, _kmeans_centroids(users, 8, 9), cfg.network.altitude_grid())
    return make


def test_end_to_end_equals_access_without_backhaul_limits(grid_service):
    gs = grid_service(link_model="none", gateway_bps=None)
    k = np.full(gs.n, 3)
    for topo in ("mst_corner", "mst_central", "balanced"):
        assert gs.end_to_end_served(k, topo)[0] == pytest.approx(gs.evaluate(k)[0])


def test_gateway_caps_delivered_traffic(grid_service):
    gs = grid_service(link_model="none", gateway_bps=1e6)
    k = np.full(gs.n, 3)
    served, _ = gs.end_to_end_served(k)
    assert served * gs.lam * gs.m * gs.r_min <= 1e6 * (1 + 1e-9)
    assert served <= gs.evaluate(k)[0]


def test_sinr_links_have_finite_positive_capacity():
    cfg = Config.from_yaml("configs/grounded.yaml")
    cfg = replace(cfg, backhaul=replace(cfg.backhaul, link_model="sinr"))
    _, bh, ch = _build_models(cfg)
    rng = np.random.default_rng(1)
    pos = np.column_stack([rng.uniform(0, 2000, (12, 2)), np.full(12, 100.0)])
    parent, _ = backhaul.mst_tree(pos, backhaul.gateway_index(pos))
    cap = backhaul.link_capacity(pos, parent, cfg, bh, ch)
    kids = parent >= 0
    assert np.all(np.isfinite(cap[kids])) and np.all(cap[kids] >= 0)
    assert np.all(cap[kids] <= cfg.backhaul.bandwidth_hz * cfg.capacity.se_max_bps_hz + 1e-6)
    assert np.isinf(cap[~kids]).all()


def test_relays_bridge_long_links():
    pos = np.array([[0, 0, 100], [350, 0, 100], [1300, 0, 100]], float)
    parent = np.array([-1, 0, 1])
    assert backhaul.relays_needed(pos, parent, 400.0) == 2       # 350 m: none; 950 m: 2
    assert backhaul.relays_needed(pos, parent, None) == 0


def test_interfaces_time_share_links():
    cfg = Config.from_yaml("configs/grounded.yaml")
    cfg = replace(cfg, backhaul=replace(cfg.backhaul, link_model="fixed", fixed_link_bps=600.0, interfaces=3))
    _, bh, ch = _build_models(cfg)
    pos = np.zeros((6, 3))
    parent = np.array([-1, 0, 0, 0, 0, 0])                       # gateway with 5 children
    cap = backhaul.link_capacity(pos, parent, cfg, bh, ch)
    np.testing.assert_allclose(cap[1:], 600.0 * 3 / 5)


def test_balanced_tree_spreads_load_over_gateway_interfaces():
    rng = np.random.default_rng(4)
    pos = np.column_stack([rng.uniform(0, 1000, (30, 2)), np.full(30, 80.0)])
    load = np.full(30, 10.0)
    g = 0
    parent = backhaul.balanced_tree(pos, load, g, interfaces=3, link_range_m=400.0)
    order = backhaul.tree_order(parent)
    assert sorted(order) == list(range(30)) and parent[g] == -1
    deg = np.bincount(parent[parent >= 0], minlength=30) + (parent >= 0)
    assert deg[g] == 3                                           # gateway uses all interfaces
    # branch loads (subtree sizes under the gateway's children) are balanced within a few drones
    sub = {c: 0 for c in np.flatnonzero(parent == g)}
    for j in range(30):
        a = j
        while parent[a] >= 0 and parent[a] != g:
            a = parent[a]
        if parent[a] == g:
            sub[a] += 1
    assert max(sub.values()) - min(sub.values()) <= 3


def test_balanced_beats_corner_mst_when_root_links_bind():
    cfg = Config.from_yaml("configs/grounded.yaml")
    cfg = replace(cfg, backhaul=replace(cfg.backhaul, link_model="fixed", fixed_link_bps=100.0,
                                        gateway_bps=None, interfaces=3, link_range_m=400.0))
    _, bh, ch = _build_models(cfg)
    xs, ys = np.meshgrid(np.linspace(100, 1900, 7), np.linspace(100, 1900, 7))
    pos = np.column_stack([xs.ravel(), ys.ravel(), np.full(49, 100.0)])
    load = np.full(49, 10.0)
    served = {t: float(load @ backhaul.end_to_end(pos, load, cfg, bh, ch, t)[0]) for t in backhaul.TOPOLOGIES}
    # on a regular grid MST ties make corner vs centre arbitrary; balancing wins against both
    assert served["balanced"] > max(served["mst_central"], served["mst_corner"])
