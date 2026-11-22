"""Tests for the network configuration: noise figure and altitude bounds."""

from dataclasses import replace

import numpy as np

from dronecomm.config import Config, NetworkConfig
from dronecomm.heuristic import sweep_kmeans_global_altitude
from dronecomm.scenario import generate_users_hotspot


def test_defaults_reproduce_icc_model():
    """Default config: no noise figure and the paper's 26-value [50, 300] m grid."""
    net = NetworkConfig()
    assert net.noise_dbm_hz == -174.0
    np.testing.assert_allclose(net.altitude_grid(), np.arange(50.0, 301.0, 10.0))


def test_noise_figure_adds_to_thermal_psd():
    assert NetworkConfig(noise_figure_db=9.0).noise_dbm_hz == -165.0


def test_altitude_grid_respects_bounds():
    grid = NetworkConfig(z_min_m=30.0, z_max_m=120.0, z_step_m=5.0).altitude_grid()
    assert grid[0] == 30.0 and grid[-1] == 120.0 and grid.size == 19


def test_sweep_stays_within_configured_cap():
    rng = np.random.default_rng(0)
    users = generate_users_hotspot(rng=rng)
    base = Config()
    cfg = replace(base, network=replace(base.network, n_drones=4, z_max_m=120.0))
    sc, entries = sweep_kmeans_global_altitude(cfg, users, seed=0)
    assert max(e.altitude_m for e in entries) == 120.0
    assert np.all(sc.drone_positions[:, 2] <= 120.0)
