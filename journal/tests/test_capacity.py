"""Tests for the load-aware capacity model."""

import numpy as np

from dronecomm.capacity import serve, served_share, truncated_shannon_se
from dronecomm.config import DemandConfig


def test_truncated_shannon_matches_tr36942_dl():
    """alpha = 0.6, SINR_min = -10 dB, cap 4.4 bit/s/Hz (reached at about 22 dB)."""
    se = truncated_shannon_se(np.array([-11.0, -10.0, 0.0, 21.0, 23.0, 40.0]))
    assert se[0] == 0.0
    np.testing.assert_allclose(se[1], 0.6 * np.log2(1.1))
    np.testing.assert_allclose(se[2], 0.6)
    assert se[3] < 4.4
    assert se[4] == se[5] == 4.4


def test_round_robin_splits_the_band():
    sinr = np.zeros(4)                          # 0.6 bit/s/Hz each
    assoc = np.array([0, 0, 0, 1])
    rate, adm = serve(sinr, assoc, n_drones=2, bandwidth_hz=10e6)
    np.testing.assert_allclose(rate, [2e6, 2e6, 2e6, 6e6])
    np.testing.assert_array_equal(adm, 1.0)


def test_points_stand_for_many_active_users():
    sinr = np.zeros(10)
    assoc = np.zeros(10, dtype=int)
    rate, _ = serve(sinr, assoc, n_drones=1, bandwidth_hz=10e6, active_per_point=0.2)
    np.testing.assert_allclose(rate, 6e6 / 2)          # 10 points x 0.2 = 2 active
    rate, _ = serve(sinr[:2], assoc[:2], n_drones=1, bandwidth_hz=10e6, active_per_point=0.2)
    np.testing.assert_allclose(rate, 6e6)              # 0.4 active -> at least one share
    rate, _ = serve(sinr, assoc, n_drones=1, bandwidth_hz=10e6, active_per_point=50.0)
    np.testing.assert_allclose(rate, 6e6 / 500)


def test_cap_admits_best_sinr_first_and_splits_a_point():
    sinr = np.array([5.0, 20.0, 10.0, 0.0])
    assoc = np.zeros(4, dtype=int)
    rate, adm = serve(sinr, assoc, n_drones=1, bandwidth_hz=10e6, max_users_per_cell=2)
    np.testing.assert_array_equal(adm, [0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(rate[1], 10e6 * truncated_shannon_se(np.array([20.0]))[0] / 2)
    # 3 active users per point, cap 4: best point fully admitted, second one third
    _, adm = serve(sinr, assoc, n_drones=1, bandwidth_hz=10e6, active_per_point=3.0, max_users_per_cell=4)
    np.testing.assert_allclose(adm, [0.0, 1.0, 1.0 / 3.0, 0.0])


def test_served_share_counts_admitted_users_above_minimum():
    rate = np.array([0.0, 1e5, 2e5, 5e5])
    assert served_share(rate, np.ones(4), 2e5) == 0.5
    assert served_share(rate, np.array([1.0, 1.0, 0.5, 0.0]), 2e5) == 0.125


def test_demand_active_per_point():
    assert DemandConfig().active_per_point(800) == 1.0
    d = DemandConfig(population=41000, device_share=0.82, activity_factor=0.2)
    np.testing.assert_allclose(d.active_per_point(800), 41000 / 800 * 0.82 * 0.2)


def test_user_sinr_matches_evaluate_single_coverage():
    from dronecomm.capacity import user_sinr_db
    from dronecomm.config import Config
    from dronecomm.heuristic import deploy_kmeans_baseline
    from dronecomm.optimize import _build_models, _evaluate_single
    from dronecomm.scenario import generate_users_hotspot
    cfg = Config.from_yaml("configs/grounded.yaml")
    users = generate_users_hotspot(rng=np.random.default_rng(3))
    sc = deploy_kmeans_baseline(cfg, users, seed=3)
    dl, bh, ch = _build_models(cfg)
    sinr_db, _ = user_sinr_db(sc, cfg, dl, ch)
    cov = _evaluate_single(sc, cfg, dl, bh, ch).coverage_fraction
    assert np.mean(sinr_db >= cfg.network.sinr_threshold_db) == cov


def test_gbr_admits_cheapest_users_until_band_is_full():
    from dronecomm.capacity import admit_gbr
    # se at 0 dB = 0.6 bit/s/Hz -> a 1.2 Mb/s user needs 1.2e6 / (10e6 * 0.6) = 0.2 of the band
    sinr = np.array([0.0] * 7)
    assoc = np.zeros(7, dtype=int)
    adm = admit_gbr(sinr, assoc, 1, 10e6, 1.2e6)
    np.testing.assert_allclose(adm, [1, 1, 1, 1, 1, 0, 0])
    # 2.5 users per point: 0.5 of the band each; 2 points fill it
    adm = admit_gbr(sinr[:3], assoc[:3], 1, 10e6, 1.2e6, active_per_point=2.5)
    np.testing.assert_allclose(adm, [1, 1, 0])


def test_gbr_prefers_high_sinr_and_skips_unusable_users():
    from dronecomm.capacity import admit_gbr
    sinr = np.array([-20.0, 0.0, 22.0, 0.0])           # -20 dB: below SINR_min, never admitted
    assoc = np.array([0, 0, 0, 1])
    adm = admit_gbr(sinr, assoc, 2, 10e6, 30e6)         # 22 dB user needs 30/44 = 0.68 of the band
    np.testing.assert_allclose(adm, [0.0, 0.0, 1.0, 0.0])   # 0 dB users need 5x the band: none fit
    adm = admit_gbr(sinr, assoc, 2, 10e6, 3e6)          # 0 dB users need 0.5, 22 dB needs 0.068
    np.testing.assert_allclose(adm, [0.0, 1.0, 1.0, 1.0])


def test_gbr_cap_limits_admitted_users():
    from dronecomm.capacity import admit_gbr
    sinr = np.full(5, 22.0)
    adm = admit_gbr(sinr, np.zeros(5, dtype=int), 1, 10e6, 38e3, active_per_point=10.0,
                    max_users_per_cell=25)
    np.testing.assert_allclose(adm, [1.0, 1.0, 0.5, 0.0, 0.0])
