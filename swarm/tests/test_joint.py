"""Tests for joint placement and pointing (dronecomm.joint)."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from dronecomm.capacity import admit_gbr
from dronecomm.channel import ChannelModel, ENVIRONMENTS
from dronecomm.antenna import ParametricAntenna
from dronecomm.config import Config
from dronecomm.interference import downlink_power_matrix
from dronecomm.joint import (Deployment, Physics, _DroneMove, admit, associate, boresight,
                             lp_association, min_enclosing_circle, nearest, optimise, rx_power,
                             score, sinr_db, tilt_deg)


def _cfg(population: float, r_min: float = 384e3) -> Config:
    base = Config.from_yaml("configs/grounded.yaml")
    return replace(base, demand=replace(base.demand, population=population, r_min_bps=r_min))


def _ground(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, np.zeros(xy.shape[0])])


def test_min_enclosing_circle_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0, 100, size=(rng.integers(1, 12), 2))
        c, r = min_enclosing_circle(p)
        assert np.all(np.hypot(*(p - c).T) <= r + 1e-6)
        best = 0.0 if len(p) == 1 else np.inf
        for a, b in combinations(p, 2):                   # circles on a pair
            o, rr = (a + b) / 2, np.hypot(*(a - b)) / 2
            if np.all(np.hypot(*(p - o).T) <= rr + 1e-6):
                best = min(best, rr)
        for a, b, cc in combinations(p, 3):               # circumcircles
            d = 2 * (a[0] * (b[1] - cc[1]) + b[0] * (cc[1] - a[1]) + cc[0] * (a[1] - b[1]))
            if abs(d) < 1e-9:
                continue
            ux = (a @ a * (b[1] - cc[1]) + b @ b * (cc[1] - a[1]) + cc @ cc * (a[1] - b[1])) / d
            uy = (a @ a * (cc[0] - b[0]) + b @ b * (a[0] - cc[0]) + cc @ cc * (b[0] - a[0])) / d
            o = np.array([ux, uy])
            rr = np.hypot(*(a - o))
            if np.all(np.hypot(*(p - o).T) <= rr + 1e-6):
                best = min(best, rr)
        assert r == pytest.approx(best, abs=1e-6)


def test_rx_power_matches_simulator_with_tilt():
    cfg = _cfg(1000.0)
    rng = np.random.default_rng(1)
    users = _ground(rng.uniform(0, 2000, size=(300, 2)))
    pos = np.column_stack([rng.uniform(0, 2000, size=(5, 2)), rng.uniform(50, 120, size=5)])
    tilt, az = rng.uniform(0, 1.2, size=5), rng.uniform(-np.pi, np.pi, size=5)
    t = np.column_stack([np.tan(tilt) * np.cos(az), np.tan(tilt) * np.sin(az)])
    phys = Physics.from_config(cfg, users.shape[0])
    ant = ParametricAntenna(cfg.antenna.dl_g_max_dbi, cfg.antenna.dl_beamwidth_deg, cfg.antenna.dl_sla_db)
    ref = downlink_power_matrix(pos, tilt, az, users, ant, ChannelModel(ENVIRONMENTS["urban"], 2.0e9),
                                10 ** ((cfg.network.p_tx_dl_dbm - 30) / 10))
    np.testing.assert_allclose(rx_power(phys, pos, boresight(t), users), ref, rtol=1e-9)
    np.testing.assert_allclose(tilt_deg(t), np.degrees(tilt), atol=1e-9)


def test_admission_matches_capacity_module():
    cfg = _cfg(20000.0)
    rng = np.random.default_rng(2)
    m, n = 500, 7
    s_db = rng.uniform(-15, 30, size=m)
    assoc = rng.integers(0, n, size=m)
    phys = Physics.from_config(cfg, m)
    adm, _ = admit(phys, s_db, assoc, n)
    ref = admit_gbr(s_db, assoc, n, phys.bandwidth_hz, phys.r_min_bps, phys.lam)
    np.testing.assert_allclose(adm, ref, atol=1e-12)


def test_nadir_max_min_power_is_at_the_enclosing_circle_centre():
    """Proposition: at nadir and fixed altitude, power falls with horizontal distance, so the
    position maximising the weakest user's power is the minimum-enclosing-circle centre."""
    rng = np.random.default_rng(7)
    users = _ground(rng.normal([1000, 900], [150, 80], size=(40, 2)))
    phys = Physics.from_config(_cfg(100.0), users.shape[0])
    c, _ = min_enclosing_circle(users)
    for z in (50.0, 120.0):
        def worst(xy):
            return rx_power(phys, np.array([[*xy, z]]), boresight(np.zeros((1, 2))), users).min()
        at_c = worst(c)
        for xy in c + rng.uniform(-200, 200, size=(200, 2)):
            assert worst(xy) <= at_c * (1 + 1e-9)


def test_pointing_can_beat_nadir_over_the_centre():
    """With a 44.5 deg beam a nadir drone at 120 m lights a 49 m disc; a wide ring is better
    served from outside it with a tilted beam. Pointing must therefore stay a decision variable."""
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    users = _ground(np.column_stack([1000 + 500 * np.cos(ang), 1000 + 500 * np.sin(ang)]))
    phys = Physics.from_config(_cfg(200.0), users.shape[0])
    centre = Deployment(np.array([[1000.0, 1000.0, 120.0]]), np.zeros((1, 2)))
    nadir = optimise(phys, users, centre, assoc_mode="nearest", move="xyz")
    tilted = optimise(phys, users, centre, assoc_mode="nearest", move="xyzt")
    assert tilted.served > nadir.served + 0.1


def test_two_far_clusters_get_one_drone_each():
    rng = np.random.default_rng(3)
    a, b = np.array([400.0, 400.0]), np.array([1500.0, 1400.0])
    users = _ground(np.vstack([rng.normal(a, 40, size=(30, 2)), rng.normal(b, 40, size=(30, 2))]))
    phys = Physics.from_config(_cfg(300.0), users.shape[0])
    mid = (a + b) / 2
    start = Deployment(np.array([[*(mid - 150), 120.0], [*(mid + 150), 120.0]]), np.zeros((2, 2)))
    res = optimise(phys, users, start, assoc_mode="free", move="xyz")
    assert res.served == pytest.approx(1.0)
    assert len(set(res.assoc[:30])) == 1 and len(set(res.assoc[30:])) == 1
    assert res.assoc[0] != res.assoc[30]


@pytest.mark.parametrize("mode", ["nearest", "free"])
def test_incremental_move_equals_full_recompute(mode):
    rng = np.random.default_rng(4)
    users = _ground(rng.uniform(0, 2000, size=(200, 2)))
    phys = Physics.from_config(_cfg(20000.0, 38e3), users.shape[0])
    dep = Deployment(np.column_stack([rng.uniform(0, 2000, size=(6, 2)), np.full(6, 100.0)]),
                     rng.normal(0, 0.2, size=(6, 2)))
    assoc, _ = associate(phys, users, dep, mode)
    power = rx_power(phys, dep.pos, dep.bore, users)
    mv = _DroneMove(phys, users, dep, power, None if mode == "nearest" else assoc, 2)
    v = np.array([700.0, 1300.0, 75.0, 0.1, -0.3])
    got = mv(v)
    moved = dep.copy()
    moved.pos[2], moved.t[2] = v[:3], v[3:]
    want = score(phys, users, moved, nearest(moved.pos, users) if mode == "nearest" else assoc)
    assert got[0] == pytest.approx(want[0], abs=1e-12)
    assert got[1] == pytest.approx(want[1], rel=1e-9)


def test_lp_bounds_every_association_and_free_beats_nearest():
    rng = np.random.default_rng(5)
    users = _ground(np.vstack([rng.normal([600, 600], 120, size=(150, 2)),
                               rng.uniform(0, 2000, size=(50, 2))]))
    phys = Physics.from_config(_cfg(15000.0), users.shape[0])
    dep = Deployment(np.column_stack([rng.uniform(300, 1700, size=(8, 2)), np.full(8, 110.0)]),
                     np.zeros((8, 2)))
    power = rx_power(phys, dep.pos, dep.bore, users)
    _, bound = lp_association(phys, power)
    near = nearest(dep.pos, users)
    free, _ = associate(phys, users, dep, "free")
    for a in (near, free, np.argmax(power, axis=0), rng.integers(0, 8, size=users.shape[0])):
        assert score(phys, users, dep, a)[0] <= bound + 1e-9
    assert score(phys, users, dep, free)[0] >= score(phys, users, dep, near)[0] - 1e-12


def test_optimise_never_serves_less_than_its_start():
    rng = np.random.default_rng(6)
    users = _ground(rng.uniform(0, 2000, size=(250, 2)))
    phys = Physics.from_config(_cfg(30000.0, 38e3), users.shape[0])
    start = Deployment(np.column_stack([rng.uniform(0, 2000, size=(5, 2)), np.full(5, 120.0)]),
                       np.zeros((5, 2)))
    base = score(phys, users, start, nearest(start.pos, users))[0]
    for mode, move in (("nearest", "xyz"), ("free", ""), ("free", "xyzt")):
        res = optimise(phys, users, start, assoc_mode=mode, move=move, max_rounds=2)
        assert res.served >= base - 1e-12
        assert res.served == pytest.approx(score(phys, users, res.dep, res.assoc)[0], abs=1e-12)
        assert np.all(res.dep.pos[:, 2] >= phys.z_min) and np.all(res.dep.pos[:, 2] <= phys.z_max)


def test_sinr_db_is_signal_over_noise_plus_others():
    phys = Physics.from_config(_cfg(1000.0), 10)
    s = np.array([1e-9])
    tot = np.array([3e-9])
    assert sinr_db(phys, s, tot)[0] == pytest.approx(10 * np.log10(1e-9 / (phys.noise_w + 2e-9)))


def test_continued_search_keeps_the_previous_association():
    rng = np.random.default_rng(8)
    users = _ground(rng.uniform(0, 2000, size=(250, 2)))
    phys = Physics.from_config(_cfg(30000.0, 38e3), users.shape[0])
    start = Deployment(np.column_stack([rng.uniform(0, 2000, size=(6, 2)), np.full(6, 120.0)]),
                       np.zeros((6, 2)))
    first = optimise(phys, users, start, assoc_mode="free", move="xyz", max_rounds=2)
    cont = optimise(phys, users, first.dep, assoc_mode="free", move="xyzt", max_rounds=1,
                    start_assoc=first.assoc)
    assert cont.served >= first.served - 1e-12
