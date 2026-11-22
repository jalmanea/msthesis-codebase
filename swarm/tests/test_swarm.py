"""Tests for dronecomm.swarm: utilities, association and best-response dynamics."""

from dataclasses import replace

import numpy as np
import pytest

from dronecomm import joint, swarm
from dronecomm.config import Config


@pytest.fixture
def problem():
    cfg = Config()
    rng = np.random.default_rng(3)
    centres = np.array([[300.0, 300.0], [700.0, 650.0], [400.0, 800.0]])
    xy = np.concatenate([c + rng.normal(0.0, 60.0, (40, 2)) for c in centres])
    users = np.column_stack([np.clip(xy, 0.0, cfg.scenario.area_size_m), np.zeros(len(xy))])
    phys = joint.Physics.from_config(cfg, users.shape[0])
    dep = joint.Deployment(np.column_stack([centres + 80.0, np.full(3, phys.z_min)]), np.zeros((3, 2)))
    return phys, users, dep


def test_strongest_association(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    assoc, adm, band = swarm.evaluate(phys, power)
    assert np.array_equal(assoc, power.argmax(axis=0))
    assert np.all((adm >= 0.0) & (adm <= 1.0)) and np.all(band >= 0.0)


def test_selfish_utilities_sum_to_global(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    total = 0.0
    for j in range(3):
        mv = swarm.AgentMove(phys, users, power, j, swarm.Utility("selfish"), dep.pos[j])
        total += mv(np.concatenate([dep.pos[j], dep.t[j]]))[0]
    assert total == pytest.approx(swarm.served(phys, users, dep))


def test_unbounded_window_is_global(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    v = np.array([500.0, 500.0, phys.z_max, 0.3, -0.1])
    win = swarm.AgentMove(phys, users, power, 1, swarm.Utility("window", 1e9), dep.pos[1])(v)
    glob = swarm.AgentMove(phys, users, power, 1, swarm.Utility("global"), dep.pos[1])(v)
    assert win == pytest.approx(glob)


def test_global_sequential_play_never_loses(problem):
    phys, users, dep = problem
    r = swarm.run(phys, users, dep, swarm.Utility("global"), "sequential", max_rounds=2, budget=60)
    assert all(b >= a - 1e-12 for a, b in zip(r.history, r.history[1:]))


def test_best_response_respects_step_bound(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    v, before, after, _ = swarm.best_response(phys, users, dep, power, 0, swarm.Utility("selfish"),
                                              pointing=False, budget=60, max_step_m=25.0)
    assert np.all(np.abs(v[:2] - dep.pos[0, :2]) <= 25.0 + 1e-9)
    assert not joint._better(before, after)


def test_unbounded_information_is_exact(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    v = np.array([450.0, 420.0, phys.z_max, 0.2, 0.4])
    for util in (swarm.Utility("selfish"), swarm.Utility("window", 200.0), swarm.Utility("global")):
        exact = swarm.AgentMove(phys, users, power, 2, util, dep.pos[2])(v)
        local = swarm.AgentMove(phys, users, power, 2, util, dep.pos[2], 1e9, dep.pos)(v)
        assert local == pytest.approx(exact)


def test_prediction_error_of_zero_is_exact(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    v = np.array([450.0, 420.0, phys.z_max, 0.2, 0.4])
    zero = swarm.PredictionError(0.0, 0.0)
    for util in (swarm.Utility("selfish"), swarm.Utility("global")):
        exact = swarm.AgentMove(phys, users, power, 2, util, dep.pos[2])(v)
        noisy = swarm.AgentMove(phys, users, power, 2, util, dep.pos[2], err=zero,
                                rng=np.random.default_rng(0))(v)
        assert noisy == pytest.approx(exact)


def test_prediction_error_scale(problem):
    err = swarm.PredictionError(10.0, 4.0)
    users = np.zeros((20000, 3))
    seen, gain = err.draw(users, np.random.default_rng(1))
    assert np.quantile(np.hypot(seen[:, 0], seen[:, 1]), 0.8) == pytest.approx(10.0, rel=0.03)
    assert np.std(10.0 * np.log10(gain)) == pytest.approx(4.0, rel=0.03)
    assert np.array_equal(seen[:, 2], users[:, 2])


def test_prediction_error_keeps_the_dynamics_stream(problem):
    """Errors come from their own stream: with errors of zero the run is the run without errors."""
    phys, users, dep = problem
    util = swarm.Utility("selfish")
    a = swarm.run(phys, users, dep, util, "simultaneous", inertia=0.5, max_rounds=2, budget=40, seed=5)
    b = swarm.run(phys, users, dep, util, "simultaneous", inertia=0.5, max_rounds=2, budget=40, seed=5,
                  err=swarm.PredictionError(0.0, 0.0))
    assert a.history == pytest.approx(b.history) and a.moves == b.moves
    c = swarm.run(phys, users, dep, util, "simultaneous", inertia=0.5, max_rounds=2, budget=40, seed=5,
                  err=swarm.PredictionError(50.0, 6.0))
    d = swarm.run(phys, users, dep, util, "simultaneous", inertia=0.5, max_rounds=2, budget=40, seed=5,
                  err=swarm.PredictionError(50.0, 6.0))
    assert c.history == d.history


def test_information_radius_hides_far_users(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    mv = swarm.AgentMove(phys, users, power, 0, swarm.Utility("global"), dep.pos[0], 1.0, dep.pos)
    assert np.all(swarm.strongest(power)[mv.cols] == 0)        # sees only its own cell


def test_agent_that_observes_no_user(problem):
    phys, users, dep = problem
    far = dep.copy()
    far.pos[0, :2] = [phys.area_m, 0.0]                          # a corner with no user near it
    power = joint.rx_power(phys, far.pos, far.bore, users)
    mv = swarm.AgentMove(phys, users, power, 0, swarm.Utility("global"), far.pos[0], 50.0, far.pos)
    assert mv.cols.size == 0
    assert mv(np.concatenate([far.pos[0], far.t[0]])) == (0.0, 0.0)


def test_beacons_carry_the_unserved_demand(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    assoc, adm, _ = swarm.evaluate(phys, power)
    cent, mass = swarm.beacons(users, assoc, adm, np.ones(3, dtype=bool), per_drone=99)
    assert mass.sum() == pytest.approx(np.sum(1.0 - adm))
    assert np.all(np.diff(mass) <= 1e-12)
    _, none = swarm.beacons(users, assoc, np.ones_like(adm), np.ones(3, dtype=bool))
    assert none.size == 0


def test_global_contribution_is_served_with_minus_without(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    mv = swarm.AgentMove(phys, users, power, 1, swarm.Utility("global"), dep.pos[1])
    mc = swarm.contribution(mv, np.concatenate([dep.pos[1], dep.t[1]]))
    without = joint.Deployment(np.delete(dep.pos, 1, axis=0), np.delete(dep.t, 1, axis=0))
    assert mc == pytest.approx(swarm.served(phys, users, dep) - swarm.served(phys, users, without))


@pytest.fixture
def heavy(problem):
    """The same users with demand such that one drone carries at most 40 % of it."""
    phys, users, dep = problem
    k = swarm.kappa(phys, users.shape[0])
    return replace(phys, lam=phys.lam * k / 0.4), users, dep


def test_exploration_finds_demand_far_away(heavy):
    phys, users, dep = heavy
    lost = dep.copy()
    lost.pos[0, :2] = [phys.area_m, 0.0]                         # a corner with no user near it
    served = {x: swarm.run_fleet(phys, users, lost, 3, swarm.Utility("global"), swarm.FleetRules(explore=x),
                                 max_rounds=2, budget=60).served for x in (False, True)}
    assert served[True] > served[False] + 1e-3


def test_land_and_call_rules(heavy):
    phys, users, dep = heavy
    shed = swarm.run_fleet(phys, users, dep, 3, swarm.Utility("global"),
                           swarm.FleetRules(explore=False, land_alpha=1e9), max_rounds=3, budget=30)
    assert shed.active.sum() == 1 and all(e[1] == "land" for e in shed.events)
    grow = swarm.run_fleet(phys, users, dep, 1, swarm.Utility("selfish"),
                           swarm.FleetRules(explore=False, call_alpha=0.0), max_rounds=3, budget=30)
    assert grow.active.sum() == 3 and all(e[1] == "call" for e in grow.events)


def test_area_call_adds_up_thin_unserved_demand(heavy):
    phys, users, dep = heavy
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    assoc, adm, _ = swarm.evaluate(phys, power)
    k = swarm.kappa(phys, users.shape[0])
    per_cell = max(np.sum(1.0 - adm[assoc == j]) for j in range(3)) / users.shape[0]
    total = np.sum(1.0 - adm) / users.shape[0]
    assert per_cell < total                                     # the unserved demand is spread over cells
    alpha = 0.5 * (per_cell + total) / k                       # above any one cell, below the area
    active = np.ones(3, dtype=bool)
    cell = swarm.FleetRules(explore=False, call_alpha=alpha)
    area = swarm.FleetRules(explore=False, call_alpha=alpha, call_area_m=1e9)
    for j in range(3):
        assert swarm.call_target(users, dep.pos, active, assoc, adm, j, cell, k) is None
        assert swarm.call_target(users, dep.pos, active, assoc, adm, j, area, k) is not None


def test_call_check_refuses_a_drone_that_would_land(heavy):
    phys, users, dep = heavy
    fleet = joint.Deployment(np.vstack([dep.pos, dep.pos[:1]]), np.zeros((4, 2)))
    loose = swarm.FleetRules(explore=False, land_alpha=1e9, call_alpha=0.0, call_area_m=1e9)
    strict = swarm.FleetRules(explore=False, land_alpha=1e9, call_alpha=0.0, call_area_m=1e9, call_check=True)
    calls = lambda r: sum(e[1] == "call" for e in r.events)
    kw = dict(max_rounds=1, budget=10)
    assert calls(swarm.run_fleet(phys, users, fleet, 3, swarm.Utility("global"), loose, **kw)) >= 1
    assert calls(swarm.run_fleet(phys, users, fleet, 3, swarm.Utility("global"), strict, **kw)) == 0


def test_hysteresis_delays_calls_and_shields_new_drones(heavy):
    phys, users, dep = heavy
    kw = dict(max_rounds=6, budget=10)
    first_call = lambda r: min(e[0] for e in r.events if e[1] == "call")
    eager = swarm.FleetRules(explore=False, call_alpha=0.0)
    patient = swarm.FleetRules(explore=False, call_alpha=0.0, call_patience=3)
    assert first_call(swarm.run_fleet(phys, users, dep, 1, swarm.Utility("selfish"), eager, **kw)) == 1
    assert first_call(swarm.run_fleet(phys, users, dep, 1, swarm.Utility("selfish"), patient, **kw)) == 3
    churn = swarm.FleetRules(explore=False, land_alpha=1e9, call_alpha=0.0, land_grace=2)
    r = swarm.run_fleet(phys, users, dep, 1, swarm.Utility("global"), churn, **kw)
    called = {}
    for rnd, kind, d in r.events:
        if kind == "call":
            called[d] = rnd
        elif d in called:
            assert rnd > called[d] + 2                          # no landing inside the grace rounds
    assert any(kind == "land" for _, kind, d in r.events if d in called)


def test_strongest_association_search(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    mv = joint._DroneMove(phys, users, dep, power, None, 1, "strongest")
    v = np.array([520.0, 480.0, phys.z_max, 0.2, 0.1])
    moved = dep.copy()
    moved.pos[1], moved.t[1] = v[:3], v[3:5]
    assert mv(v)[0] == pytest.approx(swarm.served(phys, users, moved))
    r = joint.optimise(phys, users, dep, "strongest", "xyzt", max_rounds=2, max_evals_per_drone=60)
    assert r.served == pytest.approx(swarm.served(phys, users, r.dep))
    assert r.served >= swarm.served(phys, users, dep) - 1e-12


def test_brute_response_never_loses(problem):
    phys, users, dep = problem
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    v, before, after, evals = swarm.brute_response(phys, users, dep, power, 0, swarm.Utility("selfish"),
                                                   pointing=False, xy_step_m=250.0, budget=40)
    grid = len(np.arange(0.0, phys.area_m + 1e-9, 250.0)) ** 2 * len(phys.altitude_grid())
    assert after[0] >= before[0] and evals > grid
