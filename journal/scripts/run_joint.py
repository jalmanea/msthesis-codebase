"""Joint placement and pointing (journal Sec. sec:joint) on the paper's scenarios. Runs on Ibex.

A unit is (scenario, tier, snapshot or seed, N). For each unit the paper's deployment is rebuilt
(K-means, sweep + greedy altitudes, nadir, nearest association; dronecomm.fleet) and improved by
dronecomm.joint in a chain, one row per method:

    greedy        the paper's deployment
    greedy_free   same geometry, association chosen by the LP (joint.lp_association)
    xyz_nearest   positions and altitudes optimised, nearest association (from greedy)
    xyz           positions and altitudes optimised, free association (from xyz_nearest: the free
                  association is fixed while a drone moves, so starting there is the stronger search)
    xyzt          and antenna pointing (from xyz)

Every row is also scored by the paper's own code (interference.downlink_power_matrix,
sinr.compute_sinr, capacity.admit_gbr) as served_paper_pct.

Stages (results/joint/<stage>/...):
    check   the full chain at the lower bound and at the paper's N* on one layout per cell, plus brute
            force on tiny instances; `--check` passes only if every agreement and ordering holds
    coarse  N from the capacity lower bound to just above the paper's N*, on 10 synthetic seeds or
            24 Milan snapshots
    dense   every N of configs/joint_windows.json (scripts/journal_data.py writes it from coarse)
    full_coarse, full_dense
            the same on the units of Table tab:fleet (run_fleet.py --refine: 40 synthetic seeds, 24
            snapshots x 6 K-means seeds); full_coarse adds greedy-only units on the N of the paper's
            refinement window (configs/refine_windows.json, variant tether, whose access side is the
            grounded model), so the greedy column reproduces Table tab:fleet; full_dense runs every N of
            configs/joint_windows_full.json
    e2e     end to end (Sec. sec:e2e) for every deployment of full_coarse and full_dense, read from their
            .npz: each drone's admitted load sent through backhaul.end_to_end under a tethered gateway and
            a 100 Mb/s gateway (variant grounded; the paper's 20 Mb/s satellite upload cap is applied later,
            in scripts/journal_data.py), for every tree; `--check` also requires each recomputed served
            share to equal the one the source CSV recorded
    assoc   every deployment of full_coarse and full_dense scored again with the points on the nearest drone
            (the paper's rule) and on the strongest one (what a phone selects), geometry unchanged
    multi   restarts: on the N of configs/joint_windows_multi.json (scripts/journal_data.py writes it from
            full_coarse + full_dense: the N between pointing's N* and positions' N*), the whole chain again from
            MULTI_STARTS further K-means starts (K-means seed + 1000 k, same users); the analysis keeps each
            unit's best start, the full_* run counting as start 0
    tilt    the last link (xyzt from the saved xyz deployment and association) again with the tilt capped at
            each of TILT_CAPS_DEG instead of 80 degrees, on the same N
    gmulti  the greedy deployment alone from the same MULTI_STARTS further K-means starts, on the N of the
            paper's refinement window (where greedy's N* is read), so restarts are compared like for like

python scripts/run_joint.py --stage check --list
python scripts/run_joint.py --stage coarse --shard 0/8 --procs 16
python scripts/run_joint.py --stage coarse --shard 0/8 --check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import replace
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dronecomm import backhaul, joint  # noqa: E402
from dronecomm.capacity import admit_gbr  # noqa: E402
from dronecomm.fleet import GridService, greedy, sweep  # noqa: E402
from dronecomm.heuristic import _kmeans_centroids  # noqa: E402
from dronecomm.interference import downlink_power_matrix  # noqa: E402
from dronecomm.optimize import _build_models  # noqa: E402
from dronecomm.sinr import compute_sinr  # noqa: E402
from run_fleet import (BASE_SEED, MILAN_TILES, REFINE_MILAN_SEEDS, REFINE_SYN_SEEDS, REFINE_WINDOWS,  # noqa: E402
                       SCENARIOS, TIERS, TWO_WAY, VARIANTS, _users)

OUT = ROOT / "results" / "joint"
WINDOWS = ROOT / "configs" / "joint_windows.json"
WINDOWS_FULL = ROOT / "configs" / "joint_windows_full.json"
WINDOWS_MULTI = ROOT / "configs" / "joint_windows_multi.json"
STAGES = ["check", "coarse", "dense", "full_coarse", "full_dense", "e2e", "assoc", "multi", "tilt", "gmulti"]
MULTI_STARTS = 2
MULTI_SEED_STEP = 1000           # K-means seed of start k: seed + 1000 k (no full_* unit uses these)
TILT_CAPS_DEG = (30.0, 45.0, 60.0)
CHAIN = ("xyz_nearest", "xyz", "xyzt")
E2E_SOURCES = ("full_coarse", "full_dense")
E2E_VARIANTS = ("tether", "grounded")   # tethered gateway; a 100 Mb/s gateway (the satellite cap is applied later)
# (scenario, tier) -> greedy N* of Table tab:fleet (refined); used only to choose the fleet sizes
CELLS = {
    ("urban_broad", "voice"): 16,
    ("milan_p100", "voice"): 21,
    ("milan_p075", "video"): 68,
    ("milan_p025", "broadband"): 105,
    ("shelter_clustered", "video"): 54,
    ("milan_p100", "video"): 270,
}
SYN_SEEDS = 10
COARSE_POINTS = 7
METHODS = ("greedy", "greedy_free", "xyz_nearest", "xyz", "xyzt")
COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "active_per_point", "n_drones",
           "lower_bound", "method", "served_pct", "served_paper_pct", "served_fleet_pct", "lp_bound_pct",
           "mean_altitude_m", "min_altitude_m", "max_altitude_m", "mean_tilt_deg", "max_tilt_deg",
           "n_tilted", "rounds", "evals", "time_s"]
E2E_COLUMNS = ["src", "scenario", "tier", "snapshot", "seed", "n_drones", "method", "served_pct",
               "served_src_pct", *[f"{c}_{v}_{t}" for v in E2E_VARIANTS for t in backhaul.TOPOLOGIES
                                   for c in ("served_e2e_pct", "relays")]]
ASSOC_COLUMNS = ["src", "scenario", "tier", "snapshot", "seed", "n_drones", "method", "served_pct",
                 "served_src_pct", "served_nearest_pct", "served_strongest_pct", "strongest_is_assoc_pct"]
MULTI_COLUMNS = COLUMNS + ["start", "kmeans_seed", "served_strongest_pct"]
TILT_COLUMNS = COLUMNS + ["tilt_max_deg", "served_strongest_pct"]
BRUTE_COLUMNS = ["case", "n_drones", "move", "assoc", "served_solver", "served_grid", "served_grid_median",
                 "served_grid_min", "grid_points", "time_s"]


def config_for(scenario: str, tier: str, variant: str = "grounded"):
    _, pop, share, act = SCENARIOS[scenario]
    base = VARIANTS[variant]()
    dem = replace(base.demand, population=pop, r_min_bps=TIERS[tier],
                  **({} if share is None else {"device_share": share}),
                  **({} if act is None else {"activity_factor": act}))
    return replace(base, demand=dem, backhaul=replace(base.backhaul, two_way=tier in TWO_WAY))


def n_points(scenario: str) -> int:
    return 800 if scenario.startswith("milan") else 1000


def lower_bound(scenario: str, tier: str) -> int:
    """No deployment serves the demand with fewer drones: each carries at most B x se_max of GBR traffic."""
    cfg = config_for(scenario, tier)
    traffic = cfg.demand.active_per_point(n_points(scenario)) * n_points(scenario) * TIERS[tier]
    return int(math.ceil(traffic / (cfg.network.bandwidth_hz * cfg.capacity.se_max_bps_hz) - 1e-9))


def coarse_fleet(scenario: str, tier: str) -> list[int]:
    lb, top = lower_bound(scenario, tier), CELLS[(scenario, tier)]
    ns = {int(round(x)) for x in np.geomspace(lb, top, COARSE_POINTS)} | {top, int(math.ceil(1.15 * top))}
    return sorted(ns)


def layouts(scenario: str, stage: str) -> list[tuple[int | None, int]]:
    full = stage.startswith("full")
    if scenario.startswith("milan"):
        n_snap = len(json.loads((MILAN_TILES / "tiles_metadata.json").read_text())["snapshots"])
        snaps = [0] if stage == "check" else range(n_snap)
        return [(s, BASE_SEED + r) for s in snaps for r in range(REFINE_MILAN_SEEDS if full else 1)]
    return [(None, BASE_SEED + r) for r in range(1 if stage == "check" else REFINE_SYN_SEEDS if full else SYN_SEEDS)]


def paper_window(scenario: str, tier: str) -> list[int]:
    """The N of the paper's refinement window for this cell (access side: variant tether)."""
    for w in json.loads(REFINE_WINDOWS.read_text()):
        if (w["variant"], w["scenario"], w["tier"]) == ("tether", scenario, tier):
            return w["fleet"]
    raise KeyError((scenario, tier))


def units(stage: str) -> list[dict]:
    if stage in ("e2e", "assoc"):
        out = [{**u, "kind": stage, "stage": stage, "src": src} for src in E2E_SOURCES for u in units(src)]
        return sorted(out, key=lambda u: -u["n"])
    if stage == "gmulti":
        return sorted(({"kind": stage, "stage": stage, "scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n}
                       for (sc, t) in CELLS for (snap, seed) in layouts(sc, "full") for n in paper_window(sc, t)),
                      key=lambda u: -u["n"])
    if stage in ("multi", "tilt"):
        out = [{"kind": stage, "stage": stage, "scenario": w["scenario"], "tier": w["tier"], "snapshot": snap,
                "seed": seed, "n": n}
               for w in json.loads(WINDOWS_MULTI.read_text()) for (snap, seed) in layouts(w["scenario"], "full")
               for n in w["fleet"]]
        return sorted(out, key=lambda u: -u["n"] * (8 if u["tier"] == "video" else 1))
    out = []
    windows = {}
    if stage in ("dense", "full_dense"):
        src = WINDOWS if stage == "dense" else WINDOWS_FULL
        windows = {(w["scenario"], w["tier"]): w["fleet"] for w in json.loads(src.read_text())}
    for (scen, tier) in CELLS:
        greedy_only = []
        if stage == "check":
            fleet = [lower_bound(scen, tier), CELLS[(scen, tier)]]
        elif stage in ("coarse", "full_coarse"):
            fleet = coarse_fleet(scen, tier)
            if stage == "full_coarse":
                greedy_only = sorted(set(paper_window(scen, tier)) - set(fleet))
        else:
            fleet = windows.get((scen, tier), [])
        for (snap, seed) in layouts(scen, stage):
            for kind, ns in (("chain", fleet), ("greedy", greedy_only)):
                out += [{"kind": kind, "stage": stage, "scenario": scen, "tier": tier,
                         "snapshot": snap, "seed": seed, "n": n} for n in ns]
    if stage == "check":
        out += [{"kind": "brute", "stage": stage, "case": i, "n": 1 + i % 2} for i in range(10)]
    return sorted(out, key=lambda u: -u["n"] * (1 if u["kind"] == "chain" else 0.01 if u["kind"] == "greedy" else 0))


def unit_path(u: dict) -> Path:
    if u["kind"] == "brute":
        return OUT / u["stage"] / "brute" / f"case{u['case']:02d}.csv"
    tag = f"seed_{u['seed']}" if u["snapshot"] is None else f"snap{u['snapshot']:02d}_seed_{u['seed']}"
    stage = Path(u["kind"]) / u["src"] if u["kind"] in ("e2e", "assoc") else Path(u["stage"])
    return OUT / stage / u["scenario"] / u["tier"] / f"{tag}_n{u['n']:03d}.csv"


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".csv.tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def paper_served(cfg, users: np.ndarray, dep: joint.Deployment, assoc: np.ndarray) -> float:
    """Served share of a deployment scored by the paper's simulator code."""
    dl, _, ch = _build_models(cfg)
    net, cap, dem = cfg.network, cfg.capacity, cfg.demand
    tilt = np.arctan(np.linalg.norm(dep.t, axis=1))
    az = np.arctan2(dep.t[:, 1], dep.t[:, 0])
    power = downlink_power_matrix(dep.pos, tilt, az, users, dl, ch, 10 ** ((net.p_tx_dl_dbm - 30) / 10))
    s = compute_sinr(power, assoc, noise_psd_dbm_hz=net.noise_dbm_hz, bandwidth_hz=net.bandwidth_hz).sinr_db
    adm = admit_gbr(s, assoc, dep.pos.shape[0], net.bandwidth_hz, dem.r_min_bps,
                    dem.active_per_point(users.shape[0]), cap.se_alpha, cap.se_sinr_min_db, cap.se_max_bps_hz)
    return float(np.mean(adm))


def _greedy_start(cfg, users: np.ndarray, n: int, kmeans_seed: int):
    """The paper's deployment (K-means, sweep + greedy altitudes, nadir, nearest) and fleet.py's served share."""
    grid = cfg.network.altitude_grid()
    gs = GridService(cfg, users, _kmeans_centroids(users, n, kmeans_seed), grid)
    k = greedy(gs, sweep(gs))[0]
    return joint.Deployment(np.column_stack([gs._xy, grid[k]]), np.zeros((n, 2))), gs.evaluate(k)[0]


def _chain(phys, users: np.ndarray, start: joint.Deployment, res: dict) -> dict:
    """xyz_nearest -> xyz -> xyzt from the greedy start, added to res."""
    res["xyz_nearest"] = joint.optimise(phys, users, start, "nearest", "xyz")
    res["xyz"] = joint.optimise(phys, users, res["xyz_nearest"].dep, "free", "xyz")
    res["xyzt"] = joint.optimise(phys, users, res["xyz"].dep, "free", "xyzt", start_assoc=res["xyz"].assoc)
    return res


def _greedy_result(phys, users: np.ndarray, start: joint.Deployment) -> joint.OptimiseResult:
    assoc = joint.nearest(start.pos, users)
    return joint.OptimiseResult(start, assoc, joint.score(phys, users, start, assoc)[0], float("nan"), 0, 0, 0.0)


def run_chain(u: dict) -> list[dict]:
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    start, served_fleet = _greedy_start(cfg, users, u["n"], u["seed"])
    phys = joint.Physics.from_config(cfg, users.shape[0])
    res = {"greedy": _greedy_result(phys, users, start)}
    if u["kind"] == "greedy":                       # paper-window N: the paper's method only
        return _rows(u, cfg, users, phys, served_fleet, res, ("greedy",))
    res["greedy_free"] = joint.optimise(phys, users, start, "free", "")
    _chain(phys, users, start, res)
    return _rows(u, cfg, users, phys, served_fleet, res, METHODS)


def _strongest_pct(phys, users: np.ndarray, dep: joint.Deployment) -> float:
    strongest = np.argmax(joint.rx_power(phys, dep.pos, dep.bore, users), axis=0)
    return 100.0 * joint.score(phys, users, dep, strongest)[0]


def run_multi(u: dict) -> list[dict]:
    """The chain from MULTI_STARTS further K-means starts; one row per (start, method)."""
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    phys = joint.Physics.from_config(cfg, users.shape[0])
    rows, saved = [], {}
    for k in range(1, MULTI_STARTS + 1):
        ks = u["seed"] + MULTI_SEED_STEP * k
        start, served_fleet = _greedy_start(cfg, users, u["n"], ks)
        res = _chain(phys, users, start, {"greedy": _greedy_result(phys, users, start)})
        for m in ("greedy", *CHAIN):
            rows.append({**_row(u, cfg, users, phys, served_fleet, m, res[m]), "start": k, "kmeans_seed": ks,
                         "served_strongest_pct": _strongest_pct(phys, users, res[m].dep)})
            saved |= {f"s{k}_{m}_pos": res[m].dep.pos, f"s{k}_{m}_t": res[m].dep.t, f"s{k}_{m}_assoc": res[m].assoc}
    _save(unit_path(u), saved)
    return rows


def tilt_source(u: dict) -> Path:
    """The full_* unit holding this unit's xyz deployment (full_coarse also has greedy-only units at some N)."""
    for st in E2E_SOURCES:
        p = unit_path({**u, "kind": "chain", "stage": st})
        if p.with_suffix(".npz").exists() and "xyz_pos" in np.load(p.with_suffix(".npz")).files:
            return p
    raise FileNotFoundError(f"no xyz deployment for {u}")


def run_gmulti(u: dict) -> list[dict]:
    """The greedy deployment from MULTI_STARTS further K-means starts; one row per start."""
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    phys = joint.Physics.from_config(cfg, users.shape[0])
    rows = []
    for k in range(1, MULTI_STARTS + 1):
        ks = u["seed"] + MULTI_SEED_STEP * k
        start, served_fleet = _greedy_start(cfg, users, u["n"], ks)
        g = _greedy_result(phys, users, start)
        rows.append({**_row(u, cfg, users, phys, served_fleet, "greedy", g), "start": k, "kmeans_seed": ks,
                     "served_strongest_pct": _strongest_pct(phys, users, g.dep)})
    return rows


def run_tilt(u: dict) -> list[dict]:
    """xyzt from the saved xyz deployment of full_coarse/full_dense with the tilt capped; one row per cap."""
    src = tilt_source(u)
    z = np.load(src.with_suffix(".npz"))
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    phys = joint.Physics.from_config(cfg, users.shape[0])
    served_fleet = float(next(r for r in csv.DictReader(open(src)) if r["method"] == "greedy")["served_fleet_pct"]) / 100
    xyz = joint.Deployment(z["xyz_pos"], z["xyz_t"])
    rows, saved = [], {}
    for cap in TILT_CAPS_DEG:
        r = joint.optimise(phys, users, xyz, "free", "xyzt", start_assoc=z["xyz_assoc"], tilt_max_deg=cap)
        rows.append({**_row(u, cfg, users, phys, served_fleet, "xyzt", r), "tilt_max_deg": cap,
                     "served_strongest_pct": _strongest_pct(phys, users, r.dep)})
        saved |= {f"cap{cap:g}_pos": r.dep.pos, f"cap{cap:g}_t": r.dep.t, f"cap{cap:g}_assoc": r.assoc}
    _save(unit_path(u), saved)
    return rows


def _save(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)


def _rows(u: dict, cfg, users: np.ndarray, phys, served_fleet: float, res: dict, methods) -> list[dict]:
    """One CSV row per method, and the deployments saved beside the CSV."""
    rows = [_row(u, cfg, users, phys, served_fleet, m, res[m]) for m in methods]
    _save(unit_path(u), {**{f"{m}_pos": res[m].dep.pos for m in methods}, **{f"{m}_t": res[m].dep.t for m in methods},
                         **{f"{m}_assoc": res[m].assoc for m in methods}})
    return rows


def _row(u: dict, cfg, users: np.ndarray, phys, served_fleet: float, m: str, r: joint.OptimiseResult) -> dict:
    tilt = joint.tilt_deg(r.dep.t)
    return {
        "stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
        "seed": u["seed"], "n_points": users.shape[0], "active_per_point": phys.lam, "n_drones": u["n"],
        "lower_bound": lower_bound(u["scenario"], u["tier"]), "method": m,
        "served_pct": 100.0 * r.served,
        "served_paper_pct": 100.0 * paper_served(cfg, users, r.dep, r.assoc),
        "served_fleet_pct": 100.0 * served_fleet,
        "lp_bound_pct": 100.0 * r.lp_bound,
        "mean_altitude_m": float(r.dep.pos[:, 2].mean()), "min_altitude_m": float(r.dep.pos[:, 2].min()),
        "max_altitude_m": float(r.dep.pos[:, 2].max()), "mean_tilt_deg": float(tilt.mean()),
        "max_tilt_deg": float(tilt.max()), "n_tilted": int(np.sum(tilt > 1.0)),
        "rounds": r.rounds, "evals": r.evals, "time_s": round(r.time_s, 3),
    }


def run_e2e(u: dict) -> list[dict]:
    """End to end for the deployments one source unit saved (every method it ran)."""
    src = unit_path({**u, "kind": "chain", "stage": u["src"]})
    saved = np.load(src.with_suffix(".npz"))
    recorded = {r["method"]: float(r["served_pct"]) for r in csv.DictReader(open(src))}
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    phys = joint.Physics.from_config(cfg, users.shape[0])
    models = {}
    for v in E2E_VARIANTS:
        vcfg = config_for(u["scenario"], u["tier"], v)
        models[v] = (vcfg, *_build_models(vcfg)[1:])          # backhaul antenna, channel
    cols = np.arange(users.shape[0])
    rows = []
    for m in (m for m in METHODS if f"{m}_pos" in saved):
        pos, assoc = saved[f"{m}_pos"], saved[f"{m}_assoc"]
        power = joint.rx_power(phys, pos, joint.boresight(saved[f"{m}_t"]), users)
        adm, _ = joint.admit(phys, joint.sinr_db(phys, power[assoc, cols], power.sum(axis=0)), assoc, pos.shape[0])
        # summed in fleet.GridService.end_to_end_served's order: the balanced tree breaks ties on load
        load = phys.lam * np.bincount(assoc, weights=adm, minlength=pos.shape[0]) * phys.r_min_bps
        row = {"src": u["src"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
               "seed": u["seed"], "n_drones": u["n"], "method": m, "served_pct": 100.0 * adm.mean(),
               "served_src_pct": recorded[m]}
        for v, (vcfg, bh_ant, ch) in models.items():
            for t in backhaul.TOPOLOGIES:
                share, relays = backhaul.end_to_end(pos, load, vcfg, bh_ant, ch, t)
                row[f"served_e2e_pct_{v}_{t}"] = 100.0 * float(load @ share) / (phys.r_min_bps * phys.lam * users.shape[0])
                row[f"relays_{v}_{t}"] = relays
        rows.append(row)
    return rows


def run_assoc(u: dict) -> list[dict]:
    """Served share of each saved deployment under its own association, the nearest and the strongest drone."""
    src = unit_path({**u, "kind": "chain", "stage": u["src"]})
    saved = np.load(src.with_suffix(".npz"))
    recorded = {r["method"]: float(r["served_pct"]) for r in csv.DictReader(open(src))}
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    phys = joint.Physics.from_config(cfg, users.shape[0])
    rows = []
    for m in (m for m in METHODS if f"{m}_pos" in saved):
        dep = joint.Deployment(saved[f"{m}_pos"], saved[f"{m}_t"])
        assoc = saved[f"{m}_assoc"]
        strongest = np.argmax(joint.rx_power(phys, dep.pos, dep.bore, users), axis=0)
        rows.append({"src": u["src"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
                     "seed": u["seed"], "n_drones": u["n"], "method": m,
                     "served_pct": 100.0 * joint.score(phys, users, dep, assoc)[0], "served_src_pct": recorded[m],
                     "served_nearest_pct": 100.0 * joint.score(phys, users, dep, joint.nearest(dep.pos, users))[0],
                     "served_strongest_pct": 100.0 * joint.score(phys, users, dep, strongest)[0],
                     "strongest_is_assoc_pct": 100.0 * float(np.mean(strongest == assoc))})
    return rows


def run_brute(u: dict) -> list[dict]:
    """Tiny instance: grid search against the solver. Odd cases: N = 2, nadir, nearest association, grid
    over both drones' (x, y, z); even cases: N = 1 over (x, y, z, tilt, azimuth). Users spread 250 m around
    each centre and the demand is 0.8 times what the drones carry at peak efficiency, so where the drones
    fly and point decides how much is served (the grid's median and minimum are recorded to show it)."""
    rng = np.random.default_rng(100 + u["case"])
    n = u["n"]
    centres = rng.uniform(500, 1500, size=(n, 2))
    xy = np.vstack([rng.normal(c, 250, size=(10 if n == 2 else 20, 2)) for c in centres])
    users = np.column_stack([xy, np.zeros(xy.shape[0])])
    cfg = config_for("shelter_clustered", "video")
    phys = joint.Physics.from_config(cfg, users.shape[0])
    lam = 0.8 * n * phys.bandwidth_hz * phys.se_max / phys.r_min_bps / users.shape[0]
    phys = replace(phys, lam=lam)
    zs = phys.altitude_grid()
    lo, hi = xy.min(axis=0) - 150, xy.max(axis=0) + 150
    t0 = time.perf_counter()
    if n == 1:
        gx, gy = np.linspace(lo[0], hi[0], 15), np.linspace(lo[1], hi[1], 15)
        dirs = [(0.0, 0.0)] + [(math.tan(math.radians(tl)) * math.cos(a), math.tan(math.radians(tl)) * math.sin(a))
                               for tl in (15, 30, 45, 60, 75) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        assoc = np.zeros(users.shape[0], dtype=int)
        grid_scores = [joint.score(phys, users, joint.Deployment(np.array([[x, y, z]]), np.array([t])), assoc)[0]
                       for x, y, z, t in product(gx, gy, zs, dirs)]
        start = joint.sweep_start(phys, users, xy.mean(axis=0)[None, :])
        sol = joint.optimise(phys, users, start, "nearest", "xyzt")
        move = "xyzt"
    else:
        g = [(x, y, z) for x in np.linspace(lo[0], hi[0], 9) for y in np.linspace(lo[1], hi[1], 9) for z in zs[::2]]
        grid_scores = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dep = joint.Deployment(np.array([g[i], g[j]]), np.zeros((2, 2)))
                grid_scores.append(joint.score(phys, users, dep, joint.nearest(dep.pos, users))[0])
        start = joint.sweep_start(phys, users, joint.kmeans_pp(xy, 2, seed=u["case"]))
        sol = joint.optimise(phys, users, start, "nearest", "xyz")
        move = "xyz"
    return [{"case": u["case"], "n_drones": n, "move": move, "assoc": "nearest",
             "served_solver": sol.served, "served_grid": max(grid_scores),
             "served_grid_median": float(np.median(grid_scores)), "served_grid_min": min(grid_scores),
             "grid_points": len(grid_scores),
             "time_s": round(time.perf_counter() - t0, 2)}]


def run_unit(u: dict) -> int:
    path = unit_path(u)
    if path.exists():
        return 0
    if u["kind"] in ("e2e", "assoc"):
        _write(path, *((E2E_COLUMNS, run_e2e(u)) if u["kind"] == "e2e" else (ASSOC_COLUMNS, run_assoc(u))))
        return 1
    if u["kind"] in ("multi", "tilt", "gmulti"):
        columns, fn = {"multi": (MULTI_COLUMNS, run_multi), "tilt": (TILT_COLUMNS, run_tilt),
                       "gmulti": (MULTI_COLUMNS, run_gmulti)}[u["kind"]]
        _write(path, columns, fn(u))
        return 1
    rows = run_brute(u) if u["kind"] == "brute" else run_chain(u)
    _write(path, BRUTE_COLUMNS if u["kind"] == "brute" else COLUMNS, rows)
    return 1


def check_stage(mine: list[dict]) -> list[str]:
    """Problems found in the finished check units (empty: all agreements and orderings hold)."""
    bad = []
    for u in mine:
        rows = list(csv.DictReader(open(unit_path(u))))
        if u["kind"] == "brute":
            r = rows[0]
            if float(r["served_grid_median"]) >= float(r["served_grid"]) - 1e-9:
                bad.append(f"brute case {r['case']}: degenerate, the grid median equals its best")
            if float(r["served_solver"]) < float(r["served_grid"]) - 1e-9:
                bad.append(f"brute case {r['case']} (N={r['n_drones']}): solver {float(r['served_solver']):.4f} "
                           f"< grid {float(r['served_grid']):.4f}")
            continue
        s = {r["method"]: r for r in rows}
        tag = f"{u['scenario']}/{u['tier']} N={u['n']}"
        if abs(float(s["greedy"]["served_pct"]) - float(s["greedy"]["served_fleet_pct"])) > 1e-7:
            bad.append(f"{tag}: greedy {s['greedy']['served_pct']} != fleet.py {s['greedy']['served_fleet_pct']}")
        for m, r in s.items():
            if abs(float(r["served_pct"]) - float(r["served_paper_pct"])) > 1e-7:
                bad.append(f"{tag} {m}: joint {r['served_pct']} != paper code {r['served_paper_pct']}")
            if m != "greedy" and float(r["served_pct"]) < float(s["greedy"]["served_pct"]) - 1e-7:
                bad.append(f"{tag} {m} serves less than greedy")
        for a, b in (("xyz_nearest", "xyz"), ("xyz", "xyzt")):
            if float(s[b]["served_pct"]) < float(s[a]["served_pct"]) - 1e-7:
                bad.append(f"{tag}: {b} < {a}")
    return bad


def check_e2e(mine: list[dict]) -> list[str]:
    """The served share recomputed from each saved deployment must be the one its source CSV recorded."""
    bad = []
    for u in mine:
        for r in csv.DictReader(open(unit_path(u))):
            if abs(float(r["served_pct"]) - float(r["served_src_pct"])) > 1e-7:
                bad.append(f"{u['scenario']}/{u['tier']} N={u['n']} {r['method']} ({u['src']}): "
                           f"{r['served_pct']} != recorded {r['served_src_pct']}")
    return bad


def check_restart(mine: list[dict]) -> list[str]:
    """multi / tilt / gmulti: every row scored the same by the paper's code (greedy also as fleet.py scores it); the
    chain never loses ground; tilts within the cap."""
    bad = []
    for u in mine:
        rows = list(csv.DictReader(open(unit_path(u))))
        tag = f"{u['scenario']}/{u['tier']} N={u['n']} {u['snapshot']}/{u['seed']}"
        for r in rows:
            if abs(float(r["served_pct"]) - float(r["served_paper_pct"])) > 1e-7:
                bad.append(f"{tag} {r['method']}: joint {r['served_pct']} != paper code {r['served_paper_pct']}")
            if r["method"] == "greedy" and abs(float(r["served_pct"]) - float(r["served_fleet_pct"])) > 1e-7:
                bad.append(f"{tag} start {r['start']}: greedy {r['served_pct']} != fleet.py {r['served_fleet_pct']}")
            if u["kind"] == "tilt" and float(r["max_tilt_deg"]) > float(r["tilt_max_deg"]) + 1e-6:
                bad.append(f"{tag}: tilt {r['max_tilt_deg']} above cap {r['tilt_max_deg']}")
        if u["kind"] == "multi":
            for k in range(1, MULTI_STARTS + 1):
                s = {r["method"]: float(r["served_pct"]) for r in rows if int(r["start"]) == k}
                for a, b in (("greedy", "xyz_nearest"), ("xyz_nearest", "xyz"), ("xyz", "xyzt")):
                    if s[b] < s[a] - 1e-7:
                        bad.append(f"{tag} start {k}: {b} < {a}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--shard", default="0/1", help="I/K: run the I-th of K shards")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    i, k = (int(x) for x in args.shard.split("/"))
    mine = units(args.stage)[i::k]
    done = [u for u in mine if unit_path(u).exists()]
    if args.list or args.check:
        print(f"joint {args.stage} shard {i}/{k}: {len(mine)} units, {len(done)} on disk")
        if args.check:
            ok = len(done) == len(mine)
            if ok and args.stage in ("check", "e2e", "assoc", "multi", "tilt", "gmulti"):
                bad = (check_stage(mine) if args.stage == "check" else
                       check_restart(mine) if args.stage in ("multi", "tilt", "gmulti") else check_e2e(mine))
                for b in bad:
                    print("  " + b)
                ok = not bad
            print(f"RESULT: {'PASS' if ok else 'FAIL'} joint {args.stage} shard {i}/{k}")
            sys.exit(0 if ok else 1)
        return
    todo = [u for u in mine if not unit_path(u).exists()]
    print(f"joint {args.stage} shard {i}/{k}: {len(todo)} of {len(mine)} units to run, {args.procs} procs", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        for c, _ in enumerate(pool.imap_unordered(run_unit, todo, chunksize=1), start=1):
            el = time.time() - t0
            print(f"  {c}/{len(todo)} units  {el:.0f}s  eta {el / c * (len(todo) - c):.0f}s", flush=True)


if __name__ == "__main__":
    main()
