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

from dronecomm import joint  # noqa: E402
from dronecomm.capacity import admit_gbr  # noqa: E402
from dronecomm.fleet import GridService, greedy, sweep  # noqa: E402
from dronecomm.heuristic import _kmeans_centroids  # noqa: E402
from dronecomm.interference import downlink_power_matrix  # noqa: E402
from dronecomm.optimize import _build_models  # noqa: E402
from dronecomm.sinr import compute_sinr  # noqa: E402
from run_fleet import (BASE_SEED, MILAN_TILES, REFINE_MILAN_SEEDS, REFINE_SYN_SEEDS, REFINE_WINDOWS,  # noqa: E402
                       SCENARIOS, TIERS, VARIANTS, _users)

OUT = ROOT / "results" / "joint"
WINDOWS = ROOT / "configs" / "joint_windows.json"
WINDOWS_FULL = ROOT / "configs" / "joint_windows_full.json"
STAGES = ["check", "coarse", "dense", "full_coarse", "full_dense"]
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
BRUTE_COLUMNS = ["case", "n_drones", "move", "assoc", "served_solver", "served_grid", "served_grid_median",
                 "served_grid_min", "grid_points", "time_s"]


def config_for(scenario: str, tier: str):
    _, pop, share, act = SCENARIOS[scenario]
    base = VARIANTS["grounded"]()
    dem = replace(base.demand, population=pop, r_min_bps=TIERS[tier],
                  **({} if share is None else {"device_share": share}),
                  **({} if act is None else {"activity_factor": act}))
    return replace(base, demand=dem)


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
    return OUT / u["stage"] / u["scenario"] / u["tier"] / f"{tag}_n{u['n']:03d}.csv"


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


def run_chain(u: dict) -> list[dict]:
    cfg = config_for(u["scenario"], u["tier"])
    users, _ = _users({"scenario": u["scenario"], "snapshot": u["snapshot"], "seed": u["seed"]})
    n, grid = u["n"], cfg.network.altitude_grid()
    gs = GridService(cfg, users, _kmeans_centroids(users, n, u["seed"]), grid)
    k = greedy(gs, sweep(gs))[0]
    served_fleet = gs.evaluate(k)[0]
    phys = joint.Physics.from_config(cfg, users.shape[0])
    start = joint.Deployment(np.column_stack([gs._xy, grid[k]]), np.zeros((n, 2)))
    t0 = time.perf_counter()
    base = joint.OptimiseResult(start, joint.nearest(start.pos, users),
                                joint.score(phys, users, start, joint.nearest(start.pos, users))[0],
                                float("nan"), 0, 0, time.perf_counter() - t0)
    res = {"greedy": base}
    if u["kind"] == "greedy":                       # paper-window N: the paper's method only
        return _rows(u, cfg, users, phys, served_fleet, res, ("greedy",))
    res["greedy_free"] = joint.optimise(phys, users, start, "free", "")
    res["xyz_nearest"] = joint.optimise(phys, users, start, "nearest", "xyz")
    res["xyz"] = joint.optimise(phys, users, res["xyz_nearest"].dep, "free", "xyz")
    res["xyzt"] = joint.optimise(phys, users, res["xyz"].dep, "free", "xyzt", start_assoc=res["xyz"].assoc)
    return _rows(u, cfg, users, phys, served_fleet, res, METHODS)


def _rows(u: dict, cfg, users: np.ndarray, phys, served_fleet: float, res: dict, methods) -> list[dict]:
    """One CSV row per method, and the deployments saved beside the CSV."""
    n = u["n"]
    rows = []
    for m in methods:
        r = res[m]
        tilt = joint.tilt_deg(r.dep.t)
        rows.append({
            "stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
            "seed": u["seed"], "n_points": users.shape[0], "active_per_point": phys.lam, "n_drones": n,
            "lower_bound": lower_bound(u["scenario"], u["tier"]), "method": m,
            "served_pct": 100.0 * r.served,
            "served_paper_pct": 100.0 * paper_served(cfg, users, r.dep, r.assoc),
            "served_fleet_pct": 100.0 * served_fleet,
            "lp_bound_pct": 100.0 * r.lp_bound,
            "mean_altitude_m": float(r.dep.pos[:, 2].mean()), "min_altitude_m": float(r.dep.pos[:, 2].min()),
            "max_altitude_m": float(r.dep.pos[:, 2].max()), "mean_tilt_deg": float(tilt.mean()),
            "max_tilt_deg": float(tilt.max()), "n_tilted": int(np.sum(tilt > 1.0)),
            "rounds": r.rounds, "evals": r.evals, "time_s": round(r.time_s, 3),
        })
    path = unit_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **{f"{m}_pos": res[m].dep.pos for m in methods},
                        **{f"{m}_t": res[m].dep.t for m in methods}, **{f"{m}_assoc": res[m].assoc for m in methods})
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
            if ok and args.stage == "check":
                bad = check_stage(mine)
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
