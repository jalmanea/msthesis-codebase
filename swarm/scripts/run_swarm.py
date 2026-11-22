"""Drones as agents (dronecomm.swarm): best-response dynamics on local and global utilities. Runs on Ibex.

A layout is (scenario, tier, snapshot or seed, N). Per layout, stage `central` builds the paper's
deployment (K-means, sweep + greedy altitudes, nadir; dronecomm.fleet) and the central joint optimum
(dronecomm.joint chain to xyzt, free association), and saves both. Stage `p1` then runs every agent
on one utility from one start under one dynamics:

    start      paper (the paper's deployment), depot (every drone launched from one corner),
               central (the joint optimum: is it stable under the utility?)
    utility    selfish, window(rho) for rho in RHOS, global
    dynamics   seq (sequential), sim (simultaneous), sim50 (simultaneous, inertia 0.5)

Stage `p2` repeats p1's utilities from the paper's start, sequentially, with pointing off (nadir
only): a nadir beam's interference reaches about one cell, so a small window should then match
global play. Stage `p3` limits information: each agent's model sees only the users within INFOS of
it or served by a drone within that radius (dronecomm.swarm, info_m), for the selfish and global
utilities from the paper and depot starts, sequentially.

Stage `p4a` repeats p1's paper and depot starts, sequentially, with exploration (agents also try the
unserved demand other drones advertise; dronecomm.swarm.run_fleet), at fixed N. Stage `p4b` sizes the
fleet: agents land when their own marginal contribution is below land * kappa and call a reserve
drone when their unserved demand exceeds call * kappa (FleetRules), from one drone at the depot
(`depot1`) or from the paper's deployment with 1.3 x the central N95 (`over`), the operator owning
2 x N95 drones. N95 and the served-vs-N frontier come from the joint runs (configs/swarm_frontier.json,
scripts/swarm_frontier.py), as a running maximum over N.

Stage `p4c` repeats p4b with land = 0.25 and calls judged on the unserved demand within CALL_AREA_M of
the agent rather than in its own cell. Stage `p4d` adds the call check: the caller predicts the reserve
drone's marginal contribution at the beacon and calls only if it clears the landing threshold. Stage
`p4e` repeats p4c with hysteresis instead: a called drone may not land for LAND_GRACE rounds, and
(in half the runs) an agent calls only after CALL_PATIENCE consecutive turns over the threshold.

Stage `cstrong` re-optimises the central reference under strongest-cell association (joint.optimise,
assoc_mode 'strongest'), the rule the agents play under, from three starts: the paper's deployment
(xyz then xyzt), STRONG_RESTARTS more K-means starts, the free-association optimum of `central`, and the
best final deployment any agent run of p1-p4a reached on the layout; the reference is the best of them. Stage `fstrong` does the same
along the central frontier of the self-sizing layouts (every N of configs/swarm_frontier.json), from
the joint runs' greedy and xyzt deployments (JOINT) and the K-means restarts. Stage `rstrong` polishes,
under the same rule, the xyzt deployments of the journal's further K-means starts (JOINT/multi) at every N
of the frontier that stage ran; analyze_swarm takes the better of it and cstrong or fstrong. Stage `brcheck` checks the best response: at the
final state of each p1 (pointing) and p2 (nadir) sequential run from the paper's start, on the core
layouts (every scenario and N), every agent's `best_response` against an exhaustive grid search
(swarm.brute_response); a converged run is an equilibrium only if the grid finds no gain either.

Stages `p5speed` and `p5noise` test the main ladder for realism on the core layouts, from the paper's
start, simultaneously with inertia 0.5 (the dynamics the paper recommends), for P5_UTILS. In p5speed an
agent moves at most STEPS_M horizontally per decision (speed x decision period). In p5noise an agent
predicts with ERRORS (swarm.PredictionError: user positions off by a positioning error, its own link
off by an unknown shadowing gain) while the outcome is scored exactly. The matching runs without
either are p1's `paper ... sim50`, with the same seeds.

Stage `fine` fills in the radius sweeps on every layout, from the paper's start, sequentially: the
windows FINE_RADII with pointing (as p1) and nadir only (as p2), and the global utility with information
radius FINE_RADII (as p3). With RHOS and INFOS they give every multiple of 100 m up to 1200 m. It is
not one of AGENT_STAGES, so the central reference does not move.

Served share is scored under strongest-cell association (users pick the drone they hear best).

python scripts/run_swarm.py --stage smoke                     # Mac: two short runs, a few seconds
python scripts/run_swarm.py --stage central --shard 0/4 --procs 16
python scripts/run_swarm.py --stage p1 --shard 0/8 --procs 16
python scripts/run_swarm.py --stage p1 --shard 0/8 --check
python scripts/run_swarm.py --stage p4b --shard 0/16 --procs 16
python scripts/run_swarm.py --stage brcheck --shard 0/16 --procs 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dronecomm import joint, swarm  # noqa: E402
from dronecomm.fleet import GridService, greedy, sweep  # noqa: E402
from dronecomm.heuristic import _kmeans_centroids  # noqa: E402
from run_fleet import BASE_SEED, MILAN_TILES, REFINE_MILAN_SEEDS, REFINE_SYN_SEEDS, _users  # noqa: E402
from run_joint import _write, config_for  # noqa: E402

OUT = ROOT / "results" / "swarm"
STAGES = ["smoke", "central", "p1", "p2", "p3", "p4a", "p4b", "p4c", "p4d", "p4e", "cstrong", "fstrong", "rstrong",
          "brcheck", "p5speed", "p5noise", "fine"]
FRONTIER = ROOT / "configs" / "swarm_frontier.json"
JOINT = ROOT.parent / "dronecomm-joint" / "results" / "joint"     # the joint-opt clone's runs (fstrong)
AGENT_STAGES = ["p1", "p2", "p3", "p4a"]                           # cstrong's third start: their best final state
STRONG_RESTARTS = 2                                                # extra K-means starts (seeds seed+1, ...)
MULTI_STARTS = 2                                                   # the journal's `multi` starts (s1, s2), rstrong
BR_UTILS = [swarm.Utility("selfish"), swarm.Utility("window", 300.0), swarm.Utility("window", 600.0),
            swarm.Utility("global")]
# p4b-p4e, fstrong: self-sizing layouts (scenario, tier, snapshot, seed), the p1 layouts without N: the journal's
# units (40 synthetic seeds, 24 Milan snapshots x 6 K-means seeds); CORE_SIZE_LAYOUTS are the audit's 10
_N_SNAP = len(json.loads((MILAN_TILES / "tiles_metadata.json").read_text())["snapshots"])
SIZE_LAYOUTS = ([("urban_broad", "voice", None, BASE_SEED + r) for r in range(REFINE_SYN_SEEDS)]
                + [("milan_p100", "voice", s, BASE_SEED + r) for s in range(_N_SNAP) for r in range(REFINE_MILAN_SEEDS)]
                + [("shelter_clustered", "video", None, BASE_SEED + r) for r in range(REFINE_SYN_SEEDS)])
CORE_SIZE_LAYOUTS = ([("urban_broad", "voice", None, BASE_SEED + r) for r in range(5)]
                     + [("milan_p100", "voice", s, BASE_SEED) for s in (0, 8, 16)]
                     + [("shelter_clustered", "video", None, BASE_SEED + r) for r in range(2)])
SIZE_UTILS = [swarm.Utility("selfish"), swarm.Utility("window", 600.0), swarm.Utility("window", 1200.0),
              swarm.Utility("global")]
SIZE_RULES = [swarm.FleetRules(True, 0.05, 0.25), swarm.FleetRules(True, 0.1, 0.5), swarm.FleetRules(True, 0.25, 1.0)]
# p4c: calls counted over the area within CALL_AREA_M (unserved demand spread over cells adds up)
CALL_AREA_M = 1200.0
AREA_RULES = [swarm.FleetRules(True, 0.25, a, CALL_AREA_M) for a in (0.25, 0.5, 1.0)]
# p4d: as p4c, but a call is made only if the called drone would clear the landing threshold
CHECK_RULES = [swarm.FleetRules(True, 0.25, a, CALL_AREA_M, call_check=True) for a in (0.25, 0.5, 1.0)]
# p4e: as p4c, with hysteresis (a called drone is shielded from landing; calls wait for a persistent excess)
LAND_GRACE, CALL_PATIENCE = 3, 2
HYST_RULES = [swarm.FleetRules(True, 0.25, a, CALL_AREA_M, call_patience=k, land_grace=LAND_GRACE)
              for a in (0.25, 0.5, 1.0) for k in (1, CALL_PATIENCE)]
HYST_UTILS = [swarm.Utility("selfish"), swarm.Utility("window", 600.0), swarm.Utility("global")]
SIZE_MAX_ROUNDS = 40
SIZE_STAGES = ("p4b", "p4c", "p4d", "p4e")
FLEET_COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "start", "utility", "rho_m",
                 "rules", "land_alpha", "call_alpha", "kappa", "n95_central", "n_owned", "n_start", "n_final",
                 "served_pct", "served_central_at_n_pct", "n_central_for_served", "lands", "calls", "rounds",
                 "converged", "evals", "time_s", "history"]
# (scenario, tier) -> (fleet sizes, layouts as (snapshot, seed)); fleets short of, near and above the
# central optimum's 95 % point, so the utilities have room to differ. The layouts are the journal's units
# (Table tab:fleet): 40 synthetic seeds, 24 Milan snapshots x 6 K-means seeds. CORE_LAYOUTS, the first 23
# (5 and 2 seeds, snapshots 0, 8, 16 with one K-means seed), are the ones brcheck checks exhaustively.
LAYOUTS = {
    ("urban_broad", "voice"): ([8, 11, 16], [(None, BASE_SEED + r) for r in range(REFINE_SYN_SEEDS)]),
    ("milan_p100", "voice"): ([14, 21], [(s, BASE_SEED + r) for s in range(_N_SNAP) for r in range(REFINE_MILAN_SEEDS)]),
    ("shelter_clustered", "video"): ([33], [(None, BASE_SEED + r) for r in range(REFINE_SYN_SEEDS)]),
}
CORE_LAYOUTS = {
    ("urban_broad", "voice"): [(None, BASE_SEED + r) for r in range(5)],
    ("milan_p100", "voice"): [(s, BASE_SEED) for s in (0, 8, 16)],
    ("shelter_clustered", "video"): [(None, BASE_SEED + r) for r in range(2)],
}
RHOS = [150.0, 300.0, 600.0, 1200.0]
INFOS = [150.0, 300.0, 600.0, 1200.0]
# fine: the multiples of 100 m up to 1200 m that RHOS and INFOS do not already cover
FINE_RADII = [float(r) for r in range(100, 1201, 100) if float(r) not in RHOS]
UTILS = [swarm.Utility("selfish")] + [swarm.Utility("window", r) for r in RHOS] + [swarm.Utility("global")]
DYNAMICS = {"seq": ("sequential", 1.0), "sim": ("simultaneous", 1.0), "sim50": ("simultaneous", 0.5)}
MAX_ROUNDS = 25
# p5speed, p5noise: realism of the agents (see the module docstring)
P5_UTILS = [swarm.Utility("selfish"), swarm.Utility("window", 1200.0), swarm.Utility("global")]
STEPS_M = [150.0, 600.0]          # a 10 m/s multirotor deciding every 15 s or 60 s
SPEED_MAX_ROUNDS = 40             # short steps need more rounds to arrive
# positioning: 3GPP Rel-16 commercial target (< 10 m horizontal, 80 % of UEs, TR 38.855) and the
# regulatory minimum (< 50 m); own-link gain: shadow fading of TR 38.901 UMa, LOS 4 dB and NLOS 6 dB
ERRORS = [swarm.PredictionError(10.0, 4.0, "mild"), swarm.PredictionError(50.0, 6.0, "harsh")]
DEPOT_XY = (100.0, 100.0)
CENTRAL_COLUMNS = ["scenario", "tier", "snapshot", "seed", "n_points", "n_drones", "served_paper_pct",
                   "served_central_free_pct", "served_central_pct", "central_time_s"]
STRONG_COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "n_drones", "served_central_free_pct",
                  "served_rescored_pct", "served_from_paper_pct", "served_from_free_pct", "served_from_restarts_pct",
                  "served_best_agent_pct",
                  "served_from_agent_pct", "best_agent_run", "served_strong_pct", "strong_source", "time_s"]
RSTRONG_COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "n_drones", "served_multi_pct",
                   "served_strong_pct", "time_s"]
BR_COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "n_drones", "utility", "rho_m", "pointing",
              "agent", "run_converged", "before_pp", "heuristic_pp", "brute_pp", "heuristic_gain_pp",
              "brute_gain_pp", "missed_pp", "brute_evals", "time_s"]
COLUMNS = ["stage", "scenario", "tier", "snapshot", "seed", "n_points", "n_drones", "start", "utility",
           "rho_m", "dynamics", "pointing", "info_m", "served_start_pct", "served_pct", "served_central_pct", "rounds", "converged",
           "evals", "time_s", "mean_altitude_m", "mean_tilt_deg", "n_tilted", "history", "max_step_m", "pred_err"]


def layouts(core: bool = False) -> list[dict]:
    return [{"scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n}
            for (sc, t), (ns, lays) in LAYOUTS.items() for n in ns for snap, seed in (CORE_LAYOUTS[(sc, t)] if core else lays)]


def _tag(lay: dict) -> str:
    s = f"seed_{lay['seed']}" if lay["snapshot"] is None else f"snap{lay['snapshot']:02d}_seed_{lay['seed']}"
    return f"{lay['scenario']}/{lay['tier']}/{s}_n{lay['n']:03d}"


def multi_path(scenario: str, tier: str, snapshot, seed: int, n: int) -> Path:
    s = f"seed_{seed}" if snapshot is None else f"snap{snapshot:02d}_seed_{seed}"
    return JOINT / "multi" / scenario / tier / f"{s}_n{n:03d}.npz"


def central_path(lay: dict) -> Path:
    return OUT / "central" / f"{_tag(lay)}.csv"


def units(stage: str) -> list[dict]:
    if stage in ("central", "cstrong"):
        return sorted(({"stage": stage, **lay} for lay in layouts()), key=lambda u: -u["n"])
    if stage == "fstrong":
        return sorted(({"stage": stage, "scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n}
                       for sc, t, snap, seed in SIZE_LAYOUTS for n in frontier(sc, t, snap, seed)),
                      key=lambda u: -u["n"])
    if stage == "rstrong":
        return sorted(({"stage": stage, "scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n}
                       for sc, t, snap, seed in SIZE_LAYOUTS for n in frontier(sc, t, snap, seed)
                       if multi_path(sc, t, snap, seed, n).exists()), key=lambda u: -u["n"])
    if stage == "brcheck":
        return sorted(({"stage": stage, **lay, "util": ut, "pointing": pt, "agent": j}
                       for lay in layouts(core=True)
                       for ut in BR_UTILS for pt in (True, False) for j in range(lay["n"])),
                      key=lambda u: (not u["pointing"], -u["n"]))
    if stage == "smoke":
        lay = {"scenario": "urban_broad", "tier": "voice", "snapshot": None, "seed": BASE_SEED, "n": 11}
        return [{"stage": stage, **lay, "start": "paper", "util": u, "dyn": "seq"}
                for u in (swarm.Utility("global"), swarm.Utility("selfish"))]
    out = []
    if stage == "p5speed":
        return sorted(({"stage": stage, **lay, "start": "paper", "util": u, "dyn": "sim50", "step": st}
                       for lay in layouts(core=True) for u in P5_UTILS for st in STEPS_M), key=lambda u: -u["n"])
    if stage == "p5noise":
        return sorted(({"stage": stage, **lay, "start": "paper", "util": u, "dyn": "sim50", "err": e}
                       for lay in layouts(core=True) for u in P5_UTILS for e in ERRORS), key=lambda u: -u["n"])
    if stage in SIZE_STAGES:
        rules = {"p4b": SIZE_RULES, "p4c": AREA_RULES, "p4d": CHECK_RULES, "p4e": HYST_RULES}[stage]
        utils = HYST_UTILS if stage == "p4e" else SIZE_UTILS
        for sc, t, snap, seed in SIZE_LAYOUTS:
            n95 = frontier_n(sc, t, snap, seed, 95.0)
            out += [{"stage": stage, "scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n95,
                     "start": start, "util": u, "rules": rl}
                    for start in ("depot1", "over") for u in utils
                    for rl in rules]
        return sorted(out, key=lambda u: -u["n"])
    for lay in layouts():
        if stage == "p4a":
            out += [{"stage": stage, **lay, "start": start, "util": u, "dyn": "seq", "explore": True}
                    for start in ("paper", "depot") for u in UTILS]
        elif stage == "p1":
            for start in ("paper", "depot", "central"):
                for u in UTILS:
                    for dyn in (DYNAMICS if start != "central" else ["seq"]):
                        out.append({"stage": stage, **lay, "start": start, "util": u, "dyn": dyn})
        elif stage == "fine":
            base = {"stage": stage, **lay, "start": "paper", "dyn": "seq"}
            out += [{**base, "util": swarm.Utility("window", r), "pointing": pt}
                    for r in FINE_RADII for pt in (True, False)]
            out += [{**base, "util": swarm.Utility("global"), "info": r} for r in FINE_RADII]
        elif stage == "p2":
            out += [{"stage": stage, **lay, "start": "paper", "util": u, "dyn": "seq", "pointing": False}
                    for u in UTILS]
        else:
            out += [{"stage": stage, **lay, "start": start, "util": swarm.Utility(kind), "dyn": "seq", "info": r}
                    for start in ("paper", "depot") for kind in ("selfish", "global") for r in INFOS]
    return sorted(out, key=lambda u: -u["n"])          # costliest first; round-robin shards balance


def unit_path(u: dict) -> Path:
    if u["stage"] == "central":
        return central_path(u)
    if u["stage"] in ("cstrong", "fstrong", "rstrong"):
        return OUT / u["stage"] / f"{_tag(u)}.csv"
    if u["stage"] == "brcheck":
        return OUT / "brcheck" / f"{_tag(u)}_{u['util'].label}_{'xyzt' if u['pointing'] else 'nadir'}_d{u['agent']:02d}.csv"
    if u["stage"] in SIZE_STAGES:
        s = f"seed_{u['seed']}" if u["snapshot"] is None else f"snap{u['snapshot']:02d}_seed_{u['seed']}"
        return OUT / u["stage"] / u["scenario"] / u["tier"] / f"{s}_{u['start']}_{u['util'].label}_{u['rules'].label}.csv"
    extra = (("" if u.get("pointing", True) else "_nadir") + (f"_info{u['info']:.0f}" if u.get("info") else "")
             + (f"_step{u['step']:.0f}" if u.get("step") else "") + (f"_{u['err'].label}" if u.get("err") else ""))
    return OUT / u["stage"] / f"{_tag(u)}_{u['start']}_{u['util'].label}_{u['dyn']}{extra}.csv"


def frontier(scenario: str, tier: str, snapshot, seed: int) -> dict[int, float]:
    """Central served % against N (running maximum over N) for one layout, from configs/swarm_frontier.json."""
    raw = json.loads(FRONTIER.read_text())[f"{scenario}|{tier}|{'' if snapshot is None else snapshot}|{seed}"]
    out, best = {}, -math.inf
    for n, s in sorted((int(k), v) for k, v in raw.items()):
        best = max(best, s)
        out[n] = best
    return out


def frontier_n(scenario: str, tier: str, snapshot, seed: int, target: float) -> int:
    """Smallest N on the central frontier serving at least target % (the largest N run if none does)."""
    f = frontier(scenario, tier, snapshot, seed)
    return next((n for n, s in f.items() if s >= target), max(f))


def _interp(f: dict[int, float], n: float) -> float:
    ns = np.array(sorted(f))
    if n < ns[0] or n > ns[-1]:
        return float("nan")
    return float(np.interp(n, ns, [f[k] for k in ns]))


def _n_for(f: dict[int, float], served: float) -> float:
    """Drones the central frontier needs to serve `served` % (linear between runs; nan outside)."""
    ns = sorted(f)
    if served <= f[ns[0]]:
        return float("nan") if served < f[ns[0]] else float(ns[0])
    for a, b in zip(ns, ns[1:]):
        if f[b] >= served:
            return a + (b - a) * (served - f[a]) / max(f[b] - f[a], 1e-12)
    return float("nan")


def _problem(lay: dict):
    cfg = config_for(lay["scenario"], lay["tier"])
    users, _ = _users({"scenario": lay["scenario"], "snapshot": lay["snapshot"], "seed": lay["seed"]})
    return cfg, users, joint.Physics.from_config(cfg, users.shape[0])


def paper_start(cfg, users: np.ndarray, n: int, seed: int) -> joint.Deployment:
    grid = cfg.network.altitude_grid()
    gs = GridService(cfg, users, _kmeans_centroids(users, n, seed), grid)
    k = greedy(gs, sweep(gs))[0]
    return joint.Deployment(np.column_stack([gs._xy, grid[k]]), np.zeros((n, 2)))


def run_central(u: dict) -> list[dict]:
    cfg, users, phys = _problem(u)
    start = paper_start(cfg, users, u["n"], u["seed"])
    t0 = time.perf_counter()
    xyz_n = joint.optimise(phys, users, start, "nearest", "xyz")
    xyz = joint.optimise(phys, users, xyz_n.dep, "free", "xyz")
    xyzt = joint.optimise(phys, users, xyz.dep, "free", "xyzt", start_assoc=xyz.assoc)
    path = central_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), paper_pos=start.pos, paper_t=start.t,
                        central_pos=xyzt.dep.pos, central_t=xyzt.dep.t, central_assoc=xyzt.assoc)
    return [{"scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"], "seed": u["seed"],
             "n_points": users.shape[0], "n_drones": u["n"],
             "served_paper_pct": 100.0 * swarm.served(phys, users, start),
             "served_central_free_pct": 100.0 * xyzt.served,
             "served_central_pct": 100.0 * swarm.served(phys, users, xyzt.dep),
             "central_time_s": round(time.perf_counter() - t0, 2)}]


def _strong_row(u: dict, users, phys, starts: dict, extra: dict) -> list[dict]:
    """Polish each start under strongest-cell association; keep the best. starts: name -> (Deployment, chain)."""
    t0 = time.perf_counter()
    best, src, out = None, "", {}
    for name, (dep, chain) in starts.items():
        if dep is None:
            out[name] = float("nan")
            continue
        for move in chain:
            dep = joint.optimise(phys, users, dep, "strongest", move).dep
        sc = swarm.served(phys, users, dep)
        out[name] = 100.0 * sc
        if best is None or sc > best[0]:
            best, src = (sc, dep), name
    path = unit_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), pos=best[1].pos, t=best[1].t)
    return [{"stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
             "seed": u["seed"], "n_points": users.shape[0], "n_drones": u["n"], **extra,
             "served_from_paper_pct": out["paper"], "served_from_free_pct": out["free"],
             "served_from_restarts_pct": max(out[f"restart{k}"] for k in range(1, STRONG_RESTARTS + 1)),
             "served_from_agent_pct": out.get("agent", float("nan")), "served_strong_pct": 100.0 * best[0],
             "strong_source": src, "time_s": round(time.perf_counter() - t0, 2)}]


def _restarts(cfg, users, n: int, seed: int) -> dict:
    return {f"restart{k}": (paper_start(cfg, users, n, seed + k), ["xyz", "xyzt"]) for k in range(1, STRONG_RESTARTS + 1)}


def run_cstrong(u: dict) -> list[dict]:
    cfg, users, phys = _problem(u)
    z = np.load(central_path(u).with_suffix(".npz"))
    paper, free = (joint.Deployment(z["paper_pos"], z["paper_t"]), joint.Deployment(z["central_pos"], z["central_t"]))
    best_agent, best_run = None, ""
    for st in AGENT_STAGES:
        for f in sorted((OUT / st).glob(f"{_tag(u)}_*.npz")):
            a = np.load(f)
            dep = joint.Deployment(a["pos"], a["t"])
            sc = swarm.served(phys, users, dep)
            if best_agent is None or sc > best_agent[0]:
                best_agent, best_run = (sc, dep), f"{st}/{f.stem}"
    starts = {"paper": (paper, ["xyz", "xyzt"]), "free": (free, ["xyzt"]), **_restarts(cfg, users, u["n"], u["seed"]),
              "agent": (best_agent[1] if best_agent else None, ["xyzt"])}
    return _strong_row(u, users, phys, starts, {
        "served_central_free_pct": float(next(csv.DictReader(open(central_path(u))))["served_central_free_pct"]),
        "served_rescored_pct": 100.0 * swarm.served(phys, users, free),
        "served_best_agent_pct": 100.0 * best_agent[0] if best_agent else float("nan"), "best_agent_run": best_run})


def run_fstrong(u: dict) -> list[dict]:
    cfg, users, phys = _problem(u)
    s = f"seed_{u['seed']}" if u["snapshot"] is None else f"snap{u['snapshot']:02d}_seed_{u['seed']}"
    # the joint unit with an xyzt deployment at this N (full_coarse also has greedy-only units at some N)
    src = next(f for f in (JOINT / st / u["scenario"] / u["tier"] / f"{s}_n{u['n']:03d}.npz"
                           for st in ("dense", "coarse", "full_dense", "full_coarse"))
               if f.exists() and "xyzt_pos" in np.load(f).files)
    z = np.load(src)
    free = joint.Deployment(z["xyzt_pos"], z["xyzt_t"])
    starts = {"paper": (joint.Deployment(z["greedy_pos"], z["greedy_t"]), ["xyz", "xyzt"]), "free": (free, ["xyzt"]),
              **_restarts(cfg, users, u["n"], u["seed"])}
    return _strong_row(u, users, phys, starts, {
        "served_central_free_pct": float(json.loads(FRONTIER.read_text())[
            f"{u['scenario']}|{u['tier']}|{'' if u['snapshot'] is None else u['snapshot']}|{u['seed']}"][str(u["n"])]),
        "served_rescored_pct": 100.0 * swarm.served(phys, users, free), "served_best_agent_pct": float("nan"),
        "best_agent_run": str(src.relative_to(JOINT))})


def run_rstrong(u: dict) -> list[dict]:
    """Polish the journal's further K-means starts (their xyzt deployments) under strongest-cell association."""
    t0 = time.perf_counter()
    _, users, phys = _problem(u)
    z = np.load(multi_path(u["scenario"], u["tier"], u["snapshot"], u["seed"], u["n"]))
    starts = [joint.Deployment(z[f"s{k}_xyzt_pos"], z[f"s{k}_xyzt_t"]) for k in range(1, MULTI_STARTS + 1)]
    polished = [joint.optimise(phys, users, dep, "strongest", "xyzt").dep for dep in starts]
    scores = [swarm.served(phys, users, dep) for dep in polished]
    best = polished[int(np.argmax(scores))]
    path = unit_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), pos=best.pos, t=best.t)
    return [{"stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
             "seed": u["seed"], "n_points": users.shape[0], "n_drones": u["n"],
             "served_multi_pct": 100.0 * max(swarm.served(phys, users, dep) for dep in starts),
             "served_strong_pct": 100.0 * max(scores), "time_s": round(time.perf_counter() - t0, 2)}]


def run_brcheck(u: dict) -> list[dict]:
    _, users, phys = _problem(u)
    st, suffix = ("p1", "") if u["pointing"] else ("p2", "_nadir")
    run = OUT / st / f"{_tag(u)}_paper_{u['util'].label}_seq{suffix}"
    a = np.load(run.with_suffix(".npz"))
    dep = joint.Deployment(a["pos"], a["t"])
    power = joint.rx_power(phys, dep.pos, dep.bore, users)
    t0 = time.perf_counter()
    j = u["agent"]
    _, before, heur, _ = swarm.best_response(phys, users, dep, power, j, u["util"], u["pointing"])
    _, _, brute, evals = swarm.brute_response(phys, users, dep, power, j, u["util"], u["pointing"])
    pp = lambda x: 100.0 * x[0]
    return [{"stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
             "seed": u["seed"], "n_points": users.shape[0], "n_drones": u["n"], "utility": u["util"].kind,
             "rho_m": u["util"].rho_m, "pointing": int(u["pointing"]), "agent": j,
             "run_converged": next(csv.DictReader(open(run.with_suffix(".csv"))))["converged"],
             "before_pp": pp(before), "heuristic_pp": pp(heur), "brute_pp": pp(brute),
             "heuristic_gain_pp": pp(heur) - pp(before), "brute_gain_pp": pp(brute) - pp(before),
             "missed_pp": pp(brute) - pp(heur), "brute_evals": evals, "time_s": round(time.perf_counter() - t0, 1)}]


def run_agents(u: dict) -> list[dict]:
    cfg, users, phys = _problem(u)
    cen = central_path(u).with_suffix(".npz")
    if cen.exists():
        z = np.load(cen)
        paper, central = (joint.Deployment(z["paper_pos"], z["paper_t"]),
                          joint.Deployment(z["central_pos"], z["central_t"]))
        served_central = 100.0 * swarm.served(phys, users, central)
    elif u["stage"] == "smoke":
        paper, central, served_central = paper_start(cfg, users, u["n"], u["seed"]), None, float("nan")
    else:
        raise FileNotFoundError(f"{cen}: run --stage central first")
    start = {"paper": paper, "central": central,
             "depot": swarm.depot_start(phys, u["n"], np.array(DEPOT_XY), u["seed"])}[u["start"]]
    dynamics, inertia = DYNAMICS[u["dyn"]]
    if u.get("explore"):
        r = swarm.run_fleet(phys, users, start, u["n"], u["util"], swarm.FleetRules(explore=True),
                            seed=u["seed"], max_rounds=MAX_ROUNDS)
        r.history = [h[0] for h in r.history]
    else:
        rounds = 2 if u["stage"] == "smoke" else SPEED_MAX_ROUNDS if u.get("step") else MAX_ROUNDS
        r = swarm.run(phys, users, start, u["util"], dynamics, pointing=u.get("pointing", True), inertia=inertia,
                      seed=u["seed"], max_rounds=rounds, info_m=u.get("info"), max_step_m=u.get("step"),
                      err=u.get("err"))
    tilt = joint.tilt_deg(r.dep.t)
    path = unit_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), pos=r.dep.pos, t=r.dep.t)
    return [{"stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
             "seed": u["seed"], "n_points": users.shape[0], "n_drones": u["n"], "start": u["start"],
             "utility": u["util"].kind, "rho_m": u["util"].rho_m, "dynamics": u["dyn"],
             "pointing": int(u.get("pointing", True)), "info_m": u.get("info") or "",
             "served_start_pct": 100.0 * r.history[0], "served_pct": 100.0 * r.served,
             "served_central_pct": served_central, "rounds": r.rounds, "converged": int(r.converged),
             "evals": r.evals, "time_s": round(r.time_s, 2), "mean_altitude_m": float(r.dep.pos[:, 2].mean()),
             "mean_tilt_deg": float(tilt.mean()), "n_tilted": int(np.sum(tilt > 1.0)),
             "history": json.dumps([round(100.0 * h, 4) for h in r.history]),
             "max_step_m": u.get("step") or "", "pred_err": u["err"].label if u.get("err") else ""}]


def run_selfsize(u: dict) -> list[dict]:
    cfg, users, phys = _problem(u)
    n95, rules = u["n"], u["rules"]
    owned = 2 * n95
    if u["start"] == "depot1":
        start, n_start = swarm.depot_start(phys, owned, np.array(DEPOT_XY), u["seed"]), 1
    else:
        n_start = int(math.ceil(1.3 * n95))
        air = paper_start(cfg, users, n_start, u["seed"])
        rest = swarm.depot_start(phys, owned - n_start, np.array(DEPOT_XY), u["seed"])
        start = joint.Deployment(np.vstack([air.pos, rest.pos]), np.vstack([air.t, rest.t]))
    r = swarm.run_fleet(phys, users, start, n_start, u["util"], rules, seed=u["seed"],
                        max_rounds=SIZE_MAX_ROUNDS, depot_xy=np.array(DEPOT_XY))
    f = frontier(u["scenario"], u["tier"], u["snapshot"], u["seed"])
    n_final = int(r.active.sum())
    path = unit_path(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), pos=r.dep.pos, t=r.dep.t, active=r.active)
    return [{"stage": u["stage"], "scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"],
             "seed": u["seed"], "n_points": users.shape[0], "start": u["start"], "utility": u["util"].kind,
             "rho_m": u["util"].rho_m, "rules": rules.label, "land_alpha": rules.land_alpha,
             "call_alpha": rules.call_alpha, "kappa": round(swarm.kappa(phys, users.shape[0]), 6),
             "n95_central": n95, "n_owned": owned, "n_start": n_start, "n_final": n_final,
             "served_pct": 100.0 * r.served, "served_central_at_n_pct": _interp(f, n_final),
             "n_central_for_served": _n_for(f, 100.0 * r.served),
             "lands": sum(e[1] == "land" for e in r.events), "calls": sum(e[1] == "call" for e in r.events),
             "rounds": r.rounds, "converged": int(r.converged), "evals": r.evals, "time_s": round(r.time_s, 2),
             "history": json.dumps([[round(100.0 * s, 3), n] for s, n in r.history])}]


def run_unit(u: dict) -> int:
    path = unit_path(u)
    if path.exists():
        return 0
    if u["stage"] == "central":
        _write(path, CENTRAL_COLUMNS, run_central(u))
    elif u["stage"] == "cstrong":
        _write(path, STRONG_COLUMNS, run_cstrong(u))
    elif u["stage"] == "fstrong":
        _write(path, STRONG_COLUMNS, run_fstrong(u))
    elif u["stage"] == "rstrong":
        _write(path, RSTRONG_COLUMNS, run_rstrong(u))
    elif u["stage"] == "brcheck":
        _write(path, BR_COLUMNS, run_brcheck(u))
    elif u["stage"] in SIZE_STAGES:
        _write(path, FLEET_COLUMNS, run_selfsize(u))
    else:
        _write(path, COLUMNS, run_agents(u))
    return 1


def check(mine: list[dict]) -> list[str]:
    """Invariants of finished units: the team utility never loses served share; nothing beats the bound."""
    bad = []
    for u in mine:
        r = next(csv.DictReader(open(unit_path(u))))
        if u["stage"] == "central":
            if float(r["served_central_free_pct"]) < float(r["served_paper_pct"]) - 1e-7:
                bad.append(f"{_tag(u)}: central serves less than the paper's deployment")
            continue
        if u["stage"] in ("cstrong", "fstrong"):
            if float(r["served_strong_pct"]) < float(r["served_rescored_pct"]) - 1e-7:
                bad.append(f"{_tag(u)}: the strongest-association reference serves less than its free start")
            continue
        if u["stage"] == "rstrong":
            if float(r["served_strong_pct"]) < float(r["served_multi_pct"]) - 1e-7:
                bad.append(f"{_tag(u)}: polishing lost served share")
            continue
        if u["stage"] == "brcheck":
            if float(r["heuristic_gain_pp"]) < -1e-7 or float(r["brute_gain_pp"]) < -1e-7:
                bad.append(f"{unit_path(u).name}: a search lost utility")
            continue
        if u["stage"] in SIZE_STAGES:
            if not 1 <= int(r["n_final"]) <= int(r["n_owned"]):
                bad.append(f"{unit_path(u).name}: {r['n_final']} drones in the air of {r['n_owned']} owned")
            continue
        if r["utility"] == "global" and r["dynamics"] == "seq" and not r["info_m"] and \
                float(r["served_pct"]) < float(r["served_start_pct"]) - 1e-7:
            bad.append(f"{unit_path(u).name}: sequential global play lost served share")
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
        print(f"swarm {args.stage} shard {i}/{k}: {len(mine)} units, {len(done)} on disk")
        if args.check:
            bad = check(done) + ([] if len(done) == len(mine) else [f"{len(mine) - len(done)} units missing"])
            for b in bad:
                print("  " + b)
            print(f"RESULT: {'PASS' if not bad else 'FAIL'} swarm {args.stage} shard {i}/{k}")
            sys.exit(0 if not bad else 1)
        return
    todo = [u for u in mine if not unit_path(u).exists()]
    print(f"swarm {args.stage} shard {i}/{k}: {len(todo)} of {len(mine)} units to run, {args.procs} procs", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        for c, _ in enumerate(pool.imap_unordered(run_unit, todo, chunksize=1), start=1):
            el = time.time() - t0
            print(f"  {c}/{len(todo)} units  {el:.0f}s  eta {el / c * (len(todo) - c):.0f}s", flush=True)


if __name__ == "__main__":
    main()
