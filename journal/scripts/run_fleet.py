"""WP2/WP3: how many drones does a disaster area need? Served demand against fleet size.

Every cell is (variant, scenario, tier, seed): one user layout, its demand
(people, device share, activity) and one service tier. It evaluates the
served share (GBR admission at the tier's rate), coverage, round-robin
served share and end-to-end served share (through backhaul and gateway) of the K-means / sweep / greedy altitude methods (dronecomm.fleet,
objective: served demand) at every fleet size in FLEET and
writes results/fleet/<variant>/<scenario>/<tier>/seed_<s>.csv atomically;
existing cells are skipped. Sources: docs/parameters.md (row ids below).

python scripts/run_fleet.py --variant grounded --list
python scripts/run_fleet.py --variant grounded --shard 0/4 --procs 16
python scripts/run_fleet.py --variant grounded --shard 0/4 --check
python scripts/run_fleet.py --variant grounded --extend --shard 0/8 --procs 16   # N = 350-500, edge cells
python scripts/run_fleet.py --variant all --refine --shard 0/16 --procs 16        # dense N around each N*
"""

from __future__ import annotations

import argparse
import csv
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

from dronecomm.backhaul import TOPOLOGIES  # noqa: E402
from dronecomm.config import Config  # noqa: E402
from dronecomm.fleet import service_curve  # noqa: E402
from run_heuristic_experiment import generate_users_for_distribution, make_config  # noqa: E402
import json  # noqa: E402

GROUNDED = ROOT / "configs" / "grounded.yaml"
MILAN_TILES = ROOT / "results" / "telecom_milan" / "tiles"   # scripts/extract_milan_tiles.py
FLEET = list(range(1, 11)) + [12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150,
                              200, 250, 300]
# --extend: larger fleets for the cells that reach N = 300 without 95 % (results/fleet_ext)
EXT_FLEET = [350, 400, 500]
EXT_CELLS = {(s, "broadband") for s in ("shelter_hotspot", "shelter_clustered", "urban_broad",
                                         "milan_p075", "milan_p090", "milan_p100")} | {
    ("milan_p090", "video"), ("milan_p100", "video")}
SYN_POINTS = 1000          # simulated user points per synthetic layout
SYN_SEEDS = 10
MILAN_SEEDS = 3            # K-means initialisations per snapshot
BASE_SEED = 1000
# --refine: dense N windows around each quoted N* (configs/refine_windows.json, from
# scripts/journal_data.py), with more seeds; written to results/fleet_refine
REFINE_WINDOWS = ROOT / "configs" / "refine_windows.json"
REFINE_SYN_SEEDS = 40
REFINE_MILAN_SEEDS = 6

SPHERE_M2 = 45.0           # m2 per person in a planned camp [D1]


def sphere_people(n_clusters: int, sigma_m: float) -> float:
    """People in Gaussian clusters at Sphere density over the 2-sigma disc (86 % of a 2-D Gaussian)."""
    return n_clusters * math.pi * (2.0 * sigma_m) ** 2 / SPHERE_M2


def _milan_tiles() -> dict:
    """Milan tiles at activity percentiles, each with its 2021 census residents [D10]."""
    meta = json.loads((MILAN_TILES / "tiles_metadata.json").read_text())
    return {f"milan_p{t['percentile']:03d}": (f"tile{t['percentile']:03d}", t["census_pop"], None, None)
            for t in meta["tiles"]}


# scenario -> (layout, population, device share or None for the config's, activity or None)
SCENARIOS = {
    "shelter_clustered": ("clustered", sphere_people(5, 100.0), None, None),     # [D1]
    "shelter_hotspot": ("hotspot", sphere_people(3, 50.0), None, None),          # [D1]
    "urban_broad": ("uniform", 10_000 * 4.0, 1.0, 0.2),    # TS 22.261 urban macro users/km2, 20 % [D5]
    "rural_broad": ("uniform", 100 * 4.0, 1.0, 0.2),       # TS 22.261 rural users/km2, 20 % [D5]
    **_milan_tiles(),
}
TIERS = {                  # minimum rate per active user [bit/s]
    "text": 3e3,           # real-time text, TS 26.114 Annex E [D4]
    "voice": 38e3,         # VoLTE AMR-WB over IPv6, TS 26.114 [D3]
    "video": 384e3,        # H.264 video call, TS 26.114 Annex E [D4]
    "broadband": 2e6,      # extreme-rural user-experienced DL, TR 38.913 [D6]
}
TWO_WAY = ("text", "voice", "video")   # conversational tiers: the same rate flows back (broadband: downlink)


def _grounded(**changes) -> Config:
    cfg = Config.from_yaml(GROUNDED)
    for section, fields in changes.items():
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **fields)})
    return cfg


VARIANTS = {
    "grounded": lambda: _grounded(),
    "waiver300": lambda: _grounded(network={"z_max_m": 300.0}),                  # [R1, R2]
    "cap40": lambda: _grounded(capacity={"max_users_per_cell": 40}),             # [D7]
    "lowincome": lambda: _grounded(demand={"device_share": 0.53}),               # [D12]
    "suburban": lambda: _grounded(channel={"environment": "suburban"}),          # [C1]
    "dense_urban": lambda: _grounded(channel={"environment": "dense_urban"}),    # [C1]
    # 20 MHz access carrier at the same power density: TR 36.814 gives 30 dBm per 10 MHz [N2]
    "bw20": lambda: _grounded(network={"bandwidth_hz": 20.0e6, "p_tx_dl_dbm": 33.0}),
    "tether": lambda: _grounded(backhaul={"gateway_bps": None}),                 # fiber tether [B4]
    "bh200": lambda: _grounded(backhaul={"fixed_link_bps": 2.0e8}),              # SkyHaul lower end [B3]
    # worst case: one shared 2 GHz channel, every drone transmitting at once [B2, B7]
    "bh_shared": lambda: _grounded(backhaul={"link_model": "sinr"}),
}

COLUMNS = ["variant", "scenario", "tier", "seed", "layout", "n_points", "population",
           "active_per_point", "n_drones", "method", "served_pct", "coverage_pct", "served_rr_pct",
           "served_e2e_pct", *[f"{c}_{t}" for t in TOPOLOGIES for c in ("served_e2e_pct", "relays")],
           "mean_altitude_m", "min_altitude_m", "max_altitude_m", "n_evaluations", "build_time_s"]


def cells(extend: bool = False) -> list[dict]:
    out = []
    n_snap = len(json.loads((MILAN_TILES / "tiles_metadata.json").read_text())["snapshots"])
    for scen, tier in product(SCENARIOS, TIERS):
        if extend and (scen, tier) not in EXT_CELLS:
            continue
        seeds = ([(s, BASE_SEED + r) for s in range(n_snap) for r in range(MILAN_SEEDS)]
                 if scen.startswith("milan") else [(None, BASE_SEED + r) for r in range(SYN_SEEDS)])
        for snap, seed in seeds:
            out.append({"scenario": scen, "tier": tier, "snapshot": snap, "seed": seed})
    return out


def refine_cells() -> list[dict]:
    """One cell per (window, layout, seed), carrying its variant and fleet sizes; costliest first."""
    n_snap = len(json.loads((MILAN_TILES / "tiles_metadata.json").read_text())["snapshots"])
    out = []
    for w in json.loads(REFINE_WINDOWS.read_text()):
        seeds = ([(s, BASE_SEED + r) for s in range(n_snap) for r in range(REFINE_MILAN_SEEDS)]
                 if w["scenario"].startswith("milan") else [(None, BASE_SEED + r) for r in range(REFINE_SYN_SEEDS)])
        for snap, seed in seeds:
            out.append({"variant": w["variant"], "scenario": w["scenario"], "tier": w["tier"],
                        "snapshot": snap, "seed": seed, "fleet": w["fleet"]})
    # cost grows with the number of drones evaluated; round-robin shards of a sorted list balance
    return sorted(out, key=lambda c: (-sum(c["fleet"]), c["variant"], c["scenario"], c["tier"],
                                      -1 if c["snapshot"] is None else c["snapshot"], c["seed"]))


def cell_path(out: Path, variant: str, c: dict) -> Path:
    tag = f"seed_{c['seed']}" if c["snapshot"] is None else f"snap{c['snapshot']:02d}_seed_{c['seed']}"
    return out / variant / c["scenario"] / c["tier"] / f"{tag}.csv"


def shard(all_cells: list[dict], i: int, k: int) -> list[dict]:
    return all_cells[i::k]


def _users(c: dict) -> tuple[np.ndarray, str]:
    layout = SCENARIOS[c["scenario"]][0]
    if layout.startswith("tile"):
        name = f"{layout}_snap{c['snapshot']:02d}"
        return np.load(MILAN_TILES / f"{name}.npy"), name
    cfg = make_config(1, SYN_POINTS, layout)
    return generate_users_for_distribution(layout, SYN_POINTS, cfg, np.random.default_rng(c["seed"])), layout


def run_cell(args: tuple) -> int:
    variant, c, out, fleet = args
    path = cell_path(out, variant, c)
    if path.exists():
        return 0
    _, pop, share, act = SCENARIOS[c["scenario"]]
    base = VARIANTS[variant]()
    dem = replace(base.demand, population=pop, r_min_bps=TIERS[c["tier"]],
                  **({} if share is None else {"device_share": share}),
                  **({} if act is None else {"activity_factor": act}))
    cfg = replace(base, demand=dem, backhaul=replace(base.backhaul, two_way=c["tier"] in TWO_WAY))
    users, layout = _users(c)
    rows = []
    for r in service_curve(cfg, users, fleet, seed=c["seed"]):
        rows.append({
            "variant": variant, "scenario": c["scenario"], "tier": c["tier"], "seed": c["seed"],
            "layout": layout, "n_points": users.shape[0], "population": round(pop, 1),
            "active_per_point": dem.active_per_point(users.shape[0]),
            "n_drones": r.n_drones, "method": r.method,
            "served_pct": 100.0 * r.served, "coverage_pct": 100.0 * r.coverage,
            "served_rr_pct": 100.0 * r.served_round_robin,
            "served_e2e_pct": 100.0 * r.end_to_end[cfg.backhaul.topology][0],
            **{f"served_e2e_pct_{t}": 100.0 * r.end_to_end[t][0] for t in TOPOLOGIES},
            **{f"relays_{t}": r.end_to_end[t][1] for t in TOPOLOGIES},
            "mean_altitude_m": float(r.altitudes_m.mean()), "min_altitude_m": float(r.altitudes_m.min()),
            "max_altitude_m": float(r.altitudes_m.max()), "n_evaluations": r.n_evaluations,
            "build_time_s": round(r.build_time_s, 4),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".csv.tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS) + ["all"],
                    help="'all' with --refine: every variant in the refinement windows")
    ap.add_argument("--shard", default="0/1", help="I/K: run the I-th of K shards")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results/fleet (results/fleet_ext with --extend)")
    ap.add_argument("--extend", action="store_true", help=f"N = {EXT_FLEET} for EXT_CELLS only")
    ap.add_argument("--refine", action="store_true", help="dense N windows from configs/refine_windows.json")
    ap.add_argument("--refine-variants", default=None,
                    help="with --refine: only the windows of these variants (comma-separated)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if (args.variant == "all") != args.refine:
        ap.error("--variant all goes with --refine, and --refine with --variant all")
    if args.out is None:
        args.out = ROOT / "results" / ("fleet_refine" if args.refine else "fleet_ext" if args.extend else "fleet")
    fleet = EXT_FLEET if args.extend else FLEET
    i, k = (int(x) for x in args.shard.split("/"))
    pool_cells = refine_cells() if args.refine else cells(args.extend)
    if args.refine and args.refine_variants:
        keep = set(args.refine_variants.split(","))
        pool_cells = [c for c in pool_cells if c["variant"] in keep]
    mine = shard(pool_cells, i, k)
    var = lambda c: c.get("variant", args.variant)  # noqa: E731
    done = sum(cell_path(args.out, var(c), c).exists() for c in mine)
    if args.list or args.check:
        print(f"{args.variant} fleet shard {i}/{k}: {len(mine)} cells, {done} on disk")
        if args.check:
            print(f"RESULT: {'PASS' if done == len(mine) else 'FAIL'} shard {i}/{k}")
            sys.exit(0 if done == len(mine) else 1)
        return
    todo = [(var(c), c, args.out, c.get("fleet", fleet)) for c in mine if not cell_path(args.out, var(c), c).exists()]
    print(f"{args.variant} fleet shard {i}/{k}: {len(todo)} of {len(mine)} cells to run, {args.procs} procs",
          flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        for n, _ in enumerate(pool.imap_unordered(run_cell, todo, chunksize=1), start=1):
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} cells  {el:.0f}s  eta {el / n * (len(todo) - n):.0f}s", flush=True)


if __name__ == "__main__":
    main()
