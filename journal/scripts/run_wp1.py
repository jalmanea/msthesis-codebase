"""WP1: the coverage experiments of Section IV under each model variant.

Each variant is a full system model (configs/default.yaml for the idealised model,
configs/grounded.yaml plus one change for the others; see VARIANTS). Two grids,
with the same user layouts and K-means seeds as the earlier runs, so every row
pairs across variants and with results/heuristic_50, sweep_addition,
greedy_addition and telecom_milan:

  synthetic  1,500 cells: N = 1..50 x M = 100..1000 x {clustered, uniform, hotspot}
  milan      4,992 cells: 24 snapshots x N = 5..30 x M = 100..800

Every cell runs 20 seeds x 4 altitude methods (K-means at the fixed altitude,
analytic rule, sweep, greedy) and writes one CSV,
results/wp1/<variant>/<grid>/cell_<id>.csv, atomically. Existing cells are
skipped, so a re-run resumes.

python scripts/run_wp1.py --variant grounded --grid synthetic --list
python scripts/run_wp1.py --variant grounded --grid synthetic --shard 0/8 --procs 16
python scripts/run_wp1.py --variant grounded --grid synthetic --shard 0/8 --check
"""

from __future__ import annotations

import argparse
import csv
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

from dronecomm.capacity import capacity_summary  # noqa: E402
from dronecomm.channel import ENVIRONMENTS  # noqa: E402
from dronecomm.config import Config  # noqa: E402
from dronecomm.heuristic import (  # noqa: E402
    deploy_analytic_heuristic,
    deploy_kmeans_baseline,
    deploy_kmeans_global_altitude_sweep,
    greedy_per_drone_altitude,
)
from dronecomm.optimize import _build_models, _evaluate_single  # noqa: E402
from run_heuristic_experiment import generate_users_for_distribution, make_config  # noqa: E402
from run_telecom_statistical import load_snapshot_metadata, subsample_users  # noqa: E402

GROUNDED = ROOT / "configs" / "grounded.yaml"
SEEDS = 20
BASE_SEED = 1000                     # both grids


def _grounded(**changes) -> Config:
    cfg = Config.from_yaml(GROUNDED)
    for section, fields in changes.items():
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **fields)})
    return cfg


# One change each from grounded (docs/parameters.md rows in brackets).
VARIANTS = {
    "icc": lambda: Config(),
    "grounded": lambda: _grounded(),
    "waiver300": lambda: _grounded(network={"z_max_m": 300.0}),                     # [R1, R2]
    "element1x1": lambda: _grounded(antenna={"dl_g_max_dbi": 8.0, "dl_beamwidth_deg": 65.0}),  # [A1, A2]
    "panel4x4": lambda: _grounded(antenna={"dl_g_max_dbi": 16.0, "dl_beamwidth_deg": 24.8}),   # [A2]
    **{env: (lambda env=env: _grounded(channel={"environment": env}))                # [C1]
       for env in ("suburban", "dense_urban", "highrise_urban")},
}
assert set(ENVIRONMENTS) >= {"suburban", "dense_urban", "highrise_urban"}

METHODS = ["kmeans", "analytic", "sweep", "greedy"]

SYN_DRONES = list(range(1, 51))
SYN_USERS = list(range(100, 1001, 100))
SYN_DISTS = ["clustered", "uniform", "hotspot"]
MILAN_DRONES = list(range(5, 31))
MILAN_USERS = list(range(100, 801, 100))
MILAN_SNAP = ROOT / "results" / "telecom_milan" / "snapshots"

COLUMNS = [
    "variant", "grid", "cell", "layout", "n_drones", "n_users", "method", "seed",
    "coverage_pct", "mean_sinr_db", "sinr_5th_pct_db",
    "mean_altitude_m", "min_altitude_m", "max_altitude_m", "build_time_s", "n_evaluations",
    "served_pct", "rate_median_kbps", "rate_p10_kbps", "max_cell_load",
]


def cells(grid: str) -> list[dict]:
    """The grid's cells in the order of the earlier runs' task ids."""
    if grid == "synthetic":
        return [{"cell": i, "layout": d, "n_drones": n, "n_users": m}
                for i, (d, n, m) in enumerate(product(SYN_DISTS, SYN_DRONES, SYN_USERS))]
    if grid == "milan":
        n_snap = len(load_snapshot_metadata(MILAN_SNAP))
        return [{"cell": i, "layout": f"snapshot_{s:02d}", "n_drones": n, "n_users": m}
                for i, (s, n, m) in enumerate(product(range(n_snap), MILAN_DRONES, MILAN_USERS))]
    raise ValueError(grid)


def shard(all_cells: list[dict], i: int, k: int) -> list[dict]:
    """Round-robin over cells sorted by cost (N x M, largest first): balanced, longest first."""
    order = sorted(all_cells, key=lambda c: (-c["n_drones"] * c["n_users"], c["cell"]))
    return order[i::k]


def cell_path(out: Path, variant: str, grid: str, cell: int) -> Path:
    return out / variant / grid / f"cell_{cell:05d}.csv"


def _users(grid: str, c: dict, seed: int, cache: dict) -> np.ndarray:
    if grid == "synthetic":
        # make_config's scenario fields depend only on (distribution, M): same layouts in every variant
        cfg = make_config(c["n_drones"], c["n_users"], c["layout"])
        return generate_users_for_distribution(c["layout"], c["n_users"], cfg,
                                               np.random.default_rng(seed))
    if c["layout"] not in cache:
        cache[c["layout"]] = np.load(MILAN_SNAP / f"{c['layout']}.npy")
    return subsample_users(cache[c["layout"]], c["n_users"], seed)


def run_cell(args: tuple) -> int:
    variant, grid, c, out = args
    path = cell_path(out, variant, grid, c["cell"])
    if path.exists():
        return 0
    base = VARIANTS[variant]()
    cfg = replace(base, network=replace(base.network, n_drones=c["n_drones"]))
    dl, bh, ch = _build_models(cfg)
    cache: dict = {}
    rows = []
    for s in range(SEEDS):
        seed = BASE_SEED + s
        users = _users(grid, c, seed, cache)
        for method in METHODS:
            t0 = time.perf_counter()
            n_eval = ""
            if method == "kmeans":
                sc = deploy_kmeans_baseline(cfg, users, seed=seed)
            elif method == "analytic":
                sc = deploy_analytic_heuristic(cfg, users, seed=seed)
            elif method == "sweep":
                sc = deploy_kmeans_global_altitude_sweep(cfg, users, seed=seed)
                n_eval = cfg.network.altitude_grid().size
            else:
                res = greedy_per_drone_altitude(cfg, users, seed=seed)
                sc, n_eval = res.scenario, res.n_evaluations
            build_t = time.perf_counter() - t0
            m = _evaluate_single(sc, cfg, dl, bh, ch)
            z = sc.drone_positions[:, 2]
            rows.append({
                "variant": variant, "grid": grid, "cell": c["cell"], "layout": c["layout"],
                "n_drones": c["n_drones"], "n_users": c["n_users"], "method": method, "seed": seed,
                "coverage_pct": 100.0 * m.coverage_fraction, "mean_sinr_db": m.mean_sinr_db,
                "sinr_5th_pct_db": m.sinr_5th_percentile_db,
                "mean_altitude_m": float(z.mean()), "min_altitude_m": float(z.min()),
                "max_altitude_m": float(z.max()), "build_time_s": round(build_t, 5),
                "n_evaluations": n_eval, **capacity_summary(sc, cfg, dl, ch),
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
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--grid", required=True, choices=["synthetic", "milan"])
    ap.add_argument("--shard", default="0/1", help="I/K: run the I-th of K shards")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "wp1")
    ap.add_argument("--list", action="store_true", help="print the shard's size and exit")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every cell of the shard is on disk")
    args = ap.parse_args()

    i, k = (int(x) for x in args.shard.split("/"))
    mine = shard(cells(args.grid), i, k)
    done = sum(cell_path(args.out, args.variant, args.grid, c["cell"]).exists() for c in mine)
    if args.list or args.check:
        print(f"{args.variant} {args.grid} shard {i}/{k}: {len(mine)} cells, {done} on disk")
        if args.check:
            print(f"RESULT: {'PASS' if done == len(mine) else 'FAIL'} shard {i}/{k}")
            sys.exit(0 if done == len(mine) else 1)
        return

    todo = [(args.variant, args.grid, c, args.out) for c in mine
            if not cell_path(args.out, args.variant, args.grid, c["cell"]).exists()]
    print(f"{args.variant} {args.grid} shard {i}/{k}: {len(todo)} of {len(mine)} cells to run, "
          f"{args.procs} procs", flush=True)
    t0, n = time.time(), 0
    with Pool(args.procs) as pool:
        for n, _ in enumerate(pool.imap_unordered(run_cell, todo, chunksize=1), start=1):
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} cells  {el:.0f}s  eta {el / n * (len(todo) - n):.0f}s", flush=True)


if __name__ == "__main__":
    main()
