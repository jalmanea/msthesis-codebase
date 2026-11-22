"""Run the per-drone greedy altitude heuristic on the GLOBECOM grid.

Same 1,500 configurations (N = 1..50, M = 100..1000, three distributions,
20 seeds) and the same user layouts as results/heuristic_50/ and
results/sweep_addition/, so the rows merge with both. The greedy starts from
the sweep's common altitude (`greedy_per_drone_altitude`), recorded in
`sweep_altitude_m`; the sweep's own coverage is the matching row of
results/sweep_addition/synthetic_sweep.csv.

Usage
-----
python scripts/run_greedy_addition.py --out results/greedy_addition
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from dronecomm.heuristic import greedy_per_drone_altitude
from dronecomm.optimize import _build_models, _evaluate_single
from run_heuristic_experiment import (  # noqa: E402
    generate_users_for_distribution,
    make_config,
)

# The GLOBECOM grid, verbatim.
DRONES = list(range(1, 51))
USERS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
DISTS = ["clustered", "uniform", "hotspot"]
SEEDS = 20
BASE_SEED = 1000

COLUMNS = [
    "dist", "n_drones", "n_users", "method", "seed",
    "coverage_pct", "mean_sinr_db", "sinr_5th_pct_db",
    "total_interference_dbm", "mean_altitude_m", "build_time_s",
    "sweep_altitude_m", "min_altitude_m", "max_altitude_m",
    "n_passes", "n_changes", "n_evaluations",
]


def _run_cell(args) -> list[dict]:
    dist, n_drones, n_users = args
    cfg = make_config(n_drones, n_users, dist)
    models = _build_models(cfg)
    rows = []
    for s in range(SEEDS):
        seed = BASE_SEED + s
        rng = np.random.default_rng(seed)
        users = generate_users_for_distribution(dist, n_users, cfg, rng)
        t0 = time.perf_counter()
        res = greedy_per_drone_altitude(cfg, users, seed=seed)
        build_t = time.perf_counter() - t0
        sc = res.scenario
        m = _evaluate_single(sc, cfg, *models)
        rows.append({
            "dist": dist, "n_drones": n_drones, "n_users": n_users,
            "method": "kmeans_greedy_altitude", "seed": seed,
            "coverage_pct": m.coverage_fraction * 100.0,
            "mean_sinr_db": m.mean_sinr_db,
            "sinr_5th_pct_db": m.sinr_5th_percentile_db,
            "total_interference_dbm": m.total_inter_drone_interference_dbm,
            "mean_altitude_m": float(np.mean(sc.drone_positions[:, 2])),
            "build_time_s": round(build_t, 6),
            "sweep_altitude_m": res.sweep_altitude_m,
            "min_altitude_m": float(np.min(sc.drone_positions[:, 2])),
            "max_altitude_m": float(np.max(sc.drone_positions[:, 2])),
            "n_passes": res.n_passes, "n_changes": res.n_changes,
            "n_evaluations": res.n_evaluations,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/greedy_addition")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    jobs = list(product(DISTS, DRONES, USERS))
    print(f"greedy altitude: {len(jobs)} cells x {SEEDS} seeds "
          f"(matching the GLOBECOM grid)")

    t0, rows = time.time(), []
    with Pool(args.procs) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_run_cell, jobs, chunksize=1)):
            rows.extend(chunk)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"  {i+1}/{len(jobs)} cells  {el:.0f}s  "
                      f"eta {el/(i+1)*(len(jobs)-i-1):.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_greedy.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
