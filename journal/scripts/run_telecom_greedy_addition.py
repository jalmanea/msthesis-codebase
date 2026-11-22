"""Run the per-drone greedy altitude heuristic on the paper's Milan design.

24 snapshots x N = 5..30 x M = 100..800 x 20 seeds, the design of
results/telecom_milan/main. Tasks are enumerated by the same
`generate_main_tasks`, and each seed draws the same user subsample and K-means
initialisation, so every greedy row pairs with the K-means, analytic and sweep
rows of the same task file. Resumable: finished task files are skipped.

python scripts/run_telecom_greedy_addition.py --procs 6
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_telecom_statistical as rts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "results" / "telecom_milan" / "snapshots"
OUT = ROOT / "results" / "telecom_milan_greedy"
USERS = [100, 200, 300, 400, 500, 600, 700, 800]


def _one(task: dict) -> None:
    if (OUT / "main" / f"task_{task['task_id']:04d}.csv").exists():
        return
    rts.run_main_task(task, rts.load_snapshot_metadata(SNAP), SNAP, OUT, n_seeds=20,
                      base_seed=rts.BASE_SEED, methods=["kmeans_greedy_altitude"],
                      greedy_res=rts.GREEDY_RES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    tasks = rts.generate_main_tasks(len(rts.load_snapshot_metadata(SNAP)), rts.DRONE_COUNTS, USERS)
    assert len(tasks) == 4992
    with Pool(args.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(_one, tasks, chunksize=4)):
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} tasks", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
