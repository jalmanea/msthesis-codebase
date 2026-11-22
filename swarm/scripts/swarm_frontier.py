"""configs/swarm_frontier.json: the central served-vs-N frontier per layout, from the joint runs.

Reads results/joint/{coarse,dense,full_coarse,full_dense} of the joint-opt clone (method xyzt, free association;
scripts/run_joint.py) and keeps, per (scenario, tier, snapshot, seed), the best served % at every N run (the
full_* stages cover the journal's units: 40 synthetic seeds, 24 Milan snapshots x 6 K-means seeds). run_swarm.py
stages p4b-p4d size their fleets from it.

python scripts/swarm_frontier.py [--joint ../dronecomm-joint/results/joint]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "swarm_frontier.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", default=str(ROOT.parent / "dronecomm-joint" / "results" / "joint"))
    args = ap.parse_args()
    front = defaultdict(dict)
    for stage in ("coarse", "dense", "full_coarse", "full_dense"):
        for f in sorted(Path(args.joint, stage).glob("*/*/*.csv")):
            for r in csv.DictReader(open(f)):
                if r["method"] == "xyzt":
                    key = f"{r['scenario']}|{r['tier']}|{r['snapshot']}|{r['seed']}"
                    n, s = int(r["n_drones"]), round(float(r["served_pct"]), 4)
                    front[key][n] = max(s, front[key].get(n, s))
    out = {k: dict(sorted(v.items())) for k, v in sorted(front.items())}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{OUT}: {len(out)} layouts")


if __name__ == "__main__":
    main()
