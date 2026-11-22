#!/usr/bin/env python3
"""No-interference counterfactual on the grounded model (thesis Chapter 4 sensitivity row and the
Chapter 1 scenario).  Not in the journal paper.

For each layout of a cell, place K-means drones and search their altitudes greedily, as the journal's
fleet runs do (dronecomm.fleet.greedy on the coarse altitude grid), twice:
  greedy        the grounded model (SINR with interference);
  noint         the same model whose SINR ignores interference (signal over noise);
and score the noint deployment again with interference (noint_real).  N* is read as in the journal's
refinement (journal_data.bootstrap_nstar: mean over units, smallest N reaching 95%), on the journal's
refinement seeds (run_fleet.refine seeds: 6 per Milan snapshot, 40 synthetic seeds).

    python thesis/noint_counterfactual.py run
    python thesis/noint_counterfactual.py summary
    python thesis/noint_counterfactual.py coverage

Uses the code in ../journal.  Output: thesis/noint_results/*.csv.
"""
import csv
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[1] / "journal"
sys.path[:0] = [str(J / "scripts"), str(J)]
import run_fleet as R                      # noqa: E402
import journal_data as JD                  # noqa: E402
from dronecomm import fleet as F           # noqa: E402
from dronecomm.sinr import linear_to_db    # noqa: E402

OUT = Path(__file__).resolve().parent / "noint_results"
WINDOWS = {   # dense N around both answers; the grounded N* of Table ch4:tab:sensitivity lies inside
    ("urban_broad", "voice"): list(range(8, 18)),
    ("milan_p100", "voice"): list(range(12, 23)),
    ("milan_p075", "voice"): list(range(8, 14)),
    ("milan_p075", "video"): list(range(38, 71)),
    ("milan_p025", "broadband"): list(range(50, 99, 2)) + list(range(99, 111)),
}


class NoInterference(F.GridService):
    def _sinr_db(self, signal, total):
        return linear_to_db(signal / self.noise_w)


def config(c):
    _, pop, share, act = R.SCENARIOS[c["scenario"]]
    base = R.VARIANTS["grounded"]()
    dem = replace(base.demand, population=pop, r_min_bps=R.TIERS[c["tier"]],
                  **({} if share is None else {"device_share": share}),
                  **({} if act is None else {"activity_factor": act}))
    return replace(base, demand=dem, backhaul=replace(base.backhaul, two_way=c["tier"] in R.TWO_WAY))


def run_cell(c):
    cfg = config(c)
    users, layout = R._users(c)
    unit = layout if c["scenario"].startswith("milan") else str(c["seed"])
    grid = cfg.network.altitude_grid()
    rows = []
    for n in c["fleet"]:
        xy = F._kmeans_centroids(users, n, c["seed"])
        g = F.GridService(cfg, users, xy, grid)
        k_g = F.greedy(g, F.sweep(g))[0]
        q = NoInterference(cfg, users, xy, grid)
        k_q = F.greedy(q, F.sweep(q))[0]
        for method, val in (("greedy", g.evaluate(k_g)), ("noint", q.evaluate(k_q)), ("noint_real", g.evaluate(k_q))):
            rows.append({"scenario": c["scenario"], "tier": c["tier"], "unit": unit, "seed": c["seed"],
                         "n_drones": n, "method": method, "served_pct": 100 * val[0], "coverage_pct": 100 * val[1]})
    return rows


def cells():
    seen, out = set(), []
    for c in R.refine_cells():
        key = (c["scenario"], c["tier"])
        if key in WINDOWS and c["variant"] == "tether" and (key, c["snapshot"], c["seed"]) not in seen:
            seen.add((key, c["snapshot"], c["seed"]))
            out.append({**c, "fleet": WINDOWS[key]})
    return out


def run():
    OUT.mkdir(exist_ok=True)
    todo = cells()
    rows = []
    with Pool(8) as p:
        for i, r in enumerate(p.imap_unordered(run_cell, todo), 1):
            rows += r
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} layouts", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "refined.csv", index=False)


def summary():
    d = pd.read_csv(OUT / "refined.csv")
    for (sc, t) in WINDOWS:
        line = [f"{sc} {t}:"]
        for m in ("greedy", "noint"):
            n, lo, hi = JD.bootstrap_nstar(d, sc, t, "served_pct", m)
            line.append(f"{m} N*={n:.0f} [{lo:.0f},{hi:.0f}]")
        n = JD.bootstrap_nstar(d, sc, t, "served_pct", "noint")[0]
        if np.isfinite(n):
            x = d[(d.scenario == sc) & (d.tier == t) & (d.n_drones == n)].groupby("method")["served_pct"].mean()
            line.append(f"| at noint N*={n:.0f}: believed {x['noint']:.1f}%, real {x['noint_real']:.1f}%")
        print(" ".join(line))


def coverage():
    """Coverage-sized fleet on the 75th-percentile tile, coarse N as the journal's coverage example."""
    todo = [{**c, "fleet": list(range(8, 25))} for c in R.cells() if c["scenario"] == "milan_p075" and c["tier"] == "text"]
    with Pool(8) as p:
        rows = [r for rs in p.map(run_cell, todo) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "coverage_p075.csv", index=False)
    m = d.groupby(["method", "n_drones"])["coverage_pct"].mean()
    for meth in ("greedy", "noint"):
        n = int(m[meth][m[meth] >= 90].index[0])
        print(f"{meth}: fewest drones covering 90% = {n} ({m[meth][n]:.1f}%); real SINR coverage of that "
              f"deployment {m['noint_real'][n] if meth == 'noint' else m['greedy'][n]:.1f}%")


if __name__ == "__main__":
    {"run": run, "summary": summary, "coverage": coverage}[sys.argv[1]]()
