"""Tables from results/swarm (scripts/run_swarm.py): price of locality, dynamics, pointing, information.

Every figure is a mean over the layouts of a cell (scenario, tier, N), in percentage points of
served demand; pooled gaps carry a 95 % cluster-bootstrap interval over the journal's units (a Milan
snapshot, or a synthetic seed: the K-means seeds and fleet sizes of one unit are resampled together).
--core keeps only the audit's 23 layouts (run_swarm.CORE_LAYOUTS). gap = served - central. The central reference is stage cstrong's (re-optimised under
strongest-cell association, the best of several starts) when it is on disk, else stage central's
free-association optimum scored under strongest-cell association (--ref old forces the latter).
Self-sizing (p4b-p4e) is judged against stage fstrong's frontier when it is on disk.

python scripts/analyze_swarm.py                 # all stages on disk
python scripts/analyze_swarm.py --stage p1
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "swarm"


def _label(r: dict) -> str:
    rho = float(r["rho_m"])
    return f"window{rho:.0f}" if r["utility"] == "window" else r["utility"]


CORE: set | None = None     # (scenario, tier, snapshot, seed) of the audit's layouts, with --core


def load(stage: str) -> list[dict]:
    rows = [next(csv.DictReader(open(p))) for p in sorted((OUT / stage).rglob("*.csv"))]
    return rows if CORE is None else [r for r in rows if (r["scenario"], r["tier"], r["snapshot"], r["seed"]) in CORE]


REF: dict = {}              # (scenario, tier, snapshot, seed, n) -> central served %, filled by main()
FRONT: dict = {}            # (scenario, tier, snapshot, seed) -> {n: running-max served %}, from fstrong


def _key(r: dict, n_col: str = "n_drones") -> tuple:
    return r["scenario"], r["tier"], r["snapshot"], r["seed"], r[n_col]


def central_of(r: dict) -> float:
    return REF.get(_key(r), float(r["served_central_pct"]))


def load_refs() -> None:
    for r in load("cstrong"):
        REF[_key(r)] = float(r["served_strong_pct"])
    raw = defaultdict(dict)
    for r in load("fstrong"):
        raw[_key(r)[:4]][int(r["n_drones"])] = float(r["served_strong_pct"])
    for r in load("rstrong"):                       # the journal's further starts, polished: the better of the two
        k, v = _key(r), float(r["served_strong_pct"])
        if k in REF:
            REF[k] = max(REF[k], v)
        n = int(k[4])
        if n in raw.get(k[:4], {}):
            raw[k[:4]][n] = max(raw[k[:4]][n], v)
    for k, f in raw.items():
        best, FRONT[k] = -math.inf, {}
        for n in sorted(f):
            best = max(best, f[n])
            FRONT[k][n] = best


def _interp(f: dict, n: float) -> float:
    ns = np.array(sorted(f))
    if n < ns[0] or n > ns[-1]:
        return float("nan")
    return float(np.interp(n, ns, [f[k] for k in ns]))


def _n_for(f: dict, served: float) -> float:
    ns = sorted(f)
    if served <= f[ns[0]]:
        return float("nan") if served < f[ns[0]] else float(ns[0])
    for a, b in zip(ns, ns[1:]):
        if f[b] >= served:
            return a + (b - a) * (served - f[a]) / max(f[b] - f[a], 1e-12)
    return float("nan")


def strong_table() -> None:
    """How the central reference moves: free optimum rescored vs. re-optimised under strongest association."""
    rows = load("cstrong")
    by = defaultdict(list)
    for r in rows:
        by[cell(r)].append(r)
    print("\n## cstrong: the central reference under strongest-cell association (served %, mean over layouts)\n")
    print("| cell | layouts | free opt. | rescored | from paper | restarts | from free | best agent run | "
          "from agent | reference | sources |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c, rs in sorted(by.items()):
        f = lambda k: np.nanmean([float(x[k]) for x in rs])
        src = defaultdict(int)
        for x in rs:
            src[x["strong_source"]] += 1
        print(f"| {c} | {len(rs)} | {f('served_central_free_pct'):.2f} | {f('served_rescored_pct'):.2f} | "
              f"{f('served_from_paper_pct'):.2f} | {f('served_from_restarts_pct'):.2f} | {f('served_from_free_pct'):.2f} | "
              f"{f('served_best_agent_pct'):.2f} | {f('served_from_agent_pct'):.2f} | {f('served_strong_pct'):.2f} | "
              f"{', '.join(f'{k} {v}' for k, v in sorted(src.items()))} |")
    shift = [float(x["served_strong_pct"]) - float(x["served_rescored_pct"]) for x in rows]
    print(f"\nReference shift (new - old): mean {np.mean(shift):+.2f} pp, max {np.max(shift):+.2f} pp, "
          f"{len(shift)} layouts")


def brcheck_table() -> None:
    """Heuristic best response vs. exhaustive grid at the final states of p1 (xyzt) and p2 (nadir)."""
    rows = load("brcheck")
    tol = 0.01                                  # pp: run_swarm's tol = 1e-4 of all demand
    by = defaultdict(list)
    for r in rows:
        by[("xyzt" if int(r["pointing"]) else "nadir", _label(r))].append(r)
    print("\n## brcheck: is the heuristic best response a best response? (urban N = 8, 11; per agent)\n")
    print(f"gain = best utility found - current, pp of all demand; an agent 'can gain' if gain > {tol} pp\n")
    print("| pointing | utility | agents | heur. can gain | grid can gain | grid > heur. by >tol | mean missed | "
          "max missed | converged runs | of which not an equilibrium |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (pt, u), rs in sorted(by.items()):
        f = lambda k: np.array([float(x[k]) for x in rs])
        miss = f("missed_pp")
        runs = defaultdict(list)
        for x in rs:
            runs[(x["seed"], x["n_drones"])].append(x)
        conv = [v for v in runs.values() if v[0]["run_converged"] == "1"]
        broken = sum(any(float(x["brute_gain_pp"]) > tol for x in v) for v in conv)
        print(f"| {pt} | {u} | {len(rs)} | {100 * np.mean(f('heuristic_gain_pp') > tol):.0f} % | "
              f"{100 * np.mean(f('brute_gain_pp') > tol):.0f} % | {100 * np.mean(miss > tol):.0f} % | "
              f"{miss.mean():+.3f} | {miss.max():+.3f} | {len(conv)} | {broken} |")


BOOT = 2000


def unit(r: dict) -> tuple:
    """Bootstrap unit: the Milan snapshot or the synthetic seed (as Table tab:fleet)."""
    return (r["scenario"], r["snapshot"]) if r["scenario"].startswith("milan") else (r["scenario"], r["seed"])


def boot_ci(rows: list[dict], gaps: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the mean gap, resampling units within each scenario."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: defaultdict(list))
    for r, g in zip(rows, gaps):
        by[r["scenario"]][unit(r)].append(g)
    groups = [[np.array(v) for v in u.values()] for u in by.values()]
    means = np.empty(BOOT)
    for b in range(BOOT):
        tot, cnt = 0.0, 0
        for us in groups:
            for i in rng.integers(0, len(us), len(us)):
                tot += us[i].sum()
                cnt += us[i].size
        means[b] = tot / cnt
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell(r: dict) -> str:
    return f"{r['scenario']}/{r['tier']} N={r['n_drones']}"


def central_table() -> None:
    rows = load("central")
    by = defaultdict(list)
    for r in rows:
        by[cell(r)].append(r)
    print("\n## Central references (served %, mean over layouts)\n")
    print("| cell | layouts | paper | central (free assoc.) | central (strongest) |")
    print("|---|---|---|---|---|")
    for c, rs in sorted(by.items()):
        f = lambda k: np.mean([float(x[k]) for x in rs])
        print(f"| {c} | {len(rs)} | {f('served_paper_pct'):.1f} | {f('served_central_free_pct'):.1f} | "
              f"{f('served_central_pct'):.1f} |")


def table(rows: list[dict], key, title: str) -> None:
    """One row per (cell, key(r)): mean served, mean gap to central, converged share, mean rounds."""
    by = defaultdict(list)
    for r in rows:
        by[(cell(r), key(r))].append(r)
    print(f"\n## {title}\n")
    print("| cell | run | n | served | gap to central | start | converged | rounds |")
    print("|---|---|---|---|---|---|---|---|")
    for (c, k), rs in sorted(by.items(), key=lambda q: (q[0][0], str(q[0][1]))):
        s = np.array([float(x["served_pct"]) for x in rs])
        g = s - np.array([central_of(x) for x in rs])
        st = np.mean([float(x["served_start_pct"]) for x in rs])
        conv = np.mean([int(x["converged"]) for x in rs])
        rounds = np.mean([int(x["rounds"]) for x in rs])
        print(f"| {c} | {k} | {len(rs)} | {s.mean():.1f} | {g.mean():+.1f} (worst {g.min():+.1f}) | {st:.1f} | "
              f"{100 * conv:.0f} % | {rounds:.1f} |")


def pooled(rows: list[dict], key, title: str) -> None:
    """Across all cells: mean gap to central per key(r)."""
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    print(f"\n## {title} (pooled over cells)\n")
    print("| run | n | units | mean gap | 95 % CI | median | worst | share within 1 pp |")
    print("|---|---|---|---|---|---|---|---|")
    for k, rs in sorted(by.items(), key=lambda q: str(q[0])):
        g = np.array([float(r["served_pct"]) - central_of(r) for r in rs])
        lo, hi = boot_ci(rs, g)
        print(f"| {k} | {g.size} | {len({unit(r) for r in rs})} | {g.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
              f"{np.median(g):+.2f} | {g.min():+.2f} | {100 * np.mean(g >= -1.0):.0f} % |")


def cycling(rows: list[dict]) -> None:
    """Simultaneous runs that did not converge: does served share oscillate?"""
    sim = [r for r in rows if r["dynamics"] != "seq" and not int(r["converged"])]
    if not sim:
        return
    amp = []
    for r in sim:
        h = np.array(eval(r["history"]))          # json list of floats
        tail = h[len(h) // 2:]
        amp.append(tail.max() - tail.min())
    print(f"\nNon-converged simultaneous runs: {len(sim)}; served-share swing over the second half of the "
          f"run: median {np.median(amp):.1f} pp, max {np.max(amp):.1f} pp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_stages = ["central", "cstrong", "brcheck", "p1", "p2", "p3", "p4a", "p4b", "p4c", "p4d", "p4e"]
    ap.add_argument("--stage", choices=all_stages)
    ap.add_argument("--ref", choices=["strong", "old"], default="strong",
                    help="central reference: cstrong/fstrong when on disk (default), or the old rescored optimum")
    ap.add_argument("--core", action="store_true", help="only the audit's 23 layouts (run_swarm.CORE_LAYOUTS)")
    args = ap.parse_args()
    stages = [args.stage] if args.stage else all_stages
    if args.core:
        global CORE
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_swarm import CORE_LAYOUTS
        CORE = {(sc, t, "" if snap is None else str(snap), str(seed))
                for (sc, t), lays in CORE_LAYOUTS.items() for snap, seed in lays}
    if args.ref == "strong":
        load_refs()
    print(f"central reference: {'cstrong' if REF else 'central (free optimum, rescored)'}; "
          f"self-sizing frontier: {'fstrong' if FRONT else 'configs/swarm_frontier.json (free association)'}")
    if "central" in stages and (OUT / "central").exists():
        central_table()
    if "cstrong" in stages and (OUT / "cstrong").exists():
        strong_table()
    if "brcheck" in stages and (OUT / "brcheck").exists():
        brcheck_table()
    if "p1" in stages and (OUT / "p1").exists():
        p1 = load("p1")
        seq = [r for r in p1 if r["dynamics"] == "seq"]
        pooled(seq, lambda r: f"{r['start']} {_label(r)}", "p1: utility x start, sequential")
        table([r for r in seq if r["start"] == "paper"], _label, "p1: from the paper's deployment, sequential")
        table([r for r in seq if r["start"] == "depot"], _label, "p1: from the depot, sequential")
        pooled([r for r in p1 if r["start"] != "central"], lambda r: f"{_label(r)} {r['dynamics']}",
               "p1: utility x dynamics (paper and depot starts)")
        cycling(p1)
    if "p2" in stages and (OUT / "p2").exists():
        p2 = load("p2")
        pooled(p2, _label, "p2: nadir only, from the paper's deployment, sequential")
    if "p3" in stages and (OUT / "p3").exists():
        p3 = load("p3")
        pooled(p3, lambda r: f"{r['start']} {_label(r)} info{float(r['info_m']):.0f}",
               "p3: information radius, sequential")

    if "p4a" in stages and (OUT / "p4a").exists():
        p4a = load("p4a")
        pooled(p4a, lambda r: f"{r['start']} {_label(r)}", "p4a: with exploration, fixed N, sequential")
    for st in ("p4b", "p4c", "p4d", "p4e"):
        if st in stages and (OUT / st).exists():
            selfsize(load(st), st)


def rejudge(r: dict) -> dict:
    """A self-sizing row judged against the strongest-association frontier (fstrong) when it is loaded."""
    f = FRONT.get(_key(r, "n_final")[:4])
    if not f:
        return r
    return {**r, "served_central_at_n_pct": _interp(f, int(r["n_final"])),
            "n_central_for_served": _n_for(f, float(r["served_pct"])),
            "n95_central": next((n for n, v in f.items() if v >= 95.0), max(f))}


def selfsize(rows: list[dict], stage: str = "p4b") -> None:
    """Self-sized fleets against the central frontier: drones, served, and the drones central needs for it."""
    by = defaultdict(list)
    for r in map(rejudge, rows):
        by[(r["start"], _label(r), r["rules"])].append(r)
    print(f"\n## {stage}: self-sizing (pooled over layouts)\n")
    print("excess = n_final - (drones the central frontier needs to serve the same share); "
          "N/N95 = n_final / central N95\n")
    print("| start | utility | rules | n | served [95 % CI] | gap at same N | excess drones | N/N95 [95 % CI] | lands | calls | converged |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (st, u, rl), rs in sorted(by.items()):
        f = lambda k: np.array([float(x[k]) for x in rs])
        gap = f("served_pct") - f("served_central_at_n_pct")
        excess = f("n_final") - f("n_central_for_served")
        ratio = f("n_final") / f("n95_central")
        (slo, shi), (rlo, rhi) = boot_ci(rs, f("served_pct")), boot_ci(rs, ratio)
        print(f"| {st} | {u} | {rl} | {len(rs)} | {f('served_pct').mean():.1f} [{slo:.1f}, {shi:.1f}] | "
              f"{np.nanmean(gap):+.1f} | {np.nanmean(excess):+.1f} ({int(np.sum(np.isnan(excess)))} nan) | "
              f"{ratio.mean():.2f} [{rlo:.2f}, {rhi:.2f}] | {f('lands').mean():.1f} | {f('calls').mean():.1f} | "
              f"{100 * f('converged').mean():.0f} % |")


if __name__ == "__main__":
    main()
