"""Tables from results/llm (scripts/run_llm.py): one LLM per drone against its algorithmic and scripted baselines.

Every l1 unit is (model, layout, framing, repeat): a fixed fleet from the paper's deployment, ROUNDS
simultaneous rounds with inertia 0.5, information radius INFO_M. The baselines on the same layouts and
dynamics are stage l1base (the best-response agent of dronecomm.swarm with the selfish or global utility)
and stage l1pol (scripted policies through the LLM's own tools and what_if budget). change = served at the
last round - served at the start, pp of all demand; a unit's served share at the start is the same for every
agent. Per-decision figures come from the turn traces (*.jsonl): a decision is one drone's turn in one round.

python scripts/analyze_llm.py
python scripts/analyze_llm.py --stage l1 --stage l1pol
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "llm"
STAGES = ["l1", "l1log", "l1gate", "l1base", "l1pol"]
POSE = ("x", "y", "z", "tilt_deg", "azimuth_deg")


def load(stage: str) -> list[dict]:
    return [next(csv.DictReader(open(p))) for p in sorted((OUT / stage).rglob("*.csv"))]


def _layout(r: dict) -> str:
    s = f"seed {r['seed']}" if r["snapshot"] in ("", "None") else f"snap {r['snapshot']}"
    return f"{r['scenario']} {s} N={r['n_drones']}"


def _f(rs: list[dict], k: str) -> np.ndarray:
    return np.array([float(r[k]) for r in rs])


def _size(model: str) -> float:
    """Qwen3.5 parameter count in billions from the key (qwen3.5-0.8b -> 0.8); baselines sort last."""
    try:
        return float(model.split("-")[1].rstrip("b"))
    except (IndexError, ValueError):
        return float("inf")


def served_table(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["stage"], r["model"], r["framing"])].append(r)
    print("\n## Served share: change over the run, by agent and framing (fixed fleet, paper start)\n")
    print("| stage | agent | framing | units | start | end | change | min | max | best round | "
          "units that gain |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    order = lambda k: (STAGES.index(k[0]), _size(k[1]), k[1], k[2])
    for k in sorted(by, key=order):
        rs = by[k]
        s0, s1, best = _f(rs, "served_start_pct"), _f(rs, "served_pct"), _f(rs, "best_round_pct")
        d = s1 - s0
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {s0.mean():.1f} | {s1.mean():.1f} | {d.mean():+.1f} | "
              f"{d.min():+.1f} | {d.max():+.1f} | {best.mean():.1f} | {int(np.sum(d > 0.05))} of {len(rs)} |")


def layout_table(rows: list[dict]) -> None:
    """Mean change per layout, so model differences can be read against the layout spread."""
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["framing"], _layout(r))].append(float(r["served_pct"]) - float(r["served_start_pct"]))
    lays = sorted({k[2] for k in by})
    agents = sorted({k[:2] for k in by}, key=lambda k: (_size(k[0]), k[0], k[1]))
    print("\n## Change by layout (mean over repeats, pp)\n")
    print("| agent | framing | " + " | ".join(lays) + " |")
    print("|---|---|" + "---|" * len(lays))
    for a in agents:
        print(f"| {a[0]} | {a[1]} | " + " | ".join(
            f"{np.mean(by[a + (l,)]):+.1f}" if (a + (l,)) in by else "–" for l in lays) + " |")


def _turns(stage: str, model: str) -> list[dict]:
    out = []
    for p in sorted((OUT / stage / model).glob("*.jsonl")):
        for line in open(p):
            out.extend(json.loads(line)["turns"])
    return out


def _same(a: dict, b: dict, tol: float = 0.5) -> bool:
    """Same pose up to what_if's rounding (0.1 m, 0.1 deg): within tol m and tol deg; azimuth ignored at nadir."""
    try:
        v = [float(a[k]) - float(b[k]) for k in POSE]
    except (KeyError, TypeError, ValueError):
        return False
    az = abs((v[4] + 180.0) % 360.0 - 180.0)
    nadir = max(abs(float(a["tilt_deg"])), abs(float(b["tilt_deg"]))) < tol
    return all(abs(d) <= tol for d in v[:4]) and (nadir or az <= tol)


def _checked(t: dict) -> bool:
    """A move to a pose the agent had asked what_if about in the same turn."""
    tr = t.get("trace") or []
    if t["action"] != "move_to" or not tr or tr[-1]["tool"] != "move_to":
        return False
    return any(c["tool"] == "what_if" and _same(c["args"], tr[-1]["args"]) for c in tr[:-1])


def _goal(framing: str) -> tuple[str, int]:
    """The what_if field and the people_before/after index that the framing's goal counts."""
    return ("change_in_your_cell", 0) if framing.startswith("selfish") else ("change_in_your_area", 1)


def prediction_table(stages: list[str]) -> None:
    """Moves against the agent's own what_if predictions; only runs that log tool results (from l1log on)."""
    print("\n## Moves against the agent's own what_if predictions (runs that log tool results)\n")
    print("goal = people in the cell (selfish) or the area (area, network). A move is predicted to gain when "
          "the what_if of the same pose predicted the goal to rise. realised = goal after the round - before, "
          "for moves applied that round (all drones move at once, so it includes the neighbours' moves).\n")
    print("| stage | agent | framing | moves | to predicted gain | to predicted loss | to best what_if | "
          "stays with a predicted gain | predicted, applied moves | realised, applied moves |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st in stages:
        if not (OUT / st).is_dir():
            continue
        for m in sorted((p.name for p in (OUT / st).iterdir() if p.is_dir()), key=lambda m: (_size(m), m)):
            by = defaultdict(list)
            for p in sorted((OUT / st / m).glob("*.jsonl")):
                for line in open(p):
                    by[p.stem.rsplit("_", 2)[1]].extend(json.loads(line)["turns"])
            for fr, ts in sorted(by.items()):
                if not any(c.get("result") for t in ts for c in t["trace"] or []):
                    continue
                key, i = _goal(fr)
                moves, stays_gain, pred, real = [], [], [], []
                for t in ts:
                    tr = t["trace"] or []
                    wif = [c for c in tr if c["tool"] == "what_if" and key in (c.get("result") or {})]
                    best = max((c["result"][key] for c in wif), default=None)
                    if t["action"] == "stay":
                        stays_gain.append(best is not None and best > 0)
                    if not _checked(t):
                        continue
                    d = next((c["result"][key] for c in reversed(wif) if _same(c["args"], tr[-1]["args"])), None)
                    if d is None:                  # only the what_if past the budget had that pose
                        continue
                    moves.append((d > 0, d < 0, d >= best))
                    if t["applied"] == "move" and t["people_after"]:
                        pred.append(d)
                        real.append(t["people_after"][i] - t["people_before"][i])
                mv = np.array(moves, dtype=bool).reshape(-1, 3)
                print(f"| {st} | {m} | {fr} | {len(mv)} | {100 * mv[:, 0].mean():.0f} % | {100 * mv[:, 1].mean():.0f} % | "
                      f"{100 * mv[:, 2].mean():.0f} % | {100 * np.mean(stays_gain):.0f} % | "
                      f"{np.mean(pred):+.1f} | {np.mean(real):+.1f} |")


def decision_table(stages: list[str]) -> None:
    print("\n## Per decision (one drone's turn): actions, errors, cost\n")
    print("errors: max_calls = the turn used its 12 tool calls without an action; no_tool_call = the model "
          "answered in text; refused = the request exceeded the context. An errored turn counts as stay.\n")
    print("| stage | agent | turns | move | stay | checked moves | what_ifs | tool calls | errors | "
          "error kinds | prompt tok. | completion tok. | model s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in stages:
        if st == "l1base":
            continue                       # the algorithmic agent has no turns
        for m in sorted((p.name for p in (OUT / st).iterdir() if p.is_dir()), key=lambda m: (_size(m), m)):
            ts = _turns(st, m)
            if not ts:
                continue
            act = Counter(t["action"] for t in ts)
            moves = [t for t in ts if t["action"] == "move_to"]
            kinds = Counter(str(t["error"]).split(":")[0] for t in ts if t["error"])
            f = lambda k: np.mean([t[k] for t in ts])
            print(f"| {st} | {m} | {len(ts)} | {100 * act['move_to'] / len(ts):.0f} % | "
                  f"{100 * act['stay'] / len(ts):.0f} % | "
                  f"{100 * np.mean([_checked(t) for t in moves]) if moves else float('nan'):.0f} % | "
                  f"{f('what_ifs'):.1f} | {f('calls'):.1f} | {100 * len([t for t in ts if t['error']]) / len(ts):.1f} % | "
                  f"{', '.join(f'{k} {v}' for k, v in kinds.most_common()) or '–'} | {f('prompt_tokens'):.0f} | "
                  f"{f('completion_tokens'):.0f} | {f('seconds'):.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=STAGES, action="append")
    args = ap.parse_args()
    stages = args.stage or STAGES
    rows = [r for st in stages if (OUT / st).is_dir() for r in load(st)]
    if not rows:
        raise SystemExit(f"no results under {OUT} for {stages}")
    served_table(rows)
    layout_table(rows)
    decision_table([st for st in stages if (OUT / st).is_dir()])
    prediction_table(stages)


if __name__ == "__main__":
    main()
