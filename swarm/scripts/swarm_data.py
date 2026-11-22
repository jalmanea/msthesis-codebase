"""Numbers and series for the swarm paper (paper/main_swarm.tex), shared by swarm_numbers.py and swarm_figures.py.

Fixed-N stages (p1-p4a) are judged against the central reference of stage cstrong, self-sizing stages
(p4b-p4e) against the frontier of stage fstrong, exactly as scripts/analyze_swarm.py does; intervals are its
95 % cluster bootstrap over the journal's units. LLM stages come from scripts/analyze_llm.py.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import analyze_llm  # noqa: E402
import analyze_swarm as A  # noqa: E402
from dronecomm import joint  # noqa: E402
from run_swarm import BASE_SEED, _problem  # noqa: E402

A.load_refs()

SCENARIOS = ("urban_broad", "milan_p100", "shelter_clustered")
SPREAD = ("urban_broad", "milan_p100")                    # spread demand; shelter is clustered
UTIL_ORDER = ("selfish", "window150", "window300", "window600", "window1200", "global")
# the full radius sweep: RHOS of p1-p3 plus stage fine's FINE_RADII, every multiple of 100 m and 150 m
RADII = (100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
LADDER_ORDER = ("selfish",) + tuple(f"window{r}" for r in RADII) + ("global",)
FINAL_RULE = "x_land0.25_call0.5_area1200_pat2_gr3"      # the self-sizing rule the paper adopts
INR_DIST = (100, 300, 500, 800, 1200, 2000, 3000)


@lru_cache(maxsize=None)
def fixed(stage: str) -> tuple[dict, ...]:
    """Rows of a fixed-N stage with 'label' and 'gap' (served - central reference, pp)."""
    return tuple({**r, "label": A._label(r), "gap": float(r["served_pct"]) - A.central_of(r)} for r in A.load(stage))


def sweep(kind: str) -> list[dict]:
    """One row per layout and radius, from the paper's start, sequential play, every radius of RADII:
    kind 'point' (p1 utilities with pointing), 'nadir' (p2), 'info' (team with information radius, p3)."""
    fine = fixed("fine")
    if kind == "point":
        return pick(fixed("p1"), start="paper", dynamics="seq") + pick(fine, utility="window", pointing="1")
    if kind == "nadir":
        return list(fixed("p2")) + pick(fine, utility="window", pointing="0")
    if kind == "info":
        return pick(fixed("p3"), start="paper", label="global") + pick(fine, label="global")
    raise ValueError(kind)


@lru_cache(maxsize=None)
def sized(stage: str) -> tuple[dict, ...]:
    """Rows of a self-sizing stage, re-judged against fstrong, with ratio, gap, excess and events."""
    out = []
    for r in map(A.rejudge, A.load(stage)):
        n = float(r["n_final"])
        out.append({**r, "label": A._label(r), "ratio": n / float(r["n95_central"]),
                    "gap": float(r["served_pct"]) - float(r["served_central_at_n_pct"]),
                    "excess": n - float(r["n_central_for_served"]),
                    "events": float(r["lands"]) + float(r["calls"])})
    return tuple(out)


def pick(rows, **kw) -> list[dict]:
    """Rows whose fields equal kw (a tuple value means any of)."""
    def ok(r):
        return all(r[k] in v if isinstance(v, tuple) else r[k] == v for k, v in kw.items())
    return [r for r in rows if ok(r)]


def gap_stats(rows: list[dict]) -> dict:
    g = np.array([r["gap"] for r in rows])
    lo, hi = A.boot_ci(rows, g)
    return {"n": g.size, "mean": g.mean(), "lo": lo, "hi": hi, "median": float(np.median(g)), "worst": g.min(),
            "within1": 100 * np.mean(g >= -1.0), "conv": 100 * np.mean([int(r["converged"]) for r in rows])}


def size_stats(rows: list[dict]) -> dict:
    f = lambda k: np.array([float(r[k]) for r in rows])
    served, ratio = f("served_pct"), f("ratio")
    (slo, shi), (rlo, rhi) = A.boot_ci(rows, served), A.boot_ci(rows, ratio)
    return {"n": len(rows), "served": served.mean(), "served_lo": slo, "served_hi": shi, "ratio": ratio.mean(),
            "ratio_lo": rlo, "ratio_hi": rhi, "gap": np.nanmean(f("gap")), "excess": np.nanmean(f("excess")),
            "events": f("events").mean(), "lands": f("lands").mean(), "calls": f("calls").mean(),
            "conv": 100 * f("converged").mean()}


def swing(rows: list[dict]) -> np.ndarray:
    """Served-share swing (max - min, pp) over the second half of each run."""
    out = []
    for r in rows:
        h = np.array(json.loads(r["history"]), dtype=float)
        tail = h[len(h) // 2:]
        out.append(tail.max() - tail.min())
    return np.array(out)


@lru_cache(maxsize=None)
def brcheck_runs() -> tuple[tuple, ...]:
    """(pointing, utility, scenario, converged, epsilon) per run; epsilon = the largest gain any one agent could
    still take (grid or heuristic), pp of all demand."""
    runs = defaultdict(list)
    for r in A.load("brcheck"):
        runs[("xyzt" if int(r["pointing"]) else "nadir", A._label(r), r["scenario"], r["snapshot"], r["seed"],
              r["n_drones"])].append(r)
    return tuple((pt, u, sc, rs[0]["run_converged"] == "1",
                  max(max(float(x["brute_gain_pp"]), float(x["heuristic_gain_pp"]), 0.0) for x in rs))
                 for (pt, u, sc, *_), rs in runs.items())


@lru_cache(maxsize=None)
def brcheck() -> dict:
    """epsilon per run by (pointing, utility): converged runs and runs that never settled."""
    out = defaultdict(lambda: {"conv": [], "open": []})
    for pt, u, _, conv, eps in brcheck_runs():
        out[(pt, u)]["conv" if conv else "open"].append(eps)
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in out.items()}


@lru_cache(maxsize=None)
def _physics(scenario: str = "urban_broad", tier: str = "voice"):
    return _problem({"scenario": scenario, "tier": tier, "snapshot": None, "seed": BASE_SEED})[2]


def inr_db(z: float, tilt: float, d: float) -> float:
    """Interference-to-noise ratio [dB] at a ground user d m from one drone at altitude z, beam tilted at it."""
    phys = _physics()
    bore = joint.boresight(np.array([[math.tan(math.radians(tilt)), 0.0]]))
    p = joint.rx_power(phys, np.array([[0.0, 0.0, z]]), bore, np.array([[d, 0.0, 0.0]]))[0, 0]
    return 10.0 * math.log10(p / phys.noise_w)


def coupling_radius(z: float, tilt: float, floor_db: float = 0.0) -> float:
    """Largest ground distance [m] at which one drone's beam is still floor_db above noise (1 m grid)."""
    d = np.arange(1.0, 6001.0)
    v = np.array([inr_db(z, tilt, x) for x in d])
    above = np.nonzero(v >= floor_db)[0]
    return float(d[above[-1]]) if above.size else 0.0


@lru_cache(maxsize=None)
def llm() -> tuple[dict, ...]:
    rows = []
    for st in ("l1", "l1log", "l1gate", "l1base", "l1pol"):
        for r in analyze_llm.load(st):
            rows.append({**r, "change": float(r["served_pct"]) - float(r["served_start_pct"])})
    return tuple(rows)


def llm_change(model: str, framing: str, stage: str = "l1") -> np.ndarray:
    st = (stage,) if model.startswith("qwen") else ("l1base", "l1pol")
    return np.array([r["change"] for r in llm()
                     if r["model"] == model and r["framing"] == framing and r["stage"] in st])


@lru_cache(maxsize=None)
def llm_turns(stage: str, model: str) -> dict:
    ts = analyze_llm._turns(stage, model)
    moves = [t for t in ts if t["action"] == "move_to"]
    return {"turns": len(ts), "move": 100 * len(moves) / len(ts),
            "checked": 100 * np.mean([analyze_llm._checked(t) for t in moves]) if moves else float("nan"),
            "what_ifs": np.mean([t["what_ifs"] for t in ts]), "calls": np.mean([t["calls"] for t in ts]),
            "errors": 100 * np.mean([bool(t["error"]) for t in ts]),
            "max_calls": 100 * np.mean([str(t["error"]).startswith("max_calls") for t in ts]),
            "prompt": np.mean([t["prompt_tokens"] for t in ts]),
            "completion": np.mean([t["completion_tokens"] for t in ts]), "seconds": np.mean([t["seconds"] for t in ts])}


@lru_cache(maxsize=None)
def llm_moves(stage: str, model: str, framing: str) -> tuple[dict, ...]:
    """Every move_to turn of runs that log tool results: the goal change its own what_if predicted for the pose
    (None if it never queried it), whether that was the best what_if of the turn, and for moves applied that
    round the realised change of the goal (people; cell for selfish, area otherwise)."""
    key, i = analyze_llm._goal(framing)
    out = []
    for p in sorted((analyze_llm.OUT / stage / model).glob(f"*_{framing}_r*.jsonl")):
        for line in open(p):
            for t in json.loads(line)["turns"]:
                if t["action"] != "move_to":
                    continue
                tr = t["trace"] or []
                wif = [c for c in tr if c["tool"] == "what_if" and key in (c.get("result") or {})]
                d = next((c["result"][key] for c in reversed(wif) if analyze_llm._same(c["args"], tr[-1]["args"])),
                         None) if analyze_llm._checked(t) else None
                applied = t["applied"] == "move" and bool(t["people_after"])
                out.append({"pred": d, "best": d is not None and d >= max(c["result"][key] for c in wif),
                            "applied": applied,
                            "real": t["people_after"][i] - t["people_before"][i] if applied else None})
    return tuple(out)
