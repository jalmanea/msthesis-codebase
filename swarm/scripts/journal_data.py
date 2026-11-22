"""Shared tables for the journal paper's figures and number check.

Builds (and caches) the per-N curves of every fleet variant from
results/fleet/<variant>/<scenario>/<tier>/*.csv, plus results/fleet_ext (N = 350-500
for the edge cells, run_fleet.py --extend) where present, means over seeds
(synthetic) or snapshots x seeds (Milan), and derives fleet sizes from them as
scripts/analyze_fleet.py does. Bootstrap intervals of a fleet size resample the
independent units: seeds for synthetic layouts, snapshots for Milan tiles. WP1
curves come from results/wp1/summary_*.csv (scripts/analyze_wp1.py).

python scripts/journal_data.py                  # (re)build results/fleet/curves_all.csv
python scripts/journal_data.py --joint-windows  # configs/joint_windows.json from results/joint/coarse
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_fleet import FLEET, curves, fleet_size, load  # noqa: E402

CACHE = FLEET / "curves_all.csv"
EXT = FLEET.parent / "fleet_ext"
VARIANTS = ["grounded", "waiver300", "cap40", "lowincome", "suburban", "dense_urban", "bw20",
            "tether", "bh200", "bh_shared"]
TIERS = ["text", "voice", "video", "broadband"]
MILAN = ["milan_p100", "milan_p090", "milan_p075", "milan_p050", "milan_p025"]
SYNTH = ["shelter_hotspot", "shelter_clustered", "urban_broad", "rural_broad"]
TOPOS = ["mst_corner", "mst_central", "balanced"]
TIER_RATE = {"text": 3e3, "voice": 38e3, "video": 384e3, "broadband": 2e6}


def fleet_curves(rebuild: bool = False) -> pd.DataFrame:
    """Mean curves by variant, scenario, tier, method and N (cached)."""
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE)
    cur = curves(_all_runs(VARIANTS))
    cur.to_csv(CACHE, index=False)
    return cur


def _all_runs(variants: list[str]) -> pd.DataFrame:
    df = load(variants)
    ext = [f for v in variants for f in (EXT / v).rglob("*.csv")]
    if ext:
        df = pd.concat([df] + [pd.read_csv(f) for f in ext], ignore_index=True)
    return df


def runs(variant: str, rebuild: bool = False) -> pd.DataFrame:
    """Per-run rows of one variant (cached), with the bootstrap unit of each run."""
    cache = FLEET / f"runs_{variant}.pkl"
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    df = _all_runs([variant])
    keep = ["scenario", "tier", "seed", "layout", "n_drones", "method", "served_pct"] + [
        c for c in df.columns if c.startswith(("served_e2e_pct_", "relays_"))]
    df = df[keep].copy()
    df["unit"] = np.where(df["scenario"].str.startswith("milan"), df["layout"], df["seed"].astype(str))
    df.to_pickle(cache)
    return df


def bootstrap_nstar(r: pd.DataFrame, scenario: str, tier: str, metric: str = "served_pct",
                    method: str = "greedy", target: float = 95.0, n_boot: int = 2000,
                    seed: int = 0) -> tuple[float, float, float]:
    """(N*, 2.5 %, 97.5 % bootstrap percentiles of N*); inf where the target is not reached."""
    d = r[(r["scenario"] == scenario) & (r["tier"] == tier) & (r["method"] == method)]
    m = d.pivot_table(index="unit", columns="n_drones", values=metric, aggfunc="mean")   # units x N
    ns = m.columns.to_numpy()
    x = m.to_numpy()

    def first(curves: np.ndarray) -> np.ndarray:
        ok = curves >= target
        return np.where(ok.any(axis=1), ns[np.argmax(ok, axis=1)], np.inf)
    point = first(x.mean(axis=0, keepdims=True))[0]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(x.shape[0], np.full(x.shape[0], 1.0 / x.shape[0]), size=n_boot) / x.shape[0]
    boot = first(w @ x)
    lo, hi = np.percentile(boot, [2.5, 97.5], method="inverted_cdf")
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------- refinement windows
REFINE_WINDOWS = ROOT / "configs" / "refine_windows.json"
REFINE_SENSITIVITY = [            # (variant, scenario, tier) quoted in Table V or the sensitivity text
    *[(v, sc, t) for v in ("waiver300", "bw20", "lowincome", "cap40", "suburban", "dense_urban")
      for sc, t in (("urban_broad", "voice"), ("milan_p100", "voice"), ("milan_p075", "video"),
                    ("milan_p025", "broadband"))],
    ("waiver300", "milan_p025", "voice"), ("waiver300", "rural_broad", "broadband"),
    ("waiver300", "milan_p100", "video"), ("bw20", "milan_p100", "video"),
    ("bw20", "milan_p050", "broadband"), ("cap40", "milan_p075", "voice"),
    ("lowincome", "milan_p100", "video"),
]


def dense_step(n: int) -> int:
    return 1 if n <= 40 else 2 if n <= 100 else 5


def refine_windows(cur: pd.DataFrame) -> list[dict]:
    """Dense N windows around every coarse N* the paper quotes: two grid values either side of N*.

    Table IV cells run as variant tether (its access side is the grounded model's, and it adds the tethered
    end-to-end shares); satellite end-to-end cells as grounded; sensitivity cells as their variant.
    """
    s = sizes(cur)
    grid = sorted(cur["n_drones"].unique())
    jobs = {}
    def add(v, sc, t, n):
        i = grid.index(int(n))
        lo, hi = grid[max(i - 2, 0)], grid[min(i + 2, len(grid) - 1)]
        ns = list(range(lo, hi + 1, 1))
        ns = [x for x in ns if (x - lo) % dense_step(x) == 0 or x in (lo, hi)]
        key = (v, sc, t)
        jobs[key] = sorted(set(jobs.get(key, [])) | set(ns))
    for sc in SYNTH + MILAN:
        for t in TIERS:
            n = s.loc[("grounded", sc, t, "greedy"), "n95"]
            if not pd.isna(n):
                add("tether", sc, t, n)
            e = sizes(cur, metric="served_e2e_pct_balanced").loc[("grounded", sc, t, "greedy"), "n95"]
            if t == "voice" and not pd.isna(e):
                add("grounded", sc, t, e)
    for v, sc, t in REFINE_SENSITIVITY:
        n = s.loc[(v, sc, t, "greedy"), "n95"]
        if not pd.isna(n):
            add(v, sc, t, n)
    return [{"variant": v, "scenario": sc, "tier": t, "fleet": ns} for (v, sc, t), ns in sorted(jobs.items())]


REFINE = FLEET.parent / "fleet_refine"


def refine_runs(rebuild: bool = False) -> pd.DataFrame:
    """Per-run rows of the refinement run (results/fleet_refine, run_fleet.py --refine), cached."""
    cache = REFINE / "runs.pkl"
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    files = list(REFINE.rglob("*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    keep = ["variant", "scenario", "tier", "seed", "layout", "n_drones", "method", "served_pct"] + [
        c for c in df.columns if c.startswith(("served_e2e_pct_", "relays_"))]
    df = df[keep].copy()
    df["unit"] = np.where(df["scenario"].str.startswith("milan"), df["layout"], df["seed"].astype(str))
    df.to_pickle(cache)
    return df


def refined_nstar(rr: pd.DataFrame, variant: str, scenario: str, tier: str, metric: str = "served_pct",
                  method: str = "greedy", target: float = 95.0) -> dict:
    """N* on the dense window with its bootstrap interval, the relays at N*, and window edges.

    'below' is True when the window's smallest N already reaches the target (N* <= window start),
    'above' when no N in the window does; both would mean the window missed N*.
    """
    r = rr[rr["variant"] == variant]
    n, lo, hi = bootstrap_nstar(r, scenario, tier, metric, method, target)
    d = r[(r["scenario"] == scenario) & (r["tier"] == tier) & (r["method"] == method)]
    mean = d.groupby("n_drones")[metric].mean()
    out = {"n": n, "lo": lo, "hi": hi, "n_min": int(mean.index.min()), "n_max": int(mean.index.max()),
           "below": bool(mean.iloc[0] >= target), "above": bool(mean.max() < target),
           "units": int(d["unit"].nunique()), "runs_per_n": int(len(d) / mean.size)}
    if metric.startswith("served_e2e_pct_") and np.isfinite(n):
        out["relays"] = float(d[d["n_drones"] == n][metric.replace("served_e2e_pct_", "relays_")].mean())
    return out


def fleet_cell(cur: pd.DataFrame, rr: pd.DataFrame | None, variant: str, scenario: str, tier: str,
               metric: str = "served_pct") -> dict:
    """The fleet size the paper reports for one cell: refined (dense window, more seeds) where a window
    exists, otherwise the coarse grid. Keys: n (inf if not reached), lo, hi, relays, best, n_max, source.

    The access side of the grounded model is read from the tether variant's refinement (identical
    access model; the tether run adds its end-to-end shares).
    """
    rv = "tether" if (variant == "grounded" and metric == "served_pct") else variant
    co = sizes(cur, metric=metric).loc[(variant, scenario, tier, "greedy")]
    coarse_n = co["n95"] if not pd.isna(co["n95"]) else np.inf
    nmax = int(cur[(cur["variant"] == variant) & (cur["scenario"] == scenario)
                   & (cur["tier"] == tier)]["n_drones"].max())
    base = {"best": float(co["best_served_pct"]), "n_max": nmax}
    has = rr is not None and ((rr["variant"] == rv) & (rr["scenario"] == scenario) & (rr["tier"] == tier)).any()
    if has:
        info = refined_nstar(rr, rv, scenario, tier, metric)
        if info["below"]:
            raise ValueError(f"refine window starts above N* for {rv} {scenario} {tier} {metric}")
        if info["above"]:
            # an end-to-end N* can lie beyond the access window; then the coarse grid has it
            if np.isfinite(coarse_n) and coarse_n <= info["n_max"]:
                raise ValueError(f"refine window ends below N* for {rv} {scenario} {tier} {metric}")
        else:
            return {**base, "n": info["n"], "lo": info["lo"], "hi": info["hi"],
                    "relays": info.get("relays"), "source": "refined"}
    rel = co.get("relays95") if metric.startswith("served_e2e_pct_") else None
    return {**base, "n": coarse_n, "lo": coarse_n, "hi": coarse_n,
            "relays": None if rel is None or pd.isna(rel) else float(rel), "source": "coarse"}


def lower_bound_drones(scenario: str, tier: str, variant: str = "grounded") -> int:
    """Fewest drones any deployment could use for the grounded demand: GBR admission fills at most one
    band per drone, so a drone carries at most B x se_max (the variant sets B)."""
    sys.path.insert(0, str(ROOT))
    from run_fleet import VARIANTS as RUN_VARIANTS
    cfg = RUN_VARIANTS[variant]()
    per_drone = cfg.network.bandwidth_hz * cfg.capacity.se_max_bps_hz
    return int(np.ceil(demand().loc[scenario, "active"] * TIER_RATE[tier] / per_drone - 1e-9))


def gateway_bound_bps(variant: str = "tether") -> float:
    """Most a single gateway can deliver under any tree [bit/s]: its own cell's GBR load is at most
    B x se_max, and its I backhaul interfaces carry at most I x fixed link rate (time-shared)."""
    sys.path.insert(0, str(ROOT))
    from run_fleet import VARIANTS as RUN_VARIANTS
    cfg = RUN_VARIANTS[variant]()
    bh = cfg.backhaul
    tree = bh.interfaces * bh.fixed_link_bps + cfg.network.bandwidth_hz * cfg.capacity.se_max_bps_hz
    return tree if bh.gateway_bps is None else min(tree, bh.gateway_bps)


def demand() -> pd.DataFrame:
    """People and active users per scenario, from the cell files (one row each)."""
    rows = []
    for scen in SYNTH + MILAN:
        f = next((FLEET / "grounded" / scen / "voice").glob("*.csv"))
        r = pd.read_csv(f, nrows=1).iloc[0]
        rows.append({"scenario": scen, "population": r["population"],
                     "active": r["active_per_point"] * r["n_points"]})
    return pd.DataFrame(rows).set_index("scenario")


def sizes(cur: pd.DataFrame, metric: str = "served_pct", target: float = 95.0) -> pd.DataFrame:
    """Smallest N reaching the target, indexed by (variant, scenario, tier, method)."""
    fs = fleet_size(cur, targets=(target,), metric=metric)
    return fs.set_index(["variant", "scenario", "tier", "method"])


def wp1(grid: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "wp1" / f"summary_{grid}.csv")


# --------------------------------------------------------------- joint placement (run_joint.py)
JOINT = FLEET.parent / "joint"
JOINT_WINDOWS = ROOT / "configs" / "joint_windows.json"
JOINT_METHODS = ["greedy", "greedy_free", "xyz_nearest", "xyz", "xyzt"]


def joint_runs(stages: tuple[str, ...] = ("coarse", "dense")) -> pd.DataFrame:
    """Per-unit rows of run_joint.py (one per method), with the bootstrap unit: snapshot or seed."""
    files = [f for st in stages for f in (JOINT / st).rglob("*.csv")] if JOINT.exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files if f.parent.name != "brute"), ignore_index=True)
    df["unit"] = np.where(df["scenario"].str.startswith("milan"),
                          "snap" + df["snapshot"].fillna(-1).astype(int).astype(str),
                          df["seed"].astype(int).astype(str))
    return df


def joint_nstar(r: pd.DataFrame, scenario: str, tier: str, method: str, target: float = 95.0) -> dict:
    """N* of one method on the joint runs (mean over units), its bootstrap interval, and window edges."""
    n, lo, hi = bootstrap_nstar(r, scenario, tier, "served_pct", method, target)
    d = r[(r["scenario"] == scenario) & (r["tier"] == tier) & (r["method"] == method)]
    mean = d.groupby("n_drones")["served_pct"].mean()
    return {"n": n, "lo": lo, "hi": hi, "below": bool(mean.iloc[0] >= target),
            "above": bool(mean.max() < target), "units": int(d["unit"].nunique())}


def joint_windows(r: pd.DataFrame) -> list[dict]:
    """Dense N windows from the coarse joint runs: every N (dense_step) from the coarse value below each
    method's coarse N* up to it, merged over methods."""
    out = []
    for (sc, t), d in r.groupby(["scenario", "tier"]):
        grid = sorted(d["n_drones"].unique())
        ns = set()
        for m in JOINT_METHODS:
            mean = d[d["method"] == m].groupby("n_drones")["served_pct"].mean()
            hit = mean.index[mean >= 95.0]
            if len(hit) == 0 or hit.min() == grid[0]:
                continue
            n = int(hit.min())
            lo = grid[grid.index(n) - 1]
            ns |= {x for x in range(lo + 1, n) if (x - lo) % dense_step(x) == 0}
        ns -= set(int(x) for x in grid)
        if ns:
            out.append({"scenario": sc, "tier": t, "fleet": sorted(ns)})
    return out


if __name__ == "__main__":
    import json
    if sys.argv[1:] == ["--joint-windows"]:              # after pulling results/joint/coarse
        JOINT_WINDOWS.write_text(json.dumps(joint_windows(joint_runs(("coarse",))), indent=1))
        print(f"-> {JOINT_WINDOWS}")
        sys.exit(0)
    c = fleet_curves(rebuild=True)
    REFINE_WINDOWS.write_text(json.dumps(refine_windows(c), indent=1))
    print(f"-> {REFINE_WINDOWS}")
    print(f"{len(c)} rows, variants {sorted(c['variant'].unique())} -> {CACHE}")
    for v in ("grounded", "tether"):
        print(f"{v}: {len(runs(v, rebuild=True))} runs")
