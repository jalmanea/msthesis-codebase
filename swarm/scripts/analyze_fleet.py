"""Reading of the fleet-sizing runs: how many drones each scenario and service tier needs.

Reads results/fleet/<variant>/<scenario>/<tier>/*.csv (scripts/run_fleet.py), averages the
served share over seeds (synthetic) or snapshots x seeds (Milan) at each fleet size, and
reports per variant, scenario, tier and method the smallest N whose mean served share
reaches 90 % and 95 % (">300" when none does, with the best mean share). With an
end-to-end metric of one backhaul topology (served_e2e_pct_<topology>) the fleet is
reported as access drones + mean relay drones at that N. Writes
results/fleet/fleet_size*.csv and results/fleet/curves.csv.

python scripts/analyze_fleet.py
python scripts/analyze_fleet.py --variants grounded waiver300 --target 95
python scripts/analyze_fleet.py --variants tether --metric served_e2e_pct_balanced
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FLEET = ROOT / "results" / "fleet"
SCEN_ORDER = ["rural_broad", "shelter_hotspot", "shelter_clustered", "urban_broad"]  # then Milan tiles
TIERS = ["text", "voice", "video", "broadband"]
METHODS = ["kmeans", "sweep", "greedy"]


def load(variants: list[str]) -> pd.DataFrame:
    files = [f for v in variants for f in (FLEET / v).rglob("*.csv")]
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def curves(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "scenario", "tier", "method", "n_drones"]
    g = df.groupby(keys)
    cols = [c for c in df.columns if c.startswith(("served_", "relays_"))
            or c in ("coverage_pct", "mean_altitude_m")]
    out = g[cols].mean()
    out["runs"] = g.size()
    return out.reset_index()


def fleet_size(cur: pd.DataFrame, targets=(90.0, 95.0), metric: str = "served_pct") -> pd.DataFrame:
    rows = []
    for (v, s, t, m), g in cur.groupby(["variant", "scenario", "tier", "method"]):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        relays = metric.replace("served_e2e_pct_", "relays_") if metric.startswith("served_e2e_pct_") else None
        row = {"variant": v, "scenario": s, "tier": t, "method": m, "metric": metric,
               "best_served_pct": g[metric].max(),
               "n_at_best": int(g.loc[g[metric].idxmax(), "n_drones"])}
        for tgt in targets:
            ok = g[g[metric] >= tgt]
            row[f"n{int(tgt)}"] = int(ok["n_drones"].min()) if len(ok) else None
            row[f"relays{int(tgt)}"] = (float(ok.loc[ok["n_drones"].idxmin(), relays])
                                        if len(ok) and relays else None)
        rows.append(row)
    return pd.DataFrame(rows)


def table(fs: pd.DataFrame, variant: str, target: int, metric: str = "served_pct") -> str:
    sub = fs[fs["variant"] == variant]
    head = f"{'scenario':18s} " + " ".join(f"{t:>20s}" for t in TIERS)
    lines = [f"--- {variant}: smallest N with {metric} >= {target} % (greedy / sweep / K-means; +r = relays)",
             head]
    scen = [x for x in SCEN_ORDER if x in set(sub["scenario"])] + sorted(
        (x for x in set(sub["scenario"]) if x not in SCEN_ORDER), reverse=True)
    for s in scen:
        cells = []
        for t in TIERS:
            r = sub[(sub["scenario"] == s) & (sub["tier"] == t)].set_index("method")
            if r.empty:
                cells.append("-")
                continue
            def fmt(m):
                n = r.loc[m, f"n{target}"] if m in r.index else None
                if pd.isna(n):
                    return f">300({r.loc[m, 'best_served_pct']:.0f}%)"
                rel = r.loc[m, f"relays{target}"]
                return f"{int(n)}" + (f"+{rel:.0f}r" if pd.notna(rel) else "")
            cells.append("/".join(fmt(m) for m in ["greedy", "sweep", "kmeans"]))
        lines.append(f"{s:18s} " + " ".join(f"{c:>20s}" for c in cells))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--target", type=int, default=95, choices=[90, 95])
    ap.add_argument("--metric", default="served_pct",
                    help="served_pct (access-side GBR, default), served_e2e_pct (config topology), "
                         "served_e2e_pct_<mst_corner|mst_central|balanced>, or served_rr_pct (round robin)")
    args = ap.parse_args()
    variants = args.variants or sorted(p.name for p in FLEET.iterdir() if p.is_dir())
    df = load(variants)
    cur = curves(df)
    fs = fleet_size(cur, metric=args.metric)
    runs = cur.groupby(["variant", "scenario", "tier"])["runs"].max()
    print(f"runs per (variant, scenario, tier): min {runs.min()}, max {runs.max()}")
    for v in variants:
        print("\n" + table(fs, v, args.target, args.metric))
    cur.to_csv(FLEET / "curves.csv", index=False)
    out = FLEET / ("fleet_size.csv" if args.metric == "served_pct" else f"fleet_size_{args.metric}.csv")
    fs.to_csv(out, index=False)
    print(f"\n-> {FLEET / 'curves.csv'}\n-> {out}")


if __name__ == "__main__":
    main()
