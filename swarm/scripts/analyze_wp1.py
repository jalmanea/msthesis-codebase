"""WP1 reading: which ICC findings survive each model variant.

Reads results/wp1/<variant>/<grid>/cell_*.csv (scripts/run_wp1.py) and prints, per
variant, the ICC paper's checkable claims (values in the ICC text in brackets) and
the fleet-size view the extension adds. Writes results/wp1/summary_<grid>.csv
(means by variant x layout x N x method) for the figures.

python scripts/analyze_wp1.py                  # every variant on disk
python scripts/analyze_wp1.py --variants icc grounded
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WP1 = ROOT / "results" / "wp1"
METHODS = ["kmeans", "analytic", "sweep", "greedy"]


def load(variant: str, grid: str) -> pd.DataFrame | None:
    files = sorted((WP1 / variant / grid).glob("cell_*.csv"))
    if not files:
        return None
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def pivot(df: pd.DataFrame, col: str = "coverage_pct") -> pd.DataFrame:
    """One row per (layout, N, M, seed), one column per method."""
    return df.pivot_table(index=["layout", "n_drones", "n_users", "seed"], columns="method",
                          values=col).reset_index()


def synthetic_claims(df: pd.DataFrame, expected_cells: int = 1500) -> list[tuple[str, str]]:
    p = pivot(df)
    cells = df["cell"].nunique()
    out = [("cells on disk", f"{cells} of {expected_cells}")]
    mean = p[METHODS].mean()
    out.append(("overall coverage K-means/analytic/sweep/greedy [50.4/76.7/80.6/83.6]",
                "/".join(f"{mean[m]:.1f}" for m in METHODS)))
    by = p.groupby("layout")[METHODS].mean()
    for lay, icc in [("clustered", "+37.6"), ("hotspot", "+42.5")]:
        if lay in by.index:
            out.append((f"analytic - K-means, {lay} [{icc}]", f"{by.loc[lay, 'analytic'] - by.loc[lay, 'kmeans']:+.1f}"))
    if "uniform" in by.index:
        out.append(("uniform K-means vs analytic [89.5 vs 88.5]",
                    f"{by.loc['uniform', 'kmeans']:.1f} vs {by.loc['uniform', 'analytic']:.1f}"))
    # sweep >= K-means and analytic at every N >= 4 on every layout (means over M and seeds)
    byn = p.groupby(["layout", "n_drones"])[METHODS].mean().reset_index()
    big = byn[byn["n_drones"] >= 4]
    lose = big[(big["sweep"] < big["kmeans"]) | (big["sweep"] < big["analytic"])]
    out.append(("sweep >= K-means and analytic at every N >= 4, all layouts [yes]",
                "yes" if lose.empty else f"no: {len(lose)} (layout, N) cells, e.g. "
                + ", ".join(f"{r.layout} N={r.n_drones}" for r in lose.head(3).itertuples())))
    gain = p["greedy"] - p["sweep"]
    out.append(("greedy - sweep, overall [+3.0]; runs improved [86%]",
                f"{gain.mean():+.1f}; {100 * np.mean(gain > 1e-9):.0f}%"))
    gl = (p.assign(g=gain).groupby("layout")["g"].mean())
    out.append(("greedy - sweep by layout clustered/hotspot/uniform [+1.4/+5.4/+2.2]",
                "/".join(f"{gl.get(l, np.nan):+.1f}" for l in ["clustered", "hotspot", "uniform"])))
    hot = byn[byn["layout"] == "hotspot"].set_index("n_drones")
    if 40 in hot.index:
        out.append(("hotspot K-means at N=40 [~0]", f"{hot.loc[40, 'kmeans']:.1f}"))
    # parking: hotspot runs with N >= 30 that put some drone at the altitude ceiling (greedy)
    g = df[(df["method"] == "greedy") & (df["layout"] == "hotspot") & (df["n_drones"] >= 30)]
    if len(g):
        ceil = df["max_altitude_m"].max()
        out.append((f"hotspot N>=30 greedy runs with a drone at the {ceil:.0f} m ceiling [79-87%]",
                    f"{100 * np.mean(g['max_altitude_m'] >= ceil - 1e-6):.0f}%"))
    return out


def milan_claims(df: pd.DataFrame, expected_cells: int = 4992) -> list[tuple[str, str]]:
    out = [("cells on disk", f"{df['cell'].nunique()} of {expected_cells}")]
    d = df[df["n_users"] == 800]
    p = pivot(d)
    snap = p.groupby(["layout", "n_drones"])[METHODS].mean().reset_index()   # seeds averaged
    mean_over_n = snap.groupby("n_drones")[METHODS].mean().mean()
    out.append(("M=800 mean over N greedy/sweep/K-means/analytic [92.6/90.7/89.1/85.3]",
                "/".join(f"{mean_over_n[m]:.1f}" for m in ["greedy", "sweep", "kmeans", "analytic"])))
    for a, b, icc in [("sweep", "kmeans", "all"), ("greedy", "sweep", "all")]:
        wins = snap[snap[a] > snap[b]]
        out.append((f"{a} > {b} on every snapshot at every N [{icc}]",
                    f"{len(wins)} of {len(snap)} (snapshot, N)"))
    byn = snap.groupby("n_drones")[METHODS].mean()
    lead = [int(n) for n in byn.index if byn.loc[n, "analytic"] > byn.loc[n, "kmeans"]]
    out.append(("N where analytic leads K-means [N <= 7]", _ranges(lead) or "none"))
    return out


def _ranges(ns: list[int]) -> str:
    if not ns:
        return ""
    parts, start = [], ns[0]
    for a, b in zip(ns, ns[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(parts)


def fleet_size(df: pd.DataFrame, target: float = 90.0, col: str = "coverage_pct") -> pd.DataFrame:
    """Smallest N whose mean (over M and seeds) reaches the target, per layout and method."""
    m = df.groupby(["layout", "n_drones", "method"])[col].mean().reset_index()
    rows = []
    for (lay, meth), g in m.groupby(["layout", "method"]):
        ok = g[g[col] >= target]["n_drones"]
        rows.append({"layout": lay, "method": meth, "n_min": int(ok.min()) if len(ok) else None})
    return pd.DataFrame(rows).pivot(index="layout", columns="method", values="n_min")[METHODS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*")
    args = ap.parse_args()
    variants = args.variants or sorted(p.name for p in WP1.iterdir() if p.is_dir())
    for grid, claims in [("synthetic", synthetic_claims), ("milan", milan_claims)]:
        summaries = []
        print(f"\n######## {grid} ########")
        for v in variants:
            df = load(v, grid)
            if df is None:
                print(f"\n== {v}: no cells")
                continue
            print(f"\n== {v}")
            for k, val in claims(df):
                print(f"  {k:78s} {val}")
            if grid == "synthetic":
                syn = df[df["layout"] != "hotspot"]
                print("  smallest N reaching 90% mean coverage (clustered/uniform), per method:")
                print("    " + fleet_size(syn).to_string().replace("\n", "\n    "))
            summaries.append(df.groupby(["variant", "layout", "n_drones", "method"])[
                ["coverage_pct", "served_pct", "rate_median_kbps", "rate_p10_kbps", "mean_altitude_m"]
            ].mean().reset_index())
        if summaries:
            path = WP1 / f"summary_{grid}.csv"
            pd.concat(summaries).to_csv(path, index=False)
            print(f"\n-> {path}")


if __name__ == "__main__":
    main()
