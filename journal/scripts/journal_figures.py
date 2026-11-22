"""Figures for the journal paper (paper/main_journal.tex).

  fig_j_wp1       coverage vs N: grounded (120 m limit) and grounded + 300 m waiver (sweep)
  fig_j_served    served share vs N per service tier (access side, greedy)
  fig_j_altitude  altitude chosen vs N under load (greedy, video tier)
  fig_j_e2e       end-to-end served share vs N by backhaul topology, with the single-gateway bound and the satellite cap
  fig_j_relays    relay drones vs N by backhaul topology
  fig_j_joint     served share vs N: greedy against joint placement and pointing (results/joint)

Reads results/wp1 (scripts/run_wp1.py) and results/fleet (scripts/run_fleet.py,
cached by scripts/journal_data.py). Writes paper/figures/fig_j_*.pdf.

python scripts/journal_figures.py
python scripts/journal_figures.py --deck   # slides/figures: fig_j_wp1, fig_j_e2e
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from journal_data import (MILAN, TIER_RATE, TIERS, demand, fleet_curves, gateway_bound_bps,  # noqa: E402
                          joint_runs, lower_bound_drones)

OUT = ROOT / "paper" / "figures"
COL_W, PAGE_W = 3.5, 7.16
FONT_SZ, TICK_SZ, LW, MS = 9, 8, 1.3, 3.5

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SZ, "axes.labelsize": FONT_SZ, "axes.titlesize": FONT_SZ,
    "xtick.labelsize": TICK_SZ, "ytick.labelsize": TICK_SZ,
    "legend.fontsize": TICK_SZ - 0.5, "lines.linewidth": LW, "lines.markersize": MS,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "grid.color": "#d9d9d9",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
})

# Okabe-Ito hues, each with its own dash and marker for greyscale.
INK = "#333333"
MODEL = {
    "grounded":  dict(color="#0072B2", ls="-",  marker="o", label="Grounded (120 m)"),
    "waiver300": dict(color="#D55E00", ls="--", marker="^", label="Grounded, 300 m waiver"),
}
SCEN = {
    "urban_broad":       dict(color="#000000", ls="--", marker="", label="Urban (TS 22.261)"),
    "shelter_clustered": dict(color="#D55E00", ls="-.", marker="", label="Shelter, clustered"),
    "rural_broad":       dict(color="#009E73", ls=":",  marker="", label="Rural (TS 22.261)"),
}
MILAN_LABEL = {"milan_p100": "busiest", "milan_p090": "90th pct", "milan_p075": "75th pct",
               "milan_p050": "50th pct", "milan_p025": "25th pct"}
# one sequential hue (blue), dark = more active tile
MILAN_SHADE = dict(zip(MILAN, ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]))
TOPO = {
    "access":      dict(color="#999999", ls="-",  marker="", label="Access only"),
    "mst_corner":  dict(color="#D55E00", ls=":",  marker="s", label="Corner MST"),
    "mst_central": dict(color="#0072B2", ls="--", marker="^", label="Central MST"),
    "balanced":    dict(color="#009E73", ls="-",  marker="o", label="Balanced tree"),
    "satellite":   dict(color="#000000", ls="-.", marker="", label="Satellite gateway"),
    "bound":       dict(color="#CC79A7", ls=(0, (4, 1, 1, 1)), marker="", label="Bound, any tree"),
}
TIER_TITLE = {"text": "Text, 3 kb/s", "voice": "Voice, 38 kb/s", "video": "Video call, 384 kb/s",
              "broadband": "Broadband, 2 Mb/s"}
N_TICKS = [1, 3, 10, 30, 100, 300]
# fig_j_served: x range per tier. Text and voice end where every curve has levelled off (text >= 99.9% at
# N = 10, voice >= 99% at N = 25). Video and broadband end at 60: where one tethered gateway binds, the access
# side reaches its ceiling by about 60 drones; larger fleets (up to 270 for video) are given in tab:fleet.
SERVED_X = {"text": (10, [1, 2, 5, 10]), "voice": (25, [1, 2, 5, 10, 25]),
            "video": (60, [1, 3, 10, 30, 60]), "broadband": (60, [1, 3, 10, 30, 60])}
JOINT_STYLE = {
    "greedy":      dict(color="#999999", ls="-",  marker="o", label="Greedy (K-means)"),
    "xyz_nearest": dict(color="#0072B2", ls="--", marker="^", label="Positions, nearest"),
    "xyz":         dict(color="#009E73", ls=":",  marker="v", label="Positions, free association"),
    "xyzt":        dict(color="#D55E00", ls="-",  marker="s", label="Positions and pointing"),
}
JOINT_CELLS = [("urban_broad", "voice", "Urban, voice"), ("milan_p100", "voice", "Milan busiest, voice"),
               ("shelter_clustered", "video", "Shelter clustered, video"),
               ("milan_p075", "video", "Milan 75th pct, video"), ("milan_p100", "video", "Milan busiest, video"),
               ("milan_p025", "broadband", "Milan 25th pct, broadband")]


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> {OUT / name}.pdf")


def _log_n(ax, xmax: float = 300, ticks: list[int] = N_TICKS) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(1, xmax)
    ax.grid(True, which="major")


def _target(ax, y: float = 95.0) -> None:
    ax.axhline(y, color="#777777", lw=0.7, ls=(0, (1, 2)), zorder=1)


# --------------------------------------------------------------------- WP1
def _wp1_milan_m800() -> pd.DataFrame:
    """Milan WP1 coverage at M = 800, sweep."""
    cache = ROOT / "results" / "wp1" / "milan_m800_sweep.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for v in MODEL:
        files = sorted((ROOT / "results" / "wp1" / v / "milan").glob("cell_*.csv"))
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        d = df[(df["n_users"] == 800) & (df["method"] == "sweep")]
        snap = d.groupby(["layout", "n_drones"])["coverage_pct"].mean().reset_index()
        g = snap.groupby("n_drones")["coverage_pct"].agg(["mean", "std"]).reset_index()
        rows.append(g.assign(variant=v))
    out = pd.concat(rows)
    out.to_csv(cache, index=False)
    return out


def fig_wp1() -> None:
    syn = pd.read_csv(ROOT / "results" / "wp1" / "summary_synthetic.csv")
    syn = syn[syn["method"] == "sweep"]
    mil = _wp1_milan_m800()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.0), sharey=True)
    for ax, lay, title in zip(axes[:2], ["clustered", "uniform"], ["(a) Clustered", "(b) Uniform"]):
        for v, st in MODEL.items():
            d = syn[(syn["variant"] == v) & (syn["layout"] == lay)].sort_values("n_drones")
            ax.plot(d["n_drones"], d["coverage_pct"], color=st["color"], ls=st["ls"], label=st["label"])
        ax.set_title(title)
        ax.set_xlim(1, 50)
    ax = axes[2]
    for v, st in MODEL.items():
        d = mil[mil["variant"] == v].sort_values("n_drones")
        ax.plot(d["n_drones"], d["mean"], color=st["color"], ls=st["ls"], label=st["label"])
        ax.fill_between(d["n_drones"], d["mean"] - d["std"], d["mean"] + d["std"], color=st["color"],
                        alpha=0.12, lw=0)
    ax.set_title("(c) Milan, $M=800$")
    ax.set_xlim(5, 30)
    for ax in axes:
        ax.axhline(90, color="#777777", lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.set_xlabel("Fleet size $N$")
        ax.grid(True)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(0, 102)
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_j_wp1")


# --------------------------------------------------------------------- service
def fig_served(cur: pd.DataFrame) -> None:
    g = cur[(cur["variant"] == "grounded") & (cur["method"] == "greedy")]
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 2.15), sharey=True)
    for ax, tier in zip(axes, TIERS):
        t = g[g["tier"] == tier]
        piv = t.pivot(index="n_drones", columns="scenario", values="served_pct")
        ax.fill_between(piv.index, piv[MILAN].min(axis=1), piv[MILAN].max(axis=1),
                        color=MILAN_SHADE["milan_p050"], alpha=0.15, lw=0)
        for s in MILAN:
            ax.plot(piv.index, piv[s], color=MILAN_SHADE[s], lw=1.0,
                    label=f"Milan {MILAN_LABEL[s]}")
        for s, st in SCEN.items():
            ax.plot(piv.index, piv[s], color=st["color"], ls=st["ls"], lw=1.1, label=st["label"])
        _target(ax)
        _log_n(ax, *SERVED_X[tier])
        ax.set_title(TIER_TITLE[tier])
        ax.set_xlabel("Access drones $N$")
    axes[0].set_ylabel("Served share (%)")
    axes[0].set_ylim(0, 102)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), columnspacing=1.2)
    fig.tight_layout(w_pad=0.5)
    _save(fig, "fig_j_served")


def fig_altitude(cur: pd.DataFrame) -> None:
    g = cur[(cur["method"] == "greedy") & (cur["tier"] == "video")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    scen = {"milan_p100": dict(color=MILAN_SHADE["milan_p100"], label="Milan busiest"),
            "urban_broad": dict(color=SCEN["urban_broad"]["color"], label="Urban"),
            "shelter_clustered": dict(color=SCEN["shelter_clustered"]["color"], label="Shelter, clustered")}
    for s, st in scen.items():
        for v, ls in [("grounded", "-"), ("waiver300", "--")]:
            d = g[(g["variant"] == v) & (g["scenario"] == s)].sort_values("n_drones")
            ax.plot(d["n_drones"], d["mean_altitude_m"], color=st["color"], ls=ls,
                    label=st["label"] if v == "grounded" else None)
    ax.axhline(120, color="#777777", lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.text(290, 124, "120 m limit", ha="right", fontsize=TICK_SZ - 1, color="#555555", va="bottom")
    _log_n(ax)
    ax.set_ylim(40, 310)
    ax.set_xlabel("Access drones $N$")
    ax.set_ylabel("Mean altitude (m)")
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK, ls="-"), plt.Line2D([], [], color=INK, ls="--")]
    lab += ["120 m cap", "300 m waiver"]
    ax.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.25), columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig_j_altitude")


def fig_e2e(cur: pd.DataFrame) -> None:
    panels = [("milan_p100", "voice", "(a) Milan busiest, voice"),
              ("milan_p075", "video", "(b) Milan 75th pct, video call"),
              ("urban_broad", "video", "(c) Urban, video call")]
    g = cur[cur["method"] == "greedy"]
    dem = demand()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.15), sharey=True)
    for ax, (s, tier, title) in zip(axes, panels):
        teth = g[(g["variant"] == "tether") & (g["scenario"] == s) & (g["tier"] == tier)].sort_values("n_drones")
        sat = g[(g["variant"] == "satellite") & (g["scenario"] == s) & (g["tier"] == tier)].sort_values("n_drones")
        st = TOPO["access"]
        ax.plot(teth["n_drones"], teth["served_pct"], color=st["color"], ls=st["ls"], lw=2.2, alpha=0.6,
                label=st["label"])
        for topo in ["mst_corner", "mst_central", "balanced"]:
            st = TOPO[topo]
            ax.plot(teth["n_drones"], teth[f"served_e2e_pct_{topo}"], color=st["color"], ls=st["ls"],
                    label=f"Tether, {st['label']}")
        st = TOPO["bound"]
        cap = gateway_bound_bps("tether", tier)
        bound = np.minimum(teth["served_pct"], 100.0 * cap / (dem.loc[s, "active"] * TIER_RATE[tier]))
        ax.plot(teth["n_drones"], bound, color=st["color"], ls=st["ls"], lw=1.0, label="Tether, bound for any tree")
        st = TOPO["satellite"]
        ax.plot(sat["n_drones"], sat["served_e2e_pct_balanced"], color=st["color"], ls=st["ls"],
                label="Satellite, 20 Mb/s up, any tree")
        _target(ax)
        _log_n(ax)
        ax.set_title(title)
        ax.set_xlabel("Access drones $N$")
    axes[0].set_ylabel("End-to-end served (%)")
    axes[0].set_ylim(0, 102)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    fig.tight_layout(w_pad=0.5)
    _save(fig, "fig_j_e2e")


def fig_relays(cur: pd.DataFrame) -> None:
    """Relay drones the tree needs (tether, voice: every topology delivers all traffic)."""
    g = cur[(cur["variant"] == "tether") & (cur["method"] == "greedy") & (cur["tier"] == "voice")]
    panels = [("urban_broad", "(a) Urban"), ("milan_p100", "(b) Milan busiest")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, (s, title) in zip(axes, panels):
        d = g[g["scenario"] == s].sort_values("n_drones")
        ax.plot(d["n_drones"], d["n_drones"], color="#bbbbbb", lw=0.8, ls="-", label="Relays = access drones")
        # rerooting an MST keeps its edges, so corner and central MST need the same relays
        assert np.allclose(d["relays_mst_corner"], d["relays_mst_central"])
        for topo, label in [("mst_central", "MST (either root)"), ("balanced", TOPO["balanced"]["label"])]:
            st = TOPO[topo]
            ax.plot(d["n_drones"], d[f"relays_{topo}"], color=st["color"], ls=st["ls"], label=label)
        _log_n(ax)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 1, 3, 10, 30])
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1, 10, 100])
        ax.set_title(title)
        ax.set_xlabel("Access drones $N$")
    axes[0].set_ylabel("Relay drones")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), columnspacing=0.8,
               handlelength=1.6)
    fig.tight_layout(w_pad=0.4)
    _save(fig, "fig_j_relays")


# --------------------------------------------------------------------- joint placement
def fig_joint() -> None:
    r = joint_runs()
    if r.empty:
        print("  (no results/joint: fig_j_joint skipped)")
        return
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, 4.0), sharey=True)
    for ax, (sc, tier, title) in zip(axes.flat, JOINT_CELLS):
        d = r[(r["scenario"] == sc) & (r["tier"] == tier)]
        mean = d.groupby(["method", "n_drones"])["served_pct"].mean().unstack(0)
        for m, st in JOINT_STYLE.items():
            if m in mean:
                y = mean[m].dropna()                         # each method has its own N grid
                ax.plot(y.index, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                        markersize=2.5, label=st["label"])
        ax.axvline(lower_bound_drones(sc, tier), color=INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
        _target(ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 5, 10]))   # whole drones
        ax.grid(True)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel("Access drones $N$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Served share (%)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.5)
    _save(fig, "fig_j_joint")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", action="store_true",
                    help="write fig_j_wp1 and fig_j_e2e to slides/figures")
    if ap.parse_args().deck:
        global OUT
        OUT = ROOT / "slides" / "figures"
        fig_wp1()
        fig_e2e(fleet_curves())
        return
    cur = fleet_curves()
    fig_wp1()
    fig_served(cur)
    fig_altitude(cur)
    fig_e2e(cur)
    fig_relays(cur)
    fig_joint()


if __name__ == "__main__":
    main()
