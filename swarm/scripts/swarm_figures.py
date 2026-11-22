"""Figures for the swarm paper (paper/main_swarm.tex).

  fig_s_inr     interference-to-noise ratio of one drone vs ground distance, by altitude and tilt (analytic)
  fig_s_ladder  gap to the central reference against the radius: windows with pointing and nadir, team with limited information
  fig_s_tilt    distribution of final antenna tilts by utility, spread vs clustered demand
  fig_s_size    self-sized fleets against the central served-vs-N frontier, per scenario
  fig_s_llm     served-share change of one LLM per drone by model size and framing, with baselines

Reads results/swarm and results/llm through scripts/swarm_data.py. Writes paper/figures/fig_s_*.pdf.

python scripts/swarm_figures.py
python scripts/swarm_figures.py fig_s_inr fig_s_llm
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import swarm_data as D  # noqa: E402
from dronecomm import joint  # noqa: E402

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
# Okabe-Ito hues (as in the journal figures), each with its own dash and marker for greyscale.
INK = "#333333"
SCEN_TITLE = {"urban_broad": "Urban, voice", "milan_p100": "Milan, voice", "shelter_clustered": "Shelter, video"}
SERIES = {
    "point": dict(color="#0072B2", ls="-", marker="o", label="Pointing"),
    "nadir": dict(color="#999999", ls="--", marker="s", label="Nadir only"),
    "info":  dict(color="#D55E00", ls=":", marker="^", label="Team, information radius $r$"),
}
UTIL_STYLE = {
    "global":     dict(color="#0072B2", ls="-", label="Team"),
    "window1200": dict(color="#009E73", ls="-.", label="Window 1200 m"),
    "window300":  dict(color="#E69F00", ls="--", label="Window 300 m"),
    "selfish":    dict(color="#D55E00", ls=":", label="Selfish"),
}
FRAMING = {
    "selfish": dict(color="#D55E00", ls=":", marker="v", label="Selfish framing"),
    "area":    dict(color="#009E73", ls="-.", marker="s", label="Area framing"),
    "network": dict(color="#0072B2", ls="-", marker="o", label="Network framing"),
}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def fig_s_inr() -> None:
    d = np.geomspace(50.0, 5000.0, 200)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    cases = [(50, 0, "#999999", ":", "50 m, nadir"), (120, 0, "#000000", "--", "120 m, nadir"),
             (120, 30, "#0072B2", "-", "120 m, tilt 30°"), (120, 60, "#D55E00", "-.", "120 m, tilt 60°")]
    for z, t, c, ls, lab in cases:
        ax.plot(d, [D.inr_db(z, t, x) for x in d], color=c, ls=ls, label=lab)
    ax.axhline(0.0, color=INK, lw=0.6)
    for rho in (150, 300, 600, 1200):
        ax.axvline(rho, color="#d9d9d9", lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([100, 300, 1000, 3000])
    ax.set_xticklabels(["100", "300", "1000", "3000"])
    ax.set_xlabel("Ground distance from the drone [m]")
    ax.set_ylabel("INR [dB]")
    ax.set_ylim(-25, 55)
    ax.legend(loc="upper right", ncol=1)
    save(fig, "fig_s_inr")


def fig_s_ladder() -> None:
    """Gap against the radius (window rho, or the team's information radius r) at every radius of D.RADII, per
    scenario; selfish and team play with pointing as reference lines, the 30 deg coupling radius as a guide."""
    sweeps = {k: D.sweep(k) for k in ("point", "nadir", "info")}
    radii = np.array(D.RADII, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    for ax, sc in zip(axes, D.SCENARIOS):
        for key, rows in sweeps.items():
            if key == "info":
                st = [D.gap_stats(D.pick(rows, scenario=sc, info_m=f"{r:.1f}")) for r in radii]
            else:
                st = [D.gap_stats(D.pick(rows, scenario=sc, label=f"window{r:.0f}")) for r in radii]
            m = np.array([s["mean"] for s in st])
            err = np.array([[s["mean"] - s["lo"] for s in st], [s["hi"] - s["mean"] for s in st]])
            sty = SERIES[key]
            lab = {"point": "Window, pointing", "nadir": "Window, nadir only"}.get(key, sty["label"])
            ax.errorbar(radii, m, yerr=err, color=sty["color"], ls=sty["ls"], marker=sty["marker"], capsize=1.2,
                        markersize=MS - 1, label=lab)
        for u, ls, lab in (("global", "-", "Team, pointing"), ("selfish", ":", "Selfish, pointing")):
            ax.axhline(D.gap_stats(D.pick(sweeps["point"], scenario=sc, label=u))["mean"], color=INK, ls=ls, lw=0.7,
                       label=lab)
        ax.axvline(D.coupling_radius(120, 30), color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xlim(0, 1260)
        ax.set_xticks([0, 300, 600, 900, 1200])
        ax.set_title(SCEN_TITLE[sc])
        ax.set_xlabel("Radius $\\rho$ or $r$ [m]")
        ax.grid(axis="y")
    axes[0].set_ylabel("Gap to central [pp]")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), handlelength=2.2, columnspacing=1.2)
    save(fig, "fig_s_ladder")


def _tilts(sc: tuple, u: str) -> np.ndarray:
    out = []
    for s in sc:
        for p in (D.A.OUT / "p1" / s).rglob(f"*_paper_{u}_seq.npz"):
            out.extend(joint.tilt_deg(np.load(p)["t"]))
    return np.sort(np.array(out))


def fig_s_tilt() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, (title, sc) in zip(axes, (("Urban and Milan", D.SPREAD), ("Shelter", ("shelter_clustered",)))):
        for u, sty in UTIL_STYLE.items():
            t = _tilts(sc, u)
            ax.plot(t, np.arange(1, t.size + 1) / t.size, **sty)
        ax.set_title(title)
        ax.set_xlim(0, 80)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Final tilt from nadir [°]")
        ax.grid()
    axes[0].set_ylabel("CDF over drones")
    axes[1].legend(loc="lower right")
    save(fig, "fig_s_tilt")


SIZE_MARK = [  # (stage, start, utility, rules, style)
    ("p4b", "depot1", "global", "x_land0.05_call0.25", dict(color="#999999", marker="s", label="Per cell")),
    ("p4c", "depot1", "global", "x_land0.25_call0.5_area1200", dict(color="#E69F00", marker="D", label="Area")),
    ("p4d", "depot1", "global", "x_land0.25_call0.5_area1200_chk", dict(color="#CC79A7", marker="v",
                                                                        label="Area + call check")),
    ("p4e", "depot1", "global", D.FINAL_RULE, dict(color="#0072B2", marker="o", label="Area + hysteresis")),
    ("p4e", "over", "global", D.FINAL_RULE, dict(color="#0072B2", marker="o", mfc="white",
                                                 label="Area + hyst., from $1.3N_{95}$")),
    ("p4e", "depot1", "selfish", D.FINAL_RULE, dict(color="#D55E00", marker="^", label="Area + hyst., selfish")),
]


def _frontier(sc: str, grid: np.ndarray) -> np.ndarray:
    """Mean central served share at N = ratio * N95 over the layouts of a scenario (NaN outside the frontier)."""
    curves = []
    for k, fr in D.A.FRONT.items():
        if k[0] != sc:
            continue
        ns = np.array(sorted(fr), dtype=float)
        n95 = next((n for n in sorted(fr) if fr[n] >= 95.0), max(fr))
        curves.append(np.interp(grid * n95, ns, [fr[n] for n in sorted(fr)], left=np.nan, right=np.nan))
    c = np.array(curves)
    m = np.nanmean(c, axis=0)
    m[np.mean(np.isfinite(c), axis=0) < 0.8] = np.nan          # only where most layouts reach
    return m


def fig_s_size() -> None:
    grid = np.linspace(0.1, 2.5, 97)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    for ax, sc in zip(axes, D.SCENARIOS):
        ax.plot(grid, _frontier(sc, grid), color=INK, lw=1.0, label="Central frontier")
        for st, start, u, rule, sty in SIZE_MARK:
            s = D.size_stats(D.pick(D.sized(st), scenario=sc, start=start, label=u, rules=rule))
            ax.errorbar([s["ratio"]], [s["served"]], xerr=[[s["ratio"] - s["ratio_lo"]], [s["ratio_hi"] - s["ratio"]]],
                        yerr=[[s["served"] - s["served_lo"]], [s["served_hi"] - s["served"]]], ls="none",
                        capsize=1.5, ms=MS + 0.5, **sty)
        ax.axvline(1.0, color="#d9d9d9", lw=0.6, zorder=0)
        ax.set_title(SCEN_TITLE[sc])
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("Fleet size $N/N_{95}$")
        ax.grid()
    axes[0].set_ylabel("Served share [%]")
    axes[0].set_ylim(30, 101)
    axes[0].legend(loc="lower right", fontsize=TICK_SZ - 1.5)
    save(fig, "fig_s_size")


def fig_s_llm() -> None:
    sizes = {"qwen3.5-0.8b": 0.8, "qwen3.5-2b": 2.0, "qwen3.5-4b": 4.0, "qwen3.5-9b": 9.0}
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for fr, sty in FRAMING.items():
        m = [D.llm_change(k, fr) for k in sizes]
        ax.errorbar(list(sizes.values()), [v.mean() for v in m],
                    yerr=[[v.mean() - v.min() for v in m], [v.max() - v.mean() for v in m]], capsize=1.5, **sty)
    refs = [("Best response, team", D.llm_change("br-global", "global").mean(), "#000000", "--"),
            ("Stay", 0.0, INK, "-"),
            ("Best response, selfish", D.llm_change("br-selfish", "selfish").mean(), "#999999", "--")]
    for lab, v, c, ls in refs:
        ax.axhline(v, color=c, ls=ls, lw=0.7)
        ax.text(9.6, v, lab, va="center", ha="left", fontsize=TICK_SZ - 1.5, color=c)
    ax.set_xscale("log")
    ax.set_xticks(list(sizes.values()))
    ax.set_xticklabels(["0.8", "2", "4", "9"])
    ax.set_xlim(0.7, 9.5)
    ax.set_xlabel("Model size [billion parameters]")
    ax.set_ylabel("Served-share change [pp]")
    ax.legend(loc="lower right")
    ax.grid(axis="y")
    save(fig, "fig_s_llm")


FIGS = {f.__name__: f for f in (fig_s_inr, fig_s_ladder, fig_s_tilt, fig_s_size, fig_s_llm)}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
