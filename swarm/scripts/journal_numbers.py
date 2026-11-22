"""Check every number quoted in the journal paper (paper/main_journal.tex) against results/.

Each check recomputes a value, formats it as the paper prints it, and requires that
exact snippet in the .tex source; table rows are rebuilt whole. Exits non-zero on any
mismatch. Sources: results/wp1 (scripts/run_wp1.py), results/fleet
(scripts/run_fleet.py, curves cached by scripts/journal_data.py),
results/joint (scripts/run_joint.py, Section sec:joint),
results/telecom_milan/tiles/tiles_metadata.json.

Conventions (as in the paper)
-----------------------------
- Fleet size: smallest N on the grid whose mean served share reaches 95 %
  (journal_data.sizes, as scripts/analyze_fleet.py); greedy unless stated.
- End to end: served_e2e_pct_<topology>; satellite = variant grounded, tether =
  variant tether; relays are the mean at that N, rounded.
- WP1 synthetic: means over all runs; Milan at M = 800 with seeds averaged per
  snapshot, as scripts/analyze_wp1.py.

python scripts/journal_numbers.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analyze_wp1 import METHODS as WP1_METHODS, load as wp1_load, pivot  # noqa: E402
from journal_data import (MILAN, SYNTH, TIER_RATE, TIERS, bootstrap_nstar, demand, fleet_cell,  # noqa: E402
                          fleet_curves, gateway_bound_bps, joint_nstar, joint_runs, lower_bound_drones,
                          refine_runs, runs, sizes)

TEX = ROOT / "paper" / "main_journal.tex"


class Checker:
    def __init__(self, tex: str):
        self.tex = re.sub(r"\s+", " ", tex)
        self.n = 0
        self.bad: list[str] = []

    def snippet(self, label: str, snippet: str) -> None:
        """The paper must contain this exact (whitespace-normalised) snippet."""
        self.n += 1
        s = re.sub(r"\s+", " ", snippet)
        ok = s in self.tex
        if not ok:
            self.bad.append(f"{label}: '{s}' not in tex")
        print(f"  {'ok' if ok else 'MISSING':8s} {label:58s} {s[:90]}")

    def value(self, label: str, got: str, paper: str, snippet: str | None = None) -> None:
        """Computed and quoted values agree, and the quoted form is in the paper."""
        self.n += 1
        status = "ok"
        if got != paper:
            status = "MISMATCH"
        elif snippet is not None and re.sub(r"\s+", " ", snippet) not in self.tex:
            status = "NOT IN TEX"
        if status != "ok":
            self.bad.append(f"{label}: computed {got}, paper {paper} [{status}]")
        print(f"  {status:8s} {label:58s} {got:>10s}  (paper {paper})")


def _thou(x: float) -> str:
    return f"{x:,.0f}"


def _n(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{int(x)}"


# ---------------------------------------------------------------- demand
def check_demand(c: Checker) -> pd.DataFrame:
    print("\n== Table II: scenarios and demand traffic")
    d = demand()
    names = {"rural_broad": "Rural (TS 22.261)", "shelter_hotspot": "Shelter, hotspot",
             "shelter_clustered": "Shelter, clustered", "urban_broad": "Urban (TS 22.261)",
             "milan_p025": "Milan, 25th pct", "milan_p050": "Milan, 50th pct",
             "milan_p075": "Milan, 75th pct", "milan_p090": "Milan, 90th pct",
             "milan_p100": "Milan, hottest"}
    for s, name in names.items():
        r = d.loc[s]
        mbps = {t: r["active"] * TIER_RATE[t] / 1e6 for t in TIERS}
        text = f"{mbps['text']:.1f}" if mbps["text"] < 10 else _thou(mbps["text"])
        cells = [name, _thou(r["population"]), _thou(r["active"]), text] + [_thou(mbps[t]) for t in TIERS[1:]]
        cells += [str(lower_bound_drones(s, t)) for t in TIERS[1:]]
        if lower_bound_drones(s, "text") != 1:
            c.value(f"text lower bound {s}", str(lower_bound_drones(s, "text")), "1")
        c.snippet(f"row {s}", " & ".join(cells) + r" \\")
    meta = json.loads((ROOT / "results/telecom_milan/tiles/tiles_metadata.json").read_text())
    pops = [_thou(t["census_pop"]).replace(",", "{,}") for t in meta["tiles"]]
    c.snippet("Milan residents p100..p25", f"${pops[0]}$, ${pops[1]}$, ${pops[2]}$, ${pops[3]}$ and ${pops[4]}$ residents")
    ratio = meta["tiles"][0]["census_pop"] / meta["tiles"][-1]["census_pop"]
    c.value("resident ratio hottest / 25th pct", f"{ratio:.1f}", "9.8", "a factor of $9.8$")
    c.value("Sphere clustered people", _thou(d.loc["shelter_clustered", "population"]), "13,963", "holds $13{,}963$ people")
    c.value("Sphere hotspot people", _thou(d.loc["shelter_hotspot", "population"]), "2,094", "$\\sigma = 50$\\,m) $2{,}094$")
    c.value("hottest tile active users", _thou(d.loc["milan_p100", "active"]), "11,797", "up to $11{,}797$ active users")
    return d


# ---------------------------------------------------------------- fleet tables
def _cell(cur, rr, v, sc, t, metric="served_pct", ci=True) -> str:
    """Table entry: N* [+ relays] [interval], or '>300 (x)' / '>500 (x)'."""
    f = fleet_cell(cur, rr, v, sc, t, metric)
    if not math.isfinite(f["n"]):
        return f"\\{'neverext' if f['n_max'] > 300 else 'never'}{{{f['best']:.0f}}}"
    out = _n(f["n"]) if f["relays"] is None else f"\\plr{{{int(f['n'])}}}{{{f['relays']:.0f}}}"
    if ci and f["lo"] != f["hi"]:
        hi = _n(f["hi"]) if math.isfinite(f["hi"]) else f"$>${f['n_max']}"
        out += f"\\ci{{{_n(f['lo'])}}}{{{hi}}}"
    return out


def check_fleet_table(c: Checker, cur: pd.DataFrame) -> None:
    print("\n== Table IV: smallest fleet (access; end to end, balanced tree)")
    s = sizes(cur)
    rows = [("rural_broad", "Rural (TS 22.261)"), ("shelter_hotspot", "Shelter, hotspot"),
            ("shelter_clustered", "Shelter, clustered"), ("urban_broad", "Urban (TS 22.261)"),
            ("milan_p025", "Milan, 25th pct"), ("milan_p050", "Milan, 50th pct"),
            ("milan_p075", "Milan, 75th pct"), ("milan_p090", "Milan, 90th pct"),
            ("milan_p100", "Milan, hottest")]
    rr = refine_runs()
    for sc, name in rows:
        cells = [name] + [_cell(cur, rr, "grounded", sc, t) for t in TIERS]
        cells += [_cell(cur, rr, "grounded", sc, "voice", "served_e2e_pct_balanced"),
                  _cell(cur, rr, "tether", sc, "voice", "served_e2e_pct_balanced"),
                  _cell(cur, rr, "tether", sc, "video", "served_e2e_pct_balanced")]
        c.snippet(f"row {sc}", " & ".join(cells) + r" \\")


def check_cost_table(c: Checker, cur: pd.DataFrame) -> None:
    print("\n== Table V: cost of each change")
    s = sizes(cur)
    cols = [("urban_broad", "voice"), ("milan_p100", "voice"), ("milan_p075", "video"),
            ("milan_p025", "broadband")]
    access = [("grounded", "Grounded baseline"), ("waiver300", "$300$\\,m waiver"),
              ("bw20", "$20$\\,MHz carrier"), ("lowincome", "$53\\%$ device share"),
              ("cap40", "$40$-user cell cap"), ("suburban", "Suburban channel"),
              ("dense_urban", "Dense-urban channel")]
    rr = refine_runs()
    for v, name in access:
        c.snippet(f"access {v}", " & ".join([name] + [_cell(cur, rr, v, sc, t, ci=False) for sc, t in cols]) + r" \\")
    e2e = [("grounded", "balanced", "Satellite, any tree"), ("tether", "mst_corner", "Tether, corner MST"),
           ("tether", "mst_central", "Tether, central MST"), ("tether", "balanced", "Tether, balanced tree")]
    for v, topo, name in e2e:
        c.snippet(f"e2e {v} {topo}", " & ".join([name] + [_cell(cur, rr, v, sc, t, f"served_e2e_pct_{topo}", ci=False)
                                                          for sc, t in cols]) + r" \\")
    # "any tree": the satellite ceiling is the same for all three topologies
    for sc, t in cols:
        vals = {topo: sizes(cur, metric=f"served_e2e_pct_{topo}").loc[("grounded", sc, t, "greedy"), "best_served_pct"]
                for topo in ("mst_corner", "mst_central", "balanced")}
        c.value(f"satellite ceiling equal across trees, {sc} {t}",
                "yes" if max(vals.values()) - min(vals.values()) < 0.05 else "no", "yes")


# ---------------------------------------------------------------- fleet text
def check_fleet_text(c: Checker, cur: pd.DataFrame, d: pd.DataFrame) -> None:
    print("\n== Sec. V-VI text")
    s = sizes(cur)
    g = s.xs("greedy", level="method").loc["grounded"]
    n95 = g["n95"]
    milan = ["milan_p100", "milan_p090", "milan_p075", "milan_p050", "milan_p025"]
    syn = ["rural_broad", "shelter_hotspot", "shelter_clustered", "urban_broad"]
    allsc = syn + milan
    rr = refine_runs()

    def F(v, sc, t, m="served_pct"):           # the fleet the paper reports (refined where a window exists)
        return fleet_cell(cur, rr, v, sc, t, m)

    def N(v, sc, t, m="served_pct"):
        return F(v, sc, t, m)["n"]

    def rng(t, scs=allsc):
        v = [N("grounded", sc, t) for sc in scs]
        return f"{int(min(v))}-{int(max(v))}"
    c.value("text fleet range", rng("text"), "2-6", "Text needs $2$ to $6$ drones")
    c.value("voice fleet range", rng("voice"), "3-21", "Voice needs $3$ to $21$")
    c.value("video fleet range", rng("video"), "6-270", "video calls $6$ to $270$")
    reach = sorted(sc for sc in allsc if math.isfinite(N("grounded", sc, "broadband")))
    c.value("broadband reached", ",".join(reach), "milan_p025,milan_p050,rural_broad")
    c.value("voice -> video, hottest", f"{N('grounded', 'milan_p100', 'video') / N('grounded', 'milan_p100', 'voice'):.0f}",
            "13", "multiplies the fleet by $13$")
    c.value("Milan voice range", rng("voice", milan), "6-21", "voice needs $6$ to $21$ drones and a video call $21$ to $270$")
    c.value("Milan video range", rng("video", milan), "21-270")
    c.value("video spread across tiles (x)",
            f"{N('grounded', 'milan_p100', 'video') / N('grounded', 'milan_p025', 'video'):.0f}", "13",
            "by up to $13$ times ($21$ to $270$ drones")

    # sweep vs greedy
    fl = sorted(cur["n_drones"].unique())
    import run_fleet as rf
    c.value("fleet grid size", str(len(rf.FLEET)), "29", "($29$ values")
    c.value("extended fleet sizes", ",".join(map(str, rf.EXT_FLEET)), "350,400,500", "$N \\in \\{350, 400, 500\\}$")
    sw = s.xs("sweep", level="method").loc["grounded"]["n95"]
    odd = []
    for key in n95.index:
        a, b = n95.loc[key], sw.loc[key]
        if pd.isna(a) and pd.isna(b):
            continue
        if a == b or (not pd.isna(a) and not pd.isna(b) and fl.index(int(b)) == fl.index(int(a)) + 1):
            continue
        odd.append(f"{key[0]}/{key[1]}")
    c.value("cells where sweep needs > next size", ",".join(sorted(odd)), "milan_p025/broadband",
            "the sweep needs the same fleet or the next larger size in every (scenario, tier) cell but one")
    c.value("sweep p25 broadband", _n(sw.loc[("milan_p025", "broadband")]), "150", "($150$ instead of $100$)")
    c.value("sweep hottest video", _n(sw.loc[("milan_p100", "video")]), "350",
            "and on the busiest tile it needs $350$ drones for video, against $300$ for greedy")

    # extended cells (N = 350-500)
    import run_fleet as rf
    gg = cur[(cur["variant"] == "grounded") & (cur["method"] == "greedy")]
    ext_ok = [f"{sc}/{t}" for sc, t in sorted(rf.EXT_CELLS) if t == "broadband" and not pd.isna(n95.loc[(sc, t)])]
    c.value("extended broadband cells reaching 95 %", ",".join(ext_ok) or "none", "none")
    slope = {}
    for sc in ("urban_broad", "milan_p100", "milan_p090", "milan_p075"):
        x = gg[(gg["scenario"] == sc) & (gg["tier"] == "broadband")].set_index("n_drones")["served_pct"]
        slope[sc] = (x.loc[500] - x.loc[300]) / 4
    c.value("broadband gain per 50 drones, 300-500", f"{min(slope.values()):.1f}-{max(slope.values()):.1f}", "1.0-2.3",
            "by $1.0$ to $2.3$\\pp\\ per $50$ drones")
    pair = lambda sc: gg[(gg["scenario"] == sc) & (gg["tier"] == "broadband")].set_index("n_drones")["served_pct"]  # noqa: E731
    c.value("urban / hottest broadband 300 -> 500",
            f"{pair('urban_broad').loc[300]:.0f}->{pair('urban_broad').loc[500]:.0f}/{pair('milan_p100').loc[300]:.0f}->{pair('milan_p100').loc[500]:.0f}",
            "62->71/42->51", "(urban $62 \\to 71\\%$, busiest tile $42 \\to 51\\%$ from $300$ to $500$ drones)")
    h = pair("shelter_hotspot")
    c.value("hotspot broadband peak / at 500", f"{h.max():.0f}@{h.idxmax()}/{h.loc[500]:.0f}", "68@70/21",
            "(hotspot: $68\\%$ at $70$ drones, $21\\%$ at $500$)")
    dev = 0.0
    for sc in milan:
        for t in TIERS:
            f = F("grounded", sc, t)
            if math.isfinite(f["n"]):
                dev = max(dev, abs(f["lo"] - f["n"]) / f["n"], abs(f["hi"] - f["n"]) / f["n"])
    c.value("Milan intervals within x % of N*", f"{100 * dev:.1f}", "7.1", "every bootstrap interval lies within $8\\%$ of $N^\\star$")
    def iv(sc, t):
        f = F("grounded", sc, t)
        return f"{f['lo']:.0f}-{f['hi']:.0f}@{f['n']:.0f}"
    c.value("hottest video interval", iv("milan_p100", "video"), "255-285@270", "(busiest tile, video: $255$ to $285$ around $270$)")
    c.value("clustered voice / video interval", iv("shelter_clustered", "voice") + "/" + iv("shelter_clustered", "video"),
            "5-8@6/52-58@54", "(clustered shelter: voice $5$ to $8$ around $6$, video $52$ to $58$ around $54$)")

    # lower bound
    ratio = {t: {sc: N("grounded", sc, t) / lower_bound_drones(sc, t) for sc in allsc} for t in ("voice", "video")}
    vb = [lower_bound_drones(sc, "video") for sc in allsc if lower_bound_drones(sc, "video") >= 10]
    vr = [ratio["video"][sc] for sc in allsc if lower_bound_drones(sc, "video") >= 10]
    c.value("video bounds >= 10: range", f"{min(vb)}-{max(vb)}", "11-103", "where this bound is $11$ to $103$ drones")
    c.value("video fleet / bound there", f"{min(vr):.1f}-{max(vr):.1f}", "1.7-2.7", "are $1.7$ to $2.7$ times the bound")
    c.value("voice bound max / ratio range",
            f"{max(lower_bound_drones(sc, 'voice') for sc in allsc)}/{min(ratio['voice'].values()):.1f}-{max(ratio['voice'].values()):.0f}",
            "11/1.9-4", "for voice, with bounds of at most $11$ drones, $1.9$ to $4$ times")
    c.value("hottest broadband bound", str(lower_bound_drones("milan_p100", "broadband")), "537", "at least $537$ drones")

    # altitude under load
    gv = cur[(cur["variant"] == "grounded") & (cur["method"] == "greedy") & (cur["tier"] == "video")
             & (cur["n_drones"] == 100)].set_index("scenario")["mean_altitude_m"]
    c.value("video altitude at N=100, Milan hottest / urban",
            f"{gv['milan_p100']:.0f}-{gv['urban_broad']:.0f}", "62-67", "to $62$--$67$\\,m at $100$ drones")
    km = cur[(cur["variant"] == "grounded") & (cur["scenario"] == "milan_p100") & (cur["tier"] == "video")]
    kmm = km[km["method"] == "kmeans"].set_index("n_drones")["served_pct"]
    c.value("K-means best, hottest video", f"{kmm.max():.1f}@{kmm.idxmax()}", "61.6@200",
            "at most $61.6\\%$ of the video demand, at $200$ drones")
    grm = km[km["method"] == "greedy"].set_index("n_drones")["served_pct"]
    c.value("greedy at 300, hottest video", f"{grm.loc[300]:.1f}", "95.4", "against $95.4\\%$ for greedy at $300$")

    # sensitivities
    def nv(v, sc, t):
        return _n(N(v, sc, t)) if math.isfinite(N(v, sc, t)) else "-"
    for v, sc, t, q, snip in [
        ("grounded", "urban_broad", "voice", "16", None), ("grounded", "milan_p075", "video", "68", None),
        ("grounded", "milan_p100", "video", "270", None), ("grounded", "milan_p025", "voice", "6", None),
        ("waiver300", "urban_broad", "voice", "11", "urban voice $16 \\to 11$"),
        ("waiver300", "milan_p025", "voice", "3", "Milan $25$th-percentile voice $6 \\to 3$"),
        ("waiver300", "rural_broad", "broadband", "6", "rural broadband $12 \\to 6$"),
        ("waiver300", "milan_p075", "video", "68", "(Milan video $68$, $270$)"),
        ("waiver300", "milan_p100", "video", "270", None),
        ("bw20", "milan_p075", "video", "34", "video $68 \\to 34$"),
        ("bw20", "milan_p100", "video", "94", "hottest tile $270 \\to 94$"),
        ("bw20", "milan_p050", "broadband", "86", None),
        ("cap40", "milan_p075", "voice", "160", "tile needs $160$ drones instead of $12$"),
        ("lowincome", "milan_p075", "video", "43", "($75$th percentile $68 \\to 43$"),
        ("lowincome", "milan_p100", "video", "130", "hottest $270 \\to 130$)"),
        ("suburban", "milan_p075", "video", "90", "needs $90$ drones instead of $68$"),
        ("dense_urban", "urban_broad", "voice", "20", "(urban voice $16 \\to 20$)"),
    ]:
        c.value(f"{v} {sc} {t}", nv(v, sc, t), q, snip)
    for sc in ("milan_p100", "milan_p090"):
        c.value(f"cap40 {sc} voice not reached", nv("cap40", sc, "voice"), "-")

    # 90 % target
    s90 = sizes(cur, target=90.0).xs("greedy", level="method").loc["grounded"]["n90"]
    shrink = {sc: 1 - s90.loc[(sc, "video")] / n95.loc[(sc, "video")] for sc in allsc}
    c.value("video shrink at 90 %, max", f"{100 * max(shrink.values()):.0f}", "40", "video fleets shrink by up to $40\\%$")
    c.value("hottest / urban video at 90 %", f"{_n(s90.loc[('milan_p100', 'video')])}/{_n(s90.loc[('urban_broad', 'video')])}",
            "200/100", "(hottest tile $300 \\to 200$, urban $150 \\to 100$)")
    dv = max(n95.loc[(sc, "voice")] - s90.loc[(sc, "voice")] for sc in allsc)
    c.value("voice shrink at 90 %, max drones", f"{dv:.0f}", "3", "voice fleets by at most $3$ drones (urban $16 \\to 14$)")
    c.snippet("90 % on the coarse grid", "On the coarse grid, at $90\\%$ instead of $95\\%$")
    c.value("urban voice at 90 %", _n(s90.loc[("urban_broad", "voice")]), "14")

    # gateway
    print("\n== Sec. VI text")
    sat = sizes(cur, metric="served_e2e_pct_balanced").xs("greedy", level="method").loc["grounded"]
    ok = lambda t: ",".join(sorted(sc for sc in allsc if not pd.isna(sat.loc[(sc, t), "n95"])))  # noqa: E731
    c.value("satellite serves text", ok("text"), ",".join(sorted(allsc)))
    c.value("satellite serves voice", ok("voice"),
            "milan_p025,milan_p050,rural_broad,shelter_clustered,shelter_hotspot")
    c.value("satellite serves video", ok("video"), "rural_broad")
    c.value("satellite ceiling hottest voice", f"{sat.loc[('milan_p100', 'voice'), 'best_served_pct']:.0f}", "22",
            "$22\\%$ of the voice demand of the busiest Milan tile ($100/448$\\,Mb/s)")
    c.value("100 / hottest voice demand", f"{100 / (d.loc['milan_p100', 'active'] * 38e3 / 1e6) * 100:.0f}", "22")
    c.value("satellite ceiling urban voice", f"{sat.loc[('urban_broad', 'voice'), 'best_served_pct']:.0f}", "33",
            "and $33\\%$ of the urban voice demand")
    for topo in ("mst_corner", "mst_central", "balanced"):
        a = sizes(cur, metric=f"served_e2e_pct_{topo}")
        b = a.loc[("bh200", slice(None), slice(None), "greedy"), ["n95", "best_served_pct"]].droplevel([0, 3])
        base = a.loc[("grounded", slice(None), slice(None), "greedy"), ["n95", "best_served_pct"]].droplevel([0, 3])
        same = (b["n95"].fillna(-1).eq(base["n95"].fillna(-1)) & (b["best_served_pct"] - base["best_served_pct"]).abs().lt(0.05)).all()
        c.value(f"200 Mb/s links change nothing ({topo})", "yes" if same else "no", "yes")
    teth = {t: sizes(cur, metric=f"served_e2e_pct_{t}").xs("greedy", level="method").loc["tether"]
            for t in ("mst_corner", "mst_central", "balanced")}
    c.value("tether voice served everywhere (corner)",
            "yes" if all(not pd.isna(teth["mst_corner"].loc[(sc, "voice"), "n95"]) for sc in allsc) else "no", "yes")
    same_n = all(N("tether", sc, "voice", f"served_e2e_pct_{tp}") == N("grounded", sc, "voice")
                 for sc in allsc for tp in ("mst_corner", "balanced"))
    c.value("tether voice fleet = access fleet", "yes" if same_n else "no", "yes")
    for topo, vals, snip in [
        ("mst_corner", ("50", "29", "20"), "stalls at $50\\%$ on the $75$th-percentile Milan tile, $29\\%$ for urban demand and $20\\%$ on the busiest tile"),
        ("mst_central", ("81", "56", "34"), "lifts these ceilings to $81\\%$, $56\\%$ and $34\\%$"),
    ]:
        got = tuple(f"{teth[topo].loc[(sc, 'video'), 'best_served_pct']:.0f}" for sc in ("milan_p075", "urban_broad", "milan_p100"))
        c.value(f"tether {topo} video ceilings", "/".join(got), "/".join(vals), snip)
    cen = N("tether", "milan_p050", "video", "served_e2e_pct_mst_central")
    c.value("p50 video: central MST = access fleet", f"{cen:.0f}={N('grounded', 'milan_p050', 'video'):.0f}", "33=33",
            "it serves video with the access fleet ($33$ drones)")
    c.value("p50 video: corner MST best", f"{teth['mst_corner'].loc[('milan_p050', 'video'), 'best_served_pct']:.1f}", "95.9",
            "the corner MST only just reaches $95\\%$ (at best $95.9\\%$)")
    b = teth["balanced"]
    f = F("tether", "milan_p075", "video", "served_e2e_pct_balanced")
    c.value("balanced p75 video", f"{f['n']:.0f}+{f['relays']:.1f}", "68+0.5", "($68$ drones and about one relay)")
    got = "/".join(f"{b.loc[(sc, 'video'), 'best_served_pct']:.0f}" for sc in ("urban_broad", "milan_p100"))
    c.value("balanced video ceilings urban / hottest", got, "79/54", "$79\\%$ for urban demand and $54\\%$ on the busiest tile")

    # relays
    def nr(topo, sc, t):
        f = F("tether", sc, t, f"served_e2e_pct_{topo}")
        return f"{f['n']:.0f}+{f['relays']:.0f}"
    c.value("urban voice, balanced", nr("balanced", "urban_broad", "voice"), "16+16", "$16 + 16$ for urban demand")
    c.value("p75 voice, balanced", nr("balanced", "milan_p075", "voice"), "12+13", "$12 + 13$ on the $75$th-percentile")
    tx = [F("tether", sc, "text", "served_e2e_pct_balanced")["relays"] for sc in allsc]
    ta = [N("grounded", sc, "text") for sc in allsc]
    c.value("text relays / access range", f"{min(tx):.0f}-{max(tx):.0f}/{min(ta):.0f}-{max(ta):.0f}", "2-8/2-6",
            "needs $2$ to $8$ relays besides $2$ to $6$ access drones")
    vr = [F("tether", sc, "video", "served_e2e_pct_balanced")["relays"] for sc in ("milan_p075", "milan_p050")]
    c.value("video relays p75 / p50", f"{min(vr):.0f}-{max(vr):.0f}", "1-3", "few ($1$ to $3$")
    c.value("urban voice relays, corner", f"{F('tether', 'urban_broad', 'voice', 'served_e2e_pct_mst_corner')['relays']:.0f}",
            "13", "($16$ instead of $13$ for urban voice)")
    sh = sizes(cur, metric="served_e2e_pct_balanced").loc[("bh_shared", "milan_p100", "text", "greedy"), "best_served_pct"]
    c.value("shared channel, hottest text", f"{sh:.0f}", "67", "at most $67\\%$ of its text demand")


# ---------------------------------------------------------------- WP1
def check_wp1(c: Checker) -> None:
    print("\n== Sec. IV and Table III (WP1)")
    cols = {}
    for v in ("icc", "grounded", "waiver300"):
        df = wp1_load(v, "synthetic")
        p = pivot(df)
        mean = p[WP1_METHODS].mean()
        by = p.groupby("layout")[WP1_METHODS].mean()
        gain = (p["greedy"] - p["sweep"]).mean()
        m = df.groupby(["layout", "n_drones", "method"])["coverage_pct"].mean().reset_index()
        n90 = {}
        for lay in ("clustered", "uniform"):
            ok = m[(m["layout"] == lay) & (m["method"] == "sweep") & (m["coverage_pct"] >= 90.0)]["n_drones"]
            n90[lay] = int(ok.min())
        dm = wp1_load(v, "milan")
        dm = dm[dm["n_users"] == 800]
        snap = pivot(dm).groupby(["layout", "n_drones"])[WP1_METHODS].mean().reset_index()
        mil = snap.groupby("n_drones")[WP1_METHODS].mean()
        nm = [int(n) for n in mil.index if mil.loc[n, "sweep"] >= 90.0]
        cols[v] = {
            "km_an": f"{mean['kmeans']:.1f} / {mean['analytic']:.1f}",
            "sw_gr": f"{mean['sweep']:.1f} / {mean['greedy']:.1f}",
            "clu": f"${by.loc['clustered', 'analytic'] - by.loc['clustered', 'kmeans']:+.1f}$",
            "hot": f"${by.loc['hotspot', 'analytic'] - by.loc['hotspot', 'kmeans']:+.1f}$",
            "greedy": f"${gain:+.1f}$",
            "n90": f"{n90['clustered']} / {n90['uniform']}",
            "m_gr_sw": f"{mil['greedy'].mean():.1f} / {mil['sweep'].mean():.1f}",
            "m_km_an": f"{mil['kmeans'].mean():.1f} / {mil['analytic'].mean():.1f}",
            "wins": f"{int((snap['sweep'] > snap['kmeans']).sum())}",
            "m_n90": f"{min(nm)}" if nm else "-",
            "spread": float(mil.mean().max() - mil.mean().min()),
            "alt": float(dm[dm["method"] == "sweep"]["mean_altitude_m"].mean()),
        }
    rows = [("km_an", "Synthetic: K-means / analytic"), ("sw_gr", "\\quad sweep / greedy"),
            ("clu", "Analytic gain, clustered (pp)"), ("hot", "Analytic gain, hotspot (pp)"),
            ("greedy", "Greedy gain over sweep (pp)"), ("n90", "$N$ for $90\\%$, clustered / uniform"),
            ("m_gr_sw", "Milan: greedy / sweep"), ("m_km_an", "\\quad K-means / analytic"),
            ("wins", "Sweep beats K-means, of 624"), ("m_n90", "$N$ for $90\\%$ (sweep)")]
    for key, name in rows:
        c.snippet(f"row {key}", " & ".join([name] + [cols[v][key] for v in ("icc", "grounded", "waiver300")]) + r" \\")
    one = pd.read_csv(ROOT / "results" / "wp1" / "summary_synthetic.csv")
    one = one[(one["variant"] == "icc") & (one["n_drones"] == 1) & (one["method"] == "sweep")]
    rr_cap = 10e6 * 4.4 / 384e3
    c.value("round robin serves nobody above B se_max / r_min (video)", f"{rr_cap:.0f}", "115",
            "nobody once a cell holds more than $B\\,\\mathrm{se}_{\\max}/r_{\\min}$ active users (about $115$ for video calls)")
    c.value("conference model, one drone covers (min over layouts)", f"{one['coverage_pct'].min():.0f}", "100",
            "Under that model one drone covers $100\\%$ of the users")
    c.value("grounded Milan method spread", f"{cols['grounded']['spread']:.1f}", "0.6", "within $0.6$\\pp\\ of each other")
    c.value("grounded Milan sweep altitude", f"{cols['grounded']['alt']:.0f}", "120",
            "the mean sweep altitude on Milan is $120$\\,m")
    print("   (environment and antenna variants, Milan greedy)")
    for v, q, snip in [("highrise_urban", "41.0", "covers $41.0\\%$ in the high-rise environment"),
                       ("dense_urban", "74.0", "$74.0\\%$ in dense urban"),
                       ("suburban", "87.2", "$87.2\\%$ in suburban"),
                       ("grounded", "82.7", "against $82.7\\%$ in urban"),
                       ("element1x1", "88.9", "element covers $88.9\\%$"),
                       ("panel4x4", "77.6", "panel $77.6\\%$")]:
        dm = wp1_load(v, "milan")
        dm = dm[dm["n_users"] == 800]
        snap = pivot(dm).groupby(["layout", "n_drones"])[WP1_METHODS].mean().reset_index()
        c.value(f"Milan greedy coverage, {v}", f"{snap.groupby('n_drones')['greedy'].mean().mean():.1f}", q, snip)


# ---------------------------------------------------------------- antenna
def check_antenna(c: Checker) -> None:
    print("\n== Sec. III-B antenna")
    from antenna_panel_fit import panel_rows
    from dronecomm.antenna import ParametricAntenna

    def radiated(a: ParametricAntenna) -> float:
        th = np.linspace(0.0, np.pi, 20001)
        trap = getattr(np, "trapezoid", None) or np.trapz      # numpy 1.x has only trapz
        return float(trap(a.gain_linear(th) * np.sin(th), th) / 2.0)
    icc = radiated(ParametricAntenna(18.0, 60.0, 20.0))
    c.value("ICC access pattern / isotropic", f"{icc:.1f} (+{10 * np.log10(icc):.1f} dB)", "6.2 (+7.9 dB)",
            "integrates to $6.2$ ($+7.9\\dB$)")
    rows = {n: (d, g, hp) for n, d, g, hp in panel_rows()}
    c.value("1x1 element HPBW / directivity", f"{rows[1][2]:.0f}/{rows[1][0]:.1f}", "65/9.8",
            "the single element, with a $65^\\circ$ beam, has a directivity of $9.8\\dBi$")
    c.value("element offset (dB)", f"{rows[1][0] - 8.0:.1f}", "1.8", "subtract the $1.8\\dB$")
    c.value("2x2 gain / HPBW", f"{rows[2][1]:.1f}/{rows[2][2]:.1f}", "11.0/44.5",
            "A $2{\\times}2$ panel gives $11.0\\dBi$ and $44.5^\\circ$")
    c.value("4x4 gain / HPBW", f"{rows[4][1]:.1f}/{rows[4][2]:.1f}", "16.0/24.8",
            "a $4{\\times}4$ panel gives $16.0\\dBi$ and $24.8^\\circ$")
    c.value("4x4 directivity", f"{rows[4][0]:.1f}", "17.9", "a directivity of $17.9\\dBi$ takes the $4{\\times}4$ panel")
    acc = radiated(ParametricAntenna(11.0, 44.5, 30.0))
    elem = 10 ** (-(rows[1][0] - 8.0) / 10)
    c.value("fitted access pattern / isotropic", f"{acc:.2f}", "0.67", "integrates to $0.67$, close to the $0.66$ of the 3GPP element")
    c.value("3GPP element (8 dBi on its directivity)", f"{elem:.2f}", "0.66")


# ---------------------------------------------------------------- bound, relays, placement
def check_bound_relays(c: Checker, cur: pd.DataFrame, d: pd.DataFrame) -> None:
    print("\n== Proposition 1, relays, empty clusters")
    cap = gateway_bound_bps("tether") / 1e6
    c.value("tether bound (Mb/s)", _thou(cap).replace(",", "{,}"), "2{,}444", "$44 + 3 \\times 800 = 2{,}444$\\,Mb/s")
    vid = lambda sc: 100 * cap / (d.loc[sc, "active"] * TIER_RATE["video"] / 1e6)  # noqa: E731
    c.value("bound share, hottest / urban video", f"{vid('milan_p100'):.1f}/{vid('urban_broad'):.1f}", "54.0/79.6",
            "$54.0\\%$ of the busiest tile's video demand and $79.6\\%$ of the urban one")
    g = cur[(cur["variant"] == "tether") & (cur["method"] == "greedy")]
    gaps, central = {}, {}
    for (sc, t), x in g.groupby(["scenario", "tier"]):
        b = np.minimum(x["served_pct"], 100 * cap / (d.loc[sc, "active"] * TIER_RATE[t] / 1e6)).max()
        gaps[(sc, t)] = b - x["served_e2e_pct_balanced"].max()
        central[(sc, t)] = b - x["served_e2e_pct_mst_central"].max()
        over = max((x[f"served_e2e_pct_{tp}"] - np.minimum(
            x["served_pct"], 100 * cap / (d.loc[sc, "active"] * TIER_RATE[t] / 1e6))).max()
            for tp in ("mst_corner", "mst_central", "balanced"))
        if over > 1e-6:
            c.value(f"bound violated {sc} {t}", f"{over:.3f}", "0")
    close = sum(v <= 0.2 for v in gaps.values())
    c.value("cells with balanced within 0.2 pp of bound", f"{close}/{len(gaps)}", "35/36",
            "within $0.2$\\pp\\ of this bound in $35$ of the $36$")
    worst = max(gaps, key=gaps.get)
    c.value("worst cell", f"{worst[0]} {worst[1]} {gaps[worst]:.1f}", "shelter_clustered broadband 3.8",
            "$3.8$\\pp\\ below it")
    cv = [central[(sc, "video")] for sc in ("urban_broad", "milan_p075", "milan_p090", "milan_p100")]
    c.value("central MST gap, unserved video cells", f"{min(cv):.0f}-{max(cv):.0f}", "17-24",
            "the central MST stays $17$ to $24$\\pp\\ below the bound")
    r = g[g["tier"] == "voice"]
    u = r[r["scenario"] == "urban_broad"].set_index("n_drones")["relays_balanced"]
    c.value("urban relay peak (balanced)", f"{u.max():.0f}@{u.idxmax()}", "16@16",
            "at $16$ relays for $16$ urban access drones")
    for sc, q in [("urban_broad", "40"), ("milan_p100", "60")]:
        x = r[r["scenario"] == sc].set_index("n_drones")["relays_balanced"]
        c.value(f"relays below 0.5 from N ({sc})", str(int(x[(x < 0.5) & (x.index > 3)].index.min())), q)
    c.snippet("relay vanish text", "from about $40$ access drones for urban demand and $60$ on the busiest tile")
    same = all(np.allclose(x["relays_mst_corner"], x["relays_mst_central"]) for _, x in g.groupby(["scenario", "tier"]))
    c.value("corner and central MST relays equal", "yes" if same else "no", "yes")

    # empty K-means clusters at N = 300 (Milan tiles and synthetic layouts, every seed)
    import warnings
    from scipy.cluster.vq import kmeans2
    import run_fleet as rf
    worst_empty = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in rf.cells():
            if cell["tier"] != "voice":
                continue
            users, _ = rf._users(cell)
            _, lab = kmeans2(users[:, :2], 300, minit="points", seed=cell["seed"])
            worst_empty = max(worst_empty, 300 - len(np.unique(lab)))
    c.value("most empty K-means clusters at N = 300", str(worst_empty), "2", "(at most $2$ in our runs)")


# ---------------------------------------------------------------- joint placement (Section sec:joint)
JOINT_ROWS = [("urban_broad", "voice", "Urban, voice"), ("milan_p100", "voice", "Milan p100, voice"),
              ("shelter_clustered", "video", "Shelter cl., video"),
              ("milan_p075", "video", "Milan p75, video"), ("milan_p025", "broadband", "Milan p25, bband"),
              ("milan_p100", "video", "Milan p100, video")]


def joint_table_rows(r: pd.DataFrame) -> list[str]:
    """Table tab:joint, one row per cell: bound, greedy [ci], free assoc., positions, + pointing [ci], gap closed."""
    rows = []
    for sc, t, name in JOINT_ROWS:
        q = {m: joint_nstar(r, sc, t, m) for m in ("greedy", "greedy_free", "xyz_nearest", "xyz", "xyzt")}
        lb = int(r[(r["scenario"] == sc) & (r["tier"] == t)]["lower_bound"].iloc[0])

        def ci(x):
            return _n(x["n"]) + ("" if x["lo"] == x["hi"] else f"\\ci{{{_n(x['lo'])}}}{{{_n(x['hi'])}}}")
        gap = 100 * (q["greedy"]["n"] - q["xyzt"]["n"]) / (q["greedy"]["n"] - lb)
        rows.append(" & ".join([name, str(lb), ci(q["greedy"]), _n(q["greedy_free"]["n"]), _n(q["xyz"]["n"]),
                                ci(q["xyzt"]), f"${gap:.0f}\\%$"]) + r" \\")
    return rows


def check_joint(c: Checker) -> None:
    print("\n== Section sec:joint: joint placement and pointing")
    r = joint_runs()
    c.value("joint agrees with the paper's simulator", "yes" if (r["served_pct"] - r["served_paper_pct"]).abs().max() < 1e-7
            else "no", "yes")
    w = r.pivot_table(index=["scenario", "tier", "unit", "n_drones"], columns="method", values="served_pct")
    mono = min((w[b] - w[a]).min() for a, b in (("greedy", "greedy_free"), ("greedy", "xyz_nearest"),
                                                 ("xyz_nearest", "xyz"), ("xyz", "xyzt")))
    c.value("each method serves at least its start", "yes" if mono >= -1e-9 else "no", "yes")
    for row, (sc, t, _) in zip(joint_table_rows(r), JOINT_ROWS):
        c.snippet(f"joint row {sc} {t}", row)
    n = {(sc, t): {m: joint_nstar(r, sc, t, m) for m in ("greedy", "greedy_free", "xyz_nearest", "xyz", "xyzt")}
         for sc, t, _ in JOINT_ROWS}
    for (sc, t), q in n.items():
        if any(x["below"] or x["above"] for x in q.values()):
            c.value(f"joint window holds N* ({sc} {t})", "no", "yes")
        c.value(f"nearest = free association after positions ({sc} {t})",
                "yes" if q["xyz_nearest"]["n"] == q["xyz"]["n"] else "no", "yes")
    lb = {k: int(r[(r["scenario"] == k[0]) & (r["tier"] == k[1])]["lower_bound"].iloc[0]) for k in n}
    red = {k: 100 * (1 - q["xyzt"]["n"] / q["greedy"]["n"]) for k, q in n.items()}
    gap = {k: 100 * (q["greedy"]["n"] - q["xyzt"]["n"]) / (q["greedy"]["n"] - lb[k]) for k, q in n.items()}
    c.value("fleet reduction with pointing (%)", f"{min(red.values()):.0f}-{max(red.values()):.0f}", "15-42",
            "Placement is worth $15$ to $42\\%$ of the fleet.")
    for snip in ("cuts the fleet by $15$ to $42\\%$ on six representative cells, closing $33$ to $69\\%$ of the gap",
                 "needs $15$ to $42\\%$ more drones than a searched placement",
                 "Placement saves $15$ to $42\\%$ of the access fleet",
                 "cuts the access fleet by $15$ to $42\\%$ on six representative cells"):
        c.snippet("15-42 % elsewhere", snip)
    c.value("gap closed (%)", f"{min(gap.values()):.0f}-{max(gap.values()):.0f}", "33-69",
            "The search closes $33$ to $69\\%$ of the gap")
    c.snippet("gap closed, Sec. VI", "a searched placement closes $33$ to $69\\%$ of the gap")
    rg = [q["greedy"]["n"] / lb[k] for k, q in n.items()]
    rt = [q["xyzt"]["n"] / lb[k] for k, q in n.items()]
    c.value("ratio to bound", f"{min(rg):.2f}-{max(rg):.2f}/{min(rt):.2f}-{max(rt):.2f}", "1.69-2.70/1.42-2.00",
            "from $1.69$--$2.70$ to $1.42$--$2.00$ times the bound")
    g = lambda sc, t, m: _n(n[(sc, t)][m]["n"])  # noqa: E731
    c.value("urban voice greedy -> pointing", f"{g('urban_broad', 'voice', 'greedy')}->{g('urban_broad', 'voice', 'xyzt')}",
            "16->12", "urban voice $12$ instead of $16$")
    c.value("hottest video greedy -> pointing", f"{g('milan_p100', 'video', 'greedy')}->{g('milan_p100', 'video', 'xyzt')}",
            "260->152", "video on the busiest tile $152$ instead of $260$")
    cap = [("shelter_clustered", "video"), ("milan_p075", "video"), ("milan_p025", "broadband"), ("milan_p100", "video")]
    pos = [100 * (1 - n[k]["xyz"]["n"] / n[k]["greedy"]["n"]) for k in cap]
    c.value("positions alone, capacity-bound (%)", f"{min(pos):.0f}-{max(pos):.0f}", "12-40",
            "saves $12$ to $40\\%$ of the fleet (busiest tile, video: $260 \\to 157$)")
    c.value("hottest video positions", g("milan_p100", "video", "xyz"), "157")
    pt = [n[k]["xyz"]["n"] - n[k]["xyzt"]["n"] for k in cap]
    c.value("pointing, capacity-bound (drones)", f"{min(pt):.0f}-{max(pt):.0f}", "2-5", "pointing adds $2$ to $5$ drones")
    c.value("voice: positions / pointing (urban; hottest)",
            f"{g('urban_broad', 'voice', 'greedy')}->{g('urban_broad', 'voice', 'xyz')}->{g('urban_broad', 'voice', 'xyzt')}; "
            f"{g('milan_p100', 'voice', 'greedy')}->{g('milan_p100', 'voice', 'xyz')}->{g('milan_p100', 'voice', 'xyzt')}",
            "16->15->12; 20->19->17", "three more on urban demand ($15 \\to 12$) and two on the busiest tile ($19 \\to 17$)")
    x = r[r["method"] == "xyzt"].groupby(["scenario", "tier"])[["mean_tilt_deg", "mean_altitude_m", "time_s"]].mean()
    c.value("mean tilt, voice urban / hottest", f"{x.loc[('urban_broad', 'voice'), 'mean_tilt_deg']:.1f}/"
            f"{x.loc[('milan_p100', 'voice'), 'mean_tilt_deg']:.1f}", "37.5/22.2",
            "tilt by $37.5^\\circ$ (urban) and $22.2^\\circ$ (busiest tile)")
    tv = [x.loc[k, "mean_tilt_deg"] for k in cap]
    c.value("mean tilt, video/broadband", f"{min(tv):.1f}-{max(tv):.1f}", "7.2-11.6", "against $7.2^\\circ$ to $11.6^\\circ$")
    c.value("greedy free association (hottest video; shelter)",
            f"{g('milan_p100', 'video', 'greedy_free')}; {g('shelter_clustered', 'video', 'greedy')}->"
            f"{g('shelter_clustered', 'video', 'greedy_free')}", "250; 54->50",
            "saves up to $10$ drones (busiest tile, video: $260 \\to 250$; clustered shelter, video: $54 \\to 50$)")
    fa = max(n[k]["greedy"]["n"] - n[k]["greedy_free"]["n"] for k in n)
    c.value("free association saves at most (drones)", f"{fa:.0f}", "10")
    c.value("altitude, video shelter / hottest", f"{x.loc[('shelter_clustered', 'video'), 'mean_altitude_m']:.1f}/"
            f"{x.loc[('milan_p100', 'video'), 'mean_altitude_m']:.1f}", "52.4/52.6",
            "$52.4$\\,m on average for video in the clustered shelter and $52.6$\\,m on the busiest tile")
    av = [x.loc[(sc, "voice"), "mean_altitude_m"] for sc in ("urban_broad", "milan_p100")]
    c.value("altitude, voice", f"{min(av):.0f}-{max(av):.0f}", "110-114", "and $110$ to $114$\\,m for voice")
    tv = [x.loc[(sc, "voice"), "time_s"] for sc in ("urban_broad", "milan_p100")]
    c.value("pointing time, voice / hottest video (s)", f"{np.mean(tv):.0f}/{x.loc[('milan_p100', 'video'), 'time_s']:.0f}",
            "22/184", "takes $22$\\,s per layout on average for voice and $184$\\,s for video on the busiest tile")
    c.snippet("greedy recomputed note", "($260$ instead of $270$ drones for video on the busiest tile)")


def main() -> int:
    c = Checker(TEX.read_text())
    cur = fleet_curves()
    check_antenna(c)
    d = check_demand(c)
    check_fleet_table(c, cur)
    check_cost_table(c, cur)
    check_fleet_text(c, cur, d)
    check_bound_relays(c, cur, d)
    check_wp1(c)
    check_joint(c)
    print(f"\n{c.n} checks, {len(c.bad)} problems")
    for b in c.bad:
        print("  " + b)
    return 1 if c.bad else 0


if __name__ == "__main__":
    sys.exit(main())
