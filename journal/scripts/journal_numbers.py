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
- End to end: served_e2e_pct_<topology>; tether = variant tether; satellite = variant satellite,
  the tether runs capped at 20 Mb/s up for the two-way tiers and 100 Mb/s down for broadband
  (journal_data.add_satellite); relays are the mean at that N, rounded. Variant grounded simulates a
  100 Mb/s gateway in both directions and is used here only where that is what is checked.
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
                          fleet_curves, gateway_bound_bps, joint_e2e_runs, joint_nstar, joint_runs,
                          lower_bound_drones, paper_window, refine_runs, runs, sizes, e2e_cell,
                          satellite_cap_pct)
from journal_data import JOINT  # noqa: E402

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
             "milan_p100": "Milan, busiest"}
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
    c.value("resident ratio busiest / 25th pct", f"{ratio:.1f}", "9.8", "a factor of $9.8$")
    c.value("Sphere clustered people", _thou(d.loc["shelter_clustered", "population"]), "13,963", "holds $13{,}963$ people")
    c.value("Sphere hotspot people", _thou(d.loc["shelter_hotspot", "population"]), "2,094", "$\\sigma = 50$\\,m) $2{,}094$")
    c.value("busiest tile active users", _thou(d.loc["milan_p100", "active"]), "11,797", "up to $11{,}797$ active users")
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


def _cell_e2e(cur, rr, v, sc, t, ci=True) -> str:
    """End-to-end entry with the cheapest tree (access + relays), as in Table IV."""
    f = e2e_cell(cur, rr, v, sc, t)
    if not math.isfinite(f["n"]):
        return f"\\{'neverext' if f['n_max'] > 300 else 'never'}{{{f['best']:.0f}}}"
    out = f"\\plr{{{int(f['n'])}}}{{{f['relays']:.0f}}}"
    if ci and f["lo"] != f["hi"]:
        out += f"\\ci{{{_n(f['lo'])}}}{{{_n(f['hi'])}}}"
    return out


def check_fleet_table(c: Checker, cur: pd.DataFrame) -> None:
    print("\n== Table tab:fleet: smallest fleet (access; end to end, cheapest tree)")
    s = sizes(cur)
    rows = [("rural_broad", "Rural (TS 22.261)"), ("shelter_hotspot", "Shelter, hotspot"),
            ("shelter_clustered", "Shelter, clustered"), ("urban_broad", "Urban (TS 22.261)"),
            ("milan_p025", "Milan, 25th pct"), ("milan_p050", "Milan, 50th pct"),
            ("milan_p075", "Milan, 75th pct"), ("milan_p090", "Milan, 90th pct"),
            ("milan_p100", "Milan, busiest")]
    rr = refine_runs()
    for sc, name in rows:
        cells = [name] + [_cell(cur, rr, "grounded", sc, t) for t in TIERS]
        cells += [_cell_e2e(cur, rr, "satellite", sc, "voice"),
                  _cell_e2e(cur, rr, "tether", sc, "voice"),
                  _cell_e2e(cur, rr, "tether", sc, "video")]
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
    e2e = [("satellite", "balanced", "Satellite, any tree"), ("tether", "mst_corner", "Tether, corner MST"),
           ("tether", "mst_central", "Tether, central MST"), ("tether", "balanced", "Tether, balanced tree")]
    for v, topo, name in e2e:
        c.snippet(f"e2e {v} {topo}", " & ".join([name] + [_cell(cur, rr, v, sc, t, f"served_e2e_pct_{topo}", ci=False)
                                                          for sc, t in cols]) + r" \\")
    # "any tree": the satellite ceiling is the same for all three topologies
    for sc, t in cols:
        vals = {topo: sizes(cur, metric=f"served_e2e_pct_{topo}").loc[("satellite", sc, t, "greedy"), "best_served_pct"]
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
    c.value("voice -> video, busiest", f"{N('grounded', 'milan_p100', 'video') / N('grounded', 'milan_p100', 'voice'):.0f}",
            "13", "multiplies the fleet by $13$")
    c.value("Milan voice range", rng("voice", milan), "6-21", "voice needs $6$ to $21$ drones and a video call $21$ to $270$")
    c.value("Milan video range", rng("video", milan), "21-270")
    c.value("video spread across tiles (x)",
            f"{N('grounded', 'milan_p100', 'video') / N('grounded', 'milan_p025', 'video'):.0f}", "13",
            "by up to $13$ times ($21$ to $270$ drones")

    # sweep vs greedy
    fl = sorted(cur["n_drones"].unique())
    import run_fleet as rf
    c.value("fleet grid size", str(len(rf.FLEET)), "29", "on a $29$-value grid from $1$ to $300$")
    want = (list(range(1, 11)) + list(range(12, 21, 2)) + list(range(25, 41, 5)) + list(range(50, 81, 10))
            + [100, 120, 150] + list(range(200, 301, 50)))
    c.value("fleet grid as described", "yes" if list(rf.FLEET) == want else "no", "yes")
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
    c.value("sweep busiest video", _n(sw.loc[("milan_p100", "video")]), "350",
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
    c.value("urban / busiest broadband 300 -> 500",
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
    c.value("busiest video interval", iv("milan_p100", "video"), "255-285@270", "(busiest tile, video: $255$ to $285$ around $270$)")
    c.value("clustered voice / video interval", iv("shelter_clustered", "voice") + "/" + iv("shelter_clustered", "video"),
            "5-8@6/52-58@54", "(clustered shelter: voice $5$ to $8$ around $6$, video $52$ to $58$ around $54$)")

    # lower bound
    ratio = {t: {sc: N("grounded", sc, t) / lower_bound_drones(sc, t) for sc in allsc} for t in ("voice", "video")}
    vb = [lower_bound_drones(sc, "video") for sc in allsc if lower_bound_drones(sc, "video") >= 10]
    vr = [ratio["video"][sc] for sc in allsc if lower_bound_drones(sc, "video") >= 10]
    c.value("video bounds >= 10: range", f"{min(vb)}-{max(vb)}", "10-98", "where this bound is $10$ to $98$ drones")
    c.value("video fleet / bound there", f"{min(vr):.1f}-{max(vr):.1f}", "1.8-2.8", "are $1.8$ to $2.8$ times the bound")
    rv = {sc: v for sc, v in ratio["voice"].items() if sc != "milan_p025"}
    c.value("voice bound max / ratio range without p25",
            f"{max(lower_bound_drones(sc, 'voice') for sc in allsc)}/{min(rv.values()):.1f}-{max(rv.values()):.0f}",
            "10/2.1-4", "for voice, with bounds of at most $10$ drones, $2.1$ to $4$ times")
    p25 = 0.95 * d.loc["milan_p025", "active"] * TIER_RATE["voice"] / (10e6 * 4.4)
    c.value("p25 voice: ratio / raw bound", f"{ratio['voice']['milan_p025']:.0f}/{p25:.2f}", "6/0.99",
            "and $6$ times on the $25$th-percentile tile, whose bound of $0.99$ drones rounds up to one")
    c.value("busiest broadband bound", str(lower_bound_drones("milan_p100", "broadband")), "510", "at least $510$ drones")

    # altitude under load
    gv = cur[(cur["variant"] == "grounded") & (cur["method"] == "greedy") & (cur["tier"] == "video")
             & (cur["n_drones"] == 100)].set_index("scenario")["mean_altitude_m"]
    broad = [gv[sc] for sc in ("urban_broad", "milan_p075", "milan_p090", "milan_p100")]
    c.value("video altitude at N=100, urban + three busiest / two quietest",
            f"{min(broad):.0f}-{max(broad):.0f}/{gv['milan_p050']:.0f}/{gv['milan_p025']:.0f}", "60-67/77/80",
            "to $60$--$67$\\,m at $100$ drones on urban demand and the three busiest tiles ($77$ and $80$\\,m on the two quietest)")
    km = cur[(cur["variant"] == "grounded") & (cur["scenario"] == "milan_p100") & (cur["tier"] == "video")]
    kmm = km[km["method"] == "kmeans"].set_index("n_drones")["served_pct"]
    c.value("K-means best, busiest video", f"{kmm.max():.1f}@{kmm.idxmax()}", "61.6@200",
            "at most $61.6\\%$ of the video demand, at $200$ drones")
    grm = km[km["method"] == "greedy"].set_index("n_drones")["served_pct"]
    c.value("greedy at 300, busiest video", f"{grm.loc[300]:.1f}", "95.4", "against $95.4\\%$ for greedy at $300$")

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
        ("bw20", "milan_p100", "video", "94", "busiest tile $270 \\to 94$"),
        ("bw20", "milan_p050", "broadband", "86", None),
        ("grounded", "milan_p050", "broadband", "230", None),
        ("bw20", "milan_p075", "broadband", "250", "brings broadband within reach of the $75$th-percentile tile"),
        ("cap40", "milan_p075", "voice", "160", "voice and video calls both need $160$ drones, instead of $12$ and $68$"),
        ("lowincome", "milan_p075", "video", "43", "($75$th percentile $68 \\to 43$"),
        ("lowincome", "milan_p100", "video", "130", "busiest $270 \\to 130$)"),
        ("suburban", "milan_p075", "video", "90", "needs $90$ drones instead of $68$"),
        ("dense_urban", "urban_broad", "voice", "20", "(urban voice $16 \\to 20$)"),
    ]:
        c.value(f"{v} {sc} {t}", nv(v, sc, t), q, snip)
    for sc in ("milan_p100", "milan_p090"):
        c.value(f"cap40 {sc} voice not reached", nv("cap40", sc, "voice"), "-")
    c.value("grounded milan_p075 broadband not reached", nv("grounded", "milan_p075", "broadband"), "-")
    r13 = N("grounded", "milan_p100", "video") / N("grounded", "milan_p100", "voice")
    r35 = N("grounded", "milan_p025", "video") / N("grounded", "milan_p025", "voice")
    c.value("voice -> video: busiest / 25th pct", f"{r13:.0f}/{r35:.1f}", "13/3.5",
            "($13$ times from voice to video on the busiest tile, $3.5$ times on the $25$th-percentile tile)")
    rel = {tp: F("tether", "urban_broad", "voice", f"served_e2e_pct_{tp}")["relays"] for tp in ("balanced", "mst_central")}
    c.value("rule 3: urban voice relays balanced / MST", f"{rel['balanced']:.0f}/{rel['mst_central']:.0f}", "21/15",
            "(the balanced tree needs $21$ relays for urban voice, the MSTs $15$)")
    c.value("rule 1: tether two-way capacity (Gb/s)", f"{gateway_bound_bps('tether', 'video') / 1e9:.1f}", "1.2",
            "carry about $1.2$\\,Gb/s of two-way traffic")

    # 90 % target
    s90 = sizes(cur, target=90.0).xs("greedy", level="method").loc["grounded"]["n90"]
    shrink = {sc: 1 - s90.loc[(sc, "video")] / n95.loc[(sc, "video")] for sc in allsc}
    c.value("video shrink at 90 %, max", f"{100 * max(shrink.values()):.0f}", "40", "video fleets shrink by up to $40\\%$")
    c.value("busiest / urban video at 90 %", f"{_n(s90.loc[('milan_p100', 'video')])}/{_n(s90.loc[('urban_broad', 'video')])}",
            "200/100", "(busiest tile $300 \\to 200$, urban $150 \\to 100$)")
    dv = max(n95.loc[(sc, "voice")] - s90.loc[(sc, "voice")] for sc in allsc)
    c.value("voice shrink at 90 %, max drones", f"{dv:.0f}", "3", "voice fleets by at most $3$ drones (urban $16 \\to 14$)")
    c.snippet("90 % on the coarse grid", "On the coarse grid, at $90\\%$ instead of $95\\%$")
    c.value("urban voice at 90 %", _n(s90.loc[("urban_broad", "voice")]), "14")

    # gateway
    print("\n== Sec. VI text")
    sat = sizes(cur, metric="served_e2e_pct_balanced").xs("greedy", level="method").loc["satellite"]
    ok = lambda t: ",".join(sorted(sc for sc in allsc if not pd.isna(sat.loc[(sc, t), "n95"])))  # noqa: E731
    c.value("satellite serves text", ok("text"), ",".join(sorted(set(allsc) - {"urban_broad", "milan_p090", "milan_p100"})),
            "carries text everywhere but urban demand and the two busiest Milan tiles")
    c.value("satellite serves voice", ok("voice"), "rural_broad,shelter_hotspot",
            "voice only in the rural scenario and the hotspot shelter, and video nowhere")
    c.value("satellite serves video", ok("video") or "none", "none")
    c.value("satellite ceiling busiest voice", f"{sat.loc[('milan_p100', 'voice'), 'best_served_pct']:.1f}", "4.5",
            "$4.5\\%$ of the voice demand of the busiest Milan tile ($20/448$\\,Mb/s)")
    c.value("20 / busiest voice demand", f"{satellite_cap_pct('milan_p100', 'voice'):.1f}", "4.5")
    c.value("satellite ceiling urban voice", f"{sat.loc[('urban_broad', 'voice'), 'best_served_pct']:.1f}", "6.6",
            "and $6.6\\%$ of the urban voice demand")
    c.value("abstract: under 5 % of busiest voice", "yes" if satellite_cap_pct('milan_p100', 'voice') < 5 else "no", "yes",
            "carries under $5\\%$ of the busiest Milan tile's voice traffic, whatever the fleet")
    # introduction: a coverage-sized fleet on the busiest tile (greedy, coarse grid; text tier = coverage)
    g = cur[(cur.variant == "grounded") & (cur.method == "greedy") & (cur.scenario == "milan_p100")].set_index(["tier", "n_drones"])
    n_cov = min(n for (t, n), v in g["coverage_pct"].items() if t == "text" and v >= 90)
    c.value("intro: busiest tile, fewest drones covering 90 % (text, coarse grid)", f"{n_cov}", "20",
            "$20$ drones cover $90\\%$ of the users, but the same number serves $19\\%$ of the video-call demand, and video needs $270$")
    c.value("intro: busiest tile, video served by 20 drones", f"{g.loc[('video', 20), 'served_pct']:.0f}", "19")
    import journal_data as jd
    c.value("calls on the satellite upload", f"{jd.SAT_UP_BPS / TIER_RATE['voice']:.0f}", "526",
            "carries about $500$ simultaneous voice calls")
    for topo in ("mst_corner", "mst_central", "balanced"):
        a = sizes(cur, metric=f"served_e2e_pct_{topo}")
        b = a.loc[("bh200", slice(None), slice(None), "greedy"), ["n95", "best_served_pct"]].droplevel([0, 3])
        # bh200 and grounded both simulate a 100 Mb/s gateway in both directions; the 200 Mb/s links change
        # nothing there because the gateway binds first, and a 20 Mb/s upload binds sooner still
        base = a.loc[("grounded", slice(None), slice(None), "greedy"), ["n95", "best_served_pct"]].droplevel([0, 3])
        same = (b["n95"].fillna(-1).eq(base["n95"].fillna(-1)) & (b["best_served_pct"] - base["best_served_pct"]).abs().lt(0.05)).all()
        c.value(f"200 Mb/s links change nothing ({topo})", "yes" if same else "no", "yes")
    teth = {t: sizes(cur, metric=f"served_e2e_pct_{t}").xs("greedy", level="method").loc["tether"]
            for t in ("mst_corner", "mst_central", "balanced")}
    same_n = all(N("tether", sc, "voice", f"served_e2e_pct_{tp}") == N("grounded", sc, "voice")
                 for sc in allsc for tp in ("mst_central", "balanced"))
    c.value("tether voice fleet = access fleet (central MST, balanced)", "yes" if same_n else "no", "yes",
            "the balanced tree and the central MST serve voice end to end in every scenario with the access fleet")
    c.value("corner MST, busiest voice: best share", f"{teth['mst_corner'].loc[('milan_p100', 'voice'), 'best_served_pct']:.0f}",
            "93", "the corner MST stops at $93\\%$ on the busiest tile")
    for topo, vals, snip in [
        ("mst_corner", ("29", "16", "11"), "stalls at $29\\%$ on the $75$th-percentile Milan tile, $16\\%$ for urban demand and $11\\%$ on the busiest tile"),
        ("mst_central", ("53", "33", "20"), "lifts these ceilings to $53\\%$, $33\\%$ and $20\\%$"),
        ("balanced", ("73", "40", "27"), "lifts them to $73\\%$, $40\\%$ and $27\\%$"),
    ]:
        got = tuple(f"{teth[topo].loc[(sc, 'video'), 'best_served_pct']:.0f}" for sc in ("milan_p075", "urban_broad", "milan_p100"))
        c.value(f"tether {topo} video ceilings", "/".join(got), "/".join(vals), snip)
    ab = (f"{teth['mst_central'].loc[('milan_p100', 'video'), 'best_served_pct']:.0f}/"
          f"{teth['balanced'].loc[('milan_p100', 'video'), 'best_served_pct']:.0f}")
    c.value("abstract: busiest video, best spanning tree -> balanced", ab, "20/27",
            "raises that tile's video ceiling from $20\\%$ (best spanning tree) to $27\\%$")
    c.value("conclusion: busiest video, best spanning tree -> balanced", ab, "20/27",
            "from the $20\\%$ of the best spanning tree to $27\\%$")
    best_mst = max(teth[t].loc[("milan_p100", "video"), "best_served_pct"] for t in ("mst_corner", "mst_central"))
    c.value("central MST is the best spanning tree (busiest video)",
            "yes" if best_mst == teth["mst_central"].loc[("milan_p100", "video"), "best_served_pct"] else "no", "yes")
    c.value("p50 video: central / corner MST best", f"{teth['mst_central'].loc[('milan_p050', 'video'), 'best_served_pct']:.0f}/"
            f"{teth['mst_corner'].loc[('milan_p050', 'video'), 'best_served_pct']:.0f}", "94/52",
            "the $50$th-percentile tile's video to $94\\%$, where the corner MST stops at $52\\%$")
    f = F("tether", "milan_p050", "video", "served_e2e_pct_balanced")
    c.value("balanced p50 video", f"{f['n']:.0f}+{f['relays']:.0f}", "33+13", "($33$ drones and $13$ relays)")

    # relays
    def nr(topo, sc, t):
        f = F("tether", sc, t, f"served_e2e_pct_{topo}")
        return f"{f['n']:.0f}+{f['relays']:.0f}"
    rr = refine_runs()
    def best(sc, t):
        f = e2e_cell(cur, rr, "tether", sc, t)
        return f"{f['n']:.0f}+{f['relays']:.0f}"
    c.value("voice, cheapest tree: urban / busiest / rural",
            f"{best('urban_broad', 'voice')}/{best('milan_p100', 'voice')}/{best('rural_broad', 'voice')}", "16+15/21+16/3+6",
            "$16 + 15$ for urban demand, $21 + 16$ on the busiest tile and $3 + 6$ in the rural scenario")
    ratios = []
    for sc in allsc:
        f = e2e_cell(cur, rr, "tether", sc, "voice")
        ratios.append(round(f["relays"]) / f["n"])
    c.value("voice relays / access, cheapest tree", f"{min(ratios):.1f}-{max(ratios):.0f}", "0.8-2",
            "relays add $0.8$ to $2$ times the access fleet")
    def voice_relays(sc, tp):
        return fleet_cell(cur, rr, "tether", sc, "voice", f"served_e2e_pct_{tp}")["relays"]
    mst_le = all(voice_relays(sc, "mst_central") <= voice_relays(sc, "balanced") + 1e-9 for sc in allsc)
    c.value("voice: MST relays <= balanced relays everywhere", "yes" if mst_le else "no", "yes",
            "the MSTs deliver voice in full with fewer or equal relays")
    tx = [e2e_cell(cur, rr, "tether", sc, "text")["relays"] for sc in allsc]
    ta = [N("grounded", sc, "text") for sc in allsc]
    c.value("text relays / access range (cheapest tree)", f"{min(tx):.0f}-{max(tx):.0f}/{min(ta):.0f}-{max(ta):.0f}", "3-9/2-6",
            "needs $3$ to $9$ relays besides $2$ to $6$ access drones")
    vr = [e2e_cell(cur, rr, "tether", sc, "video")["relays"] for sc in ("milan_p075", "milan_p050")]
    c.value("video relays p50 (cheapest tree)", f"{vr[1]:.0f}", "13", "($13$ for the $33$ access drones of the $50$th-percentile tile)")
    c.value("urban voice relays, corner", f"{F('tether', 'urban_broad', 'voice', 'served_e2e_pct_mst_corner')['relays']:.0f}",
            "15", "($21$ instead of $15$ for urban voice)")
    sh = sizes(cur, metric="served_e2e_pct_balanced").loc[("bh_shared", "milan_p100", "text", "greedy"), "best_served_pct"]
    c.value("shared channel, busiest text", f"{sh:.0f}", "57", "at most $57\\%$ of its text demand")
    used = sh / 100 * d.loc["milan_p100", "active"] * TIER_RATE["text"] / 1e6
    c.value("shared channel: 100 Mb/s gateway not binding", "yes" if used < 100 else "no", "yes",
            "with a gateway that does not bind, as with a tether")
    c.value("satellite cap, busiest text = shared-channel share", f"{satellite_cap_pct('milan_p100', 'text'):.0f}/{sh:.0f}", "57/57",
            "a satellite's upload alone would cap it at the same share")


# ---------------------------------------------------------------- WP1
def check_wp1(c: Checker) -> None:
    print("\n== Sec. IV and Table III (WP1)")
    cols = {}
    for v in ("grounded", "waiver300"):
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
        c.snippet(f"row {key}", " & ".join([name] + [cols[v][key] for v in ("grounded", "waiver300")]) + r" \\")
    rr_cap = 10e6 * 4.4 / 384e3
    c.value("round robin serves nobody above B se_max / r_min (video)", f"{rr_cap:.0f}", "115",
            "nobody once a cell holds more than $B\\,\\mathrm{se}_{\\max}/r_{\\min}$ active users (about $115$ for video calls)")
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
    rows = {n: (d, g, hp) for n, d, g, hp in panel_rows()}
    c.value("1x1 element HPBW / directivity", f"{rows[1][2]:.0f}/{rows[1][0]:.1f}", "65/9.8")
    c.value("element offset (dB)", f"{rows[1][0] - 8.0:.1f}", "1.8", "subtract the $1.8\\dB$")
    c.value("2x2 gain / HPBW", f"{rows[2][1]:.1f}/{rows[2][2]:.1f}", "11.0/44.5",
            "A $2{\\times}2$ panel gives $11.0\\dBi$ and $44.5^\\circ$")
    c.value("4x4 gain / HPBW", f"{rows[4][1]:.1f}/{rows[4][2]:.1f}", "16.0/24.8",
            "a $4{\\times}4$ panel gives $16.0\\dBi$ and $24.8^\\circ$")
    acc = radiated(ParametricAntenna(11.0, 44.5, 30.0))
    elem = 10 ** (-(rows[1][0] - 8.0) / 10)
    c.value("fitted access pattern / isotropic", f"{acc:.2f}", "0.67", "integrates to $0.67$, close to the $0.66$ of the 3GPP element")
    c.value("3GPP element (8 dBi on its directivity)", f"{elem:.2f}", "0.66")


# ---------------------------------------------------------------- bound, relays, placement
def check_bound_relays(c: Checker, cur: pd.DataFrame, d: pd.DataFrame) -> None:
    print("\n== Proposition 1, relays, empty clusters")
    capt = {t: gateway_bound_bps("tether", t) / 1e6 for t in TIERS}
    allsc_ = SYNTH + MILAN
    cap = capt["video"]
    c.value("tether bound, two-way / broadband (Mb/s)", f"{_thou(cap)}/{_thou(capt['broadband'])}", "1,244/2,444")
    vid = lambda sc: 100 * cap / (d.loc[sc, "active"] * TIER_RATE["video"] / 1e6)  # noqa: E731
    c.snippet("bound values", "For broadband the bound is $44 + 3 \\times 800 = 2{,}444$\\,Mb/s; for the two-way tiers, whose links carry $400$\\,Mb/s, it is $44 + 3 \\times 400 = 1{,}244$\\,Mb/s")
    c.value("bound share, busiest / urban video", f"{vid('milan_p100'):.1f}/{vid('urban_broad'):.1f}", "27.5/40.5",
            "$27.5\\%$ of the busiest tile's video demand and $40.5\\%$ of the urban one")
    g = cur[(cur["variant"] == "tether") & (cur["method"] == "greedy")]
    gaps, central = {}, {}
    for (sc, t), x in g.groupby(["scenario", "tier"]):
        b = np.minimum(x["served_pct"], 100 * capt[t] / (d.loc[sc, "active"] * TIER_RATE[t] / 1e6)).max()
        gaps[(sc, t)] = b - x["served_e2e_pct_balanced"].max()
        central[(sc, t)] = b - x["served_e2e_pct_mst_central"].max()
        over = max((x[f"served_e2e_pct_{tp}"] - np.minimum(
            x["served_pct"], 100 * capt[t] / (d.loc[sc, "active"] * TIER_RATE[t] / 1e6))).max()
            for tp in ("mst_corner", "mst_central", "balanced"))
        if over > 1e-6:
            c.value(f"bound violated {sc} {t}", f"{over:.3f}", "0")
    binding = {k for k, x in g.groupby(["scenario", "tier"])
               if 100 * capt[k[1]] / (d.loc[k[0], "active"] * TIER_RATE[k[1]] / 1e6) < x["served_pct"].max() - 1e-9}
    c.value("cells where the bound binds", f"{len(binding)}/{len(gaps)}", "10/36", "in $10$ of the $36$ (scenario, tier) cells")
    bv = sorted(sc for sc, t in binding if t == "video")
    c.value("binding video cells", ",".join(bv), "milan_p075,milan_p090,milan_p100,urban_broad",
            "video on urban demand and the three busiest tiles, and broadband in six scenarios")
    c.value("binding broadband cells", str(sum(t == "broadband" for _, t in binding)), "6")
    close = sum(gaps[k] <= 0.2 for k in binding)
    c.value("binding cells with balanced within 0.2 pp", f"{close}/{len(binding)}", "8/10",
            "In $8$ of the $10$ binding cells the balanced tree's ceiling is within $0.2$\\pp\\ of the bound")
    nb = [k for k in gaps if k not in binding]
    off = [k for k in nb if gaps[k] > 0.2]
    c.value("non-binding cells within 0.2 pp / exception", f"{len(nb) - len(off)}/{len(nb)} "
            + ",".join(f"{k[0]} {k[1]} {gaps[k]:.1f}" for k in off), "25/26 shelter_clustered video 2.0",
            "In $25$ of the other $26$ cells the balanced tree's ceiling is within $0.2$\\pp\\ of the access share")
    c.snippet("non-binding exception", "the exception is video in the clustered shelter, $2.0$\\pp\\ below it")
    gaps_b = {k: gaps[k] for k in binding}
    worst = max(gaps_b, key=gaps_b.get)
    c.value("worst cell", f"{worst[0]} {worst[1]} {gaps[worst]:.1f}", "shelter_clustered broadband 3.6",
            "$3.6$\\pp\\ below it")
    second = sorted(gaps_b, key=gaps_b.get)[-2]
    c.value("second binding cell", f"{second[0]} {second[1]} {gaps[second]:.1f}", "milan_p075 video 0.3",
            "video on the $75$th-percentile tile, $0.3$\\pp\\ below it")
    unserved = [sc for sc in allsc_ if (sc, "video") in central and
                pd.isna(sizes(cur, metric="served_e2e_pct_mst_central").loc[("tether", sc, "video", "greedy"), "n95"])]
    c.value("video cells the MSTs cannot serve", ",".join(sorted(unserved)),
            "milan_p050,milan_p075,milan_p090,milan_p100,shelter_clustered,urban_broad",
            "On the six video cells the MSTs cannot serve (the clustered shelter, urban demand and the four busiest tiles)")
    cv = [central[(sc, "video")] for sc in unserved]
    c.value("central MST gap, unserved video cells", f"{min(cv):.0f}-{max(cv):.0f}", "5-20",
            "the central MST stays $5$ to $20$\\pp\\ below the smaller of the access share and the bound")
    r = g[g["tier"] == "voice"]
    u = r[r["scenario"] == "urban_broad"].set_index("n_drones")["relays_balanced"]
    c.value("urban relay peak (balanced)", f"{u.max():.0f}@{u.idxmax()}", "26@25",
            "at $26$ relays for $25$ urban access drones")
    for sc, q in [("urban_broad", "80"), ("milan_p100", "120")]:
        x = r[r["scenario"] == sc].set_index("n_drones")["relays_balanced"]
        c.value(f"relays below 0.5 from N ({sc})", str(int(x[(x < 0.5) & (x.index > 3)].index.min())), q)
    c.snippet("relay vanish text", "from about $80$ access drones for urban demand and $120$ on the busiest tile")
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
        lb = lower_bound_drones(sc, t)   # 95 % target, as Table II

        def ci(x):
            return _n(x["n"]) + ("" if x["lo"] == x["hi"] else f"\\ci{{{_n(x['lo'])}}}{{{_n(x['hi'])}}}")
        gap = 100 * (q["greedy"]["n"] - q["xyzt"]["n"]) / (q["greedy"]["n"] - lb)
        rows.append(" & ".join([name, str(lb), ci(q["greedy"]), _n(q["greedy_free"]["n"]), _n(q["xyz"]["n"]),
                                ci(q["xyzt"]), f"${gap:.0f}\\%$"]) + r" \\")
    return rows


def check_joint(c: Checker, cur: pd.DataFrame) -> None:
    print("\n== Section sec:joint: joint placement and pointing")
    r = joint_runs()
    rr = refine_runs()
    for sc, t, _ in JOINT_ROWS:                 # the greedy column is Table IV's entry, recomputed
        q = joint_nstar(r, sc, t, "greedy")
        got = _n(q["n"]) + ("" if q["lo"] == q["hi"] else f"\\ci{{{_n(q['lo'])}}}{{{_n(q['hi'])}}}")
        c.value(f"greedy = Table IV ({sc} {t})", got, _cell(cur, rr, "grounded", sc, t))
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
    lb = {k: lower_bound_drones(*k) for k in n}
    red = {k: 100 * (1 - q["xyzt"]["n"] / q["greedy"]["n"]) for k, q in n.items()}
    gap = {k: 100 * (q["greedy"]["n"] - q["xyzt"]["n"]) / (q["greedy"]["n"] - lb[k]) for k, q in n.items()}
    c.value("fleet reduction with pointing (%)", f"{min(red.values()):.0f}-{max(red.values()):.0f}", "15-42",
            "placement is worth $15$ to $42\\%$ of the fleet.")
    # the abstract, design rules, limitations and conclusion quote the saving with restarts (check_joint_restarts)
    c.value("gap closed (%)", f"{min(gap.values()):.0f}-{max(gap.values()):.0f}", "32-66",
            "The search closes $32$ to $66\\%$ of the gap")
    c.snippet("gap closed, Sec. VI", "a searched placement closes $32$ to $66\\%$ of the gap")
    rg = [q["greedy"]["n"] / lb[k] for k, q in n.items()]
    rt = [q["xyzt"]["n"] / lb[k] for k, q in n.items()]
    c.value("ratio to bound", f"{min(rg):.2f}-{max(rg):.2f}/{min(rt):.2f}-{max(rt):.2f}", "1.84-2.84/1.55-2.11",
            "from $1.84$--$2.84$ to $1.55$--$2.11$ times the bound")
    g = lambda sc, t, m: _n(n[(sc, t)][m]["n"])  # noqa: E731
    c.value("urban voice greedy -> pointing", f"{g('urban_broad', 'voice', 'greedy')}->{g('urban_broad', 'voice', 'xyzt')}",
            "16->12", "urban voice $12$ instead of $16$")
    c.value("busiest video greedy -> pointing", f"{g('milan_p100', 'video', 'greedy')}->{g('milan_p100', 'video', 'xyzt')}",
            "270->157", "video on the busiest tile $157$ instead of $270$")
    cap = [("shelter_clustered", "video"), ("milan_p075", "video"), ("milan_p025", "broadband"), ("milan_p100", "video")]
    pos = [100 * (1 - n[k]["xyz"]["n"] / n[k]["greedy"]["n"]) for k in cap]
    c.value("positions alone, capacity-bound (%)", f"{min(pos):.0f}-{max(pos):.0f}", "15-38",
            "saves $15$ to $38\\%$ of the fleet (busiest tile, video: $270 \\to 167$)")
    c.value("busiest video positions", g("milan_p100", "video", "xyz"), "167")
    pt = [n[k]["xyz"]["n"] - n[k]["xyzt"]["n"] for k in cap]
    c.value("pointing, capacity-bound (drones)", f"{min(pt):.0f}-{max(pt):.0f}", "0-10", "pointing saves $0$ to $10$ drones more")
    pp = [100 * (n[k]["xyz"]["n"] - n[k]["xyzt"]["n"]) / n[k]["xyz"]["n"] for k in cap]
    c.value("pointing, capacity-bound, at most (%)", f"{max(pp):.0f}", "7", "at most $7\\%$")
    c.value("voice: positions / pointing (urban; busiest)",
            f"{g('urban_broad', 'voice', 'greedy')}->{g('urban_broad', 'voice', 'xyz')}->{g('urban_broad', 'voice', 'xyzt')}; "
            f"{g('milan_p100', 'voice', 'greedy')}->{g('milan_p100', 'voice', 'xyz')}->{g('milan_p100', 'voice', 'xyzt')}",
            "16->15->12; 21->19->17", "three more on urban demand ($15 \\to 12$) and two on the busiest tile ($19 \\to 17$)")
    x = r[r["method"] == "xyzt"].groupby(["scenario", "tier"])[["mean_tilt_deg", "mean_altitude_m", "time_s"]].mean()
    c.value("mean tilt, voice urban / busiest", f"{x.loc[('urban_broad', 'voice'), 'mean_tilt_deg']:.1f}/"
            f"{x.loc[('milan_p100', 'voice'), 'mean_tilt_deg']:.1f}", "37.4/22.4",
            "tilt by $37.4^\\circ$ (urban) and $22.4^\\circ$ (busiest tile)")
    tv = [x.loc[k, "mean_tilt_deg"] for k in cap]
    c.value("mean tilt, video/broadband", f"{min(tv):.1f}-{max(tv):.1f}", "7.6-11.5", "against $7.6^\\circ$ to $11.5^\\circ$")
    c.value("greedy free association (busiest video; shelter)",
            f"{g('milan_p100', 'video', 'greedy_free')}; {g('shelter_clustered', 'video', 'greedy')}->"
            f"{g('shelter_clustered', 'video', 'greedy_free')}", "255; 54->50",
            "up to $15$ (busiest tile, video: $270 \\to 255$; clustered shelter, video: $54 \\to 50$)")
    fa = max(n[k]["greedy"]["n"] - n[k]["greedy_free"]["n"] for k in n)
    c.value("free association saves at most (drones)", f"{fa:.0f}", "15")
    fp = max(n[k]["xyz_nearest"]["n"] - n[k]["xyz"]["n"] for k in n)
    c.value("free association after positions saves at most (drones)", f"{fp:.0f}", "2",
            "once positions are searched it saves at most two drones")
    c.value("altitude, video shelter / busiest", f"{x.loc[('shelter_clustered', 'video'), 'mean_altitude_m']:.1f}/"
            f"{x.loc[('milan_p100', 'video'), 'mean_altitude_m']:.1f}", "52.5/52.7",
            "$52.5$\\,m on average for video in the clustered shelter and $52.7$\\,m on the busiest tile")
    av = [x.loc[(sc, "voice"), "mean_altitude_m"] for sc in ("urban_broad", "milan_p100")]
    c.value("altitude, voice", f"{min(av):.0f}-{max(av):.0f}", "110-114", "and $110$ to $114$\\,m for voice")
    tv = [x.loc[(sc, "voice"), "time_s"] for sc in ("urban_broad", "milan_p100")]
    c.value("pointing time, voice / busiest video (s)", f"{np.mean(tv):.0f}/{x.loc[('milan_p100', 'video'), 'time_s']:.0f}",
            "23/243", "takes $23$\\,s per layout on average for voice and $243$\\,s for video on the busiest tile")
    c.snippet("greedy column is Table IV", "so its greedy column is that table's entry")


def check_joint_e2e(c: Checker) -> None:
    print("\n== Section sec:joint: end to end")
    e, r, rr = joint_e2e_runs(), joint_runs(), refine_runs()
    g = e[e["method"] == "greedy"].assign(key=e["snapshot"].fillna(-1).astype(int))
    t = rr[(rr["variant"] == "tether") & (rr["method"] == "greedy")]
    t = t.assign(key=t["layout"].str.extract(r"snap(\d+)")[0].fillna(-1).astype(int))
    m = g.merge(t, on=["scenario", "tier", "seed", "key", "n_drones"], suffixes=("", "_t4"))
    same = all((m[f"served_e2e_pct_tether_{x}"] - m[f"served_e2e_pct_{x}"]).abs().max() < 1e-9
               for x in ("mst_corner", "mst_central", "balanced"))
    c.value("greedy end to end = Table IV runs: shares", "yes" if same and len(m) > 10000 else "no", "yes")
    # relays: the balanced tree can break a load tie differently (float noise), by one relay
    dr = {x: (m[f"relays_tether_{x}"] - m[f"relays_{x}"]).abs() for x in ("mst_corner", "mst_central", "balanced")}
    c.value("greedy end to end = Table IV runs: relays differ", f"{sum(int((v > 0).sum()) for v in dr.values())}/"
            f"{len(m)}, at most {max(int(v.max()) for v in dr.values())}", "20/19976, at most 5",
            "and the relay counts in all but $20$ of $19{,}976$ runs; these differ by up to five relays")
    vb = sum(int(((v > 0) & m["tier"].eq("video") & m["scenario"].isin(["milan_p075", "milan_p100"])).sum())
             for v in dr.values())
    c.value("relay differences in bound-capped video cells", str(vb), "18",
            "$18$ of them in video cells where the gateway bound binds")
    col, rel = "served_e2e_pct_tether_balanced", "relays_tether_balanced"

    def cell(sc: str, t: str, meth: str) -> pd.DataFrame:
        d = e[(e["scenario"] == sc) & (e["tier"] == t) & (e["method"] == meth)]
        return d[d["n_drones"].isin(paper_window(sc, t))] if meth == "greedy" else d

    # end-to-end fleet of each method: the smallest N whose mean end-to-end share reaches 95 % with some tree,
    # with that tree's relays; the cheapest (access + relays) over the trees, as in Table IV
    def e2e_fleet(sc, t, meth):
        m_ = cell(sc, t, meth).groupby("n_drones").mean(numeric_only=True)
        best = None
        for tp in ("balanced", "mst_central", "mst_corner"):
            ok = m_[m_[f"served_e2e_pct_tether_{tp}"] >= 95.0]
            if ok.empty:
                continue
            n = int(ok.index.min())
            cand = (n, float(ok.loc[n, f"relays_tether_{tp}"]))
            if best is None or sum(cand) < sum(best) - 1e-9:
                best = cand
        return best
    tot = {(sc, t, meth): e2e_fleet(sc, t, meth) for sc, t, _ in JOINT_ROWS for meth in ("greedy", "xyzt")}
    served = [(sc, t) for sc, t, _ in JOINT_ROWS if tot[(sc, t, "greedy")] is not None and tot[(sc, t, "xyzt")] is not None]
    c.value("cells served end to end (tether)", str(len(served)), "4", "in the four cells whose demand the gateway can carry")
    same = [(sc, t) for sc, t in served if all(tot[(sc, t, m)][0] == joint_nstar(r, sc, t, m)["n"] for m in ("greedy", "xyzt"))]
    c.value("end-to-end N* = access N* except clustered video", ",".join(sorted(f"{sc} {t}" for sc, t in set(served) - set(same))),
            "shelter_clustered video", "with its access fleet except in the clustered shelter")
    cs = [f"{tot[('shelter_clustered', 'video', m)][0]}/{joint_nstar(r, 'shelter_clustered', 'video', m)['n']:.0f}" for m in ("greedy", "xyzt")]
    c.value("clustered video e2e / access N* (greedy, searched)", " ".join(cs), "58/54 41/40",
            "video needs $58$ drones instead of $54$ (greedy) and $41$ instead of $40$ (searched)")
    red = [100 * (1 - sum(tot[(sc, t, "xyzt")]) / sum(tot[(sc, t, "greedy")])) for sc, t in served]
    c.value("total fleet reduction with relays (%)", f"{min(red):.0f}-{max(red):.0f}", "16-26",
            "the total fleet falls by $16$ to $26\\%$")
    c.snippet("conclusion: total saving with relays", "relays included, by $16$ to $26\\%$")
    f = lambda sc, t, meth: f"{tot[(sc, t, meth)][0]:.0f}+{tot[(sc, t, meth)][1]:.0f}"  # noqa: E731
    c.value("urban voice access + relays", f"{f('urban_broad', 'voice', 'xyzt')} vs {f('urban_broad', 'voice', 'greedy')}",
            "12+11 vs 16+15", "urban voice needs $12 + 11$ drones instead of $16 + 15$")
    c.value("busiest voice access + relays", f"{f('milan_p100', 'voice', 'xyzt')} vs {f('milan_p100', 'voice', 'greedy')}",
            "17+14 vs 21+16", "the busiest tile $17 + 14$ instead of $21 + 16$")
    c.value("shelter video access + relays", f"{f('shelter_clustered', 'video', 'xyzt')} vs "
            f"{f('shelter_clustered', 'video', 'greedy')}", "41+5 vs 58+3", "clustered-shelter video $41 + 5$ instead of $58 + 3$")
    vid = [(sc, t) for sc, t in served if t != "voice"]
    more = max(tot[(sc, t, "xyzt")][1] - tot[(sc, t, "greedy")][1] for sc, t in vid)
    c.value("video/broadband relays added, at most", f"{more:.0f}", "2", "at most two relays more")
    hv = [f"{cell(sc, 'video', meth).groupby('n_drones')[col].mean().loc[joint_nstar(r, sc, 'video', meth)['n']]:.1f}"
          for meth in ("greedy", "xyzt") for sc in ("milan_p075", "milan_p100")]
    c.value("p75 / busiest video at the gateway ceiling (greedy, searched)", "/".join(hv), "72.8/27.1/73.1/27.3",
            "at each method's access fleet the balanced tree delivers $72.8\\%$ and $27.1\\%$ (greedy) and $73.1\\%$ and $27.3\\%$ (searched)")
    bnd = [100 * gateway_bound_bps("tether", "video") / (demand().loc[sc, "active"] * TIER_RATE["video"]) for sc in ("milan_p075", "milan_p100")]
    c.value("bound shares p75 / busiest video", f"{bnd[0]:.1f}/{bnd[1]:.1f}", "73.6/27.5",
            "stays just below the bound of Proposition~\\ref{prop:bound}, $73.6\\%$ and $27.5\\%$, under every method")
    sat = max(np.ptp([cell(sc, t, meth).groupby("n_drones")["served_e2e_pct_tether_balanced"].mean().clip(upper=satellite_cap_pct(sc, t))
                      .loc[joint_nstar(r, sc, t, meth)["n"]] for meth in ("greedy", "xyz", "xyzt")]) for sc, t, _ in JOINT_ROWS)
    c.value("satellite: same share for every method (pp)", "yes" if sat < 0.05 else "no", "yes",
            "caps every method at the same share")


def check_joint_assoc(c: Checker) -> None:
    print("\n== Section sec:joint: nearest and strongest association")
    a, r = joint_e2e_runs("assoc"), joint_runs()
    c.value("assoc: own association reproduces the runs", "yes" if (a["served_pct"] - a["served_src_pct"]).abs().max() < 1e-7
            else "no", "yes")

    def nstar(sc: str, t: str, meth: str, col: str) -> float:
        d = a[(a["scenario"] == sc) & (a["tier"] == t) & (a["method"] == meth)]
        d = d[d["n_drones"].isin(paper_window(sc, t))] if meth == "greedy" else d
        return bootstrap_nstar(d, sc, t, col, meth)[0]

    st = {(sc, t): nstar(sc, t, "xyzt", "served_strongest_pct") for sc, t, _ in JOINT_ROWS}
    own = {(sc, t): joint_nstar(r, sc, t, "xyzt")["n"] for sc, t, _ in JOINT_ROWS}
    same = []
    for sc, t, _ in JOINT_ROWS:
        d = a[(a["scenario"] == sc) & (a["tier"] == t) & (a["method"] == "xyzt") & (a["n_drones"] == own[(sc, t)])]
        same.append(d["strongest_is_assoc_pct"].mean())
    c.value("strongest = chosen association (% of points)", f"{min(same):.0f}-{max(same):.0f}", "97-99",
            "agrees with the strongest drone for $97$ to $99\\%$ of points")
    grown = {k: st[k] - own[k] for k in st}
    c.value("pointing fleets under strongest: unchanged cells / max growth",
            f"{sum(v == 0 for v in grown.values())}/{max(grown.values()):.0f}", "4/5",
            "unchanged in four cells and grow by at most five drones in two")
    c.value("strongest: shelter video, p25 broadband",
            f"{own[('shelter_clustered', 'video')]:.0f}->{st[('shelter_clustered', 'video')]:.0f}; "
            f"{own[('milan_p025', 'broadband')]:.0f}->{st[('milan_p025', 'broadband')]:.0f}", "40->45; 82->84",
            "(clustered shelter, video: $40 \\to 45$; Milan p25, broadband: $82 \\to 84$)")
    g = {(sc, t): joint_nstar(r, sc, t, "greedy")["n"] for sc, t, _ in JOINT_ROWS}
    lb = {(sc, t): lower_bound_drones(sc, t) for sc, t, _ in JOINT_ROWS}
    red = [100 * (1 - st[k] / g[k]) for k in st]
    gap = [100 * (g[k] - st[k]) / (g[k] - lb[k]) for k in st]
    c.value("strongest: saving / gap closed (%)", f"{min(red):.0f}-{max(red):.0f}/{min(gap):.0f}-{max(gap):.0f}",
            "15-42/26-66", "the saving over greedy stays $15$ to $42\\%$ and closes $26$ to $66\\%$ of the gap")
    near = {(sc, t): nstar(sc, t, "xyzt", "served_nearest_pct") for sc, t, _ in JOINT_ROWS}
    voice = all(near[(sc, "voice")] == g[(sc, "voice")] for sc in ("urban_broad", "milan_p100"))
    c.value("nearest: voice pointing fleets = greedy", "yes" if voice else "no", "yes",
            "the voice fleets return to the greedy size")
    top = {k: a[(a["scenario"] == k[0]) & (a["tier"] == k[1]) & (a["method"] == "xyzt")]["n_drones"].max() for k in near}
    miss = all(np.isinf(near[k]) and top[k] >= math.ceil(1.15 * g[k]) for k in near if k[1] != "voice")
    c.value("nearest: video/broadband pointing never reaches 95 % up to 1.15 x greedy", "yes" if miss else "no", "yes",
            "do not reach $95\\%$ with up to $1.15$ times the greedy fleet")


def restart_nstar(r: pd.DataFrame, sc: str, t: str, meth: str) -> dict:
    """N* of meth with each unit at the best of its first start (joint_runs) and the restarts of
    stage multi (gmulti for greedy), read on the N the restarts were run at; below: the window's first
    N already reaches 95 %, so it does not hold N*."""
    x = joint_runs(("gmulti",) if meth == "greedy" else ("multi",))
    x = x[(x["scenario"] == sc) & (x["tier"] == t) & (x["method"] == meth)]
    first = r[(r["scenario"] == sc) & (r["tier"] == t) & (r["method"] == meth) & r["n_drones"].isin(x["n_drones"])]
    keys = ["scenario", "tier", "method", "unit", "snapshot", "seed", "n_drones"]
    both = pd.concat([first, x])[keys + ["served_pct"]]
    best = both.fillna({"snapshot": -1}).groupby(keys, as_index=False)["served_pct"].max()
    n, lo, hi = bootstrap_nstar(best, sc, t, "served_pct", meth)
    mean = best.groupby("n_drones")["served_pct"].mean()
    return {"n": n, "lo": lo, "hi": hi, "below": bool(mean.iloc[0] >= 95.0)}


def paired_saving(r: pd.DataFrame, sc: str, t: str, a: str = "greedy", b: str = "xyzt", n_boot: int = 2000,
                  seed: int = 0) -> tuple[float, float, float]:
    """N*(a) - N*(b) with a paired 95 % percentile interval: units resampled jointly for both methods."""
    mats = {}
    for m in (a, b):
        rr = r
        if m == "greedy":
            rr = r[~((r["scenario"] == sc) & (r["tier"] == t) & (r["method"] == "greedy")
                     & ~r["n_drones"].isin(paper_window(sc, t)))]
        d = rr[(rr["scenario"] == sc) & (rr["tier"] == t) & (rr["method"] == m)]
        mats[m] = d.pivot_table(index="unit", columns="n_drones", values="served_pct", aggfunc="mean")
    units = sorted(set(mats[a].index) & set(mats[b].index))
    w = np.random.default_rng(seed).multinomial(len(units), np.full(len(units), 1 / len(units)), size=n_boot) / len(units)

    def first(x: np.ndarray, ns: np.ndarray) -> np.ndarray:
        ok = x >= 95.0
        return np.where(ok.any(axis=1), ns[np.argmax(ok, axis=1)], np.inf)
    point, boot = {}, {}
    for m, x in mats.items():
        x = x.loc[units]
        ns, xv = x.columns.to_numpy(), x.to_numpy()
        point[m] = first(xv.mean(axis=0, keepdims=True), ns)[0]
        boot[m] = first(w @ xv, ns)
    lo, hi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
    return float(point[a] - point[b]), float(lo), float(hi)


def check_gateway_text(c: Checker) -> None:
    import journal_data as jd
    c.value("satellite down / up (Mb/s)", f"{jd.SAT_DOWN_BPS / 1e6:.0f}/{jd.SAT_UP_BPS / 1e6:.0f}", "100/20",
            "We give the terminal $100$\\,Mb/s down and $20$\\,Mb/s up")
    c.snippet("Ookla upload facts", "its median upload reached $20$\\,Mb/s in $22$ states, and the national median upload was $21.46$\\,Mb/s in the first quarter of 2026")


def check_paired(c: Checker) -> None:
    print("\n== Section sec:joint: paired intervals of the saving")
    r = joint_runs()
    res = {(sc, t): paired_saving(r, sc, t) for sc, t, _ in JOINT_ROWS}
    c.value("paired interval excludes zero in every cell", "yes" if all(v[1] > 0 for v in res.values()) else "no", "yes",
            "the paired $95\\%$ interval of the saving excludes zero in every cell")
    f = lambda k: f"{res[k][0]:.0f} [{res[k][1]:.0f}, {res[k][2]:.0f}]"  # noqa: E731
    c.value("paired saving urban voice", f(("urban_broad", "voice")), "4 [4, 5]", "from $4$ $[4, 5]$ drones for urban voice")
    c.value("paired saving busiest video", f(("milan_p100", "video")), "113 [98, 128]",
            "to $113$ $[98, 128]$ for video on the busiest tile")


def check_joint_restarts(c: Checker) -> None:
    print("\n== Section sec:joint: restarts and tilt cap")
    r = joint_runs()
    lb = {(sc, t): lower_bound_drones(sc, t) for sc, t, _ in JOINT_ROWS}
    one = {(sc, t): {m: joint_nstar(r, sc, t, m)["n"] for m in ("greedy", "xyzt")} for sc, t, _ in JOINT_ROWS}
    best = {(sc, t): {m: restart_nstar(r, sc, t, m) for m in ("greedy", "xyzt")} for sc, t, _ in JOINT_ROWS}
    for k, q in best.items():
        if q["greedy"]["below"] or q["xyzt"]["below"]:
            c.value(f"restart window holds N* ({k[0]} {k[1]})", "no", "yes")
    g = {k: one[k]["greedy"] - q["greedy"]["n"] for k, q in best.items()}
    x = {k: one[k]["xyzt"] - q["xyzt"]["n"] for k, q in best.items()}
    p100v, p25b = ("milan_p100", "video"), ("milan_p025", "broadband")
    c.value("restarts: greedy gains at most (drones)", f"{max(g.values()):.0f}", "30", "greedy needs up to $30$ drones fewer")
    c.value("restarts: greedy busiest video, p25 broadband",
            f"{one[p100v]['greedy']:.0f}->{best[p100v]['greedy']['n']:.0f}; {one[p25b]['greedy']:.0f}->{best[p25b]['greedy']['n']:.0f}",
            "270->240; 105->96", "(busiest tile, video: $270 \\to 240$; $25$th-percentile tile, broadband: $105 \\to 96$)")
    c.value("restarts: search gains (drones), busiest video",
            f"{min(x.values()):.0f}-{max(x.values()):.0f}; {one[p100v]['xyzt']:.0f}->{best[p100v]['xyzt']['n']:.0f}",
            "1-5; 157->152", "and the search one to five fewer ($157 \\to 152$)")
    red = {k: 100 * (1 - q["xyzt"]["n"] / q["greedy"]["n"]) for k, q in best.items()}
    gap = {k: 100 * (q["greedy"]["n"] - q["xyzt"]["n"]) / (q["greedy"]["n"] - lb[k]) for k, q in best.items()}
    c.value("restarts: saving / gap closed (%)", f"{min(red.values()):.0f}-{max(red.values()):.0f}/"
            f"{min(gap.values()):.0f}-{max(gap.values()):.0f}", "15-37/33-62",
            "The saving is then $15$ to $37\\%$ and closes $33$ to $62\\%$ of the gap to the bound.")
    for snip in ("cuts the fleet by $15$ to $37\\%$ on six cells, even against the best of three K-means starts",
                 "placement saves $15$ to $37\\%$ of the access fleet",
                 "a searched placement needs $15$ to $37\\%$ fewer drones",
                 "cuts the access fleet by $15$ to $37\\%$ on six cells that span the tiers"):
        c.value("restart saving quoted", f"{min(red.values()):.0f}-{max(red.values()):.0f}", "15-37", snip)
    tl = pd.concat(pd.read_csv(f) for f in (JOINT / "tilt").rglob("*.csv"))
    tl["unit"] = np.where(tl["scenario"].str.startswith("milan"),
                          "snap" + tl["snapshot"].fillna(-1).astype(int).map("{:02d}".format), tl["seed"].astype(int).astype(str))
    ns = {cap: {(sc, t): bootstrap_nstar(tl[tl["tilt_max_deg"] == cap], sc, t, "served_pct", "xyzt")[0]
                for sc, t, _ in JOINT_ROWS} for cap in (30.0, 45.0, 60.0)}
    c.value("tilt cap: max tilt within the cap", "yes" if all(
        tl[tl["tilt_max_deg"] == cap]["max_tilt_deg"].max() <= cap + 1e-6 for cap in ns) else "no", "yes")
    same = all(ns[cap][k] == one[k]["xyzt"] for cap in (45.0, 60.0) for k in one)
    c.value("tilt cap 45/60: every N* unchanged", "yes" if same else "no", "yes",
            "capping the tilt at $45^\\circ$ or $60^\\circ$ instead of $80^\\circ$ leaves every $N^\\star$ unchanged")
    moved = {k: (one[k]["xyzt"], v) for k, v in ns[30.0].items() if v != one[k]["xyzt"]}
    c.value("tilt cap 30: cells that change", "; ".join(f"{k[0]} {k[1]} {a:.0f}->{b:.0f}" for k, (a, b) in moved.items()),
            "urban_broad voice 12->14", "at $30^\\circ$ only urban voice needs two more drones ($12 \\to 14$)")


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
    check_joint(c, cur)
    check_joint_e2e(c)
    check_joint_assoc(c)
    check_joint_restarts(c)
    check_paired(c)
    check_gateway_text(c)
    print(f"\n{c.n} checks, {len(c.bad)} problems")
    for b in c.bad:
        print("  " + b)
    return 1 if c.bad else 0


if __name__ == "__main__":
    sys.exit(main())
