"""Check every number quoted in the swarm paper (paper/main_swarm.tex) against results/.

Each check recomputes a value, formats it as the paper prints it, and requires that exact
(whitespace-normalised) snippet in the .tex source; table rows are rebuilt whole. Exits non-zero on
any mismatch. Data: scripts/swarm_data.py (results/swarm via analyze_swarm, results/llm via analyze_llm).

Conventions (as in the paper)
-----------------------------
- gap: served share minus the best-known central deployment of the layout (stages cstrong, rstrong), pp of all
  demand; fixed-N stages from the paper's deployment with sequential play unless stated.
- [a, b]: 95 % cluster bootstrap over the journal's units (a Milan snapshot or a synthetic seed).
- Self-sizing: N/N95 against the fstrong frontier; events = lands + calls per run.
- LLM change: served share after 8 rounds minus at the start, mean of 6 units (3 layouts x 2 repeats).

python scripts/swarm_numbers.py            # check
python scripts/swarm_numbers.py --rows     # print the table rows (to paste into the paper)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import swarm_data as D  # noqa: E402

TEX = ROOT / "paper" / "main_swarm.tex"
P = D.pick
SCEN_TEX = {"urban_broad": "Urban", "milan_p100": "Milan", "shelter_clustered": "Shelter"}
UTIL_TEX = {"selfish": "Selfish", "window150": r"Window, $\rho=150$\,m", "window300": r"Window, $\rho=300$\,m",
            "window600": r"Window, $\rho=600$\,m", "window1200": r"Window, $\rho=1200$\,m", "global": "Team (global)"}


def f(x: float, d: int = 1, sign: bool = True) -> str:
    """A number as the paper prints it, in math mode: $-13.0$, $+4.5$, $0.0$."""
    if not np.isfinite(x):
        return "--"
    v = round(float(x), d)
    if v == 0:
        return f"${0:.{d}f}$"
    return f"${v:+.{d}f}$" if sign else f"${v:.{d}f}$"


def ci(s: dict, d: int = 1) -> str:
    return f"{f(s['mean'], d)}\\ci{{{f(s['lo'], d)[1:-1]}}}{{{f(s['hi'], d)[1:-1]}}}"


def pct(x: float) -> str:
    return f"${x:.0f}\\%$"


# ── tables ───────────────────────────────────────────────────────────


def lkey(r: dict) -> tuple:
    return r["scenario"], r["tier"], r["snapshot"], r["seed"], r["n_drones"]


def vs_team(rows: list[dict], team: list[dict]) -> dict:
    """Paired gap minus team play's on the same layouts: mean and 95 % cluster-bootstrap interval."""
    t = {lkey(r): r["gap"] for r in team}
    d = np.array([r["gap"] - t[lkey(r)] for r in rows])
    lo, hi = D.A.boot_ci(rows, d)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi}


def ladder_label(u: str) -> str:
    return {"selfish": "Selfish", "global": "Team (global)"}.get(u, f"$\\rho={u.removeprefix('window')}$\\,m")


def ladder_rows() -> list[str]:
    """Table tab:ladder: gap by utility, pointing (p1, fine) and nadir (p2, fine), pooled, per scenario and
    (pointing) paired against team play; windows at every radius of D.RADII."""
    point, nadir = D.sweep("point"), D.sweep("nadir")
    team = P(point, label="global")
    out = []
    for u in D.LADDER_ORDER:
        cells = []
        for k, rows in enumerate((P(point, label=u), P(nadir, label=u))):
            cells.append(ci(D.gap_stats(rows)))
            cells += [f(D.gap_stats(P(rows, scenario=sc))["mean"]) for sc in D.SCENARIOS]
            if k == 0:
                cells.append("--" if u == "global" else f(vs_team(rows, team)["mean"], 2))
        conv = D.gap_stats(P(point, label=u))["conv"]
        out.append(f"{ladder_label(u)} & " + " & ".join(cells) + f" & {pct(conv)} \\\\")
    return out


def info_rows() -> list[str]:
    """Table tab:info: exact information with window rho = r (p1, fine), against team play with limited information r
    (p3, fine), every radius of D.RADII; selfish play with limited information only at p3's radii."""
    pt, inf, p3 = D.sweep("point"), D.sweep("info"), D.fixed("p3")
    out = []
    for r in D.RADII:
        w = D.gap_stats(P(pt, label=f"window{r}"))
        gi = D.gap_stats(P(inf, info_m=f"{r:.1f}"))
        srows = P(p3, start="paper", label="selfish", info_m=f"{r:.1f}")
        sel = f(D.gap_stats(srows)["mean"]) if srows else "--"
        out.append(f"${r}$ & {ci(w)} & {pct(w['conv'])} & {ci(gi)} & {pct(gi['conv'])} & {sel} \\\\")
    return out


def dyn_rows() -> list[str]:
    """Table tab:dyn: utility x dynamics from the paper's deployment."""
    p1 = D.fixed("p1")
    out = []
    for u in ("global", "window1200", "selfish"):
        cells = []
        for d in ("seq", "sim", "sim50"):
            s = D.gap_stats(P(p1, start="paper", dynamics=d, label=u))
            cells += [f(s["mean"]), pct(s["conv"])]
        out.append(f"{UTIL_TEX[u]} & " + " & ".join(cells) + " \\\\")
    return out


SIZE_CASES = [  # (label in the table, stage, start, utility, rules)
    ("Per cell $(0.05, 0.25)$", "p4b", "depot1", "global", "x_land0.05_call0.25"),
    ("Area $(0.25, 0.5)$", "p4c", "depot1", "global", "x_land0.25_call0.5_area1200"),
    ("Area + check", "p4d", "depot1", "global", "x_land0.25_call0.5_area1200_chk"),
    ("Area + hyst.", "p4e", "depot1", "global", D.FINAL_RULE),
    ("Area + hyst., from $1.3N_{95}$", "p4e", "over", "global", D.FINAL_RULE),
    ("Area + hyst., selfish", "p4e", "depot1", "selfish", D.FINAL_RULE),
]


def size_rows() -> list[str]:
    """Table tab:size: self-sizing per scenario."""
    out = []
    for sc in D.SCENARIOS:
        out.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{SCEN_TEX[sc]}}}}} \\\\")
        for lab, st, start, u, rule in SIZE_CASES:
            s = D.size_stats(P(D.sized(st), scenario=sc, start=start, label=u, rules=rule))
            out.append(f"{lab} & {f(s['ratio'], 2, False)} & {f(s['served'], 1, False)} & {f(s['gap'])} & "
                       f"{f(s['excess'])} & {f(s['events'], 0, False)} & {pct(s['conv'])} \\\\")
    return out


LLM_MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
LLM_TEX = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B",
           "qwen3.5-9b": "Qwen3.5-9B"}


def llm_rows() -> list[str]:
    """Table tab:llm: change by model and framing, with cost per decision."""
    out = []
    cases = [(LLM_TEX[m], "l1", m, ("selfish", "area", "network")) for m in LLM_MODELS] + [
        ("Qwen3.5-9B, thinking", "l1", "qwen3.5-9b-think", ("selfish", "area", "network")),
        ("Qwen3.5-9B, gated", "l1gate", "qwen3.5-9b", (None, "area+gate", None))]
    for label, st, m, frs in cases:
        ch = [f(D.llm_change(m, fr, st).mean()) if fr else "--" for fr in frs]
        t = D.llm_turns(st, m)
        out.append(f"{label} & " + " & ".join(ch) + f" & {f(t['move'], 0, False)} & "
                   f"{f(t['checked'], 0, False)} & {f(t['errors'], 1, False)} & "
                   f"{f(t['prompt'] / 1000, 1, False)} & {f(t['completion'] / 1000, 2, False)} & "
                   f"{f(t['seconds'], 1, False)} \\\\")
    br_s, br_g = D.llm_change("br-selfish", "selfish").mean(), D.llm_change("br-global", "global").mean()
    out.append(f"Best response & {f(br_s)} & -- & {f(br_g)} & & & & & & \\\\")
    pol = [D.llm_turns("l1pol", "pol-rand6")]
    out.append(f"Random what-ifs (rand6) & {f(D.llm_change('pol-rand6', 'selfish').mean())} & -- & "
               f"{f(D.llm_change('pol-rand6', 'network').mean())} & {f(pol[0]['move'], 0, False)} & "
               f"{f(pol[0]['checked'], 0, False)} & & & & \\\\")
    out.append(f"Fly to unserved (goto) & -- & -- & {f(D.llm_change('pol-goto', 'network').mean())} & & & & & & \\\\")
    return out


def real() -> dict:
    """Gap stats of p5speed / p5noise by (utility, condition), and of p1 sim50 on the same core layouts ('none')."""
    sp, no = D.fixed("p5speed"), D.fixed("p5noise")
    lay = lambda r: (r["scenario"], r["snapshot"], r["seed"], r["n_drones"])
    core = {lay(r) for r in sp}
    base = [r for r in P(D.fixed("p1"), start="paper", dynamics="sim50") if lay(r) in core]
    out = {}
    for u in ("global", "window1200", "selfish"):
        out[(u, "none")] = P(base, label=u)
        for v in ("150.0", "600.0"):
            out[(u, v)] = P(sp, label=u, max_step_m=v)
        for v in ("mild", "harsh"):
            out[(u, v)] = P(no, label=u, pred_err=v)
    assert len(core) == 23 and all(len(rs) == 23 for rs in out.values())
    return out


REAL_COLS = ("none", "150.0", "600.0", "mild", "harsh")


def real_rows() -> list[str]:
    """Table tab:real: speed limits and prediction errors on the core layouts, inertia 0.5."""
    r = real()
    return [f"{UTIL_TEX[u]} & " + " & ".join(f(D.gap_stats(r[(u, c)])["mean"]) for c in REAL_COLS) + " \\\\"
            for u in ("global", "window1200", "selfish")]


TABLES = {"tab:ladder": ladder_rows, "tab:info": info_rows, "tab:dyn": dyn_rows, "tab:real": real_rows,
          "tab:size": size_rows, "tab:llm": llm_rows}


# ── checks ───────────────────────────────────────────────────────────


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
        print(f"  {'ok' if ok else 'MISSING':8s} {label:44s} {s[:100]}")


def check_tables(c: Checker) -> None:
    for name, fn in TABLES.items():
        print(f"\n{name}")
        for i, row in enumerate(fn()):
            c.snippet(f"{name} row {i + 1}", row)


def n(x: float, d: int = 1) -> str:
    """A bare number as in running text: 13.0, 0.04, 1117."""
    return f"{round(float(x), d):.{d}f}"


def check_text(c: Checker) -> None:
    import math

    import run_swarm
    from dronecomm import joint, swarm
    p1, p2, p3, p4a = D.fixed("p1"), D.fixed("p2"), D.fixed("p3"), D.fixed("p4a")
    gs = lambda rows: D.gap_stats(rows)
    seq = lambda **kw: gs(P(p1, start="paper", dynamics="seq", **kw))
    nad = lambda **kw: gs(P(p2, **kw))

    # setup and model
    phys = D._physics()
    c.snippet("tx power, band", f"transmits ${10 * math.log10(phys.p_tx_w) + 30:.0f}$\\,dBm over a "
                                f"${phys.bandwidth_hz / 1e6:.0f}$\\,MHz band")
    c.snippet("antenna", f"${phys.g_max_dbi:.0f}$\\,dBi peak gain, a ${math.degrees(phys.beamwidth_rad):.1f}^\\circ$ "
                         f"beamwidth and a ${phys.sla_db:.0f}$\\,dB sidelobe floor")
    c.snippet("altitudes", f"Drones fly between ${phys.z_min:.0f}$ and ${phys.z_max:.0f}$\\,m")
    c.snippet("tilt cap", f"with tilts up to ${math.degrees(math.atan(joint.T_MAX)):.0f}^\\circ$ from nadir")
    c.snippet("rounds", f"Runs stop at ${run_swarm.MAX_ROUNDS}$ rounds with a fixed fleet and "
                        f"${run_swarm.SIZE_MAX_ROUNDS}$ when")
    c.snippet("brute grid", f"grid search, wider than the agents' own, of $50$\\,m in position, every altitude, nadir and "
                            f"{['', '', '', '', 'four'][len(swarm.BRUTE_TILTS_DEG)]} tilts at "
                            f"{['', '', '', '', '', '', '', '', 'eight'][swarm.BRUTE_AZIMUTHS]} azimuths, each of the "
                            f"{['', '', '', '', '', '', '', '', '', '', 'ten'][swarm.BRUTE_POLISH]} best grid points")
    c.snippet("beacon grid", f"the two heaviest cells of a ${swarm.BEACON_CELL_M:.0f}$\\,m grid, and every agent's "
                             f"search also tries the {['', '', '', '', '', '', 'six'][swarm.BEACONS_TRIED]} heaviest")
    kap = {sc: float(P(D.sized("p4e"), scenario=sc)[0]["kappa"]) for sc in D.SCENARIOS}
    c.snippet("kappa", f"(urban voice $\\kappa={n(kap['urban_broad'], 3)}$, Milan ${n(kap['milan_p100'], 3)}$, "
                       f"shelter video ${n(kap['shelter_clustered'], 3)}$)")
    cs = D.A.load("cstrong")
    rs = {D.A._key(r): float(r["served_strong_pct"]) for r in D.A.load("rstrong")}      # the journal's restarts, polished
    shift = np.mean([max(float(r["served_strong_pct"]), rs.get(D.A._key(r), -math.inf)) - float(r["served_rescored_pct"])
                     for r in cs])
    agent = sum(r["strong_source"] == "agent" and float(r["served_strong_pct"]) >= rs.get(D.A._key(r), -math.inf)
                for r in cs)
    c.snippet("reference shift", f"raises the reference by ${n(shift)}$\\,pp on average")
    c.snippet("reference from agents", f"in ${agent}$ of the ${len(cs)}$ layouts the best deployment came from an agent run")
    c.snippet("limitations: agent refs", f"${agent}$ of ${len(cs)}$ references came from agent runs")
    nfix = len(p1) + len(p2) + len(p3) + len(p4a)
    nsize = sum(len(D.sized(s)) for s in ("p4b", "p4c", "p4d", "p4e"))
    nfine = len(D.fixed("fine"))
    c.snippet("run counts", f"${nfix // 1000}{{,}}{nfix % 1000:03d}$ fixed-$N$ runs, "
                            f"${nfine // 1000}{{,}}{nfine % 1000:03d}$ more that fill in the radius sweeps")
    c.snippet("self-sizing run count", f"and ${nsize // 1000}{{,}}{nsize % 1000:03d}$ self-sizing runs")
    units = {D.A.unit(r) for r in p1}
    milan = np.mean([r["scenario"] == "milan_p100" for r in P(p1, start="paper", dynamics="seq", label="global")])
    c.snippet("units, Milan weight", f"over the ${len(units)}$ units, and pooled means weight layouts equally, so Milan "
                                     f"carries ${100 * milan:.0f}\\%$")
    lay = len(P(p1, start="paper", dynamics="seq", label="global"))
    nsl = len(P(D.sized("p4e"), start="depot1", label="global", rules=D.FINAL_RULE))
    c.snippet("layout counts", f"This gives ${lay}$ fixed-$N$ layouts and ${nsl}$ self-sizing layouts")

    # abstract
    self_, team = seq(label="selfish"), seq(label="global")
    c.snippet("abstract: layouts", f"Across ${lay}$ layouts")
    c.snippet("abstract: selfish", f"Selfish drones end ${n(-self_['mean'])}$\\,pp of demand below it")
    c.snippet("abstract: team", f"team play ends ${n(-team['mean'])}$\\,pp below")
    r0, r30, r60 = D.coupling_radius(120, 0), D.coupling_radius(120, 30), D.coupling_radius(120, 60)
    gain = {sc: seq(label="global", scenario=sc)["mean"] - nad(label="global", scenario=sc)["mean"] for sc in D.SPREAD}
    cost = {(u, sc): nad(label=u, scenario=sc)["mean"] - seq(label=u, scenario=sc)["mean"]
            for u in ("window150", "window300", "selfish") for sc in D.SPREAD}
    c.snippet("abstract: pointing", f"pointing adds ${min(gain.values()):.0f}$ to ${max(gain.values()):.0f}$\\,pp to "
                                    f"team play but costs local utilities up to ${max(cost.values()):.0f}$\\,pp")
    c.snippet("conclusion: pointing", f"a steerable antenna that adds ${min(gain.values()):.0f}$ to "
                                      f"${max(gain.values()):.0f}$\\,pp to team play")
    fin = {sc: D.size_stats(P(D.sized("p4e"), scenario=sc, start="depot1", label="global", rules=D.FINAL_RULE))
           for sc in D.SCENARIOS}
    sfin = {sc: D.size_stats(P(D.sized("p4e"), scenario=sc, start="depot1", label="selfish", rules=D.FINAL_RULE))
            for sc in D.SCENARIOS}
    c.snippet("abstract: self-sizing", f"grow from one drone to ${n(fin['urban_broad']['ratio'], 2)}$ to "
                                       f"${n(fin['milan_p100']['ratio'], 2)}$ times the central")
    c.snippet("abstract: selfish fleets", f"selfish fleets grow to ${n(sfin['urban_broad']['ratio'], 2)}$ to "
                                          f"${n(sfin['milan_p100']['ratio'], 2)}$ times it")

    # Section V: price of locality
    c.snippet("V: selfish vs team", f"they end {f(self_['mean'])}\\,pp below the central reference, against "
                                    f"{f(team['mean'])}\\,pp for team play")
    by = {sc: seq(label="selfish", scenario=sc)["mean"] for sc in D.SCENARIOS}
    c.snippet("V: selfish by scenario", f"The loss is {f(by['urban_broad'])}\\,pp on urban demand, "
                                        f"{f(by['milan_p100'])}\\,pp in Milan and {f(by['shelter_clustered'])}\\,pp")
    cself, cteam = gs(P(p1, start="central", dynamics="seq", label="selfish")), \
        gs(P(p1, start="central", dynamics="seq", label="global"))
    c.snippet("V: from central", f"walks away from it and ends {f(cself['mean'])}\\,pp below")
    c.snippet("V: team from central", f"team play stays within {f(cteam['mean'])}\\,pp")
    conv = lambda sc, nd: seq(label="selfish", scenario=sc, n_drones=nd)["conv"]
    scarce = [conv("urban_broad", "8"), conv("urban_broad", "11"), conv("milan_p100", "14")]
    assert max(scarce) - min(scarce) < 1.0, scarce
    c.snippet("V: selfish converges when scarce", f"(${np.mean(scarce):.0f}\\%$ of runs converge on urban $N=8$, $11$ "
                                                  f"and Milan $N=14$)")
    c.snippet("V: not with spare drones", f"(${conv('urban_broad', '16'):.0f}\\%$ on urban $N=16$, "
                                          f"${conv('milan_p100', '21'):.0f}\\%$ on Milan $N=21$)")
    c.snippet("V: team within 1 pp", f"within $1$\\,pp of it in ${team['within1']:.0f}\\%$ of layouts")
    b = D.brcheck()
    med = lambda k: np.median(b[k]["conv"])
    c.snippet("V: epsilon", f"median $\\epsilon$ of ${n(med(('xyzt', 'global')), 2)}$\\,pp for team play and "
                            f"${n(med(('xyzt', 'selfish')), 2)}$\\,pp for selfish play with pointing (maxima "
                            f"${n(b[('xyzt', 'global')]['conv'].max(), 2)}$ and "
                            f"${n(b[('xyzt', 'selfish')]['conv'].max(), 2)}$\\,pp), and at most "
                            f"${n(max(b[(p, u)]['conv'].max() for p, u in b if p == 'nadir' and len(b[(p, u)]['conv'])), 2)}$"
                            f"\\,pp with nadir beams")
    c.snippet("V: epsilon open windows", f"an agent can still gain a median ${n(np.median(b[('xyzt', 'window300')]['open']))}$ "
                                         f"and ${n(np.median(b[('xyzt', 'window600')]['open']))}$\\,pp")
    w = [b[('xyzt', u)] for u in ("window300", "window600")]
    c.snippet("V: windows rarely converge", f"rarely converge (${len(w[0]['conv'])}$ and ${len(w[1]['conv'])}$ of "
                                            f"${len(w[0]['conv']) + len(w[0]['open'])}$ runs)")
    c.snippet("IV: core layouts", f"a core subset of ${len(real()[('global', 'none')])}$ layouts")
    c.snippet("V: 12 pp between", f"small next to the ${team['mean'] - self_['mean']:.0f}$\\,pp between the utilities")

    # Section VI: radius
    c.snippet("VI: coupling radii", f"falls below the noise floor ${r0:.0f}$\\,m away, about one cell")
    c.snippet("VI: tilted radii", f"stays above noise to ${r30:.0f}$\\,m, and tilted $60^\\circ$ to ${r60:.0f}$\\,m")
    cells = {(r["scenario"], r["n_drones"]) for r in p1}
    wdiff = max(abs(seq(label="window1200", scenario=sc, n_drones=nd)["mean"]
                    - seq(label="global", scenario=sc, n_drones=nd)["mean"]) for sc, nd in cells)
    c.snippet("VI: 1200 matches cell by cell", f"matches team play cell by cell, within ${n(max(wdiff, 0.05))}$\\,pp")
    c.snippet("VI: 300 m on spread", f"(${n(seq(label='window300', scenario='urban_broad')['mean'])}$ and "
                                     f"${n(seq(label='window300', scenario='milan_p100')['mean'])}$\\,pp at $300$\\,m, against "
                                     f"${n(seq(label='selfish', scenario='urban_broad')['mean'])}$ and "
                                     f"${n(seq(label='selfish', scenario='milan_p100')['mean'])}$\\,pp selfish)")
    assert all(seq(label=u, scenario=sc)["conv"] == 0 for u in ("window150", "window300") for sc in D.SPREAD)
    pt = D.sweep("point")
    tm = P(pt, label="global")
    win = lambda r, **kw: gs(P(pt, label=f"window{r}", **kw))
    vt = lambda r: vs_team(P(pt, label=f"window{r}"), tm)
    rec = (win(500)["mean"] - seq(label="selfish")["mean"]) / (team["mean"] - seq(label="selfish")["mean"])
    assert 0.4 < rec < 0.6, rec                           # "about half of the pooled loss"
    c.snippet("VI: 500 half", f"A $500$\\,m window recovers about half of the pooled loss ({f(win(500)['mean'])}\\,pp)")
    assert vt(700)["mean"] < -1.0 <= vt(800)["mean"] and vt(800)["mean"] < -0.25 <= vt(900)["mean"]
    assert vt(1100)["lo"] < 0 < vt(1100)["hi"] and vt(1000)["hi"] < 0
    c.snippet("VI: 800 within 1 pp", f"From $800$\\,m the window is within $1$\\,pp of team play ({ci(vt(800), 2)}\\,pp, paired)")
    assert round(vt(900)["mean"], 2) == -0.25
    c.snippet("VI: 900 about 0.25", f"and at $900$\\,m about $0.25$\\,pp below it ({ci(vt(900), 2)}\\,pp)")
    c.snippet("VI: 1100 zero", f"At $1100$\\,m the paired difference is indistinguishable from zero ({ci(vt(1100), 2)}\\,pp)")
    import swarm_figures as F
    tgs = F._tilts(D.SPREAD, "global")
    c.snippet("VI: 30 deg comparable, team tilts", f"comparable to the ${r30:.0f}$\\,m coupling radius of a beam tilted "
                                                  f"$30^\\circ$, but the correspondence is all we claim: team agents do not "
                                                  f"tilt uniformly, keeping ${100 * np.mean(tgs <= 15):.0f}\\%$ of their beams "
                                                  f"within $15^\\circ$ of nadir and tilting ${100 * np.mean(tgs > 45):.0f}\\%$ "
                                                  f"beyond $45^\\circ$")
    sconv = lambda r: win(r, scenario=D.SPREAD)["conv"]
    assert all(sconv(r) == 0 for r in D.RADII if r <= 300) and sconv(400) < 1 and min(sconv(r) for r in (1000, 1100, 1200)) >= 97.5
    assert min(sconv(r) for r in D.RADII if r >= 1000) >= round(sconv(1000)) - 0.5
    c.snippet("VI: convergence curve", f"no run with $\\rho\\le300$\\,m converges, ${sconv(700):.0f}\\%$ do at $700$\\,m "
                                       f"and ${sconv(800):.0f}\\%$ at $800$\\,m, and ${sconv(1000):.0f}\\%$ or more from $1000$\\,m")
    unconv = lambda rs: [r for r in P(pt, label=tuple(f"window{x}" for x in rs), scenario=D.SPREAD) if r["converged"] == "0"]
    c.snippet("VI: window cycling", f"unconverged runs on spread demand with $\\rho\\le400$\\,m swing by a median "
                                    f"${n(np.median(D.swing(unconv((100, 150, 200, 300, 400)))))}$\\,pp of served share, and "
                                    f"those from $800$\\,m by ${n(np.median(D.swing(unconv((800, 900, 1000, 1100, 1200)))))}$\\,pp")
    sh = lambda u: seq(label=u, scenario="shelter_clustered")
    shc = [win(r, scenario="shelter_clustered")["conv"] for r in D.RADII if r >= 300]
    c.snippet("VI: shelter windows", f"a $150$\\,m window is only ${n(sh('global')['mean'] - sh('window150')['mean'])}$"
                                     f"\\,pp below team play ({f(sh('window150')['mean'])} against "
                                     f"{f(sh('global')['mean'])}\\,pp), and from $300$\\,m windows converge in "
                                     f"${min(shc):.0f}$ to ${max(shc):.0f}\\%$ of runs")
    c.snippet("VI: shelter 100 m", f"Only $100$\\,m is too small ({f(win(100, scenario='shelter_clustered')['mean'])}\\,pp)")
    nd = D.sweep("nadir")
    tn = P(nd, label="global")
    nwin = lambda r: gs(P(nd, label=f"window{r}"))
    nvt = lambda r: vs_team(P(nd, label=f"window{r}"), tn)
    nadm = [nwin(r)["mean"] for r in D.RADII if r >= 200]
    c.snippet("VI: nadir window spread", f"from $200$ to $1200$\\,m the pooled gap varies within "
                                         f"${math.ceil(10 * (max(nadm) - min(nadm))) / 10:.1f}$\\,pp")
    assert nvt(100)["hi"] < 0 and nvt(150)["hi"] < 0 and all(nvt(r)["hi"] > 0 for r in D.RADII if r >= 200)
    c.snippet("VI: nadir small windows lose", f"only windows of $150$\\,m or less lose against team play "
                                              f"({ci(nvt(100), 2)}\\,pp paired at $100$\\,m, mostly in the shelter)")
    up = {r: nvt(r) for r in D.RADII if 300 <= r <= 900}
    assert all(v["lo"] > 0 for v in up.values()) and max(up, key=lambda r: up[r]["mean"]) == 400
    c.snippet("VI: nadir windows above team", f"Windows of $300$ to $900$\\,m even end slightly above team play, by at most "
                                              f"{ci(up[400], 2)}\\,pp at $400$\\,m")
    nconv = [nwin(r)["conv"] for r in up]
    c.snippet("VI: nadir windows converge less", f"(${min(nconv):.0f}$ to ${max(nconv):.0f}\\%$ of runs, against "
                                                 f"${gs(tn)['conv']:.0f}\\%$)")
    c.snippet("VI: team gain from pointing", f"team play gains ${n(team['mean'] - nad(label='global')['mean'])}$\\,pp over "
                                             f"nadir pooled, ${n(gain['urban_broad'])}$\\,pp on urban demand and "
                                             f"${n(gain['milan_p100'])}$\\,pp in Milan")
    pgain = lambda r, **kw: win(r, **kw)["mean"] - gs(P(nd, label=f"window{r}", **kw))["mean"]
    worse = {sc: max(r for r in D.RADII if pgain(r, scenario=sc) < 0) for sc in D.SPREAD}
    assert all(pgain(r, scenario=sc) < 0 for sc in D.SPREAD for r in D.RADII if r <= worse[sc])
    c.snippet("VI: pointing worse than fixed", f"worse than a fixed beam for windows up to ${worse['urban_broad']}$\\,m on urban "
                                               f"demand and ${worse['milan_p100']}$\\,m in Milan")
    assert pgain(500) < 0 < pgain(600) and all(pgain(r) > 0 for r in D.RADII if r >= 600)
    c.snippet("VI: pointing helps windows", f"Pointing helps the window agents from $600$\\,m on, and from $900$\\,m nearly as "
                                            f"much as it helps the team (${n(pgain(900))}$ against "
                                            f"${n(team['mean'] - nad(label='global')['mean'])}$\\,pp pooled)")
    c.snippet("VI: pointing cost", f"pointing costs a $150$\\,m window ${n(cost[('window150', 'urban_broad')])}$ and "
                                   f"${n(cost[('window150', 'milan_p100')])}$\\,pp, a $300$\\,m window "
                                   f"${n(cost[('window300', 'urban_broad')])}$ and ${n(cost[('window300', 'milan_p100')])}$"
                                   f"\\,pp, and selfish agents ${n(cost[('selfish', 'urban_broad')])}$ and "
                                   f"${n(cost[('selfish', 'milan_p100')])}$\\,pp")
    import swarm_figures as F
    meds = [np.median(F._tilts((sc,), u)) for sc in D.SPREAD for u in ("selfish", "window300")]
    c.snippet("VI: steep local tilts", f"with a median of ${min(meds):.0f}$ to ${max(meds):.0f}^\\circ$")
    tg = F._tilts(D.SPREAD, "global")
    c.snippet("VI: team near nadir", f"Team agents keep ${100 * np.mean(tg <= 15):.0f}\\%$ of their beams within "
                                     f"$15^\\circ$ of nadir")
    smeds = [np.median(F._tilts(("shelter_clustered",), u)) for u in ("selfish", "window300")]
    c.snippet("VI: shelter tilts", f"(median ${min(smeds):.0f}$ to ${max(smeds):.0f}^\\circ$)")
    sgain = [sh(u)["mean"] - nad(label=u, scenario="shelter_clustered")["mean"] for u in D.UTIL_ORDER]
    assert min(sgain) > 0
    c.snippet("VI: shelter pointing helps", f"pointing helps every utility, by ${min(sgain):.0f}$ to "
                                            f"${max(sgain):.0f}$\\,pp")
    i150 = gs(P(p3, start="paper", label="global", info_m="150.0"))
    i600 = gs(P(p3, start="paper", label="global", info_m="600.0"))
    w150, w600 = seq(label="window150"), seq(label="window600")
    inf = D.sweep("info")
    iw = lambda r: gs(P(inf, info_m=f"{float(r):.1f}"))
    assert all(iw(r)["mean"] < win(r)["mean"] for r in D.RADII if r <= 500) and iw(600)["mean"] > win(600)["mean"]
    cost = [win(r)["mean"] - iw(r)["mean"] for r in D.RADII if r <= 300]
    c.snippet("VI: info vs scope", f"Up to $500$\\,m, missing information costs more than missing scope, "
                                   f"${min(cost):.0f}$ to ${max(cost):.0f}$\\,pp at and below $300$\\,m "
                                   f"({f(i150['mean'])} against {f(w150['mean'])}\\,pp at $150$\\,m)")
    assert iw(400)["mean"] - iw(700)["mean"] < 0 and (iw(700)["mean"] - iw(400)["mean"]) > (win(700)["mean"] - win(400)["mean"])
    c.snippet("VI: info transition", f"The transition is steeper than for windows: the gap moves from {f(iw(400)['mean'])}\\,pp "
                                     f"at $400$\\,m to {f(iw(700)['mean'])}\\,pp at $700$\\,m, against "
                                     f"{f(win(400)['mean'])} to {f(win(700)['mean'])}\\,pp for the window")
    ivt = lambda r: vs_team(P(inf, info_m=f"{float(r):.1f}"), tm)
    assert ivt(600)["hi"] < 0 and ivt(700)["lo"] < 0 < ivt(700)["hi"]
    assert all(ivt(r)["lo"] < 0 < ivt(r)["hi"] for r in D.RADII if r >= 700)
    tmk = {lkey(r): r["gap"] for r in tm}
    far = 100 * np.mean([r["gap"] - tmk[lkey(r)] < -1.0 for r in P(inf, info_m="700.0")])
    c.snippet("VI: info 700 matches team", f"at $700$\\,m it matches team play on average ({ci(ivt(700), 2)}\\,pp, paired), "
                                           f"where the $700$\\,m window still loses ${n(-vt(700)['mean'])}$\\,pp; "
                                           f"${far:.0f}\\%$ of layouts still end more than $1$\\,pp below team play there")
    iconv = [iw(r)["conv"] for r in D.RADII if r >= 900]
    c.snippet("VI: info convergence", f"${iw(700)['conv']:.0f}\\%$ of runs converge at $700$\\,m, and "
                                      f"${min(iconv):.0f}$ to ${max(iconv):.0f}\\%$ from $900$\\,m")
    c.snippet("tab:info caption team", f"Team play with full information: {ci(gs(tm))}\\,pp.")
    c.snippet("VI: info 600", f"({f(i600['mean'])} against {f(w600['mean'])}\\,pp)")
    si = [gs(P(p3, start="paper", label="selfish", info_m=f"{r:.1f}"))["mean"] for r in (150, 300, 600, 1200)]
    # abstract, contribution, design rule (ii), conclusion: "about 1.1 km" / "about 1 km" for windows, 700 m for
    # information, 500 m for pointing against nadir
    assert vt(1000)["hi"] < 0 < vt(1100)["hi"] and vt(1100)["lo"] < 0 and 1050 < r30 < 1150
    c.snippet("abstract: window radius", f"a window matches team play at about $1.1$\\,km, comparable to the reach of a "
                                         f"$30^\\circ$ beam")
    assert ivt(600)["hi"] < 0 and ivt(700)["lo"] < 0 < ivt(700)["hi"]
    c.snippet("abstract: info radius", "observing $700$\\,m suffices for a drone that values the whole network")
    assert vt(1000)["mean"] > -0.15                       # "about 1 km": within 0.15 pp of team play at 1000 m
    pg = {}
    for r in D.RADII:
        rows = P(pt, label=f"window{r}")
        d = np.array([x["gap"] - {lkey(y): y["gap"] for y in P(nd, label=f"window{r}")}[lkey(x)] for x in rows])
        lo, hi = D.A.boot_ci(rows, d)
        pg[r] = (d.mean(), lo, hi)
    assert all(pg[r][2] < 0 for r in D.RADII if r < 500) and pg[500][1] < 0 < pg[500][2] and pg[600][1] > 0
    c.snippet("contribution: radii", "it must value about $1$\\,km of the network, or value all of it and observe about "
                                     "$700$\\,m, and below about $500$\\,m a steerable antenna")
    c.snippet("design rule radii", "drones need to value about $1$\\,km of the network, or to value all of it while "
                                   "observing about $700$\\,m. If the inter-drone link covers only $500$\\,m or less")
    c.snippet("design rule: 500 m", "since pointing then gives local play nothing and below $500$\\,m costs it")
    c.snippet("conclusion radii", "values the network within about $1$\\,km, or values all of it but sees only $700$\\,m")
    c.snippet("VI: selfish info", f"({f(min(si))} to {f(max(si))}\\,pp from $150$ to $1200$\\,m)")
    shi = gs(P(p3, start="paper", label="global", info_m="150.0", scenario="shelter_clustered"))["mean"]
    assert sh("global")["mean"] - shi < 1.0
    c.snippet("VI: shelter info", "In the shelter, $150$\\,m of information is already within $1$\\,pp")

    # Section VII: dynamics
    dyn = lambda u, d: gs(P(p1, start="paper", dynamics=d, label=u))
    c.snippet("VII: synchronous team", f"team agents end {f(dyn('global', 'sim')['mean'])}\\,pp below the reference and "
                                       f"${dyn('global', 'sim')['conv']:.0f}\\%$ of runs converge")
    c.snippet("VII: synchronous selfish", f"selfish agents collapse to {f(dyn('selfish', 'sim')['mean'])}\\,pp")
    nc = [r for r in P(p1, start="paper", dynamics=("sim", "sim50")) if r["converged"] == "0"]
    sw = D.swing(nc)
    c.snippet("VII: swing", f"swing by a median ${n(np.median(sw))}$\\,pp of served share over their second half, up to "
                            f"${n(sw.max())}$\\,pp")
    c.snippet("VII: inertia team", f"team play ends {f(dyn('global', 'sim50')['mean'])}\\,pp below the reference")
    c.snippet("VII: inertia converges", f"and ${dyn('global', 'sim50')['conv']:.0f}\\%$ of runs converge")
    c.snippet("VII: inertia selfish", f"Inertia does nothing for selfish agents ({f(dyn('selfish', 'sim50')['mean'])}\\,pp)")
    dep = gs(P(p1, start="depot", dynamics="seq", label="global"))
    c.snippet("VII: depot", f"({f(dep['mean'])}\\,pp for team play)")

    # Section VII: speed limits and prediction errors
    rl = real()
    m = lambda u, v: D.gap_stats(rl[(u, v)])["mean"]
    step = sorted(run_swarm.STEPS_M)
    c.snippet("VII: step", f"caps each move at ${step[0]:.0f}$ or ${step[1]:.0f}$\\,m: a drone flying at $10$\\,m/s")
    c.snippet("VII: periods", f"that decides every ${step[0] / 10:.0f}$ or ${step[1] / 10:.0f}$\\,s. Runs may then take "
                              f"${run_swarm.SPEED_MAX_ROUNDS}$ rounds")
    err = {e.name: e for e in run_swarm.ERRORS}
    c.snippet("VII: errors pos", f"$80$th percentile is ${err['mild'].pos_r80_m:.0f}$\\,m, the 3GPP commercial target "
                                 f"for outdoor positioning, or ${err['harsh'].pos_r80_m:.0f}$\\,m")
    c.snippet("VII: errors gain", f"log-normal error of ${err['mild'].gain_sigma_db:.0f}$ or "
                                  f"${err['harsh'].gain_sigma_db:.0f}$\\,dB")
    shift = max(abs(m(u, v) - m(u, "none")) for u in ("global", "window1200", "selfish") for v in ("150.0", "600.0"))
    assert shift < 1.0, shift
    conv = [D.gap_stats(rl[(u, v)])["conv"] for u in ("global", "window1200") for v in ("150.0", "600.0")]
    rounds = [np.mean([float(r["rounds"]) for r in rl[(u, v)] if r["converged"] == "1"])
              for u in ("global", "window1200") for v in ("150.0", "600.0")]
    c.snippet("VII: step converge", f"converge in ${min(conv):.0f}$ to ${max(conv):.0f}\\%$ of runs, within "
                                    f"${min(rounds):.0f}$ to ${max(rounds):.0f}$ rounds on average")
    cost = lambda u, v: m(u, "none") - m(u, v)
    al = ("global", "window1200")
    c.snippet("VII: error cost", f"Errors cost aligned agents ${n(min(cost(u, 'mild') for u in al))}$ to "
                                 f"${n(max(cost(u, 'mild') for u in al))}$\\,pp when mild and "
                                 f"${n(min(cost(u, 'harsh') for u in al))}$ to ${n(max(cost(u, 'harsh') for u in al))}$"
                                 f"\\,pp when harsh, and selfish agents ${n(cost('selfish', 'mild'))}$ and "
                                 f"${n(cost('selfish', 'harsh'))}$\\,pp")
    sh = [np.mean([r["gap"] for r in rl[(u, "harsh")] if r["scenario"] == "shelter_clustered"])
          - np.mean([r["gap"] for r in rl[(u, "none")] if r["scenario"] == "shelter_clustered"]) for u in al]
    c.snippet("VII: shelter errors", f"harsh errors cost aligned agents about ${-np.mean(sh):.0f}$\\,pp there")
    assert not any(r["converged"] == "1" for u in rl for r in rl[u] if u[1] in ("mild", "harsh"))

    # Section VIII: self-sizing
    ex = gs(P(p4a, start="depot", label="global"))
    c.snippet("VIII: exploration", f"end {f(ex['mean'])}\\,pp below the reference, against {f(dep['mean'])}\\,pp "
                                   f"without, and ${ex['within1']:.0f}\\%$ of runs")
    es = [gs(P(p4a, start=s, label="selfish"))["mean"] for s in ("paper", "depot")]
    c.snippet("VIII: selfish exploration", f"Selfish agents end ${n(es[0])}$ and ${n(es[1])}$\\,pp below")
    c.snippet("VIII: stranded", f"still strands drones in the shelter "
                                f"({f(gs(P(p4a, start='depot', label='window1200', scenario='shelter_clustered'))['mean'])}\\,pp)")
    st = lambda stg, sc, start="depot1", u="global", rule=D.FINAL_RULE: D.size_stats(
        P(D.sized(stg), scenario=sc, start=start, label=u, rules=rule))
    pc = {sc: st("p4b", sc, rule="x_land0.05_call0.25") for sc in D.SPREAD}
    c.snippet("VIII: per cell", f"stops at ${n(pc['urban_broad']['ratio'], 2)}$ and "
                                f"${n(pc['milan_p100']['ratio'], 2)}\\,N_{{95}}$ on urban and Milan demand")
    pc1 = D.size_stats(P(D.sized("p4b"), start="depot1", label="global", rules="x_land0.25_call1"))
    c.snippet("VIII: per cell high", f"at ${n(pc1['ratio'], 2)}\\,N_{{95}}$ pooled with $(0.25,1)$")
    ar = {sc: st("p4c", sc, rule="x_land0.25_call0.5_area1200") for sc in D.SPREAD}
    c.snippet("VIII: area", f"aligned fleets grow to ${n(ar['urban_broad']['ratio'], 2)}$ and "
                            f"${n(ar['milan_p100']['ratio'], 2)}\\,N_{{95}}$ on urban and Milan demand")
    c.snippet("VIII: area gap", f"within ${n(max(-ar[sc]['gap'] for sc in D.SPREAD))}$\\,pp at the same $N$")
    ck = [st("p4d", sc, rule="x_land0.25_call0.5_area1200_chk")["ratio"] for sc in D.SPREAD]
    c.snippet("VIII: call check", f"stalls growth at ${n(np.mean(ck), 2)}\\,N_{{95}}$")
    c.snippet("VIII: hysteresis", f"aligned fleets reach ${n(fin['urban_broad']['ratio'], 2)}$ and "
                                  f"${n(fin['milan_p100']['ratio'], 2)}\\,N_{{95}}$ with at most ${n(max(fin[sc]['excess'] for sc in D.SPREAD))}$ "
                                  f"excess drones and a gap of at most ${n(max(-fin[sc]['gap'] for sc in D.SPREAD))}$\\,pp")
    c.snippet("VIII: events", f"lands plus calls fall from ${ar['urban_broad']['events']:.0f}$ and "
                              f"${ar['milan_p100']['events']:.0f}$ to ${fin['urban_broad']['events']:.0f}$ and "
                              f"${fin['milan_p100']['events']:.0f}$ per run")
    ov = {sc: st("p4e", sc, start="over") for sc in D.SPREAD}
    c.snippet("VIII: shrinking", f"lands aligned fleets to ${n(ov['urban_broad']['ratio'], 2)}$ and "
                                 f"${n(ov['milan_p100']['ratio'], 2)}\\,N_{{95}}$ at "
                                 f"${n(ov['urban_broad']['served'])}$ and ${n(ov['milan_p100']['served'])}\\%$ served")
    c.snippet("VIII: selfish fleets", f"selfish fleets end at ${n(sfin['urban_broad']['ratio'], 2)}$ and "
                                      f"${n(sfin['milan_p100']['ratio'], 2)}\\,N_{{95}}$ on urban and Milan demand, "
                                      f"${-sfin['milan_p100']['gap']:.0f}$ to ${n(-sfin['urban_broad']['gap'])}$\\,pp below")
    c.snippet("VIII: shelter kappa", f"one drone serves only ${100 * kap['shelter_clustered']:.0f}\\%$ of the demand")
    sa, sf = st("p4c", "shelter_clustered", rule="x_land0.25_call0.5_area1200"), fin["shelter_clustered"]
    sk = st("p4d", "shelter_clustered", rule="x_land0.25_call0.5_area1200_chk")
    assert sa["ratio"] > 1 and sf["ratio"] > 1 and sk["ratio"] < 1      # area rules over-provision; the checked one stops short
    c.snippet("VIII: shelter call check", f"every area rule without a call check churns and over-provisions; the one "
                                          f"with a call check stalls at ${n(sk['ratio'], 2)}\\,N_{{95}}$")
    c.snippet("VIII: shelter churn", f"from ${sa['events']:.0f}$ to ${sf['events']:.0f}$ per run and closes the gap at "
                                     f"the same $N$ from {f(sa['gap'])} to {f(sf['gap'])}\\,pp")
    c.snippet("VIII: shelter size", f"still ends at ${n(sf['ratio'], 2)}\\,N_{{95}}$ and only ${sf['conv']:.0f}\\%$")
    hi = st("p4e", "urban_broad", rule="x_land0.25_call1_area1200_pat2_gr3")
    c.snippet("VIII: higher threshold", f"stops at ${n(hi['ratio'], 2)}\\,N_{{95}}$ on urban demand")

    # Section IX: LLM
    l1 = [r for r in D.llm() if r["stage"] == "l1" and r["model"] in LLM_MODELS]
    nturn = sum(D.llm_turns("l1", m)["turns"] for m in LLM_MODELS)
    c.snippet("IX: runs", f"${len(l1)}$ runs and ${nturn}$ decisions")
    c.snippet("IX: start", f"The served share starts at ${n(np.mean([float(r['served_start_pct']) for r in l1]))}\\%$")
    c.snippet("IX: best response", f"gains {f(D.llm_change('br-global', 'global').mean())}\\,pp in eight rounds")
    c.snippet("IX: rand6", f"gains {f(D.llm_change('pol-rand6', 'network').mean())}\\,pp")
    c.snippet("IX: goto", f"loses ${n(-D.llm_change('pol-goto', 'network').mean())}$\\,pp")
    up = sum(r["change"] > 0.05 for r in l1)
    c.snippet("IX: runs above start", f"Of the ${len(l1)}$ runs only {['no', 'one', 'two'][up]} ends above its start")
    best = max(((m, fr) for m in LLM_MODELS for fr in ("selfish", "area", "network")),
               key=lambda k: D.llm_change(*k).mean())
    bv = D.llm_change(*best).mean()
    assert best == ("qwen3.5-9b", "area")
    c.snippet("IX: best cell", f"the $9$B model with the area framing, ends {f(bv)}\\,pp")
    c.snippet("IX: below BR", f"${n(D.llm_change('br-global', 'global').mean() - bv)}$\\,pp below the best-response agent")
    c.snippet("IX: scale", f"(network framing: from {f(D.llm_change('qwen3.5-0.8b', 'network').mean())}\\,pp at $0.8$B "
                           f"to {f(D.llm_change('qwen3.5-9b', 'network').mean())}\\,pp at $9$B)")
    d = [D.llm_change(m, "selfish").mean() - D.llm_change(m, fr).mean() for m in ("qwen3.5-4b", "qwen3.5-9b")
         for fr in ("area", "network")]
    c.snippet("IX: framing", f"the selfish framing is ${-max(d):.0f}$ to ${-min(d):.0f}$\\,pp worse")
    c.snippet("IX: selfish vs BR", f"({f(D.llm_change('qwen3.5-9b', 'selfish').mean())} against "
                                   f"{f(D.llm_change('br-selfish', 'selfish').mean())}\\,pp)")
    t9, tr = D.llm_turns("l1", "qwen3.5-9b"), D.llm_turns("l1pol", "pol-rand6")
    c.snippet("IX: what-ifs", f"asks a mean of ${n(t9['what_ifs'])}$ what-if questions per turn and moves to a pose it "
                              f"had queried in ${t9['checked']:.0f}\\%$ of its moves, yet it moves in "
                              f"${t9['move']:.0f}\\%$ of turns")
    c.snippet("IX: rand6 moves", f"moves only on a predicted gain, moves in ${tr['move']:.0f}\\%$")
    log = D.llm_moves("l1log", "qwen3.5-9b", "area")
    c.snippet("IX: rerun", f"The rerun ends {f(D.llm_change('qwen3.5-9b', 'area', 'l1log').mean())}\\,pp against "
                           f"{f(D.llm_change('qwen3.5-9b', 'area').mean())}\\,pp in the sweep")
    gain = lambda x: x["pred"] is not None and x["pred"] > 0
    c.snippet("IX: follows queries", f"Of its moves, ${100 * np.mean([gain(x) for x in log]):.0f}\\%$ go to a pose its own "
                                     f"query predicted to gain and ${100 * np.mean([x['best'] for x in log]):.0f}\\%$ to "
                                     f"the best pose it queried")
    ap = [x for x in log if x["applied"]]
    good, bad = [x for x in ap if gain(x)], [x for x in ap if x["pred"] is None or x["pred"] < 0]
    c.snippet("IX: good and bad moves", f"${len(good)}$ went to a predicted gain and raised the people served in the "
                                        f"movers' areas by ${sum(x['real'] for x in good)}$; ${len(bad)}$ went to a "
                                        f"predicted loss or to a pose never queried and lowered it by "
                                        f"${-sum(x['real'] for x in bad)}$")
    q = [x for x in ap if x["pred"] is not None]
    c.snippet("IX: realised", f"(${np.mean([x['real'] for x in q]):+.0f}$ against ${np.mean([x['pred'] for x in q]):+.0f}$ "
                              f"people per applied move)")
    tk = D.llm_turns("l1", "qwen3.5-9b-think")
    c.snippet("IX: thinking cost", f"writes ${tk['completion'] / 1000:.1f}$ thousand completion tokens and takes "
                                   f"${tk['seconds']:.0f}$\\,s per turn, ${tk['seconds'] / t9['seconds']:.1f}$ times as long")
    c.snippet("IX: thinking moves", f"It moves in fewer turns (${tk['move']:.0f}\\%$)")
    fr3 = ("selfish", "area", "network")
    c.snippet("IX: thinking change", "ends no better: " + ", ".join(
        f(D.llm_change("qwen3.5-9b-think", fr).mean()) for fr in fr3[:2]) + " and " +
        f(D.llm_change("qwen3.5-9b-think", "network").mean()) + "\\,pp under the three framings, against " + ", ".join(
        f(D.llm_change("qwen3.5-9b", fr).mean()) for fr in fr3[:2]) + " and " +
        f(D.llm_change("qwen3.5-9b", "network").mean()) + "\\,pp without thinking")
    tn = D.llm_moves("l1", "qwen3.5-9b-think", "network")
    c.snippet("IX: thinking network", f"predicted to lose in ${100 * np.mean([x['pred'] is not None and x['pred'] < 0 for x in tn]):.0f}"
                                      f"\\%$ of its moves")
    from dronecomm import llmagent
    c.snippet("IX: gate text", "``" + llmagent.GATE_TEXT.replace("_", "\\_") + "''")
    gt, gm = D.llm_turns("l1gate", "qwen3.5-9b"), D.llm_moves("l1gate", "qwen3.5-9b", "area+gate")
    gch = D.llm_change("qwen3.5-9b", "area+gate", "l1gate")
    c.snippet("IX: gated", f"makes ${sum(x['pred'] is not None and x['pred'] < 0 for x in gm)}$ of its ${len(gm)}$ moves to a "
                           f"predicted loss, moves in ${gt['move']:.0f}\\%$ of turns and gains {f(gch.mean())}\\,pp; "
                           f"${int(np.sum(gch > 0.05))}$ of ${len(gch)}$ runs end above their start")
    c.snippet("IX: gated vs refs", f"({f(D.llm_change('pol-rand6', 'network').mean())}\\,pp) and "
                                   f"${n(D.llm_change('br-global', 'global').mean() - gch.mean())}$\\,pp below best response")
    mc = [D.llm_turns("l1", m)["max_calls"] for m in ("qwen3.5-0.8b", "qwen3.5-2b")]
    c.snippet("IX: max calls", f"${min(mc):.0f}$ to ${max(mc):.0f}\\%$ of their turns use up the twelve tool calls")
    c.snippet("IX: cost", f"about ${t9['prompt'] / 1000:.0f}$ thousand prompt tokens and "
                          f"{['', 'one'][round(t9['completion'] / 1000)]} thousand completion tokens, and "
                          f"${t9['seconds']:.0f}$\\,s of model time")

    # Discussion and conclusion
    c.snippet("X: selfish loss", f"the selfish loss of about ${-self_['mean']:.0f}$\\,pp")
    c.snippet("XI: selfish loss", f"loses about ${-self_['mean']:.0f}$\\,pp of demand")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", action="store_true", help="print the table rows and exit")
    args = ap.parse_args()
    if args.rows:
        for name, fn in TABLES.items():
            print(f"% {name}")
            print("\n".join(fn()))
        return
    c = Checker(TEX.read_text())
    check_tables(c)
    print("\ntext")
    check_text(c)
    print(f"\n{c.n - len(c.bad)} of {c.n} checks pass")
    if c.bad:
        print("\n".join(c.bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
