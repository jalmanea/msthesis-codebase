# Fleet run v3: Milan across the city, backhaul topologies and relays

Run 2026-10-25 on Ibex (job 54554870, repo `0fed2f8`): 10 variants × 1,600 cells (4 synthetic scenarios ×
4 tiers × 10 seeds; 5 Milan tiles × 4 tiers × 24 snapshots × 3 seeds) × N = 1–300 × 3 altitude methods × 3
backhaul topologies. 58 tasks ended `RESULT: PASS`; tasks 50 and 52 (bh200) finished all cells (driver rc=0,
`--check` PASS from the login node) but their in-job check stalled and they were cancelled. Output:
`results/fleet/reading_v3.txt` (`scripts/analyze_fleet.py`). Earlier runs: `results/fleet_v1`, `results/fleet_v2`.

Milan tiles (2.35 km, 2021 census residents, `docs/parameters.md` D13): p100 71,932; p90 58,185; p75 26,848;
p50 12,637; p25 7,352. Backhaul: 60 GHz links, 800 Mb/s, 400 m range (longer hops get relays), 3 interfaces.

## Access side: smallest fleet for 95 % served (greedy), grounded

| Scenario | Text | Voice | Video call | Broadband |
|---|---|---|---|---|
| Milan p100 (hottest) | 6 | 20 | 300 | never (42 %) |
| Milan p90 | 5 | 18 | 250 | never (47 %) |
| Milan p75 | 4 | 12 | 70 | never (81 %) |
| Milan p50 | 3 | 8 | 35 | 250 |
| Milan p25 | 3 | 6 | 25 | 100 |
| Urban broad (TS 22.261) | 5 | 16 | 150 | never (62 %) |
| Shelter clustered | 3 | 7 | 60 | never (59 %) |
| Rural broad | 3 | 3 | 6 | 12 |

With 20 MHz: Milan p100 voice 14, video 100; p25 broadband 50. With the 300 m waiver: Milan p100 voice 18, p25
voice 3 (coverage-bound fleets shrink; capacity-bound ones do not).

## End to end: access drones + relays (greedy)

| Scenario, tier | Satellite gateway (any topology) | Tether, corner MST | Tether, central MST | Tether, balanced |
|---|---|---|---|---|
| Milan p100 voice | ceiling 22 % | 20 + 9 | 20 + 9 | 20 + 12 |
| Milan p100 video | ceiling 2 % | ceiling 20 % | ceiling 34 % | ceiling 54 % |
| Milan p75 video | ceiling 6 % | ceiling 50 % | ceiling 81 % | **70 + 0** |
| Milan p50 video | ceiling 2 % | 80 + 1 | 35 + 2 | 35 + 3 |
| Urban broad video | ceiling 3 % | ceiling 29 % | ceiling 56 % | ceiling 79 % |
| Shelter clustered video | ceiling 11 % | ceiling 87 % | 60 + 2 | 60 + 2 |
| Milan p100 text | 6 + 6 | 6 + 6 | 6 + 6 | 6 + 8 |
| Rural broadband | ceiling 62 % | 12 + 11 | 12 + 11 | 12 + 14 |

## Findings

1. **Demand across one city spans an order of magnitude, and so does the fleet.** Voice needs 6 drones on the
   quietest tile and 20 on the busiest; video 25 to 300. A single "Milan" number hides this.
2. **Relays are a large part of the fleet.** With 400 m 60 GHz hops, a voice fleet carries about as many relays as
   access drones (urban 16 + 16, Milan p75 12 + 14); even text on 2 km needs 3–6 access drones plus 4–8 relays.
3. **The satellite gateway caps everything above text** in dense areas: Milan p100 voice reaches 22 % end to end;
   p50 and p25 voice fit (8 + 7, 6 + 6).
4. **With a tether, the topology decides video.** Balancing the tree lifts the ceiling from 20 → 54 % (p100),
   50 % → served with 70 drones (p75), 29 → 79 % (urban), and halves the p50 fleet (80 → 35). The balanced tree pays
   for it with a few more relays.
5. **A shared 2 GHz backhaul channel is unusable** (Milan p100 text ≤ 67 % end to end).

## Extension to N = 500 (2026-10-26)

Ibex job 54569249 (8/8 PASS, repo `39ffb46`): `run_fleet.py --extend`, N = 350/400/500 for the cells that reach
N = 300 without 95 % or only there (`run_fleet.EXT_CELLS`: broadband for both shelters, urban and Milan p75/p90/p100;
video for Milan p90/p100), variants grounded and tether, 780 cells in `results/fleet_ext`.

- No broadband cell reaches 95 % at 500. Broad demand creeps up 1.0–2.3 pp per 50 drones (urban 62 → 71 %, Milan
  p100 42 → 51 %); the shelters peak and fall (hotspot 68 % at 70 drones, 21 % at 500): a single-band ceiling.
- Milan p100 video: greedy 300 confirmed (96.4 % at 500); the sweep reaches 95 % at 350.
- Bootstrap intervals of N* (resampling seeds or snapshots, `journal_data.bootstrap_nstar`): one grid value on
  Milan except three cells (one step wide); synthetic shelters wider (clustered text 2–7).
- Single-gateway bound (paper Proposition 1): delivered ≤ min(G, B·se_max + I·C) = 2,444 Mb/s with a tether.
  The balanced tree's ceiling is within 0.2 pp of it in 35 of 36 (scenario, tier) cells (clustered-shelter
  broadband: 3.8 pp); no run of any topology exceeds it.

## Refinement: dense N around every quoted N*, more seeds (2026-10-26)

Ibex job 54572566 (48/48 PASS, repo `6592cb4`): `run_fleet.py --variant all --refine`, 65 windows from
`configs/refine_windows.json` (`journal_data.refine_windows`: two coarse grid values either side of each N* in
Tables IV-V and the sensitivity text; steps of 1 to N = 40, 2 to 100, 5 beyond), 40 synthetic seeds and 6
K-means seeds per Milan snapshot; 6,968 cells in `results/fleet_refine`. No window missed its N* (none starts
at or above 95 %). Paper tables and text now take N* from here (`journal_data.fleet_cell`), else the coarse grid.

| Cell | coarse | refined [95 % bootstrap] |
|---|---|---|
| Milan p100 video | 300 | 270 [255, 285] |
| Milan p90 video | 250 | 210 [200, 225] |
| Urban video | 150 | 125 [120, 125] |
| Shelter clustered video | 60 | 54 [52, 58] |
| Milan p75 / p50 / p25 video | 70 / 35 / 25 | 68 / 33 / 21 |
| Milan p50 / p25 broadband | 250 / 100 | 230 [225, 240] / 105 [100, 105] |
| Milan p100 voice | 20 | 21 [20, 21] |

Lower bound (`journal_data.lower_bound_drones`, Table II): N ≥ ⌈traffic / (B·se_max)⌉, 44 Mb/s per drone. Video
fleets are 1.7–2.7× it where it is ≥ 10 drones, voice 1.9–4×; Milan p100 broadband needs ≥ 537 drones.
