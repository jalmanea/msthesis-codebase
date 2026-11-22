# WP1 reading: which ICC findings survive a grounded model

Run 2026-10-25 on Ibex (job 54537317, 24/24 tasks PASS, repo `b976aea`), 8 variants × (1,500 synthetic +
4,992 Milan cells) × 20 seeds × 4 methods. Full output: `results/wp1/reading.txt` (from
`scripts/analyze_wp1.py`). Variants: `icc` (the ICC model), `grounded` (`configs/grounded.yaml`), and grounded
with one change each: `waiver300` (altitude to 300 m), `element1x1` (8 dBi / 65° element), `panel4x4`
(16 dBi / 24.8°), `suburban`, `dense_urban`, `highrise_urban`.

**Validation.** `icc` reproduces every checked ICC number exactly: synthetic 50.4/76.7/80.6/83.6 %, analytic gains
+37.6/+42.5 pp, greedy +3.0 pp in 86 % of runs, Milan 92.6/90.7/89.1/85.3 % with the snapshot-level wins.

## Synthetic grid (means over N = 1–50, M = 100–1000, 20 seeds)

| Claim [ICC value] | icc | grounded | waiver300 |
|---|---|---|---|
| Coverage K-means / analytic / sweep / greedy [50.4/76.7/80.6/83.6] | same | 60.1/76.7/81.8/82.6 | 60.1/77.1/86.3/87.8 |
| Analytic − K-means, clustered [+37.6] | +37.6 | +6.8 | +4.9 |
| Analytic − K-means, hotspot [+42.5] | +42.5 | +43.0 | +44.4 |
| Greedy − sweep; share of runs improved [+3.0; 86 %] | same | +0.8; 62 % | +1.5; 76 % |
| Greedy − sweep, hotspot [+5.4] | +5.4 | +0.4 | +2.0 |
| Hotspot N ≥ 30 greedy runs with a drone at the ceiling [79–87 %] | 83 % | 22 % (120 m) | 57 % (300 m) |
| Smallest N for 90 % coverage, clustered / uniform (sweep) | 1 / 1 | 5 / 20 | 2 / 4 |

## Milan (M = 800, 24 snapshots, N = 5–30)

| | icc | grounded | waiver300 |
|---|---|---|---|
| Mean over N, greedy / sweep / K-means / analytic [92.6/90.7/89.1/85.3] | same | 82.7/82.1/82.1/82.2 | 93.9/92.4/82.1/85.5 |
| Sweep > K-means, (snapshot, N) pairs [all 624] | 624 | 28 | 624 |
| Greedy > sweep [all 624] | 624 | 566 | 624 |
| Smallest N for 90 % coverage (sweep / greedy) | 5 / 5 | 21 / 20 | 5 / 5 |
| Mean sweep altitude | 156 m | 120 m | 185 m |

## What this means

1. **The drone count now matters.** With a physical antenna and a noise figure, one drone no longer covers
   everything: 90 % coverage needs about 5 drones on clustered demand and 20 on uniform or Milan demand (ICC: 1).
2. **The 120 m cap is the single largest effect.** Under it, every method flies at or near the ceiling on broad
   demand, so on Milan the four methods are within 0.6 pp and the altitude decision nearly disappears. With a
   300 m waiver the sweep and greedy search regain their lead, and the fleet for 90 % on Milan falls from 21 to 5:
   the altitude limit costs about four times the drones.
3. **What survives:** the ranking sweep ≥ analytic ≥ K-means overall, greedy ≥ sweep by construction, and the
   analytic rule's large gain on hotspots (+43 pp). **What does not:** the analytic rule's clustered gain
   (+37.6 → +6.8), the greedy search's +3.0 pp (→ +0.8) and its drone parking (83 % → 22 %), and the Milan
   snapshot-level dominance of the sweep over K-means (624 → 28 of 624).
4. **Environment and antenna move coverage more than any method.** On Milan, high-rise urban gives about 41 %,
   dense urban 74 %, suburban 85–87 %. A narrow 4×4 panel never reaches 90 % on uniform demand at N ≤ 50 except
   with the greedy search (N = 49); the wide single element reaches it at N = 14–16.

Capacity columns (served share, rates) are in the CSVs but use the ICC user counts (≤ 1,000); they become
meaningful with the grounded demand of WP3.
