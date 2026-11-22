# Fleet sizing reading (WP2/WP3): how many drones does a disaster area need?

> Run v1 (access side only, no backhaul). Its CSVs are kept in `results/fleet_v1/`; the access-side numbers
> below are unchanged in v2, which adds the end-to-end (backhaul + gateway) served share.

Run 2026-10-25 on Ibex (job 54539946, 24/24 tasks PASS, repo `5bc6636`): 6 variants × 448 cells (5 scenarios
× 4 service tiers × 10 seeds, Milan 24 snapshots × 3 seeds) × N = 1–300 × 3 altitude methods. Service model: GBR
admission at the tier's rate (`capacity.admit_gbr`). Output: `results/fleet_v1/reading.txt`,
`results/fleet_v1/{curves,fleet_size}.csv` (`scripts/analyze_fleet.py`). Demand and tier sources:
`scripts/run_fleet.py` and `docs/parameters.md`.

## Smallest fleet serving 95 % of active demand, grounded model (greedy / sweep / K-means at 120 m)

| Scenario (active users) | Text 3 kb/s | Voice 38 kb/s | Video call 384 kb/s | Broadband 2 Mb/s |
|---|---|---|---|---|
| Rural broad, 400 users (80 active) | 3/3/3 | 3/3/3 | 6/6/6 | 12/12/12 |
| Shelter hotspot, 2,094 people (343) | 2/2/2 | 3/3/3 | 6/6/never (82 %) | never (68 %) |
| Shelter clustered, 13,963 people (2,290) | 3/3/3 | 7/7/7 | 60/60/never (63 %) | never (59 %) |
| Milan crop, 41,000 people (6,724) | 5/5/5 | 16/16/16 | 120/150/never (82 %) | never (65 %) |
| Urban broad, 40,000 users (8,000) | 5/5/5 | 16/16/16 | 150/150/never (84 %) | never (62 %) |

"never (x %)": not reached at N ≤ 300; x is the best mean share.

## Findings

1. **Voice is cheap, video is not.** Voice for a Milan-sized crop needs 16 drones; a video call for everyone active
   needs 120–150, and 2 Mb/s broadband is not reachable on one 10 MHz band (≤ 68 % at 300 drones). The fleet is set by
   the service tier far more than by coverage.
2. **Altitude matters again, in the opposite direction.** Under heavy load the searches fly low (Milan video: 58 m at
   N = 120, 55 m at N = 150) to shrink cells and cut interference. K-means at the 120 m ceiling saturates at 80–84 %
   for video. Coverage wants altitude; capacity wants dense, low cells.
3. **The 120 m cap costs drones only where coverage binds.** The 300 m waiver cuts small fleets (Milan voice 16 → 10,
   urban voice 16 → 12, rural broadband 12 → 6) and changes nothing for video, where the fleets fly low anyway.
4. **The per-cell user cap dominates when it binds.** With Hydher et al.'s 40 users per aerial BS, Milan and urban
   demand need about 300 drones even for text; the shelters 18–100. Cell admission limits must be stated with any
   fleet-size claim.
5. **Demand assumptions move the answer by tens of percent:** low-income device share (53 %) cuts Milan video from 120
   to 70 drones; dense-urban propagation adds a few drones at every tier.

## Open
- WP5: backhaul and gateway capacity may cap the useful fleet, especially at video load (120–150 drones × busy-hour
  traffic).
- Milan population is the city-mean density; per-crop census counts (ISTAT sections) are next.
- A 20 MHz variant (TR 36.814 also allows it) would show how far spectrum, rather than drones, lifts the broadband tier.
