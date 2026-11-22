# Backhaul and spectrum reading (WP5): what the drones cannot fix

Run 2026-10-25 on Ibex (job 54552858, 40/40 tasks PASS, repo `00b9177`): 10 variants × 448 cells × N = 1–300 ×
3 altitude methods. Baseline (`configs/grounded.yaml`): 60 GHz drone-to-drone links at 800 Mb/s (SkyHaul's
measured upper end), MST rooted at the corner gateway drone, one 100 Mb/s satellite ground link (Starlink US
median). Output: `results/fleet/reading_e2e.txt` (end-to-end), `results/fleet/reading_access.txt`
(access side, grounded vs 20 MHz), `scripts/analyze_fleet.py --metric served_e2e_pct`.

## Demand traffic the gateway must carry (all active users at the tier's rate)

| Scenario | Active users | Text | Voice | Video call | Broadband |
|---|---|---|---|---|---|
| Rural broad | 80 | 0.2 Mb/s | 3 Mb/s | 31 Mb/s | 160 Mb/s |
| Shelter hotspot | 343 | 1 Mb/s | 13 Mb/s | 132 Mb/s | 687 Mb/s |
| Shelter clustered | 2,290 | 7 Mb/s | 87 Mb/s | 879 Mb/s | 4.6 Gb/s |
| Milan crop | 6,724 | 20 Mb/s | 256 Mb/s | 2.6 Gb/s | 13.4 Gb/s |
| Urban broad | 8,000 | 24 Mb/s | 304 Mb/s | 3.1 Gb/s | 16 Gb/s |

## Smallest fleet for 95 % end-to-end service (greedy), or the ceiling reached at N ≤ 300

| Scenario | Text | Voice | Video call | Broadband |
|---|---|---|---|---|
| Rural broad | 3 | 3 | 6 | ceiling 62 % |
| Shelter hotspot | 2 | 3 | ceiling 76 % | ceiling 15 % |
| Shelter clustered | 3 | 7 | ceiling 11 % | ceiling 2 % |
| Milan crop | 5 | ceiling 39 % | ceiling 4 % | ceiling 1 % |
| Urban broad | 5 | ceiling 33 % | ceiling 3 % | ceiling 1 % |

Every ceiling equals 100 Mb/s divided by the table above (Milan voice 100/256 = 39 %): the gateway, not the drone
count, decides the outcome as soon as demand traffic exceeds the ground link. 200 Mb/s drone links (`bh200`)
change nothing, because the gateway binds first.

## Findings

1. **One satellite gateway serves text everywhere, voice only for shelters and rural areas, and video only for a
   small shelter.** For a Milan-sized crop it carries 39 % of the voice demand; no number of drones changes that.
2. **With an unlimited ground link (tether), the tree's root becomes the bottleneck.** Voice is then served
   everywhere (Milan and urban: 16 drones), but video stays at 29–34 % for Milan and urban demand and 92 % for the
   clustered shelter: the corner gateway has few children in the MST, and their 800 Mb/s links carry the whole swarm.
3. **A shared 2 GHz backhaul channel with every drone transmitting is unusable at scale** (Milan voice ≤ 16 %,
   falling with fleet size): scheduling or dedicated links are a precondition, not a refinement.
4. **Spectrum substitutes for drones on the access side.** A 20 MHz carrier (33 dBm) halves the access fleets
   (Milan voice 16 → 10, video 120 → 60; K-means at 120 m now reaches video at 100) and lifts broadband from 65 % to
   89 % at 300 drones; end to end it changes nothing while the gateway binds.

## Implications for the design question
- Size the ground link before the fleet: required gateway capacity is the demand traffic above.
- For video-class service the tree topology matters: a gateway at the corner with a distance-only MST funnels all
  traffic through one or two links. A backhaul-aware topology (central gateway, balanced subtrees) is the natural
  method contribution here.
