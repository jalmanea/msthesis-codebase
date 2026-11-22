# Half-duplex 60 GHz links (2026-11-04)

Why: the satellite gateway now carries two-way calls (text, voice, video) in both directions, so its 20 Mb/s
upload binds. The drone-to-drone links had still carried one direction only. 802.11ad links are time-division
duplexed, so a call's return stream takes the same airtime: for the two-way tiers a link carries half its rate
(400 of 800 Mb/s) in the direction the tree carries; broadband (downlink) keeps 800 Mb/s.
Code: `BackhaulConfig.half_duplex` (`configs/grounded.yaml`: true) and `two_way` (set per cell from the tier by
`run_fleet.py` and `run_joint.py`), commit `7fd467d`.

## Runs (Ibex, repo `7fd467d`, all `RESULT: PASS`)

| Stage | Job | Tasks | Ibex output |
|---|---|---|---|
| fleet, coarse (tether, grounded, bh200, bh_shared) | 55162173 | 24 | `results/hd/fleet` |
| fleet, `--extend` (tether, grounded) | 55162599 | 8 | `results/hd/fleet_ext` |
| fleet, `--refine --refine-variants tether,grounded` | 55163465 | 16 | `results/hd/fleet_refine` |
| joint, `--stage e2e` | 55162185 | 16 | `results/joint/e2e` (the earlier run moved to `results/joint/e2e_fd`) |

On the Mac the new results are under the standard names; the full-duplex ones are kept:

| Standard path (half-duplex) | Full-duplex run, kept as |
|---|---|
| `results/fleet/{tether,grounded,bh200,bh_shared}` | `results/fleet_fd/…` |
| `results/fleet_ext/{tether,grounded}` | `results/fleet_ext_fd/…` |
| `results/fleet_refine/{tether,grounded}` | `results/fleet_refine_fd/…` |
| `results/joint/e2e` | `results/joint/e2e_fd` |

The other sensitivity variants (waiver300, bw20, lowincome, cap40, suburban, dense_urban) were not rerun; the
paper uses only their access side, which half-duplex does not touch.

## Checks

- Access side identical: coarse runs to 9e-14 and refinement runs to 6e-14 (142,800 rows).
- Refinement windows regenerated from the new coarse + extend results equal `configs/refine_windows.json` (65/65).
- Broadband end to end unchanged except 7 of about 35,000 rows (at most 0.31 pp), where the balanced tree breaks a
  load tie differently: float differences between nodes, as noted in `docs/fleet_r320.md`.
- Mac smoke test before submission (urban, seed 1000): video end to end at N = 60, balanced 72.4 -> 40.5 %.

## What changed

- Gateway bound (Proposition 2) for the two-way tiers: 44 + 3 x 400 = 1,244 Mb/s (broadband 2,444); busiest-tile
  video 27.5 %, urban 40.5 %. It binds in 10 of 36 cells; the balanced tree is within 0.2 pp in 8.
- Tethered video ceilings (corner / central MST / balanced): 75th-percentile tile 29 / 53 / 73 %, urban
  16 / 33 / 40 %, busiest tile 11 / 20 / 27 %. The 75th-percentile tile's video is no longer served end to end.
- Voice: the corner MST stops at 93 % on the busiest tile; the cheapest tree (central MST) and its relays are
  unchanged.
- Joint placement end to end: four cells served; total fleet with relays falls by 16-26 %.
