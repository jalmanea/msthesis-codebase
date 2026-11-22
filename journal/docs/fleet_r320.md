# Backhaul rerun at a 320 m link range (2026-11-03)

Why: SkyHaul (arXiv 2006.11307) reports no 60 GHz link past 360 m, where the median is about 200 Mb/s
(Fig. 4); 800 Mb/s holds on the median to about 320 m. `configs/grounded.yaml` changed `link_range_m`
400 → 320 at `82349a8` (`docs/parameters.md` B3; audit: `docs/citation_audit_2026-11-02.md`).
Everything else is unchanged, including the 384 kb/s video tier.

## Runs (Ibex, repo `82349a8`, all `RESULT: PASS`)

| Stage | Job | Tasks | Cells / units | Ibex output |
|---|---|---|---|---|
| fleet, coarse | 55037260 | 60 (10 variants × 6 shards) | 16,000 | `results/r320/fleet` |
| fleet, `--extend` | 55038796 | 8 (grounded, tether) | 780 | `results/r320/fleet_ext` |
| fleet, `--refine` | 55039254 | 48 | 6,968 | `results/r320/fleet_refine` |
| joint, `--stage e2e` | 55037289 | 16 | 24,096 | `results/joint/e2e` in the `dronecomm-journal` clone, reading `full_coarse` and `full_dense` of the `dronecomm-joint` clone through symlinks |

On the Mac the 320 m results are in place under the standard names and the 400 m results kept:

| Standard path (320 m) | 400 m run, kept as |
|---|---|
| `results/fleet` | `results/fleet_v3` (the run of `docs/fleet_v3.md`) |
| `results/fleet_ext` | `results/fleet_ext_v1` |
| `results/fleet_refine` | `results/fleet_refine_v1` |
| `results/joint/e2e` | `results/joint/e2e_v2` |

## Checks against the 400 m run

- Access side reproduces exactly (differences ≤ 1e-13) in coarse (1,392,000 rows), refine (390,936) and
  the joint served shares (61,344).
- Refine windows regenerated from the 320 m coarse + extend results equal `configs/refine_windows.json` (65/65).
- MST end-to-end shares are unchanged (a tree's edges do not depend on the range; only its relay count does);
  satellite-gateway shares are unchanged; balanced-tree tether shares change in about 1.5 % of rows
  (up to 2.9 pp refine, 5.1 pp joint); the shared-channel sensitivity moves most.
- Relays roughly double: mean per row, refine MST 1.91 → 4.22, balanced 2.64 → 6.40.
- Joint e2e vs fleet refine, greedy: shares identical; relay counts differ by one in 10 of 19,976 runs
  (nine on busiest-tile video at N ≥ 200), most likely where the balanced tree breaks a load tie that
  differs by float noise. `scripts/journal_numbers.py` reports this count instead of requiring equality.
- `scripts/journal_numbers.py`: 214 checks, 0 problems.
