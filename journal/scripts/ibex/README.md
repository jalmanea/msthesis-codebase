# Running the journal-extension experiments on KAUST Ibex

Same conventions as `compositionswarmdaas/sim/ns3/ibex/`: jobs run under `/ibex/user/$USER/dronecomm`
(never `$HOME`), every job sources `env.sh` and prints a provenance banner, every script ends with
`RESULT: PASS|FAIL`, jobs are resumable, and the submit and pull scripts are dry unless given `--go`.

## Layout on Ibex

```
/ibex/user/$USER/dronecomm/
  env.sh               module load python/3.11.0; source venv/bin/activate
  venv/                Python 3.11.0: numpy, scipy, matplotlib, pyyaml, cma, pytest
  journal-ext.bundle   the last code shipped by sync_code.sh
  dronecomm-journal/   clone of branch journal-ext (from the bundle; nothing is pushed to GitHub)
    results/telecom_milan/snapshots/   the 24 Milan snapshots (rsynced, untracked)
    results/wp1/<variant>/<grid>/cell_<id>.csv
  logs/wp1/            Slurm output and task tables (must exist before sbatch)
```

## WP1 (the coverage experiments under each model variant)

```bash
# Mac: ship committed code and the snapshots
bash scripts/ibex/sync_code.sh --go
# Ibex login node
cd /ibex/user/$USER/dronecomm/dronecomm-journal
bash scripts/ibex/submit_wp1.sh            # dry: table, shard sizes, sbatch --test-only
bash scripts/ibex/submit_wp1.sh --go
bash scripts/ibex/status_wp1.sh            # per-line completion; lists lines to resubmit
bash scripts/ibex/submit_wp1.sh --array-list 3,7 --go   # after the job has left the queue
# Mac: fetch the CSVs
bash scripts/ibex/pull_results.sh --go
```

The `icc` variant (the idealised model, `configs/default.yaml`) reproduces the earlier results row for row (checked on a synthetic and a Milan cell
against `results/sweep_addition`, `results/greedy_addition`, `results/telecom_milan{,_greedy}`).

## Joint placement and pointing (branch joint-opt, its own clone)

The joint-placement runs (`scripts/run_joint.py`, journal Sec. `sec:joint`) live on branch `joint-opt`
in a second clone, `/ibex/user/$USER/dronecomm/dronecomm-joint`, sharing `env.sh` and `venv/`; logs go
to `logs/joint/`. The check stage must pass before coarse or dense is submitted.
The e2e stage reads the deployments full_coarse and full_dense saved (`.npz`), so it runs after both.

```bash
# Mac, in the joint-opt worktree
BRANCH=joint-opt REMOTE_DIR=dronecomm-joint bash scripts/ibex/sync_code.sh --go
# Ibex login node (ssh cpu)
cd /ibex/user/$USER/dronecomm/dronecomm-joint
bash scripts/ibex/submit_joint.sh --stage check --go
bash scripts/ibex/status_joint.sh
python3 scripts/run_joint.py --stage check --check     # agreements, orderings, brute force
bash scripts/ibex/submit_joint.sh --stage coarse --go
# Mac
REMOTE_DIR=dronecomm-joint PULL_DIRS=joint bash scripts/ibex/pull_results.sh --go
```
