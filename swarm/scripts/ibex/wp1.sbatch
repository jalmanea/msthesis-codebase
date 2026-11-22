#!/bin/bash
#SBATCH --job-name=dc-wp1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/wp1/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#
# wp1.sbatch TABLE -- ONE array task = ONE line of TABLE: "<variant> <grid> <I>/<K>".
# Always submitted by submit_wp1.sh, which writes TABLE and sizes the array. Runs
# scripts/run_wp1.py on that shard with --procs = SLURM_CPUS_PER_TASK, then
# run_wp1.py --check decides the exit code ("RESULT: PASS|FAIL shard ..."). Resumable: a re-run
# skips every cell whose CSV exists (CSVs are written atomically).
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-journal}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
TABLE="$1"; I="${SLURM_ARRAY_TASK_ID}"
[ -f "$TABLE" ] && [ -n "$I" ] || ibex_fail "usage: sbatch --array=0-\$((N-1)) wp1.sbatch TABLE (via submit_wp1.sh)"
LINE="$(sed -n "$((I + 1))p" "$TABLE")"
read -r VARIANT GRID SHARD <<< "$LINE"
[ -n "$SHARD" ] || ibex_fail "no line $((I + 1)) in $TABLE"
ibex_load_env
ibex_banner "wp1 $VARIANT $GRID shard $SHARD"
ibex_cd_repo
RC=0
python3 -u scripts/run_wp1.py --variant "$VARIANT" --grid "$GRID" --shard "$SHARD" \
    --procs "${SLURM_CPUS_PER_TASK:-1}" || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_wp1.py --variant "$VARIANT" --grid "$GRID" --shard "$SHARD" --check
