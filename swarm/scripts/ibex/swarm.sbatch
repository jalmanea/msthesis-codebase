#!/bin/bash
#SBATCH --job-name=dc-swarm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/swarm/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#
# swarm.sbatch TABLE -- ONE array task = ONE line of TABLE: "<stage> <I>/<K>".
# Always submitted by submit_swarm.sh. Runs scripts/run_swarm.py on that shard with
# --procs = SLURM_CPUS_PER_TASK, then run_swarm.py --check decides the exit code. Resumable.
# Runs in the swarm-agents clone ($DC_ROOT/dronecomm-swarm).
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-swarm}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
TABLE="$1"; I="${SLURM_ARRAY_TASK_ID}"
[ -f "$TABLE" ] && [ -n "$I" ] || ibex_fail "usage: sbatch --array=0-\$((N-1)) swarm.sbatch TABLE (via submit_swarm.sh)"
read -r STAGE SHARD <<< "$(sed -n "$((I + 1))p" "$TABLE")"
[ -n "$SHARD" ] || ibex_fail "no line $((I + 1)) in $TABLE"
ibex_load_env
ibex_banner "swarm $STAGE shard $SHARD"
ibex_cd_repo
RC=0
python3 -u scripts/run_swarm.py --stage "$STAGE" --shard "$SHARD" --procs "${SLURM_CPUS_PER_TASK:-1}" || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_swarm.py --stage "$STAGE" --shard "$SHARD" --check
