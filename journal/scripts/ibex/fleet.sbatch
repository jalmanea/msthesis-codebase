#!/bin/bash
#SBATCH --job-name=dc-fleet
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/fleet/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#
# fleet.sbatch TABLE -- ONE array task = ONE line of TABLE: "<variant> <I>/<K> [--extend]".
# Always submitted by submit_fleet.sh. Runs scripts/run_fleet.py on that shard with
# --procs = SLURM_CPUS_PER_TASK, then run_fleet.py --check decides the exit code. Resumable.
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-journal}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
TABLE="$1"; I="${SLURM_ARRAY_TASK_ID}"
[ -f "$TABLE" ] && [ -n "$I" ] || ibex_fail "usage: sbatch --array=0-\$((N-1)) fleet.sbatch TABLE (via submit_fleet.sh)"
read -r VARIANT SHARD EXTRA <<< "$(sed -n "$((I + 1))p" "$TABLE")"
[ -n "$SHARD" ] || ibex_fail "no line $((I + 1)) in $TABLE"
ibex_load_env
ibex_banner "fleet $VARIANT shard $SHARD $EXTRA"
ibex_cd_repo
RC=0
python3 -u scripts/run_fleet.py --variant "$VARIANT" --shard "$SHARD" $EXTRA --procs "${SLURM_CPUS_PER_TASK:-1}" || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_fleet.py --variant "$VARIANT" --shard "$SHARD" $EXTRA --check
