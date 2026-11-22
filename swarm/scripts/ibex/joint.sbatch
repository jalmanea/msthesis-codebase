#!/bin/bash
#SBATCH --job-name=dc-joint
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/joint/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#
# joint.sbatch TABLE -- ONE array task = ONE line of TABLE: "<stage> <I>/<K>".
# Always submitted by submit_joint.sh. Runs scripts/run_joint.py on that shard with
# --procs = SLURM_CPUS_PER_TASK, then run_joint.py --check decides the exit code. Resumable.
# Runs in the joint-opt clone ($DC_ROOT/dronecomm-joint), not the journal-ext one.
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-joint}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
TABLE="$1"; I="${SLURM_ARRAY_TASK_ID}"
[ -f "$TABLE" ] && [ -n "$I" ] || ibex_fail "usage: sbatch --array=0-\$((N-1)) joint.sbatch TABLE (via submit_joint.sh)"
read -r STAGE SHARD <<< "$(sed -n "$((I + 1))p" "$TABLE")"
[ -n "$SHARD" ] || ibex_fail "no line $((I + 1)) in $TABLE"
ibex_load_env
ibex_banner "joint $STAGE shard $SHARD"
ibex_cd_repo
RC=0
python3 -u scripts/run_joint.py --stage "$STAGE" --shard "$SHARD" --procs "${SLURM_CPUS_PER_TASK:-1}" || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_joint.py --stage "$STAGE" --shard "$SHARD" --check
