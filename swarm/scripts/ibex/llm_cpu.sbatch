#!/bin/bash
#SBATCH --job-name=dc-llm-cpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/llm/%x_%j.out
#
# llm_cpu.sbatch STAGE -- the CPU stages of scripts/run_llm.py (l1base, l1pol): the algorithmic and scripted
# baselines of the LLM arm. Always submitted by submit_llm.sh. Resumable; run_llm.py --check decides the exit code.
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-swarm}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
STAGE="$1"
case "$STAGE" in l1base|l1pol) ;; *) ibex_fail "usage: sbatch llm_cpu.sbatch l1base|l1pol (via submit_llm.sh)" ;; esac
ibex_load_env
ibex_banner "llm $STAGE (CPU)"
ibex_cd_repo
RC=0
python3 -u scripts/run_llm.py --stage "$STAGE" --procs "${SLURM_CPUS_PER_TASK:-1}" || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_llm.py --stage "$STAGE" --check
