#!/bin/bash
#SBATCH --job-name=dc-llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=a100
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/user/%u/dronecomm/logs/llm/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#
# llm.sbatch MODEL STAGE -- serve MODEL with vLLM on this node's GPUs and run its units of STAGE
# (scripts/run_llm.py --serve). Always submitted by submit_llm.sh, which sets --gres from the model's
# tensor-parallel size. Resumable: finished units are skipped. Runs in the swarm-agents clone.
set -eo pipefail
REPO_DIR="${REPO_DIR:-${DC_ROOT:-/ibex/user/$USER/dronecomm}/dronecomm-swarm}"
# shellcheck source=ibex_common.sh
. "$REPO_DIR/scripts/ibex/ibex_common.sh"
MODEL="$1"; STAGE="$2"
[ -n "$MODEL" ] && [ -n "$STAGE" ] || ibex_fail "usage: sbatch llm.sbatch MODEL STAGE (via submit_llm.sh)"
# the vLLM environment, not the CPU venv: env_llm.sh activates venv-llm and sets HF_HOME
. /etc/profile >/dev/null 2>&1 || true
set +e; source "$DC_ROOT/env_llm.sh"; set -e
case "$(command -v python3)" in "$DC_ROOT/venv-llm/"*) ;; *) ibex_fail "environment: python3 is not venv-llm's" ;; esac
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ibex_banner "llm $MODEL $STAGE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
ibex_cd_repo
RC=0
python3 -u scripts/run_llm.py --model "$MODEL" --stage "$STAGE" --serve || RC=$?
echo "=== driver exited rc=$RC $(date '+%Y-%m-%d %H:%M:%S %z') ==="
python3 scripts/run_llm.py --model "$MODEL" --stage "$STAGE" --check
