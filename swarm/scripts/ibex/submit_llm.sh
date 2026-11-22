#!/bin/bash
# submit_llm.sh -- submit one LLM-agent job per model (scripts/ibex/llm.sbatch). Ibex GPU login node (ssh gpu),
# swarm-agents clone. Dry unless --go. Models must already be in $DC_ROOT/hf (download_models.sh).
#
#   bash scripts/ibex/submit_llm.sh --stage smoke --models qwen3.5-4b
#   bash scripts/ibex/submit_llm.sh --stage l1 --models qwen3.5-4b,qwen3.5-9b --go
#   bash scripts/ibex/submit_llm.sh --stage l1pol --go        # CPU baselines (l1base, l1pol): llm_cpu.sbatch, no models
# Options: --stage smoke|l1|l2|l1base|l1pol (required), --models a,b (required for the GPU stages), --time T.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
STAGE=""; MODELS=""; GO=0; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --go) GO=1; shift ;;
        *) ibex_fail "unknown option $1" ;;
    esac
done
case "$STAGE" in smoke|l1|l2|l1base|l1pol) ;; *) ibex_fail "--stage smoke|l1|l2|l1base|l1pol is required" ;; esac
ibex_load_env
ibex_cd_repo
git diff --quiet HEAD || ibex_fail "$REPO_DIR has uncommitted changes: jobs must run a committed HEAD"
if [ "$STAGE" = l1base ] || [ "$STAGE" = l1pol ]; then
    mkdir -p "$IBEX_LOGS/llm"
    echo "repo HEAD:  $(git rev-parse HEAD)"
    python3 scripts/run_llm.py --stage "$STAGE" --list
    CMD=(sbatch --job-name="dc-llm-$STAGE" ${TIME:+--time=$TIME} scripts/ibex/llm_cpu.sbatch "$STAGE")
    echo "command:    ${CMD[*]}"
    if [ "$GO" = "1" ]; then
        "${CMD[@]}"; echo "RESULT: PASS submitted"
    else
        sbatch --test-only ${TIME:+--time=$TIME} scripts/ibex/llm_cpu.sbatch "$STAGE" 2>&1 | tail -n 1
        echo "RESULT: PASS dry run (nothing submitted; add --go)"
    fi
    exit 0
fi
[ -n "$MODELS" ] || ibex_fail "--models is required"
[ -f "$DC_ROOT/env_llm.sh" ] && [ -d "$DC_ROOT/venv-llm" ] || ibex_fail "no vLLM environment (env_llm.sh, venv-llm)"
mkdir -p "$IBEX_LOGS/llm"
echo "repo HEAD:  $(git rev-parse HEAD)"
for m in ${MODELS//,/ }; do
    TP="$(python3 -c "import sys; sys.path.insert(0, 'scripts'); from run_llm import MODELS; MODELS['$m']; print(1)")"   # every model of the table fits one GPU \
        || ibex_fail "unknown model $m"
    python3 scripts/run_llm.py --model "$m" --stage "$STAGE" --list
    CMD=(sbatch --job-name="dc-llm-$m" --gres="gpu:a100:$TP" ${TIME:+--time=$TIME} scripts/ibex/llm.sbatch "$m" "$STAGE")
    echo "command:    ${CMD[*]}"
    if [ "$GO" = "1" ]; then
        "${CMD[@]}"
    else
        sbatch --test-only --gres="gpu:a100:$TP" ${TIME:+--time=$TIME} scripts/ibex/llm.sbatch "$m" "$STAGE" 2>&1 | tail -n 1
    fi
done
[ "$GO" = "1" ] && echo "RESULT: PASS submitted" || echo "RESULT: PASS dry run (nothing submitted; add --go)"
