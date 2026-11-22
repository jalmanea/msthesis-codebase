#!/bin/bash
# run_queue.sh STAGE MODEL... -- on the NVIDIA Brev instance, in the clone: for each model in turn, serve it in
# the pinned vLLM image, run the smoke stage and then STAGE (both resumable), stop it. Committed HEAD only.
#   setsid nohup bash scripts/brev/run_queue.sh l1 qwen3.5-0.8b qwen3.5-2b > ~/logs/queue.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
STAGE="$1"; shift
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
git diff --quiet HEAD || { echo "RESULT: FAIL uncommitted changes: runs use a committed HEAD"; exit 1; }
echo "repo HEAD $(git rev-parse HEAD), image $VLLM_IMAGE, $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
for m in "$@"; do
    for st in smoke "$STAGE"; do
        echo "=== $m $st $(date '+%Y-%m-%d %H:%M:%S %z')"
        .venv/bin/python -u scripts/run_llm.py --model "$m" --stage "$st" --serve || echo "driver exited rc=$?"
        .venv/bin/python scripts/run_llm.py --model "$m" --stage "$st" --check || true
    done
done
