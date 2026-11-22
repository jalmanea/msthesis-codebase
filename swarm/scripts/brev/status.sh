#!/bin/bash
# status.sh -- on the Mac: progress of the LLM queue on the Brev instance (units per model and stage, the
# current unit, server throughput, GPU), read-only.
#   bash scripts/brev/status.sh          # BREV_HOST (default dc-llm)
set -eo pipefail
H="${BREV_HOST:-dc-llm}"
ssh -o BatchMode=yes "$H" 'cd dronecomm-swarm
echo "queue: $(pgrep -f scripts/brev/run_queue.sh >/dev/null && echo running || echo stopped)$(pgrep -f retry_l1.sh >/dev/null && echo ", a follow-up queue waiting")   $(date -u "+%Y-%m-%d %H:%M UTC")"
for d in results/llm/smoke results/llm/l1 results/llm/l2 results/llm/l1log results/llm/l1gate; do
    [ -d "$d" ] || continue
    for m in "$d"/*/; do
        [ -d "$m" ] || continue
        case "$m" in */br-*|*/pol-*) continue ;; esac
        printf "  %-6s %-20s %3d units\n" "$(basename "$d")" "$(basename "$m")" "$(ls "$m"/*.csv 2>/dev/null | wc -l)"
    done
done
Q=$(ls -t ~/logs/queue_*.log 2>/dev/null | head -1)
echo "last lines of $(basename "$Q"):"; grep -E "^===|unit [0-9]+/|round [0-9]+|RESULT|rc=" "$Q" | tail -4 | sed "s/^/  /"
echo "failures so far:"; grep -h "RESULT: FAIL" ~/logs/queue_*.log 2>/dev/null | sed "s/^/  /" || true
L=$(ls -t results/llm/serve_logs/*.log 2>/dev/null | head -1)
[ -n "$L" ] && echo "server ($(basename "$L")): $(grep -o "Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs" "$L" | tail -1)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed "s/^/gpu: /"' 2>&1 | grep -v "Pseudo-terminal"
