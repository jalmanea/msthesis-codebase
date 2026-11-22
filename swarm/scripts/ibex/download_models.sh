#!/bin/bash
# download_models.sh -- fetch model weights into $DC_ROOT/hf (HF_HOME) on the Ibex login node, detached.
# Dry unless --go. Progress: $DC_ROOT/logs/llm/download_<model>.log, ending in DOWNLOAD_DONE or DOWNLOAD_FAILED.
#
#   bash scripts/ibex/download_models.sh qwen3.5-4b qwen3.5-9b          # dry
#   bash scripts/ibex/download_models.sh qwen3.5-4b qwen3.5-9b --go
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
DC_ROOT="${DC_ROOT:-/ibex/user/$USER/dronecomm}"
GO=0; KEYS=()
for a in "$@"; do [ "$a" = "--go" ] && GO=1 || KEYS+=("$a"); done
mkdir -p "$DC_ROOT/logs/llm"
for k in "${KEYS[@]}"; do
    HF="$(cd "$REPO_DIR" && python3 -c "import sys; sys.path.insert(0, 'scripts'); from run_llm import MODELS; print(MODELS['$k'].hf)")"
    LOG="$DC_ROOT/logs/llm/download_$k.log"
    echo "$k -> $HF ($LOG)"
    [ "$GO" = "1" ] || continue
    setsid -f bash -c "source '$DC_ROOT/env_llm.sh' && hf download '$HF' --exclude '*.gguf' --exclude '*.pth' --exclude 'original/*' \
        && echo DOWNLOAD_DONE || echo DOWNLOAD_FAILED" > "$LOG" 2>&1 < /dev/null
done
[ "$GO" = "1" ] && echo "RESULT: PASS downloads started" || echo "RESULT: PASS dry run (add --go)"
