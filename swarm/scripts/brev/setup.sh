#!/bin/bash
# setup.sh -- on the Mac: prepare a fresh NVIDIA Brev instance for the LLM arm, as the first one was set up
# (2026-10-28): pull the pinned vLLM image, install uv, ship the committed branch and its data (sync.sh),
# build .venv, and download the model weights into the instance's Hugging Face cache. Dry unless --go;
# the image pull and the downloads run detached on the instance (logs ~/docker_pull.log, ~/hf_download.log).
#   bash scripts/brev/setup.sh qwen3.5-9b qwen3.5-4b          # dry: what would run
#   bash scripts/brev/setup.sh qwen3.5-9b qwen3.5-4b --go
#   bash scripts/brev/setup.sh --check                        # image, weights, venv on the instance
# Environment: BREV_HOST (default dc-llm), VLLM_IMAGE (default vllm/vllm-openai:v0.31.0).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
H="${BREV_HOST:-dc-llm}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
GO=0; CHECK=0; KEYS=()
for a in "$@"; do
    case "$a" in --go) GO=1 ;; --check) CHECK=1 ;; *) KEYS+=("$a") ;; esac
done
if [ "$CHECK" = "1" ]; then
    ssh -o BatchMode=yes "$H" "docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -F '$IMAGE' || echo 'image: not pulled yet'
tail -c 200 ~/docker_pull.log 2>/dev/null | tr '\r' '\n' | tail -1
tr '\r' '\n' < ~/hf_download.log 2>/dev/null | grep -E '^(done|FAILED)' || echo 'weights: none done yet'
du -sh ~/.cache/huggingface/hub/* 2>/dev/null
cd dronecomm-swarm 2>/dev/null && git log -1 --oneline && .venv/bin/python -c 'import dronecomm, numpy; print(\"venv ok\")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1 | grep -v Pseudo
    exit 0
fi
[ "${#KEYS[@]}" -gt 0 ] || { echo "RESULT: FAIL name the model keys to download (scripts/run_llm.py --list)"; exit 1; }
HFS=""
for k in "${KEYS[@]}"; do
    HFS="$HFS $(python3 -c "import sys; sys.path.insert(0, 'scripts'); from run_llm import MODELS; print(MODELS['$k'].hf)")"
done
HFS="$(echo $HFS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
echo "host $H: pull $IMAGE, install uv, sync $(git rev-parse --short HEAD), build .venv, download:$HFS"
[ "$GO" = "1" ] || { echo "RESULT: PASS dry run (add --go)"; exit 0; }
ssh -o BatchMode=yes "$H" "nohup docker pull $IMAGE > ~/docker_pull.log 2>&1 < /dev/null &
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > ~/uv_install.log 2>&1
~/.local/bin/uv --version" 2>&1 | grep -v Pseudo
bash scripts/brev/sync.sh
ssh -o BatchMode=yes "$H" "set -e; mkdir -p ~/logs; cd dronecomm-swarm
[ -x .venv/bin/python ] || ~/.local/bin/uv venv -q -p 3.11 .venv
~/.local/bin/uv pip install -q -p .venv -e '.[dev]' pandas 'huggingface_hub[hf_xet]'
.venv/bin/python -c 'import dronecomm, numpy, scipy; print(\"env ok\", numpy.__version__, scipy.__version__)'
nohup sh -c 'for m in $HFS; do .venv/bin/hf download \$m --exclude \"*.gguf\" > /dev/null && echo done \$m || echo FAILED \$m; done' > ~/hf_download.log 2>&1 < /dev/null &" 2>&1 | grep -v Pseudo
echo "RESULT: PASS setup started; follow it with: bash scripts/brev/setup.sh --check"
