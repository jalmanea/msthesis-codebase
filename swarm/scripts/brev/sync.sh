#!/bin/bash
# sync.sh -- on the Mac: ship the committed swarm-agents branch to the Brev instance as a git bundle (nothing is
# pushed) with the data the LLM arm reads (results/swarm/central, the Milan tiles), or pull its results back.
#   bash scripts/brev/sync.sh            # push code + data
#   bash scripts/brev/sync.sh --pull     # results/llm back to the Mac
# Environment: BREV_HOST (default dc-llm, the alias `brev refresh` writes to ~/.brev/ssh_config).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
H="${BREV_HOST:-dc-llm}"
if [ "${1:-}" = "--pull" ]; then
    rsync -a "$H:dronecomm-swarm/results/llm/" results/llm/ && echo "RESULT: PASS pulled results/llm"; exit 0
fi
git diff --quiet HEAD || { echo "RESULT: FAIL uncommitted changes: commit first"; exit 1; }
B="$(mktemp -t swarm).bundle"
git bundle create "$B" swarm-agents
scp -q "$B" "$H:swarm.bundle"; rm -f "$B"
ssh "$H" 'set -e; [ -d dronecomm-swarm/.git ] || git clone -q -b swarm-agents ~/swarm.bundle dronecomm-swarm
cd dronecomm-swarm; git diff --quiet HEAD || { echo "RESULT: FAIL remote clone has local changes"; exit 1; }
git fetch -q ~/swarm.bundle swarm-agents:refs/remotes/bundle/swarm-agents && git merge -q --ff-only bundle/swarm-agents
mkdir -p results/swarm results/telecom_milan; git log -1 --oneline'
rsync -a -L results/swarm/central "$H:dronecomm-swarm/results/swarm/"
rsync -a -L results/telecom_milan/snapshots results/telecom_milan/tiles "$H:dronecomm-swarm/results/telecom_milan/"
echo "RESULT: PASS synced $(git rev-parse HEAD)"
