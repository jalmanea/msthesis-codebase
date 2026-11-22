#!/bin/bash
# sync_code.sh -- run on the Mac. Ship the committed journal-ext branch to Ibex as a git bundle
# (nothing is pushed to GitHub) and the Milan snapshots (results/, not tracked). Dry by default.
#
#   bash scripts/ibex/sync_code.sh          # show what would happen
#   bash scripts/ibex/sync_code.sh --go     # bundle -> scp -> clone or fast-forward on Ibex; rsync snapshots
# Environment: IBEX_SSH (default cpu, the ~/.ssh/config alias), IBEX_ROOT (/ibex/user/elmaneaa/dronecomm),
# BRANCH (journal-ext), REMOTE_DIR (dronecomm-journal; the clone under IBEX_ROOT). The joint-placement
# branch has its own clone:
#   BRANCH=joint-opt REMOTE_DIR=dronecomm-joint bash scripts/ibex/sync_code.sh --go
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IBEX_SSH="${IBEX_SSH:-cpu}"
IBEX_ROOT="${IBEX_ROOT:-/ibex/user/elmaneaa/dronecomm}"
BRANCH="${BRANCH:-journal-ext}"
REMOTE_DIR="${REMOTE_DIR:-dronecomm-journal}"
BUNDLE="$BRANCH.bundle"
GO=0; [ "${1:-}" = "--go" ] && GO=1
cd "$REPO"
git diff --quiet HEAD || { echo "RESULT: FAIL uncommitted changes in $REPO: commit first"; exit 1; }
HEAD="$(git rev-parse "$BRANCH")"
SNAP="$REPO/results/telecom_milan/snapshots/"
echo "branch $BRANCH at $HEAD -> $IBEX_SSH:$IBEX_ROOT/$REMOTE_DIR"
echo "snapshots $SNAP -> $IBEX_SSH:$IBEX_ROOT/$REMOTE_DIR/results/telecom_milan/snapshots/"
[ "$GO" = "1" ] || { echo "RESULT: PASS dry run (add --go)"; exit 0; }
B="$(mktemp -t dc-journal).bundle"
git bundle create "$B" "$BRANCH"
scp -q "$B" "$IBEX_SSH:$IBEX_ROOT/$BUNDLE"
rm -f "$B"
ssh "$IBEX_SSH" "set -e; cd $IBEX_ROOT
if [ -d $REMOTE_DIR/.git ]; then
  cd $REMOTE_DIR; git diff --quiet HEAD || { echo 'RESULT: FAIL remote clone has local changes'; exit 1; }
  git fetch -q ../$BUNDLE $BRANCH:refs/remotes/bundle/$BRANCH && git merge -q --ff-only bundle/$BRANCH
else
  git clone -q -b $BRANCH $BUNDLE $REMOTE_DIR
fi
cd $IBEX_ROOT/$REMOTE_DIR; test \"\$(git rev-parse HEAD)\" = $HEAD && echo \"remote HEAD \$(git rev-parse HEAD)\""
ssh "$IBEX_SSH" "mkdir -p $IBEX_ROOT/$REMOTE_DIR/results/telecom_milan/snapshots $IBEX_ROOT/$REMOTE_DIR/results/telecom_milan/tiles $IBEX_ROOT/logs/wp1 $IBEX_ROOT/logs/fleet $IBEX_ROOT/logs/joint $IBEX_ROOT/logs/swarm"
rsync -a --itemize-changes -L "$SNAP" "$IBEX_SSH:$IBEX_ROOT/$REMOTE_DIR/results/telecom_milan/snapshots/"
if [ -d "$REPO/results/telecom_milan/tiles" ]; then                 # scripts/extract_milan_tiles.py
    rsync -a --itemize-changes "$REPO/results/telecom_milan/tiles/" "$IBEX_SSH:$IBEX_ROOT/$REMOTE_DIR/results/telecom_milan/tiles/" | tail -n 2
fi
echo "RESULT: PASS synced $HEAD"
