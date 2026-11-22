#!/bin/bash
# pull_results.sh -- run on the Mac. rsync Ibex results/{wp1,fleet} (and job logs) into this clone.
# PULL_DIRS="fleet" limits it to some result directories; REMOTE_DIR (dronecomm-journal) picks the clone:
#   REMOTE_DIR=dronecomm-joint PULL_DIRS=joint bash scripts/ibex/pull_results.sh --go
# Dry by default (rsync -n, itemized); --go copies. Never deletes anything locally; skips *.tmp*.
#
#   bash scripts/ibex/pull_results.sh            # dry
#   bash scripts/ibex/pull_results.sh --go
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IBEX_SSH="${IBEX_SSH:-cpu}"
IBEX_ROOT="${IBEX_ROOT:-/ibex/user/elmaneaa/dronecomm}"
REMOTE_DIR="${REMOTE_DIR:-dronecomm-journal}"
DRY="-n"; [ "${1:-}" = "--go" ] && DRY=""
for d in ${PULL_DIRS:-wp1 fleet}; do
    mkdir -p "$REPO/results/$d" "$REPO/results/ibex_logs/$d"
    rsync -a $DRY --stats "--exclude=*.tmp*" "$IBEX_SSH:$IBEX_ROOT/$REMOTE_DIR/results/$d/" "$REPO/results/$d/" | grep -E "Number of regular files transferred|Total transferred"
    rsync -a $DRY --stats "$IBEX_SSH:$IBEX_ROOT/logs/$d/" "$REPO/results/ibex_logs/$d/" | grep -E "Number of regular files transferred"
done
[ -n "$DRY" ] && echo "RESULT: PASS dry run (add --go)" || echo "RESULT: PASS pulled"
