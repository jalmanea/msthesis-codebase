#!/bin/bash
# status_joint.sh -- queue state and per-line completion of the last joint submission (login node).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
ibex_load_env
ibex_cd_repo
[ -f "$IBEX_LOGS/joint/last_submit.env" ] || ibex_fail "no submission recorded in $IBEX_LOGS/joint/last_submit.env"
# shellcheck disable=SC1091
. "$IBEX_LOGS/joint/last_submit.env"
echo "job $JOB, stage $STAGE, table $TABLE, HEAD $HEAD"
squeue -h -j "$JOB" -o '%.18i %.8T %.10M %.20R' 2>/dev/null | sort | uniq -c -f1 | head -n 20 || true
FAIL=""; i=0
while read -r st s; do
    if python3 scripts/run_joint.py --stage "$st" --shard "$s" --check >/dev/null 2>&1; then v=done; else v=open; FAIL="$FAIL,$i"; fi
    printf '  %3d  %-6s %s\n' "$i" "$v" "$(python3 scripts/run_joint.py --stage "$st" --shard "$s" --list)"
    i=$((i + 1))
done < "$TABLE"
[ -z "$FAIL" ] && echo "RESULT: PASS all lines complete" || echo "RESULT: FAIL open lines ${FAIL#,} (resubmit with --array-list ${FAIL#,} once the job has left the queue)"
