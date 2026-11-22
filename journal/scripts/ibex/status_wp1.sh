#!/bin/bash
# status_wp1.sh -- queue state and per-line completion of the last WP1 submission (login node).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
ibex_load_env
ibex_cd_repo
[ -f "$IBEX_LOGS/wp1/last_submit.env" ] || ibex_fail "no submission recorded in $IBEX_LOGS/wp1/last_submit.env"
# shellcheck disable=SC1091
. "$IBEX_LOGS/wp1/last_submit.env"
echo "job $JOB, table $TABLE, HEAD $HEAD"
squeue -h -j "$JOB" -o '%.18i %.8T %.10M %.20R' 2>/dev/null | sort | uniq -c -f1 | head -n 20 || true
FAIL=""; i=0
while read -r v g s; do
    if python3 scripts/run_wp1.py --variant "$v" --grid "$g" --shard "$s" --check >/dev/null 2>&1; then st=done; else st=open; FAIL="$FAIL,$i"; fi
    printf '  %3d  %-8s %s\n' "$i" "$st" "$(python3 scripts/run_wp1.py --variant "$v" --grid "$g" --shard "$s" --list)"
    i=$((i + 1))
done < "$TABLE"
[ -z "$FAIL" ] && echo "RESULT: PASS all lines complete" || echo "RESULT: FAIL open lines ${FAIL#,} (resubmit with --array-list ${FAIL#,} once the job has left the queue)"
