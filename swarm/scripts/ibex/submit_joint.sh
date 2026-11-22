#!/bin/bash
# submit_joint.sh -- build the joint-placement task table and (only with --go) submit it as one Slurm array.
# Run on the Ibex LOGIN node (ssh cpu), in the joint-opt clone. Same conventions as submit_fleet.sh.
#
#   bash scripts/ibex/submit_joint.sh --stage check            # DEFAULT: dry (table, sizes, sbatch --test-only)
#   bash scripts/ibex/submit_joint.sh --stage check --go
# Options: --stage check|coarse|dense|full_coarse|full_dense (required), --shards K (default 1 for check, 16 otherwise),
#          --array-list L, --cap N (default 24), --time T (sbatch limit).
# Submit any stage but check only after `run_joint.py --stage check --check` passes.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
STAGE=""; K=""; ALIST=""; CAP=24; GO=0; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --shards) K="$2"; shift 2 ;;
        --array-list) ALIST="$2"; shift 2 ;;
        --cap) CAP="$2"; shift 2 ;;
        --go) GO=1; shift ;;
        --time) TIME="$2"; shift 2 ;;
        *) ibex_fail "unknown option $1" ;;
    esac
done
case "$STAGE" in check|coarse|dense|full_coarse|full_dense) ;; *) ibex_fail "--stage check|coarse|dense|full_coarse|full_dense is required" ;; esac
[ -n "$K" ] || { [ "$STAGE" = check ] && K=1 || K=16; }
ibex_load_env
ibex_cd_repo
if squeue -h -u "$USER" -n dc-joint 2>/dev/null | grep -q .; then
    ibex_fail "a dc-joint job is already queued or running (squeue -u $USER -n dc-joint)"
fi
git diff --quiet HEAD || ibex_fail "$REPO_DIR has uncommitted changes: jobs must run a committed HEAD"
[ -f results/telecom_milan/tiles/tiles_metadata.json ] || ibex_fail "Milan tiles missing (run sync_code.sh on the Mac)"
if [ "$STAGE" != check ]; then
    python3 scripts/run_joint.py --stage check --check >/dev/null || ibex_fail "the check stage has not passed (run_joint.py --stage check --check)"
fi
mkdir -p "$IBEX_LOGS/joint"
TABLE="$IBEX_LOGS/joint/table_${STAGE}_$(date '+%Y%m%d_%H%M%S').txt"
: > "$TABLE"
for i in $(seq 0 $((K - 1))); do echo "$STAGE $i/$K" >> "$TABLE"; done
N=$(wc -l < "$TABLE")
echo "repo HEAD:  $(git rev-parse HEAD)"
echo "table:      $TABLE ($N tasks)"
i=0
while read -r st s; do
    printf '  %3d  ' "$i"; python3 scripts/run_joint.py --stage "$st" --shard "$s" --list
    i=$((i + 1))
done < "$TABLE"
ARRAY="0-$((N - 1))"; [ -n "$ALIST" ] && ARRAY="$ALIST"
CMD=(sbatch --array="$ARRAY%$CAP" ${TIME:+--time=$TIME} scripts/ibex/joint.sbatch "$TABLE")
echo "command:    ${CMD[*]}"
if [ "$GO" = "1" ]; then
    OUT="$("${CMD[@]}")"; echo "$OUT"
    printf 'TABLE=%s\nJOB=%s\nHEAD=%s\nSTAGE=%s\n' "$TABLE" "${OUT##* }" "$(git rev-parse HEAD)" "$STAGE" > "$IBEX_LOGS/joint/last_submit.env"
    echo "RESULT: PASS submitted"
else
    sbatch --test-only --array="$ARRAY%$CAP" ${TIME:+--time=$TIME} scripts/ibex/joint.sbatch "$TABLE" 2>&1 | tail -n 1
    echo "RESULT: PASS dry run (nothing submitted; add --go)"
fi
