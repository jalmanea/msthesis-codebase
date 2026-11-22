#!/bin/bash
# submit_wp1.sh -- build the WP1 task table and (only with --go) submit it as one Slurm array.
# Run on the Ibex LOGIN node, from anywhere. Light: seconds.
#
#   bash scripts/ibex/submit_wp1.sh                 # DEFAULT: table + shard sizes + sbatch --test-only;
#                                                   #   submits NOTHING
#   bash scripts/ibex/submit_wp1.sh --go            # submit
# Options:
#   --variants "a b"   variants (default: all of run_wp1.VARIANTS)
#   --grids "a b"      grids (default: synthetic milan)
#   --shards-synthetic K   shards per variant for the synthetic grid (default 1)
#   --shards-milan K       shards per variant for the Milan grid (default 2)
#   --array-list L     submit only these table lines (0-based), e.g. 0,3,5 (resubmissions)
#   --cap N            concurrent tasks (%N; default 24)
# Refuses if a dc-wp1 job is queued or running. Writes $IBEX_LOGS/wp1/table_<stamp>.txt and
# $IBEX_LOGS/wp1/last_submit.env (the table) for status_wp1.sh.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"

VARIANTS=""; GRIDS="synthetic milan"; KS=1; KM=2; ALIST=""; CAP=24; GO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --variants) VARIANTS="$2"; shift 2 ;;
        --grids) GRIDS="$2"; shift 2 ;;
        --shards-synthetic) KS="$2"; shift 2 ;;
        --shards-milan) KM="$2"; shift 2 ;;
        --array-list) ALIST="$2"; shift 2 ;;
        --cap) CAP="$2"; shift 2 ;;
        --go) GO=1; shift ;;
        *) ibex_fail "unknown option $1" ;;
    esac
done
ibex_load_env
ibex_cd_repo
[ -n "$VARIANTS" ] || VARIANTS="$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import run_wp1; print(" ".join(run_wp1.VARIANTS))')"
if squeue -h -u "$USER" -n dc-wp1 2>/dev/null | grep -q .; then
    ibex_fail "a dc-wp1 job is already queued or running (squeue -u $USER -n dc-wp1)"
fi
git diff --quiet HEAD || ibex_fail "$REPO_DIR has uncommitted changes: jobs must run a committed HEAD"
[ -f results/telecom_milan/snapshots/snapshot_metadata.json ] || ibex_fail "Milan snapshots missing (run sync_code.sh on the Mac)"

mkdir -p "$IBEX_LOGS/wp1"
TABLE="$IBEX_LOGS/wp1/table_$(date '+%Y%m%d_%H%M%S').txt"
: > "$TABLE"
for v in $VARIANTS; do
    for g in $GRIDS; do
        K=$KS; [ "$g" = "milan" ] && K=$KM
        for i in $(seq 0 $((K - 1))); do echo "$v $g $i/$K" >> "$TABLE"; done
    done
done
N=$(wc -l < "$TABLE")
echo "repo HEAD:  $(git rev-parse HEAD)"
echo "table:      $TABLE ($N tasks)"
i=0
while read -r v g s; do
    printf '  %3d  ' "$i"; python3 scripts/run_wp1.py --variant "$v" --grid "$g" --shard "$s" --list
    i=$((i + 1))
done < "$TABLE"
ARRAY="0-$((N - 1))"; [ -n "$ALIST" ] && ARRAY="$ALIST"
CMD=(sbatch --array="$ARRAY%$CAP" scripts/ibex/wp1.sbatch "$TABLE")
echo "command:    ${CMD[*]}"
if [ "$GO" = "1" ]; then
    OUT="$("${CMD[@]}")"; echo "$OUT"
    printf 'TABLE=%s\nJOB=%s\nHEAD=%s\n' "$TABLE" "${OUT##* }" "$(git rev-parse HEAD)" > "$IBEX_LOGS/wp1/last_submit.env"
    echo "RESULT: PASS submitted"
else
    sbatch --test-only --array="$ARRAY%$CAP" scripts/ibex/wp1.sbatch "$TABLE" 2>&1 | tail -n 2
    echo "RESULT: PASS dry run (nothing submitted; add --go)"
fi
