#!/bin/bash
# submit_swarm.sh -- build the swarm task tables and (only with --go) submit them as Slurm arrays.
# Run on the Ibex LOGIN node (ssh cpu), in the swarm-agents clone. Same conventions as submit_joint.sh.
#
#   bash scripts/ibex/submit_swarm.sh --stage p1,p2,p3          # DEFAULT: dry (tables, sizes, sbatch --test-only)
#   bash scripts/ibex/submit_swarm.sh --stage p1,p2,p3 --go     # central array, then one array per stage after it
# Options: --stage central or a comma list of p1,p2,p3,p4a,p4b,p4c,p4d,p4e,cstrong,fstrong,rstrong,brcheck,p5speed,p5noise,fine (required; central is
#          submitted first unless every central unit is on disk, and the other stages wait for it to succeed),
#          --shards K (default 16; central and cstrong use 4, fstrong and rstrong 8), --cap N (default 24), --time T (sbatch limit).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
STAGE=""; K=16; CAP=24; GO=0; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --shards) K="$2"; shift 2 ;;
        --cap) CAP="$2"; shift 2 ;;
        --go) GO=1; shift ;;
        --time) TIME="$2"; shift 2 ;;
        *) ibex_fail "unknown option $1" ;;
    esac
done
for st in ${STAGE//,/ }; do
    case "$st" in central|p1|p2|p3|p4a|p4b|p4c|p4d|p4e|cstrong|fstrong|rstrong|brcheck|p5speed|p5noise|fine) ;;
        *) ibex_fail "--stage central or a comma list of p1,p2,p3,p4a,p4b,p4c,p4d,p4e,cstrong,fstrong,rstrong,brcheck,p5speed,p5noise,fine is required" ;; esac
done
ibex_load_env
ibex_cd_repo
if squeue -h -u "$USER" -n dc-swarm 2>/dev/null | grep -q .; then
    ibex_fail "a dc-swarm job is already queued or running (squeue -u $USER -n dc-swarm)"
fi
git diff --quiet HEAD || ibex_fail "$REPO_DIR has uncommitted changes: jobs must run a committed HEAD"
[ -f results/telecom_milan/tiles/tiles_metadata.json ] || ibex_fail "Milan tiles missing (run sync_code.sh on the Mac)"
python3 -m pytest -q tests/test_swarm.py >/dev/null || ibex_fail "tests/test_swarm.py fails"
mkdir -p "$IBEX_LOGS/swarm"
STAGES="${STAGE//,/ }"
if [ "$STAGE" != central ] && ! python3 scripts/run_swarm.py --stage central --check >/dev/null; then
    STAGES="central $STAGES"
fi
echo "repo HEAD:  $(git rev-parse HEAD)"
DEP=""
for st in $STAGES; do
    TABLE="$IBEX_LOGS/swarm/table_${st}_$(date '+%Y%m%d_%H%M%S').txt"
    KS=$K; case "$st" in central|cstrong) KS=4 ;; fstrong|rstrong) KS=8 ;; esac
    : > "$TABLE"
    for i in $(seq 0 $((KS - 1))); do echo "$st $i/$KS" >> "$TABLE"; done
    echo "table:      $TABLE ($KS tasks)"
    while read -r s sh; do printf '    '; python3 scripts/run_swarm.py --stage "$s" --shard "$sh" --list; done < "$TABLE"
    CMD=(sbatch --array="0-$((KS - 1))%$CAP" ${TIME:+--time=$TIME} ${DEP:+--dependency=afterok:$DEP} scripts/ibex/swarm.sbatch "$TABLE")
    echo "command:    ${CMD[*]}"
    if [ "$GO" = "1" ]; then
        OUT="$("${CMD[@]}")"; echo "$OUT"; [ "$st" = central ] && DEP="${OUT##* }"
        printf 'TABLE=%s\nJOB=%s\nHEAD=%s\nSTAGE=%s\n' "$TABLE" "$DEP" "$(git rev-parse HEAD)" "$st" >> "$IBEX_LOGS/swarm/last_submit.env"
    else
        [ -z "$DEP" ] && sbatch --test-only --array="0-$((KS - 1))%$CAP" ${TIME:+--time=$TIME} scripts/ibex/swarm.sbatch "$TABLE" 2>&1 | tail -n 1
        [ "$st" = central ] && DEP="(central job id)"
    fi
done
[ "$GO" = "1" ] && echo "RESULT: PASS submitted" || echo "RESULT: PASS dry run (nothing submitted; add --go)"
