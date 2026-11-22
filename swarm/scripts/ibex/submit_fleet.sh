#!/bin/bash
# submit_fleet.sh -- build the fleet-sizing task table and (only with --go) submit it as one Slurm array.
# Run on the Ibex LOGIN node. Same conventions as submit_wp1.sh.
#
#   bash scripts/ibex/submit_fleet.sh                 # DEFAULT: dry (table, sizes, sbatch --test-only)
#   bash scripts/ibex/submit_fleet.sh --go
# Options: --variants "a b" (default: all of run_fleet.VARIANTS), --shards K (per variant, default 4),
#          --array-list L, --cap N (default 24), --extend (N = 350-500 on the edge cells,
#          results/fleet_ext; see run_fleet.EXT_CELLS), --refine (dense N windows of
#          configs/refine_windows.json, all variants, results/fleet_refine), --time T (sbatch limit).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export REPO_DIR
# shellcheck source=ibex_common.sh
. "$HERE/ibex_common.sh"
VARIANTS=""; K=4; ALIST=""; CAP=24; GO=0; EXT=""; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --variants) VARIANTS="$2"; shift 2 ;;
        --shards) K="$2"; shift 2 ;;
        --array-list) ALIST="$2"; shift 2 ;;
        --cap) CAP="$2"; shift 2 ;;
        --go) GO=1; shift ;;
        --extend) EXT="--extend"; shift ;;
        --refine) EXT="--refine"; VARIANTS="all"; shift ;;
        --time) TIME="$2"; shift 2 ;;
        *) ibex_fail "unknown option $1" ;;
    esac
done
ibex_load_env
ibex_cd_repo
[ -n "$VARIANTS" ] || VARIANTS="$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import run_fleet; print(" ".join(run_fleet.VARIANTS))')"
if squeue -h -u "$USER" -n dc-fleet 2>/dev/null | grep -q .; then
    ibex_fail "a dc-fleet job is already queued or running (squeue -u $USER -n dc-fleet)"
fi
git diff --quiet HEAD || ibex_fail "$REPO_DIR has uncommitted changes: jobs must run a committed HEAD"
[ -f results/telecom_milan/tiles/tiles_metadata.json ] || ibex_fail "Milan tiles missing (run extract_milan_tiles.py and sync_code.sh on the Mac)"
mkdir -p "$IBEX_LOGS/fleet"
TABLE="$IBEX_LOGS/fleet/table_$(date '+%Y%m%d_%H%M%S').txt"
: > "$TABLE"
for v in $VARIANTS; do for i in $(seq 0 $((K - 1))); do echo "$v $i/$K $EXT" >> "$TABLE"; done; done
N=$(wc -l < "$TABLE")
echo "repo HEAD:  $(git rev-parse HEAD)"
echo "table:      $TABLE ($N tasks)"
i=0
while read -r v s x; do
    printf '  %3d  ' "$i"; python3 scripts/run_fleet.py --variant "$v" --shard "$s" $x --list
    i=$((i + 1))
done < "$TABLE"
ARRAY="0-$((N - 1))"; [ -n "$ALIST" ] && ARRAY="$ALIST"
CMD=(sbatch --array="$ARRAY%$CAP" ${TIME:+--time=$TIME} scripts/ibex/fleet.sbatch "$TABLE")
echo "command:    ${CMD[*]}"
if [ "$GO" = "1" ]; then
    OUT="$("${CMD[@]}")"; echo "$OUT"
    printf 'TABLE=%s\nJOB=%s\nHEAD=%s\n' "$TABLE" "${OUT##* }" "$(git rev-parse HEAD)" > "$IBEX_LOGS/fleet/last_submit.env"
    echo "RESULT: PASS submitted"
else
    sbatch --test-only --array="$ARRAY%$CAP" ${TIME:+--time=$TIME} scripts/ibex/fleet.sbatch "$TABLE" 2>&1 | tail -n 1
    echo "RESULT: PASS dry run (nothing submitted; add --go)"
fi
