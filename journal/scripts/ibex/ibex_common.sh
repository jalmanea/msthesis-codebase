# ibex_common.sh -- sourced (bash) by every script in scripts/ibex/. Not executable on its own.
# Mirrors compositionswarmdaas/sim/ns3/ibex/ibex_common.sh, without the ns-3 parts.
#
# Paths (override with environment variables):
#   DC_ROOT    /ibex/user/$USER/dronecomm         project root on Ibex (NOT $HOME)
#   REPO_DIR   $DC_ROOT/dronecomm-journal         the git clone; jobs run here
#   IBEX_LOGS  $DC_ROOT/logs                      Slurm stdout/stderr (must exist before sbatch)
#   IBEX_MAIL  abdulrahman.almanea@kaust.edu.sa   --mail-type=END,FAIL
#
# No `set -u` anywhere (module scripts read unset variables). Callers use `set -eo pipefail`.

DC_ROOT="${DC_ROOT:-/ibex/user/$USER/dronecomm}"
REPO_DIR="${REPO_DIR:-$DC_ROOT/dronecomm-journal}"
IBEX_LOGS="${IBEX_LOGS:-$DC_ROOT/logs}"
IBEX_MAIL="${IBEX_MAIL:-abdulrahman.almanea@kaust.edu.sa}"

ibex_fail() {            # machine-readable verdict, non-zero exit
    echo "RESULT: FAIL $*"
    exit "${IBEX_FAIL_RC:-1}"
}

# Source env.sh (module python/3.11.0 + $DC_ROOT/venv), tolerate module noise, then assert the result.
ibex_load_env() {
    [ -f "$DC_ROOT/env.sh" ] || ibex_fail "environment: $DC_ROOT/env.sh not found (set DC_ROOT)"
    if ! type module >/dev/null 2>&1 && [ -f /etc/profile ]; then
        . /etc/profile >/dev/null 2>&1 || true
    fi
    local had_e=0
    case $- in *e*) had_e=1 ;; esac
    set +e
    # shellcheck disable=SC1091
    source "$DC_ROOT/env.sh"
    [ "$had_e" = "1" ] && set -e
    local py
    py="$(command -v python3 || true)"
    case "$py" in "$DC_ROOT/venv/"*) ;; *) ibex_fail "environment: python3 is '$py', not under $DC_ROOT/venv/";; esac
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
}

ibex_cpu_model() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1
    else
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown
    fi
}

ibex_banner() {
    echo "==================== provenance ($1) ===================="
    echo "date:        $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "hostname:    $(hostname)"
    echo "cpu model:   $(ibex_cpu_model)"
    echo "slurm:       job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none} nodelist ${SLURM_JOB_NODELIST:-none} cpus-per-task ${SLURM_CPUS_PER_TASK:-?} constraint ${SLURM_JOB_CONSTRAINTS:-?}"
    echo "python3:     $(command -v python3) $(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)"
    echo "numpy/scipy: $(python3 -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)' 2>/dev/null || echo none)"
    echo "repo HEAD:   $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)$(git -C "$REPO_DIR" diff --quiet HEAD 2>/dev/null || echo ' (DIRTY)')"
    echo "========================================================="
}

ibex_cd_repo() {
    cd "$REPO_DIR" || ibex_fail "cannot cd to $REPO_DIR (set REPO_DIR / DC_ROOT)"
}
