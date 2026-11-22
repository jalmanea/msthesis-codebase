#!/bin/bash
#SBATCH --job-name=telecom_stat
#SBATCH --output=logs/telecom_stat_%A_%a.out
#SBATCH --error=logs/telecom_stat_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --mail-user=abdulrahman.almanea@kaust.edu.sa
#SBATCH --mail-type=ALL

# ============================================================================
# Statistically Rigorous Telecom Experiment
#
# Phase: main
#   Task grid: 48 snapshots × 26 drone counts × 3 user densities = 3744 tasks
#   Per task:  6 methods × 30 seeds = 180 evaluations
#   Wall time: ~2-90 min (worst case N=30, M=800 with greedy: ~90-140 min on Ibex)
#   Submit:    sbatch --array=0-3743 scripts/submit_telecom_statistical.sh
#
# Phase: sensitivity
#   Task grid: 48 snapshots (all hyperparams within each)
#   Per task:  (6 beta + 4 tier) × 20 seeds = 200 evaluations (~3 min)
#   Submit:    sbatch --array=0-47 scripts/submit_telecom_statistical.sh --phase sensitivity
#
# Without greedy (much faster — worst case becomes <2 min instead of 90 min):
#   sbatch --array=0-3743 scripts/submit_telecom_statistical.sh --no-greedy
#
# More seeds for higher power:
#   sbatch --array=0-3743 scripts/submit_telecom_statistical.sh --n-seeds 50
#
# Prerequisites:
#   1. Extract snapshots: python scripts/extract_telecom_snapshots.py
#   2. mkdir -p logs
# ============================================================================

set -e

echo "=============================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "=============================================="

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

python scripts/run_telecom_statistical.py \
    --task-id "$TASK_ID" \
    "$@"

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
