#!/usr/bin/env bash
# =============================================================================
# smoke_3_sddip_highs.sh: small SDDiP run on toy with HiGHS (Layer B)
#
# Purpose : check the Julia/SDDP.jl bridge on its own, without Gurobi: juliacall
#           starts, the Julia packages (SDDP, JuMP, HiGHS, JSON) load, the model
#           builds, training runs for 5 iterations or until the 600 s
#           --hard-time-limit is reached, and the results are written.
#
# The first Julia call precompiles the packages (~3–4 min); once they are
# compiled into the Julia depot, later runs skip this step.
#
# Expected runtime: ~10–20 min on a rome node (Julia startup dominates).
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_3_sddip_highs.sh
# =============================================================================
#SBATCH --job-name=smoke3_sddip
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke3_sddip_%j.out
#SBATCH --error=logs/smoke3_sddip_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dm %02ds" $(( s/60 )) $(( s%60 )); }

SEED=42
ROOT="experiments/smoke_snellius/sddip_highs"
mkdir -p "$ROOT"

echo "============================================================"
echo "SMOKE 3 — tiny SDDiP (HiGHS) on toy"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── SDDiP toy (5 iters; Julia startup may take ~3–4 min) ─────"
T0=$(date +%s)
$PYTHON SDDiP_main.py \
    --network toy \
    --solver highs \
    --iterations 5 \
    --hard-time-limit 600 \
    --seed "$SEED" \
    --output-dir "$ROOT" \
    --no-plots
echo "[$(ts)] DONE  SDDiP  $(elapsed $(( $(date +%s)-T0 )))"

echo ""
echo "============================================================"
echo "SMOKE 3 complete — SDDiP / SDDP.jl (HiGHS) OK."
echo "Results in $ROOT"
echo "============================================================"
