#!/usr/bin/env bash
# =============================================================================
# smoke_1_highs_pipeline.sh: small end-to-end pipeline on the default solver
#
# Purpose : run the SAA and DFL entry points end-to-end on the toy (5-bus)
#           network with the default solvers (HiGHS). Exercises (layer names as
#           in solver_config.py):
#             - SAA in Wait-and-See mode (--mode ws, 5 scenarios): one Pyomo
#               MIP on appsi_highs (Layer A)
#             - DFL data generation: single-scenario MIPs in parallel workers
#               on appsi_highs (Layer A)
#             - DFL training
#             - out-of-sample rollouts of SAA, DFL, AlwaysRepair and NoRepair on
#               20 test scenarios: stage DC-OPF LPs on scipy's HiGHS (Layer C)
#             - the comparison plots
#
# The script sets no solver, so the solver_config.py defaults apply. smoke_2 runs
# the same pipeline on Gurobi.
#
# Expected runtime: ~5–15 min on a rome node.
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_1_highs_pipeline.sh
# =============================================================================
#SBATCH --job-name=smoke1_highs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke1_highs_%j.out
#SBATCH --error=logs/smoke1_highs_%j.err

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
EVAL_SEED=1000
N_EVAL=20
ROOT="experiments/smoke_snellius/highs"
# Separate cache under $ROOT, so the smoke run never reads or writes the shared
# data_cache/. On a fresh run the cache is empty and the data-generation MIPs are
# solved; a rerun reuses the cached scenarios, so remove $ROOT/cache to solve
# them again.
CACHE="$ROOT/cache/training_data.pkl"
mkdir -p "$ROOT/cache"

echo "============================================================"
echo "SMOKE 1 — tiny pipeline on the default solver"
$PYTHON -c "import solver_config as s; print(' ', s.describe())"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── SAA wait-and-see MIP (toy) ────────────────────────────"
T0=$(date +%s)
$PYTHON main.py \
    --network toy \
    --mode ws \
    --n-scenarios 5 \
    --seed "$SEED" \
    --time-limit 60 \
    --output-dir "$ROOT/SAA" \
    --evaluate --eval-seed "$EVAL_SEED" --n-test "$N_EVAL" \
    --no-plots
echo "[$(ts)] DONE  SAA  $(elapsed $(( $(date +%s)-T0 )))"

echo ""
echo "[$(ts)] ── DFL (data-gen + train + eval + compare, toy) ─────────────"
T0=$(date +%s)
$PYTHON Pseudo_DFL_main.py \
    --network toy \
    --n-train 60 \
    --n-eval "$N_EVAL" \
    --eval-seed "$EVAL_SEED" \
    --seed "$SEED" \
    --cache-path "$CACHE" \
    --saa-dir "$ROOT/SAA" \
    --output-dir "$ROOT/DFL"
echo "[$(ts)] DONE  DFL  $(elapsed $(( $(date +%s)-T0 )))"

echo ""
echo "============================================================"
echo "SMOKE 1 complete — default (HiGHS) pipeline OK."
echo "Results in $ROOT"
echo "============================================================"
