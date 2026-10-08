#!/usr/bin/env bash
# =============================================================================
# smoke_2_gurobi_pipeline.sh: the smoke_1 pipeline on Gurobi
#
# Purpose : run the same toy pipeline as smoke_1 with Gurobi in every layer it
#           uses, which also checks that the Gurobi module, the license and
#           gurobipy work on a compute node. Exercises (layer names as in
#           solver_config.py):
#             - SAA in Wait-and-See mode (--mode ws): Pyomo MIP on gurobi (Layer A)
#             - DFL data generation: single-scenario MIPs in parallel workers
#               on gurobi (Layer A)
#             - out-of-sample rollouts of SAA, DFL, AlwaysRepair and NoRepair:
#               stage DC-OPF LPs through gurobipy (Layer C)
#
# The solver comes from SOLVER (default gurobi, see the solver selection below),
# and the results go to experiments/smoke_snellius/$SOLVER.
#
# Expected runtime: ~5–15 min on a rome node.
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_2_gurobi_pipeline.sh
# =============================================================================
#SBATCH --job-name=smoke2_gurobi
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke2_gurobi_%j.out
#SBATCH --error=logs/smoke2_gurobi_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Solver selection, the same block as in the scripts under scripts/slurm/unbudgeted/
# and scripts/slurm/budgeted/. SOLVER defaults to gurobi and is exported as
# ED_OPF_SOLVER, from which solver_config.py picks the solver of every layer. A
# name that contains "gurobi" also loads the Gurobi module. Override at
# submission, e.g. with another Gurobi interface:
#   sbatch --export=ALL,SOLVER=gurobi_direct scripts/slurm/smoke/smoke_2_gurobi_pipeline.sh
SOLVER="${SOLVER:-gurobi}"
if [[ "$SOLVER" == *gurobi* ]]; then
    module load Gurobi/10.0.1-GCCcore-12.3.0
    # Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
    # otherwise the cluster's default Gurobi license applies.
    [[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi
export ED_OPF_SOLVER="$SOLVER"

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
ROOT="experiments/smoke_snellius/$SOLVER"
# Separate cache per solver under $ROOT, so the smoke run never reads or writes
# the shared data_cache/. On a fresh run the cache is empty and the
# data-generation MIPs are solved by $SOLVER; a rerun reuses the cached
# scenarios, so remove $ROOT/cache to solve them again.
CACHE="$ROOT/cache/training_data.pkl"
mkdir -p "$ROOT/cache"

echo "============================================================"
echo "SMOKE 2 — tiny pipeline forced onto '$SOLVER'"
$PYTHON -c "import solver_config as s; print(' ', s.describe())"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── SAA wait-and-see MIP (toy) — Layer A ─────────────────"
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
echo "[$(ts)] ── DFL (worker MIPs = Layer A, LP rollout = Layer C) ────────"
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
echo "SMOKE 2 complete — '$SOLVER' pipeline OK (Layers A + C)."
echo "Results in $ROOT"
echo "============================================================"
