#!/usr/bin/env bash
# =============================================================================
# sddip.sh: SDDiP on toy, case14 at s = 45 and case14 at s = 55 (no repair budget)
#
# Runs as a 3-task array (0-2), all in parallel. Task, network, repair-cost
# scale s (REPAIR_BASE_SCALE) and output folder:
#   0  toy     (5-bus)        s = 100  toy/
#   1  case14  (IEEE 14-bus)  s = 45   case14/
#   2  case14  (IEEE 14-bus)  s = 55   case14_scale55/
#
# SDDiP_main.py trains an SDDP.jl policy with Lagrangian cuts (its default duality
# handler). The iteration limit (10000) is effectively unlimited, so the time
# limit below stops training. SDDiP_main.py then writes the cuts (cuts.json) and
# the bounds and gap (sddip_solution.json), simulates the trained policy (500
# replications) for the expected repairs and dispatch, and overwrites
# sddip_solution.json with the full solution.
#
# SLURM walltime : 117h 36m (#SBATCH --time)
# Training limit : 109h 36m (--hard-time-limit). The 8 h margin lets training
#                  return and the cuts, bounds and policy simulation be written
#                  before SLURM kills the job.
#
# Output: experiments/paper_unbudgeted/<folder>/SDDiP/
#
# Submit from the repository root:
#   sbatch scripts/slurm/unbudgeted/sddip.sh
#
# Submit a single task (e.g. task 1, case14 at s = 45):
#   sbatch --array=1 scripts/slurm/unbudgeted/sddip.sh
# =============================================================================
#SBATCH --job-name=paper_sddip
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=117:36:00
#SBATCH --mem=112G
#SBATCH --array=0-2
#SBATCH --output=logs/unb_paper_sddip_%A_%a.out
#SBATCH --error=logs/unb_paper_sddip_%A_%a.err

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection. SOLVER (default gurobi) is exported as ED_OPF_SOLVER, which
# solver_config.py reads; a SOLVER containing "gurobi" also loads the Gurobi
# module. Here it picks the optimizer of the SDDP.jl subproblems; Gurobi needs
# Gurobi.jl in the Julia environment (scripts/slurm/install_env.sh adds it).
# Override at submission:
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/sddip.sh
SOLVER="${SOLVER:-gurobi}"
if [[ "$SOLVER" == *gurobi* ]]; then
    module load Gurobi/10.0.1-GCCcore-12.3.0
    # Use ~/gurobi.lic (e.g. a personal WLS license) if it exists, otherwise the
    # cluster's default Gurobi license.
    [[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi
export ED_OPF_SOLVER="$SOLVER"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dh %02dm %02ds" $((s/3600)) $(( (s%3600)/60 )) $((s%60)); }

# ---------------------------------------------------------------------------
# Network / scale tables
# ---------------------------------------------------------------------------
# Index 2 runs case14 a second time, at scale 55 (index 1 is case14 at scale 45).
# NET_DIRS sends it to the scale-tagged folder case14_scale55, so the two scales
# never share an output tree.
NETWORKS=(toy   case14 case14)
SCALES=(  100   45     55    )
NET_DIRS=( toy   case14 case14_scale55)

NET="${NETWORKS[$SLURM_ARRAY_TASK_ID]}"
SCALE="${SCALES[$SLURM_ARRAY_TASK_ID]}"
NET_DIR="${NET_DIRS[$SLURM_ARRAY_TASK_ID]}"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="unb_paper_sddip_${NET_DIR}" || true

ROOT="experiments/paper_unbudgeted"
# SEED draws the failure model (ModelParameters) and seeds Julia's RNG. It must
# match SEED in dfl_multiseed.sh, so that SDDiP solves the same instance as DFL.
SEED=42
ITERATIONS=10000                              # effectively unlimited
# Failure realizations sampled per stage (--n-samples). Networks with fewer than
# 10 components enumerate all 2^n outcomes instead, so the toy value is ignored.
N_SAMPLES=(1024  256    256   )
N_SAMP="${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}"
SLURM_LIMIT=$(( 117 * 3600 + 36 * 60 ))       # 117h 36m, matches #SBATCH --time
HARD_LIMIT=$(( SLURM_LIMIT - 8 * 3600 ))      # 109h 36m, 8 h margin (see the header)

SDDIP_DIR="${ROOT}/${NET_DIR}/SDDiP"
mkdir -p "$SDDIP_DIR"

GLOBAL_START=$(date +%s)

echo "============================================================"
echo "SDDiP paper run"
echo "  Task         : ${SLURM_ARRAY_TASK_ID}"
echo "  Network      : ${NET}"
echo "  Scale        : ${SCALE}"
echo "  Iterations   : ${ITERATIONS} (effectively unlimited)"
echo "  Solver limit : ${HARD_LIMIT}s (109h 36m, SDDP.jl training only)"
echo "  SLURM limit  : ${SLURM_LIMIT}s (117h 36m)"
echo "  Output       : ${SDDIP_DIR}"
echo "Node           : $(hostname)"
echo "CPUs           : $SLURM_CPUS_PER_TASK"
echo "Python         : $($PYTHON --version)"
echo "Started        : $(date)"
echo "------------------------------------------------------------"
echo "Job array  : $SLURM_ARRAY_JOB_ID"
echo "Monitor    : squeue -j $SLURM_ARRAY_JOB_ID"
echo "Cancel all : scancel $SLURM_ARRAY_JOB_ID"
echo "Cancel this: scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "This log   : $PROJECT_DIR/logs/unb_paper_sddip_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "============================================================"
echo ""

# A task is skipped once sddip_solution.json exists. SDDiP_main.py writes the file
# as soon as training ends (bounds and gap) and overwrites it after the simulation.
if [ -f "${SDDIP_DIR}/sddip_solution.json" ]; then
    echo "[$(ts)] SKIP  SDDiP ${NET}  (sddip_solution.json already exists)"
else
    echo "[$(ts)] RUN   SDDiP ${NET}  [iter=${ITERATIONS}, scale=${SCALE}, limit=${HARD_LIMIT}s]"
    T0=$(date +%s)
    $PYTHON -u SDDiP_main.py \
        --network           "$NET" \
        --iterations        "$ITERATIONS" \
        --n-samples         "$N_SAMP" \
        --hard-time-limit   "$HARD_LIMIT" \
        --seed              "$SEED" \
        --repair-base-scale "$SCALE" \
        --output-dir        "$SDDIP_DIR" \
        --no-plots
    echo "[$(ts)] DONE  SDDiP ${NET}  $(elapsed $(( $(date +%s)-T0 )))"
fi

GLOBAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo ""
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${NET} SDDiP done.  $(elapsed $GLOBAL_ELAPSED)"
echo "Results in ${SDDIP_DIR}"
echo "============================================================"
