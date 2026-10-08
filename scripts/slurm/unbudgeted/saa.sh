#!/usr/bin/env bash
# =============================================================================
# saa.sh: tree SAA on toy, case14 at s = 45 and case14 at s = 55 (no repair budget)
#
# Runs as a 3-task array (0-2), all in parallel. Task, network, repair-cost
# scale s (REPAIR_BASE_SCALE) and output folder:
#   0  toy     (5-bus)        s = 100  toy/
#   1  case14  (IEEE 14-bus)  s = 45   case14/
#   2  case14  (IEEE 14-bus)  s = 55   case14_scale55/
#
# main.py solves the SAA extensive-form MIP on a sampled scenario tree with B = 2
# children per node (2^14 = 16384 sample paths at T = 14) and rolls the tree
# policy out on the same 500 test scenarios (seed 1000) as the DFL runs.
#
# SLURM walltime : 117h 36m (#SBATCH --time)
# Solver limit   : 116h 36m (--time-limit). The 1 h margin covers the work outside
#                  the solver: tree and model build, saving, evaluation and plots.
#
# Output: experiments/paper_unbudgeted/<folder>/SAA/
#
# Submit from the repository root:
#   sbatch scripts/slurm/unbudgeted/saa.sh
#
# Submit a single task (e.g. task 1, case14 at s = 45):
#   sbatch --array=1 scripts/slurm/unbudgeted/saa.sh
# =============================================================================
#SBATCH --job-name=paper_saa
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=117:36:00
#SBATCH --mem=112G
#SBATCH --array=0-2
#SBATCH --output=logs/unb_paper_saa_%A_%a.out
#SBATCH --error=logs/unb_paper_saa_%A_%a.err

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection. SOLVER (default gurobi) is exported as ED_OPF_SOLVER, which
# solver_config.py reads; a SOLVER containing "gurobi" also loads the Gurobi
# module. Here it picks the solver of the SAA MIP and the backend of the stage
# DC-OPF LPs in the out-of-sample evaluation. Override at submission:
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/saa.sh
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
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="unb_paper_saa_${NET_DIR}" || true

ROOT="experiments/paper_unbudgeted"
# SEED draws the failure model (ModelParameters) and samples the scenario tree.
# SEED, EVAL_SEED and N_EVAL must match dfl_multiseed.sh, so that SAA and DFL are
# scored on the same failure model and the same test scenarios.
SEED=42
EVAL_SEED=1000
N_EVAL=500
# Values for --n-scenarios. Tree mode ignores them (the tree size is B^T); they
# appear only in the log and in solver_metadata of solution.json.
N_SCEN=(  2000  300    300   )
N_SCENARIOS="${N_SCEN[$SLURM_ARRAY_TASK_ID]}"
SLURM_LIMIT=$(( 117 * 3600 + 36 * 60 ))   # 117h 36m, matches #SBATCH --time
TIME_LIMIT=$(( SLURM_LIMIT - 3600 ))       # 116h 36m, 1 h margin (see the header)

SAA_DIR="${ROOT}/${NET_DIR}/SAA"
mkdir -p "$SAA_DIR"

GLOBAL_START=$(date +%s)

echo "============================================================"
echo "SAA paper run"
echo "  Task         : ${SLURM_ARRAY_TASK_ID}"
echo "  Network      : ${NET}"
echo "  Scale        : ${SCALE}"
echo "  Mode         : tree SAA (B=2, 2^14=16384 sample paths; --n-scenarios inert)"
echo "  Solver limit : ${TIME_LIMIT}s (116h 36m)"
echo "  SLURM limit  : ${SLURM_LIMIT}s (117h 36m)"
echo "  Output       : ${SAA_DIR}"
echo "Node           : $(hostname)"
echo "CPUs           : $SLURM_CPUS_PER_TASK"
echo "Python         : $($PYTHON --version)"
echo "Started        : $(date)"
echo "------------------------------------------------------------"
echo "Job array  : $SLURM_ARRAY_JOB_ID"
echo "Monitor    : squeue -j $SLURM_ARRAY_JOB_ID"
echo "Cancel all : scancel $SLURM_ARRAY_JOB_ID"
echo "Cancel this: scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "This log   : $PROJECT_DIR/logs/unb_paper_saa_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "============================================================"
echo ""

# A task is skipped once solution.json exists. main.py writes the file right after
# the solve and adds the out-of-sample costs to it when the evaluation finishes.
if [ -f "${SAA_DIR}/solution.json" ]; then
    echo "[$(ts)] SKIP  SAA ${NET}  (solution.json already exists)"
else
    echo "[$(ts)] RUN   SAA ${NET}  [n=${N_SCENARIOS}, scale=${SCALE}, limit=${TIME_LIMIT}s]"
    T0=$(date +%s)
    # --mode tree --branching 2: multistage SAA on the sampled binary scenario tree.
    # --checkpoint-interval 0 runs one uninterrupted solve. Checkpointing restarts
    # the solver every interval and repeats presolve, which on the large tree MIP
    # uses up every chunk before branching starts. At --time-limit the solver
    # returns its best incumbent; without one, saa_solver.py falls back to the
    # NoRepair solution.
    $PYTHON main.py \
        --network           "$NET" \
        --mode              tree \
        --branching         2 \
        --n-scenarios       "$N_SCENARIOS" \
        --seed              "$SEED" \
        --time-limit        "$TIME_LIMIT" \
        --mip-gap           0.10 \
        --checkpoint-interval 0 \
        --repair-base-scale "$SCALE" \
        --output-dir        "$SAA_DIR" \
        --evaluate \
        --eval-seed         "$EVAL_SEED" \
        --n-test            "$N_EVAL"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(ts)] FAIL  SAA ${NET}  (exit code $rc)  $(elapsed $(( $(date +%s)-T0 )))"
        exit $rc
    fi
    echo "[$(ts)] DONE  SAA ${NET}  $(elapsed $(( $(date +%s)-T0 )))"
fi

GLOBAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo ""
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${NET} SAA done.  $(elapsed $GLOBAL_ELAPSED)"
echo "Results in ${SAA_DIR}"
echo "============================================================"
