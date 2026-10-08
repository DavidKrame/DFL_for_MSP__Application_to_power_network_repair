#!/usr/bin/env bash
# =============================================================================
# dfl_scaling.sh: DFL training-data scaling experiment, no repair budget
#
# Trains and evaluates DFL with one init seed (DEFAULT_INIT_SEED = 42 in
# config.py) at increasing training-set sizes, n_train = 25 to 120000. Each array
# task is one lane, i.e. a network with its repair-cost scale s
# (--repair-base-scale). Every run also evaluates AlwaysRepair, NoRepair and
# Near-sighted MPC on the same 500 test scenarios. A size whose evaluation.json
# already exists is skipped, so a resubmitted job runs only the missing sizes.
#
# The training data come from a cache shared by all runs with the same network,
# scale s and data seed: data_cache_scale<s>/<net>/seed42/training_data.pkl.
# Each run solves only the single-scenario MIPs that the cache is missing, so
# the data generation cost is paid once across the whole sequence.
#
# Run this before dfl_multiseed.sh, which reads the same cache. Its five seed
# tasks per lane then find all training data in place instead of solving the
# same MIPs in parallel.
#
# Array tasks (network, scale s, output folder):
#   0  toy (5-bus network)   s = 100   toy/
#   1  case14 (IEEE 14-bus)  s = 45    case14/
#   2  case14 (IEEE 14-bus)  s = 55    case14_scale55/
# Results go to experiments/paper_unbudgeted/dfl_scaling/<folder>/dfl_train<N>/.
#
# Submit all three lanes (from the repository root):
#   sbatch scripts/slurm/unbudgeted/dfl_scaling.sh
#
# Submit a single lane (e.g. case14 at s = 45, task 1):
#   sbatch --array=1 scripts/slurm/unbudgeted/dfl_scaling.sh
# =============================================================================
#SBATCH --job-name=paper_dfl_scaling
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=117:36:00
#SBATCH --mem=112G
#SBATCH --array=0-2
#SBATCH --output=logs/unb_paper_dfl_scaling_%A_%a.out
#SBATCH --error=logs/unb_paper_dfl_scaling_%A_%a.err

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection. SOLVER defaults to gurobi and is exported as ED_OPF_SOLVER,
# from which solver_config.py picks the MIP solver (training labels) and the LP
# backend (stage DC-OPF). A name that contains "gurobi" also loads the Gurobi
# module. Override at submission, e.g.
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/dfl_scaling.sh
SOLVER="${SOLVER:-gurobi}"
if [[ "$SOLVER" == *gurobi* ]]; then
    module load Gurobi/10.0.1-GCCcore-12.3.0
    # Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
    # otherwise the cluster's default Gurobi license applies.
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
# Network / scale tables  (index 0-2)
# ---------------------------------------------------------------------------
# Index 2 is a second case14 lane at scale 55, alongside case14 at scale 45 in
# index 1. NET_DIRS routes it to a scale-tagged sibling directory so the two
# scales never share an output tree.
NETWORKS=(toy case14 case14)
SCALES=(  100 45     55    )
NET_DIRS=( toy case14 case14_scale55)

NET="${NETWORKS[$SLURM_ARRAY_TASK_ID]}"
SCALE="${SCALES[$SLURM_ARRAY_TASK_ID]}"
NET_DIR="${NET_DIRS[$SLURM_ARRAY_TASK_ID]}"
# Near-sighted MPC baseline. With the config.py default MPC_SCHEDULE_CAP_RATIO = 1
# it enumerates every repair schedule on toy; on case14, a stage with more
# schedules than the cap evaluates the no-repair schedule plus cap - 1 random ones.
# The result does not depend on n_train, so the first run computes it and caches
# it next to the training data (mpc_W<W>.json); later runs read it from there.
MPC_FLAG="--with-mpc"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="unb_paper_dfl_scaling_${NET_DIR}" || true

# Increasing order is required so the cache grows incrementally
N_TRAIN_SIZES=(25 100 250 500 1000 2000 3000 5000 10000 15000 30000 60000 80000 100000 120000)

SEED=42
EVAL_SEED=1000
N_EVAL=500
ROOT="experiments/paper_unbudgeted/dfl_scaling"

GLOBAL_START=$(date +%s)

echo "============================================================"
echo "DFL scaling experiment"
echo "  Task    : ${SLURM_ARRAY_TASK_ID}"
echo "  Network : ${NET}"
echo "  Scale   : ${SCALE}"
echo "  Sizes   : ${N_TRAIN_SIZES[*]}"
echo "  Output  : ${ROOT}/${NET_DIR}/"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Python    : $($PYTHON --version)"
echo "Started   : $(date)"
echo "------------------------------------------------------------"
echo "Job array  : $SLURM_ARRAY_JOB_ID"
echo "Monitor    : squeue -j $SLURM_ARRAY_JOB_ID"
echo "Cancel all : scancel $SLURM_ARRAY_JOB_ID"
echo "Cancel this: scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "This log   : $PROJECT_DIR/logs/unb_paper_dfl_scaling_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Sequential loop: the cache grows from the smallest to the largest n_train
# ---------------------------------------------------------------------------
N_DONE=0
N_SKIP=0
N_FAIL=0

for N_TRAIN in "${N_TRAIN_SIZES[@]}"; do
    OUT_DIR="${ROOT}/${NET_DIR}/dfl_train${N_TRAIN}"
    mkdir -p "$OUT_DIR"

    if [ -f "$OUT_DIR/evaluation.json" ]; then
        echo "[$(ts)] SKIP  n_train=${N_TRAIN}  ${NET}"
        N_SKIP=$(( N_SKIP + 1 ))
        continue
    fi

    echo "[$(ts)] RUN   n_train=${N_TRAIN}  ${NET}  [scale=${SCALE}]"
    T0=$(date +%s)

    if $PYTHON Pseudo_DFL_main.py \
            --network           "$NET" \
            --n-train           "$N_TRAIN" \
            --n-eval            "$N_EVAL" \
            --eval-seed         "$EVAL_SEED" \
            --seed              "$SEED" \
            --repair-base-scale "$SCALE" \
            --output-dir        "$OUT_DIR" $MPC_FLAG; then
        echo "[$(ts)] DONE  n_train=${N_TRAIN}  ${NET}  $(elapsed $(( $(date +%s)-T0 )))"
        N_DONE=$(( N_DONE + 1 ))
    else
        rc=$?
        echo "[$(ts)] FAIL  n_train=${N_TRAIN}  ${NET}  (exit code $rc) — continuing to next size"
        N_FAIL=$(( N_FAIL + 1 ))
    fi
    echo ""
done

GLOBAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${NET} done.  $(elapsed $GLOBAL_ELAPSED)"
echo "  Completed : ${N_DONE}  |  Skipped : ${N_SKIP}  |  Failed : ${N_FAIL}"
echo "Results in ${ROOT}/${NET_DIR}/"
echo "============================================================"
