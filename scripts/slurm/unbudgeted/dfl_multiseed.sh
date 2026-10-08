#!/usr/bin/env bash
# =============================================================================
# dfl_multiseed.sh: DFL training-data scaling over five init seeds, no repair budget
#
# Repeats the dfl_scaling.sh sweep (same lanes, n_train sizes, training data and
# test set) for five init seeds, one array task per (lane, init seed). A lane is
# a network with its repair-cost scale s. The init seed only sets the initial
# weights of the neural network; the data seed stays 42 for all five.
#
# 15 tasks (array 0-14):
#   task_id // 5 -> lane index (0 = toy s=100, 1 = case14 s=45, 2 = case14 s=55)
#   task_id %  5 -> seed index (init seeds: 1 2 3 4 42)
# Tasks 0-4 run toy, 5-9 case14 at s=45 and 10-14 case14 at s=55.
#
# Each task trains and evaluates its init seed for all n_train sizes, writing
#   experiments/paper_unbudgeted/dfl_scaling_multiseed/<folder>/dfl_train<N>/seed_<s>/
# with <folder> = toy, case14 or case14_scale55. A size whose
# evaluation.json already exists is skipped, so a resubmitted job runs only the
# missing sizes.
#
# Run dfl_scaling.sh first. It fills the training-data cache that these tasks
# share, so they solve no MIPs.
#
# Submit all 15 tasks (from the repository root):
#   sbatch scripts/slurm/unbudgeted/dfl_multiseed.sh
#
# Submit a single task (e.g. case14 at s=45 with init seed 3 is task 7):
#   sbatch --array=7 scripts/slurm/unbudgeted/dfl_multiseed.sh
#
# Submit and chain the aggregation:
#   JOB=$(sbatch --parsable scripts/slurm/unbudgeted/dfl_multiseed.sh)
#   sbatch --dependency=afterok:$JOB scripts/slurm/unbudgeted/dfl_aggregate.sh
# =============================================================================
#SBATCH --job-name=paper_dfl_scaling_ms
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=117:36:00
#SBATCH --mem=112G
#SBATCH --array=0-14
#SBATCH --output=logs/unb_paper_dfl_scaling_ms_%A_%a.out
#SBATCH --error=logs/unb_paper_dfl_scaling_ms_%A_%a.err

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
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/dfl_multiseed.sh
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
# Decode the task ID into (lane, init seed)
# ---------------------------------------------------------------------------
# NET_IDX 2 (tasks 10-14) is a second case14 lane at scale 55, alongside case14
# at scale 45 in NET_IDX 1 (tasks 5-9). NET_DIRS routes it to a scale-tagged
# sibling directory so the two scales never share an output tree.
NETWORKS=(toy   case14 case14)
SCALES=(  100   45     55    )
NET_DIRS=( toy   case14 case14_scale55)
INIT_SEEDS=(1 2 3 4 42)

NET_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

NET="${NETWORKS[$NET_IDX]}"
SCALE="${SCALES[$NET_IDX]}"
NET_DIR="${NET_DIRS[$NET_IDX]}"
# Near-sighted MPC baseline. With the config.py default MPC_SCHEDULE_CAP_RATIO = 1
# it enumerates every repair schedule on toy; on case14, a stage with more
# schedules than the cap evaluates the no-repair schedule plus cap - 1 random ones.
# The result depends on neither n_train nor the init seed, so it is read from the
# cache that dfl_scaling.sh writes (and computed here if the cache is missing).
MPC_FLAG="--with-mpc"
SEED_VAL="${INIT_SEEDS[$SEED_IDX]}"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="unb_paper_dfl_ms_${NET_DIR}_s${SEED_VAL}" || true

N_TRAIN_SIZES=(25 100 250 500 1000 2000 3000 5000 10000 15000 30000 60000 80000 100000 120000)

# Data seed: failure model, training scenarios, train/val split, batch order and
# dropout. It is part of the cache path and the same for every init seed.
SEED=42
EVAL_SEED=1000
N_EVAL=500
ROOT="experiments/paper_unbudgeted/dfl_scaling_multiseed"

GLOBAL_START=$(date +%s)

echo "============================================================"
echo "DFL scaling experiment (multi-seed, 15-task array)"
echo "  Task       : ${SLURM_ARRAY_TASK_ID}"
echo "  Network    : ${NET}"
echo "  Scale      : ${SCALE}"
echo "  Init seed  : ${SEED_VAL}"
echo "  Sizes      : ${N_TRAIN_SIZES[*]}"
echo "  Output     : ${ROOT}/${NET_DIR}/dfl_train<N>/seed_${SEED_VAL}/"
echo "Node         : $(hostname)"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Python       : $($PYTHON --version)"
echo "Started      : $(date)"
echo "------------------------------------------------------------"
echo "Job array  : $SLURM_ARRAY_JOB_ID"
echo "Monitor    : squeue -j $SLURM_ARRAY_JOB_ID"
echo "Cancel all : scancel $SLURM_ARRAY_JOB_ID"
echo "Cancel this: scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "This log   : $PROJECT_DIR/logs/unb_paper_dfl_scaling_ms_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
N_DONE=0
N_SKIP=0
N_FAIL=0

for N_TRAIN in "${N_TRAIN_SIZES[@]}"; do
    OUT_DIR="${ROOT}/${NET_DIR}/dfl_train${N_TRAIN}"
    SEED_DIR="${OUT_DIR}/seed_${SEED_VAL}"
    mkdir -p "$SEED_DIR"

    if [ -f "${SEED_DIR}/evaluation.json" ]; then
        echo "[$(ts)] SKIP  n_train=${N_TRAIN}  ${NET}  seed=${SEED_VAL}"
        N_SKIP=$(( N_SKIP + 1 ))
        continue
    fi

    echo "[$(ts)] RUN   n_train=${N_TRAIN}  ${NET}  seed=${SEED_VAL}  [scale=${SCALE}]"
    T0=$(date +%s)

    # A single --init-seeds value makes Pseudo_DFL_main.py write straight into
    # SEED_DIR (several values would add another seed_<s>/ level).
    if $PYTHON Pseudo_DFL_main.py \
            --network           "$NET" \
            --n-train           "$N_TRAIN" \
            --n-eval            "$N_EVAL" \
            --eval-seed         "$EVAL_SEED" \
            --seed              "$SEED" \
            --repair-base-scale "$SCALE" \
            --init-seeds        "$SEED_VAL" \
            --output-dir        "$SEED_DIR" $MPC_FLAG; then
        echo "[$(ts)] DONE  n_train=${N_TRAIN}  ${NET}  seed=${SEED_VAL}  $(elapsed $(( $(date +%s)-T0 )))"
        N_DONE=$(( N_DONE + 1 ))
    else
        rc=$?
        echo "[$(ts)] FAIL  n_train=${N_TRAIN}  ${NET}  seed=${SEED_VAL}  (exit code $rc) — continuing"
        N_FAIL=$(( N_FAIL + 1 ))
    fi
    echo ""
done

GLOBAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${NET} seed=${SEED_VAL} done.  $(elapsed $GLOBAL_ELAPSED)"
echo "  Completed : ${N_DONE}  |  Skipped : ${N_SKIP}  |  Failed : ${N_FAIL}"
echo "Results in ${ROOT}/${NET_DIR}/"
echo "============================================================"
