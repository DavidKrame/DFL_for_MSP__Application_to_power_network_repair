#!/usr/bin/env bash
# =============================================================================
# sensitivity_sweep.sh: (AR - DFL) / DFL across repair-cost scales s
#
# Two modes, both run from the repository root:
#   sbatch scripts/slurm/sensitivity/sensitivity_sweep.sh
#       -> SLURM array of 12 tasks, one per scale, run in parallel on Snellius
#   bash scripts/slurm/sensitivity/sensitivity_sweep.sh
#       -> all 12 scales one after another (local run, no SLURM needed)
#
# Each scale task runs toy (5-bus) and case14 (IEEE 14-bus) one after the other,
# each with n_train = 120000. Repair costs are set with
# --repair-base-scale, which gives each scale its own training-data cache,
# data_cache_scale<s>/<net>/seed42/ (data_cache/<net>/seed42/ for s = 1). The MIP
# labels depend on the repair costs, so scales must not share a cache; a single
# cache shared by the parallel array tasks would also be written by several jobs
# at once and could be corrupted.
#
# Scale table (x = mean of the two repair-cost bases / LOAD_SHEDDING_COST, in MWh):
#   ID | scale | x = 0.45 * scale
#    0 |   0.1 |   0.045
#    1 |   0.2 |   0.090
#    2 |   0.5 |   0.225
#    3 |   1.0 |   0.450   <- baseline (config.py defaults)
#    4 |   2.0 |   0.900
#    5 |   5.0 |   2.250
#    6 |  10.0 |   4.500
#    7 |  20.0 |   9.000
#    8 |  50.0 |  22.500
#    9 |  70.0 |  31.500
#   10 | 100.0 |  45.000
#   11 | 150.0 |  67.500
#
# Output layout:
#   experiments/sensitivity/scale_<s>/toy/evaluation.json
#   experiments/sensitivity/scale_<s>/case14/evaluation.json
#   (each next to config_snapshot.json, model.pt and the run's plots; the
#   training data stays in the cache above)
#
# Submit via:
#   bash scripts/slurm/sensitivity/submit_sensitivity.sh   # array + plot job after it
# =============================================================================
#SBATCH --job-name=dfl_sensitivity
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G     # 120k data-gen for both networks with 64 parallel Pyomo workers
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/dfl_sensitivity_%A_%a.out
#SBATCH --error=logs/dfl_sensitivity_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup: inside a SLURM job, load the Snellius modules and the
# project's .venv. A local run uses the Python environment it is started from.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    VENV_DIR="$PROJECT_DIR/.venv"
    module purge
    module load 2023
    module load Python/3.11.3-GCCcore-12.3.0
    source "$VENV_DIR/bin/activate"

    # Solver selection. SOLVER defaults to appsi_highs here, not gurobi as in the
    # unbudgeted/ and budgeted/ scripts: data generation runs one MIP worker per
    # CPU, which can be more simultaneous solves than a Gurobi license allows.
    # HiGHS gives the same training labels (DATA_MIP_REL_GAP = 1e-6); Gurobi pays
    # off only for the single large SAA and SDDiP solves. When SOLVER contains
    # "gurobi" the Gurobi module is loaded. ED_OPF_SOLVER is exported for the
    # Python code. Override at submission with
    #   sbatch --export=ALL,SOLVER=gurobi scripts/slurm/sensitivity/sensitivity_sweep.sh
    SOLVER="${SOLVER:-appsi_highs}"
    if [[ "$SOLVER" == *gurobi* ]]; then
        module load Gurobi/10.0.1-GCCcore-12.3.0
        # Use ~/gurobi.lic (e.g. a Gurobi WLS license) when present instead of the
        # cluster's default Gurobi license
        [[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"
    fi
    export ED_OPF_SOLVER="$SOLVER"
    mkdir -p "$PROJECT_DIR/logs"
    cd "$PROJECT_DIR"
else
    # Local run: the repository root is three levels above this script. The
    # solver follows ED_OPF_SOLVER from the calling environment (HiGHS if unset).
    PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
    cd "$PROJECT_DIR"
fi

export MPLBACKEND=Agg
PYTHON="python3"

# ---------------------------------------------------------------------------
# Scale table (SLURM_ARRAY_TASK_ID indexes SCALES; keep #SBATCH --array in sync)
# ---------------------------------------------------------------------------
SCALES=( 0.1  0.2  0.5  1.0  2.0  5.0  10.0  20.0  50.0  70.0  100.0  150.0 )

SEED=42
EVAL_SEED=1000
N_EVAL=500

ts()      { date '+%H:%M:%S'; }
elapsed() { printf '%dh%02dm%02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

# ---------------------------------------------------------------------------
# run_scale: run toy and case14 at one scale. GEN_BASE, LINE_BASE and
# X_RATIO are only printed; Pseudo_DFL_main.py sets the repair-cost bases
# (550 s, 350 s) from --repair-base-scale.
# ---------------------------------------------------------------------------
run_scale() {
    local SCALE=$1
    local GEN_BASE LINE_BASE X_RATIO
    GEN_BASE=$($PYTHON  -c "print(${SCALE} * 550)")
    LINE_BASE=$($PYTHON -c "print(${SCALE} * 350)")
    X_RATIO=$($PYTHON   -c "print(round(${SCALE} * 0.45, 4))")

    local OUTBASE="${PROJECT_DIR}/experiments/sensitivity/scale_${SCALE}"

    echo ""
    echo "============================================================"
    echo " scale=${SCALE}  x_ratio=${X_RATIO}"
    echo "   gen_repair_base=${GEN_BASE}  line_repair_base=${LINE_BASE}"
    echo "   (training labels already cached under data_cache*/ are reused)"
    echo " Started: $(date)"
    echo "============================================================"

    run_net "$SCALE" toy    120000 "$GEN_BASE" "$LINE_BASE" "$OUTBASE"
    run_net "$SCALE" case14 120000 "$GEN_BASE" "$LINE_BASE" "$OUTBASE"

    echo ""
    echo "[$(ts)] scale=${SCALE} — all networks done."
}

# ---------------------------------------------------------------------------
# run_net: full Pseudo_DFL_main.py pipeline (data, training, evaluation, plots)
# for one network at the given scale
# ---------------------------------------------------------------------------
run_net() {
    local SCALE=$1 NET=$2 N_TRAIN=$3 GEN_BASE=$4 LINE_BASE=$5 OUTBASE=$6
    local OUT_DIR="${OUTBASE}/${NET}"
    mkdir -p "$OUT_DIR"

    # Skip guard: a (network, scale) whose evaluation.json already exists is
    # skipped, so a resubmit resumes instead of redoing finished networks.
    # Deleting that evaluation.json reruns the pair; its training data then comes
    # from the cache.
    if [ -f "$OUT_DIR/evaluation.json" ]; then
        echo "[$(ts)] SKIP   ${NET}  scale=${SCALE}  (evaluation.json already exists)"
        return 0
    fi

    echo ""
    echo "[$(ts)] START  ${NET}  n=${N_TRAIN}  scale=${SCALE}"
    local T0
    T0=$(date +%s)

    $PYTHON Pseudo_DFL_main.py \
        --network          "$NET" \
        --n-train          "$N_TRAIN" \
        --seed             "$SEED" \
        --init-seeds       "$SEED" \
        --n-eval           "$N_EVAL" \
        --eval-seed        "$EVAL_SEED" \
        --output-dir       "$OUT_DIR" \
        --repair-base-scale "$SCALE"

    echo "[$(ts)] DONE   ${NET}  n=${N_TRAIN}  $(elapsed $(( $(date +%s)-T0 )))"
}

# ---------------------------------------------------------------------------
# Dispatch: one scale per SLURM array task, or all scales in sequence otherwise
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    run_scale "${SCALES[$SLURM_ARRAY_TASK_ID]}"
else
    echo "No SLURM_ARRAY_TASK_ID detected — running all ${#SCALES[@]} scales sequentially."
    for SCALE in "${SCALES[@]}"; do
        run_scale "$SCALE"
    done
    echo ""
    echo "[$(ts)] All sensitivity scales complete."
    echo "Run:  python3 scripts/sensitivity_plot.py"
fi
