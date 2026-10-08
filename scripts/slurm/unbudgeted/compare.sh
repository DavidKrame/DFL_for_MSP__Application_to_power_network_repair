#!/usr/bin/env bash
# =============================================================================
# compare.sh: comparison folders for the multiseed DFL results (no repair budget)
#
# Runs as a 3-task array (0-2), one task per result folder under
# experiments/paper_unbudgeted/dfl_scaling_multiseed/ (toy, case14 and
# case14_scale55), all in parallel. Each task calls run_comparison_folders.py
# --network <folder>, which runs compare_methods.py on the aggregated result of
# every training size and on each of its seeds and writes a comparison/ folder
# next to each evaluation.json.
#
# Run it after dfl_aggregate.sh, which writes the aggregated evaluation.json
# files; a training size without one gets only the per-seed comparisons.
#
# Submit from the repository root. Without SAA and SDDiP (DFL and the baselines
# stored in evaluation.json: AlwaysRepair, NoRepair, Near-sighted MPC):
#   sbatch scripts/slurm/unbudgeted/compare.sh
#
# With the SAA and SDDiP results of saa.sh and sddip.sh:
#   R=experiments/paper_unbudgeted
#   sbatch --export=ALL,SAA_ROOT=$R,SDDIP_ROOT=$R scripts/slurm/unbudgeted/compare.sh
#
# Chained after the aggregation job:
#   AGG_JOB=$(sbatch --parsable scripts/slurm/unbudgeted/dfl_aggregate.sh)
#   sbatch --dependency=afterok:$AGG_JOB scripts/slurm/unbudgeted/compare.sh
# =============================================================================
#SBATCH --job-name=paper_dfl_compare
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=28G
#SBATCH --array=0-2
#SBATCH --output=logs/unb_paper_dfl_compare_%A_%a.out
#SBATCH --error=logs/unb_paper_dfl_compare_%A_%a.err

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection. SOLVER (default gurobi) is exported as ED_OPF_SOLVER, which
# solver_config.py reads; a SOLVER containing "gurobi" also loads the Gurobi
# module. Here it sets the backend of the stage DC-OPF LPs in the rollouts that
# compare_methods.py simulates for its plots. Override at submission:
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/compare.sh
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

# Folder names under ROOT, one per array task. run_comparison_folders.py reads the
# network and the repair-cost scale s from the name (case14_scale55: case14, s = 55).
NETWORKS=(toy case14 case14_scale55)
NET="${NETWORKS[$SLURM_ARRAY_TASK_ID]}"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="unb_paper_dfl_compare_${NET}" || true

ROOT="experiments/paper_unbudgeted/dfl_scaling_multiseed"

# SAA_ROOT and SDDIP_ROOT come from sbatch --export (see the header). Left empty,
# the comparison has no SAA or SDDiP entries.
SAA_ROOT="${SAA_ROOT:-}"
SDDIP_ROOT="${SDDIP_ROOT:-}"

echo "============================================================"
echo "DFL comparison folder generation"
echo "  Task    : ${SLURM_ARRAY_TASK_ID}"
echo "  Network : ${NET}"
echo "  Root    : ${ROOT}"
echo "  SAA     : ${SAA_ROOT:-<not provided>}"
echo "  SDDiP   : ${SDDIP_ROOT:-<not provided>}"
echo "Started   : $(date)"
echo "============================================================"
echo ""

CMD=(
    $PYTHON scripts/slurm/unbudgeted/run_comparison_folders.py
    --root    "$ROOT"
    --network "$NET"
)
[ -n "$SAA_ROOT"   ] && CMD+=(--saa-root   "$SAA_ROOT")
[ -n "$SDDIP_ROOT" ] && CMD+=(--sddip-root "$SDDIP_ROOT")

"${CMD[@]}"

echo ""
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${NET} comparison done."
echo "============================================================"
